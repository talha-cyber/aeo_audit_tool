import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.normalizer import BrandNormalizer, NormalizationResult
from ..core.sentiment import SentimentAnalyzer
from ..core.similarity import SimilarityEngine
from ..market_adapters.base import BaseMarketAdapter, MarketAdapterFactory
from ..models.brand_mention import (
    BrandContext,
    BrandMention,
    DetectionMethod,
    DetectionResult,
)
from ..utils.cache_manager import get_global_cache
from ..utils.performance import performance_monitor

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration for brand detection operations"""

    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.8
    max_context_window: int = 200
    enable_fuzzy_matching: bool = True
    enable_semantic_similarity: bool = True
    enable_sentiment_analysis: bool = True
    market_code: str = "DE"
    language_code: str = "de"
    max_mentions_per_brand: int = 50
    enable_caching: bool = True
    cache_ttl: int = 1800  # 30 minutes


class BrandDetectionEngine:
    """Advanced multi-modal brand detection engine"""

    def __init__(self, openai_api_key: str, config: DetectionConfig = None):
        self.config = config or DetectionConfig()

        # Initialize components
        self.similarity_engine = SimilarityEngine(openai_api_key)
        self.normalizer = BrandNormalizer()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Performance tracking
        self.detection_stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    @performance_monitor()
    async def detect_brands(
        self, text: str, target_brands: List[str], config: DetectionConfig = None
    ) -> DetectionResult:
        """Main brand detection method"""

        detection_config = config or self.config
        start_time = time.perf_counter()

        # Input validation
        if not text or not target_brands:
            return DetectionResult(
                text_analyzed="",
                language=detection_config.language_code,
                processing_time_ms=0,
                mentions=[],
                total_brands_found=0,
                confidence_threshold=detection_config.confidence_threshold,
                market_adapter_used=detection_config.market_code,
            )

        # Check cache
        cache_key = self._generate_cache_key(text, target_brands, detection_config)
        if detection_config.enable_caching:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.detection_stats["cache_hits"] += 1
                return cached_result
            self.detection_stats["cache_misses"] += 1

        try:
            # Get market adapter
            market_adapter = MarketAdapterFactory.create_adapter(
                detection_config.market_code
            )

            # Preprocess text
            preprocessed_text = market_adapter.preprocess_text(text)

            # Normalize all target brands
            normalized_brands = await self._normalize_brands(
                target_brands, market_adapter
            )

            # Run detection methods in parallel
            detection_tasks = []

            if detection_config.enable_fuzzy_matching:
                detection_tasks.append(
                    self._detect_with_fuzzy_matching(
                        preprocessed_text, normalized_brands, market_adapter
                    )
                )

            if detection_config.enable_semantic_similarity:
                detection_tasks.append(
                    self._detect_with_semantic_similarity(
                        preprocessed_text, normalized_brands, market_adapter
                    )
                )

            # Execute detection methods
            detection_results = await asyncio.gather(
                *detection_tasks, return_exceptions=True
            )

            # Merge and deduplicate results
            all_mentions = []
            for result in detection_results:
                if isinstance(result, list):
                    all_mentions.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Detection method failed: {result}")

            # Deduplicate mentions
            unique_mentions = self._deduplicate_mentions(all_mentions, detection_config)

            # Enhance with sentiment analysis
            if detection_config.enable_sentiment_analysis:
                enhanced_mentions = await self._enhance_with_sentiment(
                    unique_mentions, preprocessed_text, detection_config
                )
            else:
                enhanced_mentions = unique_mentions

            # Filter by confidence threshold
            filtered_mentions = [
                mention
                for mention in enhanced_mentions
                if mention.confidence >= detection_config.confidence_threshold
            ]

            # Limit mentions per brand
            limited_mentions = self._limit_mentions_per_brand(
                filtered_mentions, detection_config.max_mentions_per_brand
            )

            # Create result
            processing_time = (time.perf_counter() - start_time) * 1000

            result = DetectionResult(
                text_analyzed=text,
                language=detection_config.language_code,
                processing_time_ms=processing_time,
                mentions=limited_mentions,
                total_brands_found=len(limited_mentions),
                confidence_threshold=detection_config.confidence_threshold,
                market_adapter_used=detection_config.market_code,
            )

            # Cache result
            if detection_config.enable_caching:
                self._cache_result(cache_key, result, detection_config.cache_ttl)

            self.detection_stats["successful_detections"] += 1
            self.detection_stats["total_detections"] += 1

            return result

        except Exception as e:
            logger.error(f"Brand detection failed: {e}")
            self.detection_stats["failed_detections"] += 1
            self.detection_stats["total_detections"] += 1

            processing_time = (time.perf_counter() - start_time) * 1000
            return DetectionResult(
                text_analyzed=text,
                language=detection_config.language_code,
                processing_time_ms=processing_time,
                mentions=[],
                total_brands_found=0,
                confidence_threshold=detection_config.confidence_threshold,
                market_adapter_used=detection_config.market_code,
            )

    async def _normalize_brands(
        self, brands: List[str], market_adapter: BaseMarketAdapter
    ) -> Dict[str, NormalizationResult]:
        """Normalize all target brands"""
        normalized = {}

        for brand in brands:
            try:
                result = self.normalizer.normalize_comprehensive(
                    brand, market_adapter.market_code
                )
                normalized[brand] = result
            except Exception as e:
                logger.error(f"Normalization failed for brand {brand}: {e}")
                # Fallback normalization
                normalized[brand] = NormalizationResult(
                    original=brand,
                    normalized_forms={brand, brand.lower()},
                    primary_form=brand,
                    confidence=0.5,
                    market_specific_forms={},
                    metadata={"error": str(e)},
                )

        return normalized

    @performance_monitor()
    async def _detect_with_fuzzy_matching(
        self,
        text: str,
        normalized_brands: Dict[str, NormalizationResult],
        market_adapter: BaseMarketAdapter,
    ) -> List[BrandMention]:
        """Detect brands using fuzzy string matching"""

        mentions = []

        try:
            # Extract potential brand candidates from text
            candidates = self._extract_brand_candidates(text, market_adapter)

            for brand, norm_result in normalized_brands.items():
                all_brand_forms = norm_result.normalized_forms

                for candidate in candidates:
                    best_match = None
                    best_score = 0.0

                    # Compare candidate against all normalized forms
                    for brand_form in all_brand_forms:
                        similarity = self.similarity_engine.calculate_fuzzy_similarity(
                            candidate["text"], brand_form
                        )

                        if similarity.score > best_score:
                            best_score = similarity.score
                            best_match = (brand_form, similarity)

                    # Check if match is good enough
                    if best_match and best_score >= self.config.similarity_threshold:
                        # Additional validation
                        if market_adapter.validate_brand_mention(
                            candidate["text"],
                            candidate["context"],
                            DetectionMethod.FUZZY,
                        ):
                            mention = BrandMention(
                                brand=brand,
                                original_text=candidate["text"],
                                confidence=best_score,
                                detection_method=DetectionMethod.FUZZY,
                                language=market_adapter.language_code,
                            )

                            # Add context
                            context = BrandContext(
                                text=candidate["context"],
                                start_position=candidate["start"],
                                end_position=candidate["end"],
                                sentence=candidate["sentence"],
                            )
                            mention.add_context(context)

                            mentions.append(mention)

        except Exception as e:
            logger.error(f"Fuzzy matching detection failed: {e}")

        return mentions

    @performance_monitor()
    async def _detect_with_semantic_similarity(
        self,
        text: str,
        normalized_brands: Dict[str, NormalizationResult],
        market_adapter: BaseMarketAdapter,
    ) -> List[BrandMention]:
        """Detect brands using semantic similarity"""

        mentions = []

        try:
            # Extract potential brand candidates
            candidates = self._extract_brand_candidates(text, market_adapter)

            # Batch process for efficiency
            batch_size = 20  # Process 20 candidates at a time

            for i in range(0, len(candidates), batch_size):
                batch_candidates = candidates[i : i + batch_size]

                # Create a list of all comparisons to be made in this batch
                comparison_pairs = []
                for brand, norm_result in normalized_brands.items():
                    brand_forms = list(norm_result.normalized_forms)[
                        :3
                    ]  # Limit to top 3 forms
                    for candidate in batch_candidates:
                        for brand_form in brand_forms:
                            comparison_pairs.append(
                                {
                                    "brand": brand,
                                    "candidate": candidate,
                                    "brand_form": brand_form,
                                }
                            )

                # Run batch similarity check
                if not comparison_pairs:
                    continue

                texts1 = [p["candidate"]["text"] for p in comparison_pairs]
                texts2 = [p["brand_form"] for p in comparison_pairs]

                # Get batch of similarity results
                similarity_results = (
                    await self.similarity_engine.calculate_batch_hybrid_similarity(
                        texts1, texts2
                    )
                )

                for i, similarity in enumerate(similarity_results):
                    pair_info = comparison_pairs[i]
                    candidate = pair_info["candidate"]
                    brand = pair_info["brand"]

                    if (
                        similarity
                        and similarity.score >= self.config.similarity_threshold
                        and market_adapter.validate_brand_mention(
                            candidate["text"],
                            candidate["context"],
                            DetectionMethod.SEMANTIC,
                        )
                    ):
                        mention = BrandMention(
                            brand=brand,
                            original_text=candidate["text"],
                            confidence=similarity.score,
                            detection_method=DetectionMethod.SEMANTIC,
                            language=market_adapter.language_code,
                        )

                        # Add context with semantic metadata
                        context = BrandContext(
                            text=candidate["context"],
                            start_position=candidate["start"],
                            end_position=candidate["end"],
                            sentence=candidate["sentence"],
                        )
                        mention.add_context(context)

                        # Store semantic similarity metadata
                        mention.market_specific_data["semantic_similarity"] = {
                            "method": similarity.method,
                            "confidence": similarity.confidence,
                            "metadata": similarity.metadata,
                        }

                        mentions.append(mention)

        except Exception as e:
            logger.error(f"Semantic similarity detection failed: {e}")

        return mentions

    def _extract_brand_candidates(
        self, text: str, market_adapter: BaseMarketAdapter
    ) -> List[Dict[str, Any]]:
        """Extract potential brand candidates from text"""
        candidates = []

        # Pattern-based extraction
        patterns = [
            # Capitalized words (potential brand names)
            r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b",
            # Words with specific suffixes
            r"\b\w+(?:GmbH|AG|Inc|Corp|LLC|Ltd)\b",
            # Quoted strings (often brand names)
            r'"([^"]+)"',
            r"'([^']+)'",
            # CamelCase words
            r"\b[a-z]+[A-Z][a-zA-Z]*\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                candidate_text = match.group()

                # Skip if too short or too long
                if len(candidate_text.strip()) < 2 or len(candidate_text) > 50:
                    continue

                # Extract context
                start = match.start()
                end = match.end()
                context_start = max(0, start - self.config.max_context_window // 2)
                context_end = min(len(text), end + self.config.max_context_window // 2)
                context = text[context_start:context_end]

                # Extract sentence
                sentence = self._extract_sentence(text, start, end)

                candidates.append(
                    {
                        "text": candidate_text.strip(),
                        "start": start,
                        "end": end,
                        "context": context,
                        "sentence": sentence,
                    }
                )

        # Deduplicate candidates
        unique_candidates = []
        seen_texts = set()

        for candidate in candidates:
            if candidate["text"].lower() not in seen_texts:
                unique_candidates.append(candidate)
                seen_texts.add(candidate["text"].lower())

        return unique_candidates

    def _extract_sentence(self, text: str, start: int, end: int) -> str:
        """Extract the sentence containing the brand mention"""
        # Find sentence boundaries
        sentence_start = start
        sentence_end = end

        # Look backwards for sentence start
        for i in range(start - 1, -1, -1):
            if text[i] in ".!?":
                sentence_start = i + 1
                break

        # Look forwards for sentence end
        for i in range(end, len(text)):
            if text[i] in ".!?":
                sentence_end = i + 1
                break

        return text[sentence_start:sentence_end].strip()

    def _deduplicate_mentions(
        self, mentions: List[BrandMention], config: DetectionConfig
    ) -> List[BrandMention]:
        """Remove duplicate brand mentions"""
        if not mentions:
            return []

        # Group by brand and original text
        groups = {}
        for mention in mentions:
            key = (mention.brand, mention.original_text.lower())
            if key not in groups:
                groups[key] = []
            groups[key].append(mention)

        deduplicated = []
        for mention_group in groups.values():
            # Keep the mention with highest confidence
            best_mention = max(mention_group, key=lambda m: m.confidence)

            # Use the context from the best mention, but count all occurrences
            all_contexts = []
            for mention in mention_group:
                all_contexts.extend(mention.contexts)

            # Deduplicate all found contexts by their position
            unique_contexts = []
            seen_positions = set()
            for context in all_contexts:
                pos_key = (context.start_position, context.end_position)
                if pos_key not in seen_positions:
                    unique_contexts.append(context)
                    seen_positions.add(pos_key)

            # Assign the unique contexts from all occurrences to the best mention
            best_mention.contexts = unique_contexts
            best_mention.mention_count = len(unique_contexts)

            deduplicated.append(best_mention)

        return deduplicated

    async def _enhance_with_sentiment(
        self, mentions: List[BrandMention], text: str, config: DetectionConfig
    ) -> List[BrandMention]:
        """Enhance mentions with sentiment analysis"""

        enhanced_mentions = []

        for mention in mentions:
            try:
                # Analyze sentiment for each context
                context_sentiments = []

                for context in mention.contexts:
                    sentiment_result = (
                        await self.sentiment_analyzer.analyze_sentiment_hybrid(
                            context.text, mention.brand, config.language_code
                        )
                    )
                    context_sentiments.append(sentiment_result)

                # Calculate overall sentiment
                if context_sentiments:
                    avg_score = sum(s.score for s in context_sentiments) / len(
                        context_sentiments
                    )

                    # Determine dominant polarity
                    polarity_counts = {}
                    for sentiment in context_sentiments:
                        polarity_counts[sentiment.polarity] = (
                            polarity_counts.get(sentiment.polarity, 0) + 1
                        )

                    dominant_polarity = max(
                        polarity_counts.keys(), key=lambda k: polarity_counts[k]
                    )

                    mention.sentiment_score = avg_score
                    mention.sentiment_polarity = dominant_polarity

                    # Store detailed sentiment data
                    mention.market_specific_data["sentiment_analysis"] = {
                        "individual_contexts": [s.__dict__ for s in context_sentiments],
                        "polarity_distribution": {
                            k.value: v for k, v in polarity_counts.items()
                        },
                    }

                enhanced_mentions.append(mention)

            except Exception as e:
                logger.error(
                    f"Sentiment enhancement failed for mention {mention.brand}: {e}"
                )
                enhanced_mentions.append(mention)  # Keep original mention

        return enhanced_mentions

    def _limit_mentions_per_brand(
        self, mentions: List[BrandMention], max_per_brand: int
    ) -> List[BrandMention]:
        """Limit number of mentions per brand"""
        brand_counts = {}
        limited_mentions = []

        # Sort by relevance score (best first)
        sorted_mentions = sorted(
            mentions, key=lambda m: m.calculate_relevance_score(), reverse=True
        )

        for mention in sorted_mentions:
            brand_count = brand_counts.get(mention.brand, 0)

            if brand_count < max_per_brand:
                limited_mentions.append(mention)
                brand_counts[mention.brand] = brand_count + 1

        return limited_mentions

    def _generate_cache_key(
        self, text: str, brands: List[str], config: DetectionConfig
    ) -> str:
        """Generate cache key for detection result"""
        import hashlib

        # Create stable hash from inputs
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        brands_str = ",".join(sorted(b.lower() for b in brands))
        config_str = f"{config.confidence_threshold}:{config.similarity_threshold}:{config.market_code}"

        content = f"detect:{config_str}:{brands_str}:{text_hash}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[DetectionResult]:
        """Get cached detection result"""
        cache = get_global_cache()
        if cache:
            return cache.get(cache_key)
        return None

    def _cache_result(self, cache_key: str, result: DetectionResult, ttl: int):
        """Cache detection result"""
        cache = get_global_cache()
        if cache:
            cache.set(cache_key, result, ttl)

    async def detect_brands_batch(
        self, texts: List[str], target_brands: List[str], config: DetectionConfig = None
    ) -> List[DetectionResult]:
        """Detect brands in multiple texts efficiently"""

        if not texts or not target_brands:
            return []

        detection_config = config or self.config

        # Create detection tasks
        tasks = [
            self.detect_brands(text, target_brands, detection_config) for text in texts
        ]

        # Execute in batches to avoid overwhelming the system
        batch_size = 5
        results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, DetectionResult):
                    results.append(result)
                else:
                    logger.error(f"Batch detection failed: {result}")
                    # Add empty result for failed detection
                    results.append(
                        DetectionResult(
                            text_analyzed="",
                            language=detection_config.language_code,
                            processing_time_ms=0,
                            mentions=[],
                            total_brands_found=0,
                            confidence_threshold=detection_config.confidence_threshold,
                            market_adapter_used=detection_config.market_code,
                        )
                    )

        return results

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection engine statistics"""
        total = self.detection_stats["total_detections"]
        success_rate = (
            (self.detection_stats["successful_detections"] / total * 100)
            if total > 0
            else 0
        )
        cache_hit_rate = (
            (
                self.detection_stats["cache_hits"]
                / (
                    self.detection_stats["cache_hits"]
                    + self.detection_stats["cache_misses"]
                )
                * 100
            )
            if (
                self.detection_stats["cache_hits"]
                + self.detection_stats["cache_misses"]
            )
            > 0
            else 0
        )

        return {
            "detection_stats": self.detection_stats,
            "success_rate_percent": success_rate,
            "cache_hit_rate_percent": cache_hit_rate,
            "similarity_engine_stats": self.similarity_engine.get_cache_stats(),
            "normalizer_stats": self.normalizer.get_normalization_stats(),
        }

    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("Brand detection engine cleaned up")


class BrandDetectionOrchestrator:
    """High-level orchestrator for brand detection operations"""

    def __init__(self, openai_api_key: str):
        self.detection_engine = BrandDetectionEngine(openai_api_key)
        self.configs = {
            "high_precision": DetectionConfig(
                confidence_threshold=0.8,
                similarity_threshold=0.85,
                enable_fuzzy_matching=True,
                enable_semantic_similarity=True,
                max_mentions_per_brand=20,
            ),
            "high_recall": DetectionConfig(
                confidence_threshold=0.6,
                similarity_threshold=0.75,
                enable_fuzzy_matching=True,
                enable_semantic_similarity=True,
                max_mentions_per_brand=50,
            ),
            "fast": DetectionConfig(
                confidence_threshold=0.7,
                similarity_threshold=0.8,
                enable_fuzzy_matching=True,
                enable_semantic_similarity=False,  # Faster without semantic
                max_mentions_per_brand=30,
                enable_caching=True,
            ),
        }

    async def detect_with_profile(
        self,
        text: str,
        target_brands: List[str],
        profile: str = "high_precision",
        market_code: str = "DE",
    ) -> DetectionResult:
        """Detect brands using predefined profile"""

        if profile not in self.configs:
            profile = "high_precision"

        config = self.configs[profile]
        config.market_code = market_code
        config.language_code = "de" if market_code == "DE" else "en"

        return await self.detection_engine.detect_brands(text, target_brands, config)

    async def detect_competitive_analysis(
        self,
        text: str,
        client_brand: str,
        competitors: List[str],
        market_code: str = "DE",
    ) -> Dict[str, Any]:
        """Perform competitive brand analysis"""

        all_brands = [client_brand] + competitors

        # Use high precision for competitive analysis
        result = await self.detect_with_profile(
            text, all_brands, "high_precision", market_code
        )

        # Analyze competitive context
        competitive_analysis = {
            "client_brand": client_brand,
            "competitors": competitors,
            "total_mentions": result.total_brands_found,
            "brand_breakdown": {},
            "competitive_insights": [],
            "sentiment_comparison": {},
        }

        # Break down by brand
        for brand in all_brands:
            brand_mentions = [m for m in result.mentions if m.brand == brand]

            if brand_mentions:
                avg_sentiment = sum(m.sentiment_score for m in brand_mentions) / len(
                    brand_mentions
                )

                competitive_analysis["brand_breakdown"][brand] = {
                    "mention_count": len(brand_mentions),
                    "average_confidence": sum(m.confidence for m in brand_mentions)
                    / len(brand_mentions),
                    "average_sentiment": avg_sentiment,
                    "detection_methods": list(
                        set(m.detection_method.value for m in brand_mentions)
                    ),
                    "contexts": [
                        m.contexts[0].text[:100] + "..." if m.contexts else ""
                        for m in brand_mentions[:3]
                    ],
                }

                competitive_analysis["sentiment_comparison"][brand] = avg_sentiment

        # Generate insights
        client_mentions = len([m for m in result.mentions if m.brand == client_brand])
        competitor_mentions = len(
            [m for m in result.mentions if m.brand in competitors]
        )

        if client_mentions + competitor_mentions > 0:
            client_share = (
                client_mentions / (client_mentions + competitor_mentions) * 100
            )
            competitive_analysis["competitive_insights"].append(
                f"{client_brand} has {client_share:.1f}% of brand mentions"
            )

        # Find top competitor
        competitor_counts = {
            comp: len([m for m in result.mentions if m.brand == comp])
            for comp in competitors
        }

        if competitor_counts:
            top_competitor = max(
                competitor_counts.keys(), key=lambda k: competitor_counts[k]
            )
            competitive_analysis["competitive_insights"].append(
                f"Top competitor: {top_competitor} with {competitor_counts[top_competitor]} mentions"
            )

        return competitive_analysis

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "detection_engine_stats": self.detection_engine.get_detection_statistics(),
            "available_profiles": list(self.configs.keys()),
            "profile_configs": {
                name: {
                    "confidence_threshold": config.confidence_threshold,
                    "similarity_threshold": config.similarity_threshold,
                    "max_mentions_per_brand": config.max_mentions_per_brand,
                    "enabled_methods": [
                        method
                        for method, enabled in [
                            ("fuzzy_matching", config.enable_fuzzy_matching),
                            ("semantic_similarity", config.enable_semantic_similarity),
                        ]
                        if enabled
                    ],
                }
                for name, config in self.configs.items()
            },
        }
