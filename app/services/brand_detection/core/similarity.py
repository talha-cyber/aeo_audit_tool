import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import openai
from cachetools import TTLCache
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of similarity comparison"""

    score: float
    method: str
    confidence: float
    metadata: Dict[str, any] = None


class SimilarityEngine:
    """Multi-modal similarity engine for brand matching"""

    def __init__(self, openai_api_key: str, similarity_threshold: float = 0.7):
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = 85
        self.fuzzy_weight = 0.4
        self.semantic_weight = 0.6

        # Cache for embeddings to avoid repeated API calls
        self.embedding_cache = TTLCache(maxsize=1024, ttl=3600)

    async def _get_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts, using cache"""

        # Identify which texts are not in cache
        texts_to_fetch = [t for t in texts if t not in self.embedding_cache]

        if texts_to_fetch:
            try:
                # Fetch embeddings for new texts from OpenAI
                response = await self.openai_client.embeddings.create(
                    input=texts_to_fetch, model="text-embedding-3-small"
                )

                # Store new embeddings in cache
                for i, text in enumerate(texts_to_fetch):
                    self.embedding_cache[text] = response.data[i].embedding

            except Exception as e:
                logger.error(f"Failed to get embeddings for batch: {e}")
                # On failure, return empty embeddings for all texts in batch
                for text in texts_to_fetch:
                    self.embedding_cache[text] = []

        # Return embeddings for all original texts from cache
        return [self.embedding_cache.get(t, []) for t in texts]

    async def _get_semantic_similarity_batch(
        self, texts1: List[str], texts2: List[str]
    ) -> List[float]:
        """Calculate semantic similarity for a batch of text pairs"""

        if not texts1 or not texts2:
            return []

        # Get all embeddings in one go
        all_texts = list(set(texts1 + texts2))
        embeddings_dict = {
            text: embedding
            for text, embedding in zip(
                all_texts, await self._get_embedding_batch(all_texts)
            )
        }

        # Calculate cosine similarity for each pair
        similarities = []
        for t1, t2 in zip(texts1, texts2):
            emb1 = embeddings_dict.get(t1)
            emb2 = embeddings_dict.get(t2)

            if emb1 and emb2:
                # Convert to numpy arrays for calculation
                emb1_np = np.array(emb1)
                emb2_np = np.array(emb2)

                # Calculate cosine similarity
                cosine_sim = np.dot(emb1_np, emb2_np) / (
                    np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np)
                )
                similarities.append(float(cosine_sim))
            else:
                similarities.append(0.0)  # Could not compute similarity

        return similarities

    def calculate_fuzzy_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """Calculate fuzzy string similarity"""

        # Multiple fuzzy matching algorithms
        ratio_score = fuzz.ratio(text1.lower(), text2.lower()) / 100
        partial_ratio = fuzz.partial_ratio(text1.lower(), text2.lower()) / 100
        token_sort = fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100
        token_set = fuzz.token_set_ratio(text1.lower(), text2.lower()) / 100

        # Weighted combination
        weights = [0.3, 0.2, 0.25, 0.25]
        scores = [ratio_score, partial_ratio, token_sort, token_set]
        final_score = sum(w * s for w, s in zip(weights, scores))

        # Confidence based on score consistency
        score_variance = np.var(scores)
        confidence = max(0.0, 1.0 - score_variance)

        return SimilarityResult(
            score=final_score,
            method="fuzzy_matching",
            confidence=confidence,
            metadata={
                "ratio": ratio_score,
                "partial_ratio": partial_ratio,
                "token_sort": token_sort,
                "token_set": token_set,
                "variance": score_variance,
            },
        )

    async def calculate_semantic_similarity(
        self, text1: str, text2: str
    ) -> SimilarityResult:
        """Calculate semantic similarity using OpenAI embeddings"""
        try:
            emb1 = await self.get_openai_embedding(text1)
            emb2 = await self.get_openai_embedding(text2)

            if emb1 and emb2:
                cosine_sim = self._cosine_similarity(emb1, emb2)
                return SimilarityResult(
                    score=cosine_sim,
                    method="openai_embedding",
                    confidence=0.9,  # High confidence for OpenAI embeddings
                    metadata={"embedding_model": "text-embedding-3-small"},
                )
        except Exception as e:
            logger.error(f"Semantic similarity failed: {e}")

        # Fallback to fuzzy matching
        return self.calculate_fuzzy_similarity(text1, text2)

    async def calculate_batch_hybrid_similarity(
        self, texts1: List[str], texts2: List[str]
    ) -> List[Optional[SimilarityResult]]:
        """Calculate hybrid similarity for a batch of text pairs"""
        if not texts1 or not texts2 or len(texts1) != len(texts2):
            return [None] * len(texts1)

        try:
            # Get semantic similarity scores in a batch
            semantic_scores = await self._get_semantic_similarity_batch(texts1, texts2)

            results = []
            for i in range(len(texts1)):
                fuzzy_result = self.calculate_fuzzy_similarity(texts1[i], texts2[i])
                semantic_score = semantic_scores[i] if semantic_scores else 0.0

                # Combine scores
                final_score = (self.fuzzy_weight * fuzzy_result.score) + (
                    self.semantic_weight * semantic_score
                )

                results.append(
                    SimilarityResult(
                        score=final_score,
                        method="hybrid",
                        confidence=max(fuzzy_result.confidence, semantic_score),
                        metadata={
                            "fuzzy_score": fuzzy_result.score,
                            "semantic_score": semantic_score,
                        },
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Batch hybrid similarity calculation failed: {e}")
            # Fallback to individual fuzzy scores for the whole batch
            return [
                self.calculate_fuzzy_similarity(t1, t2)
                for t1, t2 in zip(texts1, texts2)
            ]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            return float(dot_product / (norm_v1 * norm_v2))
        except Exception as e:
            logger.error(f"Cosine similarity calculation error: {e}")
            return 0.0

    def get_cache_stats(self) -> Dict[str, int]:
        """Get embedding cache statistics"""
        return {
            "similarity_threshold": self.similarity_threshold,
            "fuzzy_threshold": self.fuzzy_threshold,
            "embedding_cache_size": len(self.embedding_cache),
        }
