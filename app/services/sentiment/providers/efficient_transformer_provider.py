"""
Efficient Transformer-based Sentiment Analysis Provider.

Cost-optimized transformer provider with quantization, model caching,
batch processing, and efficient inference strategies for running locally at minimal cost.
"""

import asyncio
import hashlib
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from app.utils.logger import get_logger

from ..core.models import (
    AnalysisContext,
    ContextType,
    SentimentMethod,
    SentimentPolarity,
    SentimentResult,
)
from .base import ProviderConfigurationError, SentimentProvider, SentimentProviderError

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = get_logger(__name__)


class ModelCache:
    """Efficient model caching with disk persistence"""

    def __init__(self, cache_dir: str = "model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache = {}

    def _get_cache_key(self, model_name: str, quantization: bool = False) -> str:
        """Generate cache key for model"""
        key_data = f"{model_name}_{quantization}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_cached_model_path(
        self, model_name: str, quantization: bool = False
    ) -> Optional[Path]:
        """Check if model is cached locally"""
        cache_key = self._get_cache_key(model_name, quantization)
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            return cache_path
        return None

    def cache_model(
        self, model_name: str, model, tokenizer, quantization: bool = False
    ):
        """Cache model and tokenizer to disk"""
        try:
            cache_key = self._get_cache_key(model_name, quantization)
            cache_path = self.cache_dir / cache_key
            cache_path.mkdir(exist_ok=True)

            # Save model and tokenizer
            model.save_pretrained(cache_path / "model")
            tokenizer.save_pretrained(cache_path / "tokenizer")

            # Save metadata
            metadata = {
                "model_name": model_name,
                "quantization": quantization,
                "cached_at": torch.utils.data.datetime.datetime.now().isoformat(),
            }

            with open(cache_path / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"Cached model {model_name} to {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to cache model {model_name}: {e}")

    def load_cached_model(self, model_name: str, quantization: bool = False):
        """Load model and tokenizer from cache"""
        cache_path = self.get_cached_model_path(model_name, quantization)
        if not cache_path:
            return None, None

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(cache_path / "tokenizer")

            # Load model
            if quantization:
                # Load with quantization config
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    cache_path / "model",
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    cache_path / "model"
                )

            logger.info(f"Loaded cached model {model_name} from {cache_path}")
            return model, tokenizer

        except Exception as e:
            logger.warning(f"Failed to load cached model {model_name}: {e}")
            return None, None


class EfficientTransformerProvider(SentimentProvider):
    """
    Cost-optimized transformer provider with multiple efficiency strategies:

    1. Model quantization (4-bit, 8-bit)
    2. Intelligent caching
    3. Batch processing optimization
    4. CPU-optimized inference
    5. Model size selection
    6. Memory management
    """

    # Cost-effective model recommendations by use case
    COST_EFFECTIVE_MODELS = {
        "ultra_small": {
            "model": "papluca/xlm-roberta-base-language-detection",  # Very small
            "description": "Minimal model for basic sentiment",
            "memory_mb": 50,
            "speed": "very_fast",
        },
        "small": {
            "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "description": "Good balance of accuracy and speed",
            "memory_mb": 150,
            "speed": "fast",
        },
        "efficient": {
            "model": "microsoft/DialoGPT-medium",
            "description": "Medium accuracy with good efficiency",
            "memory_mb": 300,
            "speed": "medium",
        },
        "accurate": {
            "model": "nlptown/bert-base-multilingual-uncased-sentiment",
            "description": "High accuracy, multilingual",
            "memory_mb": 500,
            "speed": "slow",
        },
    }

    def __init__(
        self,
        model_size: str = "small",  # ultra_small, small, efficient, accurate
        quantization: str = "none",  # none, 8bit, 4bit
        device: str = "auto",
        max_length: int = 256,  # Reduced for efficiency
        batch_size: int = 8,  # Smaller batches for memory efficiency
        enable_caching: bool = True,
        cache_dir: str = "sentiment_model_cache",
    ):
        if model_size not in self.COST_EFFECTIVE_MODELS:
            model_size = "small"

        self.model_config = self.COST_EFFECTIVE_MODELS[model_size]
        model_name = self.model_config["model"]

        super().__init__(
            SentimentMethod.TRANSFORMER, f"EfficientTransformer({model_size})"
        )

        self.model_name = model_name
        self.model_size = model_size
        self.quantization = quantization
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.enable_caching = enable_caching

        # Initialize cache
        self.cache = ModelCache(cache_dir) if enable_caching else None

        # Model components
        self.tokenizer = None
        self.model = None
        self.device_obj = None

        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.memory_usage_mb = 0

        # Label mappings
        self.label_mappings = {
            "LABEL_0": SentimentPolarity.NEGATIVE,
            "LABEL_1": SentimentPolarity.NEUTRAL,
            "LABEL_2": SentimentPolarity.POSITIVE,
            "NEGATIVE": SentimentPolarity.NEGATIVE,
            "NEUTRAL": SentimentPolarity.NEUTRAL,
            "POSITIVE": SentimentPolarity.POSITIVE,
            "0": SentimentPolarity.NEGATIVE,
            "1": SentimentPolarity.NEUTRAL,
            "2": SentimentPolarity.POSITIVE,
        }

    async def initialize(self) -> None:
        """Initialize with cost optimization strategies"""
        try:
            logger.info(f"Initializing efficient transformer: {self.model_name}")
            logger.info(
                f"Configuration: size={self.model_size}, quantization={self.quantization}"
            )

            # Setup device with cost considerations
            self.device_obj = self._setup_cost_effective_device()

            # Try to load from cache first
            if self.cache:
                model, tokenizer = self.cache.load_cached_model(
                    self.model_name, self.quantization != "none"
                )
                if model and tokenizer:
                    self.model = model
                    self.tokenizer = tokenizer
                    logger.info("Loaded model from cache")
                    await self._setup_model()
                    return

            # Load model with optimization
            await self._load_optimized_model()

            # Cache the loaded model
            if self.cache and self.model and self.tokenizer:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.cache.cache_model(
                        self.model_name,
                        self.model,
                        self.tokenizer,
                        self.quantization != "none",
                    ),
                )

            await self._setup_model()

            # Log resource usage
            self._log_resource_usage()

            logger.info("Efficient transformer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize efficient transformer: {e}")
            raise ProviderConfigurationError(self.name, f"Initialization failed: {e}")

    def _setup_cost_effective_device(self) -> torch.device:
        """Setup device with cost optimization"""
        if self.device == "auto":
            # Prioritize CPU for cost efficiency
            if torch.cuda.is_available() and self.quantization != "none":
                device = torch.device("cuda")
                logger.info("Using GPU with quantization for cost efficiency")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple Silicon for cost efficiency")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU for maximum cost efficiency")
        else:
            device = torch.device(self.device)

        return device

    async def _load_optimized_model(self):
        """Load model with optimization strategies"""
        loop = asyncio.get_event_loop()

        # Load tokenizer
        self.tokenizer = await loop.run_in_executor(
            None, lambda: AutoTokenizer.from_pretrained(self.model_name)
        )

        # Load model with quantization if requested
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = await loop.run_in_executor(
                None,
                lambda: AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                ),
            )
        elif self.quantization == "8bit":
            self.model = await loop.run_in_executor(
                None,
                lambda: AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, load_in_8bit=True, device_map="auto"
                ),
            )
        else:
            # Standard loading
            self.model = await loop.run_in_executor(
                None,
                lambda: AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16
                    if self.device_obj.type == "cuda"
                    else torch.float32,
                ),
            )

    async def _setup_model(self):
        """Final model setup"""
        if self.quantization == "none":
            self.model = self.model.to(self.device_obj)

        self.model.eval()

        # Test model
        await self._test_model()

    async def _test_model(self):
        """Test model functionality"""
        try:
            test_text = "This is a test."
            result = await self._analyze_text(test_text)

            if result.confidence == 0:
                raise ValueError("Model test returned zero confidence")

            logger.info("Model test successful")

        except Exception as e:
            raise ValueError(f"Model test failed: {e}")

    def _log_resource_usage(self):
        """Log current resource usage"""
        if self.model:
            # Estimate memory usage
            param_count = sum(p.numel() for p in self.model.parameters())

            if self.quantization == "4bit":
                self.memory_usage_mb = (
                    param_count * 0.5 / (1024 * 1024)
                )  # 4-bit = 0.5 bytes
            elif self.quantization == "8bit":
                self.memory_usage_mb = param_count / (1024 * 1024)  # 8-bit = 1 byte
            else:
                self.memory_usage_mb = (
                    param_count * 4 / (1024 * 1024)
                )  # float32 = 4 bytes

            logger.info(f"Estimated memory usage: {self.memory_usage_mb:.1f} MB")
            logger.info(f"Model parameters: {param_count:,}")

    async def _analyze_text(
        self, text: str, context: Optional[AnalysisContext] = None
    ) -> SentimentResult:
        """Efficient single text analysis"""
        if not self.model or not self.tokenizer:
            raise SentimentProviderError(
                "Model not initialized", self.name, retryable=False
            )

        start_time = asyncio.get_event_loop().time()

        try:
            cleaned_text = self._validate_text(text)

            # Tokenize with efficiency settings
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=False,  # No padding for single text
            )

            if self.quantization == "none":
                inputs = inputs.to(self.device_obj)

            # Run inference
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None, self._run_efficient_inference, inputs
            )

            # Update performance tracking
            inference_time = asyncio.get_event_loop().time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time

            # Convert to result
            result = self._predictions_to_result(predictions, cleaned_text)
            result.processing_time = inference_time

            return result

        except Exception as e:
            logger.error(f"Efficient analysis failed: {e}")
            raise SentimentProviderError(f"Analysis failed: {e}", self.name)

    def _run_efficient_inference(self, inputs) -> torch.Tensor:
        """Run inference with memory optimization"""
        with torch.no_grad():
            # Enable inference mode for better performance
            with torch.inference_mode():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return predictions.cpu()

    def _predictions_to_result(
        self, predictions: torch.Tensor, text: str
    ) -> SentimentResult:
        """Convert predictions to result with efficiency tracking"""
        scores = predictions[0]
        predicted_class_id = torch.argmax(scores).item()
        confidence = torch.max(scores).item()

        # Map to polarity
        if hasattr(self.model.config, "id2label"):
            predicted_label = self.model.config.id2label[predicted_class_id]
        else:
            predicted_label = f"LABEL_{predicted_class_id}"

        polarity = self.label_mappings.get(predicted_label, SentimentPolarity.NEUTRAL)

        # Create score
        if polarity == SentimentPolarity.POSITIVE:
            score = confidence
        elif polarity == SentimentPolarity.NEGATIVE:
            score = -confidence
        else:
            score = 0.0

        # Efficiency metadata
        avg_inference_time = (
            self.total_inference_time / self.inference_count
            if self.inference_count > 0
            else 0
        )

        metadata = {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "quantization": self.quantization,
            "predicted_label": predicted_label,
            "confidence_raw": float(confidence),
            "device": str(self.device_obj),
            "memory_usage_mb": self.memory_usage_mb,
            "avg_inference_time": avg_inference_time,
            "inference_count": self.inference_count,
            "text_truncated": len(text) > self.max_length,
            "cost_optimization": True,
        }

        return SentimentResult(
            polarity=polarity,
            score=score,
            confidence=confidence,
            method=SentimentMethod.TRANSFORMER,
            context_type=ContextType.GENERAL,
            text_length=len(text),
            metadata=metadata,
        )

    async def analyze_batch(
        self, texts: List[str], context: Optional[AnalysisContext] = None
    ) -> List[SentimentResult]:
        """Highly optimized batch processing"""
        if not texts:
            return []

        if not self.model or not self.tokenizer:
            raise SentimentProviderError(
                "Model not initialized", self.name, retryable=False
            )

        try:
            results = []

            # Process in optimized batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                batch_results = await self._process_efficient_batch(batch_texts, i)
                results.extend(batch_results)

                # Memory cleanup between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return results

        except Exception as e:
            logger.error(f"Efficient batch analysis failed: {e}")
            raise SentimentProviderError(f"Batch analysis failed: {e}", self.name)

    async def _process_efficient_batch(
        self, texts: List[str], batch_start_idx: int
    ) -> List[SentimentResult]:
        """Process batch with maximum efficiency"""
        try:
            cleaned_texts = [self._validate_text(text) for text in texts]

            # Tokenize batch efficiently
            inputs = self.tokenizer(
                cleaned_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            )

            if self.quantization == "none":
                inputs = inputs.to(self.device_obj)

            # Run batch inference
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None, self._run_efficient_batch_inference, inputs
            )

            # Convert results
            results = []
            for i, (text, pred) in enumerate(zip(cleaned_texts, predictions)):
                try:
                    result = self._predictions_to_result(pred.unsqueeze(0), text)
                    result.metadata["batch_index"] = batch_start_idx + i
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed batch item {i}: {e}")
                    results.append(self._create_error_result(str(e)))

            return results

        except Exception as e:
            logger.warning(f"Efficient batch processing failed: {e}")
            return [self._create_error_result(str(e)) for _ in texts]

    def _run_efficient_batch_inference(self, inputs) -> torch.Tensor:
        """Run batch inference with memory optimization"""
        with torch.no_grad():
            with torch.inference_mode():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return predictions.cpu()

    async def cleanup(self) -> None:
        """Cleanup with memory management"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Efficient transformer cleaned up")

    def get_cost_metrics(self) -> Dict:
        """Get cost and efficiency metrics"""
        avg_inference_time = (
            self.total_inference_time / self.inference_count
            if self.inference_count > 0
            else 0
        )

        return {
            "model_size": self.model_size,
            "quantization": self.quantization,
            "memory_usage_mb": self.memory_usage_mb,
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "avg_inference_time": avg_inference_time,
            "cost_per_inference": avg_inference_time * 0.0001,  # Estimated cost
            "efficiency_score": 1.0
            / (self.memory_usage_mb * avg_inference_time + 0.01),
            "device": str(self.device_obj),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }

    def supports_language(self, language: str) -> bool:
        """Language support based on model"""
        if "multilingual" in self.model_name.lower():
            return True
        return language.lower() in ["en", "english"]

    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        if "multilingual" in self.model_name.lower():
            return ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru"]
        return ["en"]

    def supports_batch_processing(self) -> bool:
        """Optimized for batch processing"""
        return True

    def get_capabilities(self) -> dict:
        """Get capabilities with cost information"""
        base_capabilities = super().get_capabilities()

        cost_metrics = self.get_cost_metrics()
        model_info = self.model_config.copy()

        base_capabilities.update(
            {
                "model_type": "efficient_transformer",
                "cost_optimized": True,
                "quantization_support": True,
                "local_inference": True,
                "cloud_free": True,
                **model_info,
                **cost_metrics,
            }
        )

        return base_capabilities


# Factory function for easy model creation
def create_cost_effective_sentiment_provider(
    cost_preference: str = "balanced",  # ultra_low, low, balanced, high_accuracy
    **kwargs,
) -> EfficientTransformerProvider:
    """
    Factory function to create cost-effective sentiment providers.

    Args:
        cost_preference: Cost vs accuracy preference
            - ultra_low: Minimal cost, basic accuracy
            - low: Low cost, good accuracy
            - balanced: Balanced cost/accuracy
            - high_accuracy: Higher cost, best accuracy
    """

    configs = {
        "ultra_low": {
            "model_size": "ultra_small",
            "quantization": "4bit",
            "batch_size": 4,
            "max_length": 128,
        },
        "low": {
            "model_size": "small",
            "quantization": "8bit",
            "batch_size": 8,
            "max_length": 256,
        },
        "balanced": {
            "model_size": "small",
            "quantization": "none",
            "batch_size": 16,
            "max_length": 512,
        },
        "high_accuracy": {
            "model_size": "accurate",
            "quantization": "none",
            "batch_size": 32,
            "max_length": 512,
        },
    }

    config = configs.get(cost_preference, configs["balanced"])
    config.update(kwargs)  # Allow overrides

    return EfficientTransformerProvider(**config)
