"""
Transformer-based Sentiment Analysis Provider.

Implements modern transformer-based sentiment analysis using HuggingFace models.
Provides state-of-the-art accuracy with local inference capabilities.
"""

import asyncio
import logging
from typing import List, Optional

import torch

from app.utils.logger import get_logger

from ..core.models import (
    AnalysisContext,
    ContextType,
    SentimentMethod,
    SentimentPolarity,
    SentimentResult,
)
from .base import ProviderConfigurationError, SentimentProvider, SentimentProviderError

# Suppress transformers warnings for cleaner logs
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = get_logger(__name__)


class TransformerSentimentProvider(SentimentProvider):
    """
    Transformer-based sentiment analysis provider using HuggingFace models.

    Features:
    - State-of-the-art accuracy with transformer models
    - Automatic GPU/CPU device selection
    - Batch processing optimization
    - Model caching and reuse
    - Multiple model support

    Default model: cardiffnlp/twitter-roberta-base-sentiment-latest
    - Fine-tuned on Twitter data
    - Good general performance
    - Handles social media text well
    """

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        super().__init__(SentimentMethod.TRANSFORMER, f"Transformer({model_name})")
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        # Model components (initialized in initialize())
        self.tokenizer = None
        self.model = None
        self.device_obj = None

        # Label mappings for different model types
        self.label_mappings = {
            # Default RoBERTa Twitter model
            "LABEL_0": SentimentPolarity.NEGATIVE,
            "LABEL_1": SentimentPolarity.NEUTRAL,
            "LABEL_2": SentimentPolarity.POSITIVE,
            # Alternative mappings
            "NEGATIVE": SentimentPolarity.NEGATIVE,
            "NEUTRAL": SentimentPolarity.NEUTRAL,
            "POSITIVE": SentimentPolarity.POSITIVE,
            # Numeric mappings
            "0": SentimentPolarity.NEGATIVE,
            "1": SentimentPolarity.NEUTRAL,
            "2": SentimentPolarity.POSITIVE,
        }

    async def initialize(self) -> None:
        """Initialize the transformer model and tokenizer"""
        try:
            logger.info(f"Initializing transformer model: {self.model_name}")

            # Import transformers here to handle import errors gracefully
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as e:
                raise ProviderConfigurationError(
                    self.name,
                    f"Required packages not installed: {e}. Install with: pip install transformers torch",
                )

            # Determine device
            self.device_obj = self._setup_device()
            logger.info(f"Using device: {self.device_obj}")

            # Load tokenizer and model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            # Load tokenizer
            self.tokenizer = await loop.run_in_executor(
                None, lambda: AutoTokenizer.from_pretrained(self.model_name)
            )

            # Load model
            self.model = await loop.run_in_executor(
                None,
                lambda: AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                ),
            )

            # Move model to device
            self.model = self.model.to(self.device_obj)
            self.model.eval()  # Set to evaluation mode

            # Test the model
            await self._test_model()

            logger.info(
                f"Transformer model {self.model_name} initialized successfully on {self.device_obj}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize transformer model: {e}")
            raise ProviderConfigurationError(
                self.name, f"Model initialization failed: {e}"
            )

    def _setup_device(self) -> torch.device:
        """Setup compute device (GPU/CPU)"""
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("CUDA available, using GPU")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("MPS available, using Apple Silicon GPU")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(self.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device("cpu")

        return device

    async def _test_model(self):
        """Test model with a simple input"""
        try:
            test_text = "This is a test sentence."
            test_inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            ).to(self.device_obj)

            with torch.no_grad():
                outputs = self.model(**test_inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            if predictions.shape[1] < 2:
                raise ValueError("Model output has insufficient classes")

            logger.info("Model test successful")

        except Exception as e:
            raise ValueError(f"Model test failed: {e}")

    async def cleanup(self) -> None:
        """Clean up model resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        # Clear CUDA cache if using GPU
        if self.device_obj and self.device_obj.type == "cuda":
            import torch

            torch.cuda.empty_cache()

        logger.info("Transformer model resources cleaned up")

    async def _analyze_text(
        self, text: str, context: Optional[AnalysisContext] = None
    ) -> SentimentResult:
        """
        Analyze single text using transformer model.

        Args:
            text: Text to analyze
            context: Analysis context

        Returns:
            SentimentResult with transformer analysis
        """
        if not self.model or not self.tokenizer:
            raise SentimentProviderError(
                "Transformer model not initialized", self.name, retryable=False
            )

        try:
            cleaned_text = self._validate_text(text)

            # Tokenize input
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            ).to(self.device_obj)

            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(None, self._run_inference, inputs)

            # Convert predictions to result
            result = self._predictions_to_result(predictions, cleaned_text)
            return result

        except Exception as e:
            logger.error(f"Transformer analysis failed: {e}")
            raise SentimentProviderError(f"Analysis failed: {e}", self.name)

    def _run_inference(self, inputs) -> torch.Tensor:
        """Run model inference (executed in thread pool)"""
        import torch

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return predictions.cpu()  # Move to CPU for processing

    def _predictions_to_result(
        self, predictions: torch.Tensor, text: str
    ) -> SentimentResult:
        """Convert model predictions to SentimentResult"""
        import torch

        # Get prediction scores and labels
        scores = predictions[0]  # First (and only) example
        predicted_class_id = torch.argmax(scores).item()
        confidence = torch.max(scores).item()

        # Map class ID to label
        class_labels = list(self.model.config.id2label.values())
        predicted_label = self.model.config.id2label[predicted_class_id]

        # Map to our polarity system
        polarity = self.label_mappings.get(predicted_label, SentimentPolarity.NEUTRAL)

        # Convert confidence and create score
        # For transformers, we create a score from the polarity and confidence
        if polarity == SentimentPolarity.POSITIVE:
            score = confidence
        elif polarity == SentimentPolarity.NEGATIVE:
            score = -confidence
        else:  # NEUTRAL or other
            score = 0.0

        # Create detailed metadata
        metadata = {
            "model_name": self.model_name,
            "predicted_label": predicted_label,
            "predicted_class_id": predicted_class_id,
            "class_scores": {
                label: float(scores[i]) for i, label in enumerate(class_labels)
            },
            "device": str(self.device_obj),
            "text_length": len(text),
            "truncated": len(text) > self.max_length,
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

    def supports_language(self, language: str) -> bool:
        """
        Language support depends on the specific model.
        Default RoBERTa model supports English primarily.
        """
        # This could be made configurable based on model
        if "multilingual" in self.model_name.lower():
            return True
        return language.lower() in ["en", "english"]

    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        if "multilingual" in self.model_name.lower():
            return [
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
            ]  # Common multilingual model languages
        return ["en"]

    def supports_batch_processing(self) -> bool:
        """Transformer models are optimized for batch processing"""
        return True

    async def analyze_batch(
        self, texts: List[str], context: Optional[AnalysisContext] = None
    ) -> List[SentimentResult]:
        """
        Efficient batch processing for transformer models.

        Processes texts in batches to maximize GPU utilization.
        """
        if not texts:
            return []

        if not self.model or not self.tokenizer:
            raise SentimentProviderError(
                "Transformer model not initialized", self.name, retryable=False
            )

        try:
            results = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                batch_results = await self._process_batch(batch_texts, i)
                results.extend(batch_results)

            return results

        except Exception as e:
            logger.error(f"Transformer batch analysis failed: {e}")
            raise SentimentProviderError(f"Batch analysis failed: {e}", self.name)

    async def _process_batch(
        self, texts: List[str], batch_start_idx: int
    ) -> List[SentimentResult]:
        """Process a single batch of texts"""
        try:
            # Validate and clean texts
            cleaned_texts = [self._validate_text(text) for text in texts]

            # Tokenize batch
            inputs = self.tokenizer(
                cleaned_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            ).to(self.device_obj)

            # Run batch inference
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None, self._run_batch_inference, inputs
            )

            # Convert each prediction to result
            results = []
            for i, (text, pred) in enumerate(zip(cleaned_texts, predictions)):
                try:
                    result = self._predictions_to_result(pred.unsqueeze(0), text)
                    result.metadata["batch_index"] = batch_start_idx + i
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to process batch item {i}: {e}")
                    error_result = SentimentResult(
                        polarity=SentimentPolarity.NEUTRAL,
                        score=0.0,
                        confidence=0.0,
                        method=SentimentMethod.TRANSFORMER,
                        context_type=ContextType.ERROR,
                        metadata={
                            "error": str(e),
                            "batch_index": batch_start_idx + i,
                            "provider": self.name,
                        },
                    )
                    results.append(error_result)

            return results

        except Exception as e:
            logger.warning(f"Batch processing failed: {e}")
            # Return error results for all texts in batch
            return [
                SentimentResult(
                    polarity=SentimentPolarity.NEUTRAL,
                    score=0.0,
                    confidence=0.0,
                    method=SentimentMethod.TRANSFORMER,
                    context_type=ContextType.ERROR,
                    metadata={
                        "error": str(e),
                        "batch_index": batch_start_idx + i,
                        "provider": self.name,
                    },
                )
                for i in range(len(texts))
            ]

    def _run_batch_inference(self, inputs) -> torch.Tensor:
        """Run batch inference (executed in thread pool)"""
        import torch

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return predictions.cpu()

    def get_capabilities(self) -> dict:
        """Get transformer provider capabilities"""
        base_capabilities = super().get_capabilities()

        # Add model-specific information
        model_info = {}
        if self.model:
            model_info = {
                "model_name": self.model_name,
                "model_type": "transformer",
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "num_labels": self.model.config.num_labels
                if hasattr(self.model.config, "num_labels")
                else "unknown",
                "max_length": self.max_length,
                "device": str(self.device_obj),
                "batch_size": self.batch_size,
            }

        base_capabilities.update(
            {
                "model_type": "transformer_neural",
                "supports_batch_optimization": True,
                "supports_gpu": torch.cuda.is_available()
                if "torch" in globals()
                else False,
                "resource_requirements": "medium_to_high",
                "accuracy": "high",
                **model_info,
            }
        )

        return base_capabilities
