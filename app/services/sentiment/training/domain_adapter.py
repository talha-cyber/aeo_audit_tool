"""
Domain Adaptation Training for Sentiment Analysis.

Cost-effective fine-tuning and domain adaptation capabilities for improving
sentiment analysis on specific domains, brands, or use cases using minimal data.
"""

import asyncio
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from app.utils.logger import get_logger
from ..core.models import SentimentPolarity, SentimentResult
from ..providers.efficient_transformer_provider import EfficientTransformerProvider

logger = get_logger(__name__)


class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis training"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DomainAdapter:
    """
    Cost-effective domain adaptation for sentiment analysis.

    Features:
    - Few-shot learning with minimal data
    - Efficient fine-tuning strategies
    - Domain-specific pattern learning
    - Cost tracking and optimization
    - Incremental training capabilities
    """

    def __init__(
        self,
        base_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        cache_dir: str = "domain_models",
        max_length: int = 256,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = "auto"
    ):
        self.base_model = base_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model components
        self.tokenizer = None
        self.model = None

        # Training history
        self.training_history = []
        self.domain_models = {}

    async def initialize(self):
        """Initialize the domain adapter"""
        try:
            logger.info(f"Initializing domain adapter with base model: {self.base_model}")

            loop = asyncio.get_event_loop()

            # Load tokenizer and model
            self.tokenizer = await loop.run_in_executor(
                None, lambda: AutoTokenizer.from_pretrained(self.base_model)
            )

            self.model = await loop.run_in_executor(
                None, lambda: AutoModelForSequenceClassification.from_pretrained(
                    self.base_model,
                    num_labels=3  # Negative, Neutral, Positive
                )
            )

            self.model.to(self.device)
            logger.info(f"Domain adapter initialized on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize domain adapter: {e}")
            raise

    def prepare_training_data(
        self,
        texts: List[str],
        sentiments: List[Union[str, int, SentimentPolarity]],
        validation_split: float = 0.2
    ) -> Tuple[SentimentDataset, SentimentDataset]:
        """
        Prepare training data from text and sentiment pairs.

        Args:
            texts: List of text samples
            sentiments: List of sentiment labels (various formats accepted)
            validation_split: Fraction for validation set

        Returns:
            Tuple of (train_dataset, val_dataset)
        """

        # Convert sentiments to integer labels
        labels = []
        for sentiment in sentiments:
            if isinstance(sentiment, SentimentPolarity):
                if sentiment == SentimentPolarity.NEGATIVE:
                    labels.append(0)
                elif sentiment == SentimentPolarity.NEUTRAL:
                    labels.append(1)
                else:  # POSITIVE
                    labels.append(2)
            elif isinstance(sentiment, str):
                sentiment_lower = sentiment.lower()
                if sentiment_lower in ['negative', 'neg', 'bad', '0']:
                    labels.append(0)
                elif sentiment_lower in ['neutral', 'neu', 'mixed', '1']:
                    labels.append(1)
                else:  # positive, pos, good, 2
                    labels.append(2)
            else:  # Assume integer
                labels.append(int(sentiment))

        # Split data
        if validation_split > 0:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=validation_split, random_state=42, stratify=labels
            )
        else:
            train_texts, val_texts = texts, []
            train_labels, val_labels = labels, []

        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length) if val_texts else None

        logger.info(f"Prepared training data: {len(train_texts)} train, {len(val_texts)} validation")

        return train_dataset, val_dataset

    async def fine_tune_domain(
        self,
        domain_name: str,
        train_dataset: SentimentDataset,
        val_dataset: Optional[SentimentDataset] = None,
        cost_limit: float = 1.0,  # Hours of training time limit
        save_model: bool = True
    ) -> Dict:
        """
        Fine-tune model for specific domain with cost controls.

        Args:
            domain_name: Name for the domain-specific model
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            cost_limit: Maximum training time in hours
            save_model: Whether to save the fine-tuned model

        Returns:
            Training results and metrics
        """

        if not self.model or not self.tokenizer:
            await self.initialize()

        start_time = time.time()
        logger.info(f"Starting domain adaptation for: {domain_name}")

        try:
            # Setup data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0  # Avoid multiprocessing issues
            )

            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0
                )

            # Setup optimizer and scheduler
            optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

            total_steps = len(train_loader) * self.num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )

            # Training loop with cost monitoring
            training_losses = []
            validation_metrics = []

            for epoch in range(self.num_epochs):
                # Check cost limit
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours > cost_limit:
                    logger.warning(f"Training stopped due to cost limit: {elapsed_hours:.2f}h")
                    break

                # Training phase
                train_loss = await self._train_epoch(train_loader, optimizer, scheduler)
                training_losses.append(train_loss)

                # Validation phase
                if val_loader:
                    val_metrics = await self._validate_epoch(val_loader)
                    validation_metrics.append(val_metrics)

                    logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs}: "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.4f}, "
                        f"Val F1: {val_metrics['f1']:.4f}"
                    )
                else:
                    logger.info(f"Epoch {epoch+1}/{self.num_epochs}: Train Loss: {train_loss:.4f}")

            # Calculate total training time and cost
            total_time = time.time() - start_time
            estimated_cost = total_time * 0.0001  # Rough estimate for local training

            # Save model if requested
            model_path = None
            if save_model:
                model_path = await self._save_domain_model(domain_name)

            # Compile results
            results = {
                "domain_name": domain_name,
                "training_time": total_time,
                "estimated_cost": estimated_cost,
                "epochs_completed": len(training_losses),
                "final_train_loss": training_losses[-1] if training_losses else None,
                "training_losses": training_losses,
                "validation_metrics": validation_metrics,
                "model_path": str(model_path) if model_path else None,
                "training_samples": len(train_dataset),
                "validation_samples": len(val_dataset) if val_dataset else 0
            }

            # Store in history
            self.training_history.append(results)

            # Register domain model
            if model_path:
                self.domain_models[domain_name] = {
                    "path": model_path,
                    "created_at": time.time(),
                    "performance": validation_metrics[-1] if validation_metrics else None
                }

            logger.info(f"Domain adaptation completed for {domain_name}")
            logger.info(f"Training time: {total_time:.2f}s, Estimated cost: ${estimated_cost:.4f}")

            return results

        except Exception as e:
            logger.error(f"Domain adaptation failed for {domain_name}: {e}")
            raise

    async def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        loop = asyncio.get_event_loop()

        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            optimizer.zero_grad()

            outputs = await loop.run_in_executor(
                None,
                lambda: self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            optimizer.step()
            scheduler.step()

        return total_loss / len(train_loader)

    async def _validate_epoch(self, val_loader: DataLoader) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        predictions = []
        true_labels = []

        loop = asyncio.get_event_loop()

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = await loop.run_in_executor(
                    None,
                    lambda: self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                )

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return {
            "accuracy": accuracy,
            "f1": f1,
            "classification_report": classification_report(
                true_labels, predictions, output_dict=True
            )
        }

    async def _save_domain_model(self, domain_name: str) -> Path:
        """Save domain-specific model"""
        model_dir = self.cache_dir / domain_name
        model_dir.mkdir(exist_ok=True)

        loop = asyncio.get_event_loop()

        # Save model and tokenizer
        await loop.run_in_executor(
            None, lambda: self.model.save_pretrained(model_dir / "model")
        )
        await loop.run_in_executor(
            None, lambda: self.tokenizer.save_pretrained(model_dir / "tokenizer")
        )

        # Save metadata
        metadata = {
            "domain_name": domain_name,
            "base_model": self.base_model,
            "created_at": time.time(),
            "training_config": {
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs
            }
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved domain model to: {model_dir}")
        return model_dir

    async def load_domain_model(self, domain_name: str) -> bool:
        """Load a previously trained domain model"""
        model_dir = self.cache_dir / domain_name

        if not model_dir.exists():
            logger.warning(f"Domain model not found: {domain_name}")
            return False

        try:
            loop = asyncio.get_event_loop()

            # Load tokenizer and model
            self.tokenizer = await loop.run_in_executor(
                None, lambda: AutoTokenizer.from_pretrained(model_dir / "tokenizer")
            )

            self.model = await loop.run_in_executor(
                None, lambda: AutoModelForSequenceClassification.from_pretrained(
                    model_dir / "model"
                )
            )

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded domain model: {domain_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load domain model {domain_name}: {e}")
            return False

    def create_domain_provider(self, domain_name: str) -> Optional[EfficientTransformerProvider]:
        """Create a sentiment provider using a domain-specific model"""
        model_dir = self.cache_dir / domain_name

        if not model_dir.exists():
            logger.warning(f"Domain model not found: {domain_name}")
            return None

        try:
            # Create provider with domain model path
            provider = EfficientTransformerProvider(
                model_size="custom",
                quantization="none",
                enable_caching=True
            )

            # Override model name to use local path
            provider.model_name = str(model_dir / "model")
            provider.name = f"DomainTransformer({domain_name})"

            return provider

        except Exception as e:
            logger.error(f"Failed to create domain provider for {domain_name}: {e}")
            return None

    def get_available_domains(self) -> List[str]:
        """Get list of available domain models"""
        domains = []

        for domain_dir in self.cache_dir.iterdir():
            if domain_dir.is_dir() and (domain_dir / "model").exists():
                domains.append(domain_dir.name)

        return domains

    def get_training_history(self) -> List[Dict]:
        """Get training history"""
        return self.training_history

    def estimate_training_cost(
        self,
        num_samples: int,
        num_epochs: int = 3,
        batch_size: int = 8
    ) -> Dict:
        """Estimate training cost and time"""

        # Rough estimates based on local training
        samples_per_hour = 10000  # Depends on hardware

        total_steps = (num_samples / batch_size) * num_epochs
        estimated_hours = total_steps / (samples_per_hour / batch_size)

        # Cost estimates (very rough)
        gpu_cost_per_hour = 0.10  # Local GPU electricity cost
        cpu_cost_per_hour = 0.02  # Local CPU cost

        gpu_cost = estimated_hours * gpu_cost_per_hour
        cpu_cost = estimated_hours * cpu_cost_per_hour

        return {
            "estimated_hours": estimated_hours,
            "estimated_gpu_cost": gpu_cost,
            "estimated_cpu_cost": cpu_cost,
            "total_steps": total_steps,
            "samples_per_hour": samples_per_hour,
            "recommendation": "GPU" if estimated_hours > 0.5 else "CPU"
        }

    async def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Domain adapter cleaned up")


# Utility functions for data preparation
def prepare_business_data(
    brand_mentions: List[str],
    sentiments: List[str],
    context_type: str = "business"
) -> Tuple[List[str], List[str]]:
    """
    Prepare business-specific training data.

    Args:
        brand_mentions: List of texts mentioning brands
        sentiments: List of sentiment labels
        context_type: Type of business context

    Returns:
        Tuple of (prepared_texts, prepared_labels)
    """

    prepared_texts = []
    prepared_labels = []

    for text, sentiment in zip(brand_mentions, sentiments):
        # Add business context prefix
        if context_type == "business":
            prepared_text = f"[BUSINESS] {text}"
        elif context_type == "competitive":
            prepared_text = f"[COMPETITIVE] {text}"
        else:
            prepared_text = text

        prepared_texts.append(prepared_text)
        prepared_labels.append(sentiment)

    return prepared_texts, prepared_labels


def augment_training_data(
    texts: List[str],
    labels: List[str],
    augmentation_factor: int = 2
) -> Tuple[List[str], List[str]]:
    """
    Simple data augmentation for increasing training samples.

    Args:
        texts: Original texts
        labels: Original labels
        augmentation_factor: How many augmented samples per original

    Returns:
        Tuple of (augmented_texts, augmented_labels)
    """

    augmented_texts = texts.copy()
    augmented_labels = labels.copy()

    for _ in range(augmentation_factor - 1):
        for text, label in zip(texts, labels):
            # Simple augmentation strategies
            augmented_text = text

            # Add punctuation variations
            if not text.endswith(('.', '!', '?')):
                augmented_text += "."

            # Add whitespace variations
            augmented_text = " ".join(augmented_text.split())

            augmented_texts.append(augmented_text)
            augmented_labels.append(label)

    return augmented_texts, augmented_labels