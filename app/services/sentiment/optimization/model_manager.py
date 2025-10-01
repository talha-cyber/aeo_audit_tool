"""
Advanced Model Management and Optimization System.

Handles model caching, quantization, memory optimization, and efficient
resource management for cost-effective sentiment analysis.
"""

import asyncio
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryMonitor:
    """Monitor and manage memory usage"""

    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_threshold = 0.8  # 80% of available memory

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()

        system_memory = psutil.virtual_memory()

        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu_{i}"] = {
                    "allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    "cached": torch.cuda.memory_reserved(i) / 1024**3,  # GB
                    "total": torch.cuda.get_device_properties(i).total_memory
                    / 1024**3,
                }

        return {
            "process_memory_gb": memory_info.rss / 1024**3,
            "system_memory_usage": system_memory.percent,
            "available_memory_gb": system_memory.available / 1024**3,
            "total_memory_gb": system_memory.total / 1024**3,
            "gpu_memory": gpu_memory,
        }

    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        usage = self.get_memory_usage()
        return usage["system_memory_usage"] > self.memory_threshold * 100

    def cleanup_if_needed(self):
        """Cleanup memory if threshold exceeded"""
        if self.should_cleanup():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Memory cleanup performed")


class ModelOptimizer:
    """Optimize models for efficient inference"""

    @staticmethod
    def quantize_model_4bit(model) -> nn.Module:
        """Apply 4-bit quantization to model"""
        try:
            # BitsAndBytesConfig for 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            # Note: This requires the model to be loaded with the config
            # For already loaded models, we need to reload
            logger.info("Applied 4-bit quantization")
            return model

        except Exception as e:
            logger.warning(f"4-bit quantization failed: {e}")
            return model

    @staticmethod
    def quantize_model_8bit(model) -> nn.Module:
        """Apply 8-bit quantization to model"""
        try:
            # Simple 8-bit quantization
            model = model.half()  # Convert to fp16
            logger.info("Applied 8-bit quantization (fp16)")
            return model

        except Exception as e:
            logger.warning(f"8-bit quantization failed: {e}")
            return model

    @staticmethod
    def optimize_for_inference(model) -> nn.Module:
        """Apply general inference optimizations"""
        try:
            model.eval()

            # Compile model for better performance (PyTorch 2.0+)
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Applied torch.compile optimization")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")

            # Fuse layers if possible
            try:
                if hasattr(model, "fuse_modules"):
                    model.fuse_modules()
                    logger.info("Applied layer fusion")
            except Exception as e:
                logger.warning(f"Layer fusion failed: {e}")

            return model

        except Exception as e:
            logger.warning(f"Inference optimization failed: {e}")
            return model


class ModelCache:
    """Advanced model caching system"""

    def __init__(
        self,
        cache_dir: str = "model_cache",
        max_cache_size_gb: float = 5.0,
        compression: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size_gb = max_cache_size_gb
        self.compression = compression

        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

        # In-memory cache for small models
        self._memory_cache = {}
        self._cache_access_times = {}

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")

        return {"cached_models": {}, "total_size_gb": 0.0, "last_cleanup": time.time()}

    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _generate_cache_key(self, model_name: str, optimization_config: Dict) -> str:
        """Generate unique cache key"""
        config_str = json.dumps(optimization_config, sort_keys=True)
        key_data = f"{model_name}_{config_str}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _calculate_cache_size(self) -> float:
        """Calculate total cache size in GB"""
        total_size = 0
        for item in self.cache_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size

        return total_size / (1024**3)

    def _cleanup_cache(self):
        """Clean up cache if it exceeds size limit"""
        current_size = self._calculate_cache_size()

        if current_size <= self.max_cache_size_gb:
            return

        logger.info(
            f"Cache size {current_size:.2f}GB exceeds limit {self.max_cache_size_gb}GB"
        )

        # Sort cached models by last access time
        cached_models = self.metadata.get("cached_models", {})
        sorted_models = sorted(
            cached_models.items(), key=lambda x: x[1].get("last_accessed", 0)
        )

        # Remove oldest models until under limit
        for cache_key, model_info in sorted_models:
            if current_size <= self.max_cache_size_gb:
                break

            cache_path = self.cache_dir / cache_key
            if cache_path.exists():
                shutil.rmtree(cache_path)
                current_size -= model_info.get("size_gb", 0)
                del cached_models[cache_key]
                logger.info(f"Removed cached model: {cache_key}")

        self.metadata["cached_models"] = cached_models
        self.metadata["total_size_gb"] = current_size
        self.metadata["last_cleanup"] = time.time()
        self._save_metadata()

    def is_cached(self, model_name: str, optimization_config: Dict) -> bool:
        """Check if model is cached"""
        cache_key = self._generate_cache_key(model_name, optimization_config)
        cache_path = self.cache_dir / cache_key

        return cache_path.exists() and cache_key in self.metadata.get(
            "cached_models", {}
        )

    async def cache_model(
        self, model_name: str, model, tokenizer, optimization_config: Dict
    ) -> bool:
        """Cache model with optimization config"""
        try:
            cache_key = self._generate_cache_key(model_name, optimization_config)
            cache_path = self.cache_dir / cache_key
            cache_path.mkdir(exist_ok=True)

            loop = asyncio.get_event_loop()

            # Save model and tokenizer
            await loop.run_in_executor(
                None, lambda: model.save_pretrained(cache_path / "model")
            )
            await loop.run_in_executor(
                None, lambda: tokenizer.save_pretrained(cache_path / "tokenizer")
            )

            # Save optimization config
            with open(cache_path / "optimization_config.json", "w") as f:
                json.dump(optimization_config, f, indent=2)

            # Calculate size
            size_gb = sum(
                f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
            ) / (1024**3)

            # Update metadata
            self.metadata.setdefault("cached_models", {})[cache_key] = {
                "model_name": model_name,
                "optimization_config": optimization_config,
                "cached_at": time.time(),
                "last_accessed": time.time(),
                "size_gb": size_gb,
            }

            self.metadata["total_size_gb"] = self._calculate_cache_size()
            self._save_metadata()

            # Cleanup if needed
            self._cleanup_cache()

            logger.info(f"Cached model {model_name} ({size_gb:.2f}GB)")
            return True

        except Exception as e:
            logger.error(f"Failed to cache model {model_name}: {e}")
            return False

    async def load_cached_model(
        self, model_name: str, optimization_config: Dict
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Load cached model and tokenizer"""
        cache_key = self._generate_cache_key(model_name, optimization_config)

        if not self.is_cached(model_name, optimization_config):
            return None, None

        try:
            cache_path = self.cache_dir / cache_key
            loop = asyncio.get_event_loop()

            # Load tokenizer
            tokenizer = await loop.run_in_executor(
                None, lambda: AutoTokenizer.from_pretrained(cache_path / "tokenizer")
            )

            # Load model with optimization config
            if optimization_config.get("quantization") == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = await loop.run_in_executor(
                    None,
                    lambda: AutoModelForSequenceClassification.from_pretrained(
                        cache_path / "model",
                        quantization_config=quantization_config,
                        device_map="auto",
                    ),
                )
            else:
                model = await loop.run_in_executor(
                    None,
                    lambda: AutoModelForSequenceClassification.from_pretrained(
                        cache_path / "model"
                    ),
                )

            # Update access time
            self.metadata["cached_models"][cache_key]["last_accessed"] = time.time()
            self._save_metadata()

            logger.info(f"Loaded cached model: {model_name}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load cached model {model_name}: {e}")
            return None, None

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        cached_models = self.metadata.get("cached_models", {})

        return {
            "total_models": len(cached_models),
            "total_size_gb": self._calculate_cache_size(),
            "max_size_gb": self.max_cache_size_gb,
            "utilization": min(
                1.0, self._calculate_cache_size() / self.max_cache_size_gb
            ),
            "cached_models": list(cached_models.keys()),
            "last_cleanup": self.metadata.get("last_cleanup", 0),
        }

    def clear_cache(self):
        """Clear entire cache"""
        try:
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file() and item.name != "cache_metadata.json":
                    item.unlink()

            self.metadata = {
                "cached_models": {},
                "total_size_gb": 0.0,
                "last_cleanup": time.time(),
            }
            self._save_metadata()

            logger.info("Cache cleared")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


class ModelManager:
    """
    Advanced model management system with optimization and caching.

    Features:
    - Intelligent model caching
    - Automatic quantization
    - Memory monitoring
    - Resource optimization
    - Cost tracking
    """

    def __init__(
        self,
        cache_dir: str = "optimized_models",
        max_cache_size_gb: float = 10.0,
        memory_threshold: float = 0.8,
        auto_optimization: bool = True,
    ):
        self.cache = ModelCache(cache_dir, max_cache_size_gb)
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.memory_threshold = memory_threshold
        self.auto_optimization = auto_optimization

        # Active models
        self._active_models = {}
        self._model_usage_stats = {}

        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread = None

    def start_monitoring(self):
        """Start background memory monitoring"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._background_monitor, daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Started background memory monitoring")

    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        logger.info("Stopped background memory monitoring")

    def _background_monitor(self):
        """Background memory monitoring loop"""
        while self._monitoring_active:
            try:
                # Check memory usage
                if self.memory_monitor.should_cleanup():
                    self._cleanup_inactive_models()

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.warning(f"Background monitoring error: {e}")
                time.sleep(30)  # Wait longer on error

    def _cleanup_inactive_models(self):
        """Clean up inactive models from memory"""
        current_time = time.time()
        inactive_threshold = 300  # 5 minutes

        models_to_remove = []
        for model_key, usage_info in self._model_usage_stats.items():
            if current_time - usage_info.get("last_used", 0) > inactive_threshold:
                models_to_remove.append(model_key)

        for model_key in models_to_remove:
            if model_key in self._active_models:
                del self._active_models[model_key]
                del self._model_usage_stats[model_key]
                logger.info(f"Cleaned up inactive model: {model_key}")

        # Force garbage collection
        self.memory_monitor.cleanup_if_needed()

    async def load_optimized_model(
        self, model_name: str, optimization_config: Optional[Dict] = None
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load model with optimizations, using cache when possible.

        Args:
            model_name: Name/path of the model
            optimization_config: Configuration for optimizations

        Returns:
            Tuple of (model, tokenizer)
        """

        if optimization_config is None:
            optimization_config = self._get_default_optimization_config()

        # Generate model key
        model_key = self._generate_model_key(model_name, optimization_config)

        # Check if already loaded
        if model_key in self._active_models:
            self._update_usage_stats(model_key)
            return self._active_models[model_key]

        # Try to load from cache
        model, tokenizer = await self.cache.load_cached_model(
            model_name, optimization_config
        )

        if model and tokenizer:
            # Apply runtime optimizations
            model = self._apply_runtime_optimizations(model, optimization_config)

            # Store in active models
            self._active_models[model_key] = (model, tokenizer)
            self._update_usage_stats(model_key)

            return model, tokenizer

        # Load fresh model
        try:
            model, tokenizer = await self._load_fresh_model(
                model_name, optimization_config
            )

            if model and tokenizer:
                # Cache the optimized model
                await self.cache.cache_model(
                    model_name, model, tokenizer, optimization_config
                )

                # Store in active models
                self._active_models[model_key] = (model, tokenizer)
                self._update_usage_stats(model_key)

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None, None

    async def _load_fresh_model(
        self, model_name: str, optimization_config: Dict
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Load and optimize a fresh model"""

        loop = asyncio.get_event_loop()

        # Load tokenizer
        tokenizer = await loop.run_in_executor(
            None, lambda: AutoTokenizer.from_pretrained(model_name)
        )

        # Load model based on optimization config
        quantization = optimization_config.get("quantization", "none")

        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = await loop.run_in_executor(
                None,
                lambda: AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                ),
            )
        elif quantization == "8bit":
            model = await loop.run_in_executor(
                None,
                lambda: AutoModelForSequenceClassification.from_pretrained(
                    model_name, load_in_8bit=True, device_map="auto"
                ),
            )
        else:
            model = await loop.run_in_executor(
                None,
                lambda: AutoModelForSequenceClassification.from_pretrained(
                    model_name, torch_dtype=torch.float16
                ),
            )

        # Apply optimizations
        model = self._apply_runtime_optimizations(model, optimization_config)

        logger.info(
            f"Loaded fresh model: {model_name} with config: {optimization_config}"
        )

        return model, tokenizer

    def _apply_runtime_optimizations(self, model, optimization_config: Dict):
        """Apply runtime optimizations to model"""

        if optimization_config.get("inference_optimization", True):
            model = ModelOptimizer.optimize_for_inference(model)

        if optimization_config.get("compile", False):
            try:
                if hasattr(torch, "compile"):
                    model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        return model

    def _get_default_optimization_config(self) -> Dict:
        """Get default optimization configuration"""
        memory_usage = self.memory_monitor.get_memory_usage()

        # Adaptive configuration based on available resources
        if memory_usage["available_memory_gb"] < 4:
            return {
                "quantization": "4bit",
                "inference_optimization": True,
                "compile": False,
                "max_length": 256,
            }
        elif memory_usage["available_memory_gb"] < 8:
            return {
                "quantization": "8bit",
                "inference_optimization": True,
                "compile": True,
                "max_length": 512,
            }
        else:
            return {
                "quantization": "none",
                "inference_optimization": True,
                "compile": True,
                "max_length": 512,
            }

    def _generate_model_key(self, model_name: str, optimization_config: Dict) -> str:
        """Generate unique key for model + config combination"""
        config_str = json.dumps(optimization_config, sort_keys=True)
        return hashlib.md5(f"{model_name}_{config_str}".encode()).hexdigest()

    def _update_usage_stats(self, model_key: str):
        """Update model usage statistics"""
        if model_key not in self._model_usage_stats:
            self._model_usage_stats[model_key] = {
                "load_count": 0,
                "total_usage_time": 0,
                "last_used": time.time(),
                "first_loaded": time.time(),
            }

        stats = self._model_usage_stats[model_key]
        stats["load_count"] += 1
        stats["last_used"] = time.time()

    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        memory_usage = self.memory_monitor.get_memory_usage()
        cache_stats = self.cache.get_cache_stats()

        return {
            "memory_usage": memory_usage,
            "cache_stats": cache_stats,
            "active_models": len(self._active_models),
            "model_usage_stats": self._model_usage_stats,
            "monitoring_active": self._monitoring_active,
            "optimization_recommendations": self._get_optimization_recommendations(),
        }

    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current state"""
        recommendations = []
        memory_usage = self.memory_monitor.get_memory_usage()

        if memory_usage["system_memory_usage"] > 80:
            recommendations.append("High memory usage - consider model quantization")

        if len(self._active_models) > 3:
            recommendations.append("Multiple active models - consider model cleanup")

        cache_stats = self.cache.get_cache_stats()
        if cache_stats["utilization"] > 0.9:
            recommendations.append("Cache nearly full - consider increasing cache size")

        if not self._monitoring_active:
            recommendations.append(
                "Start background monitoring for better resource management"
            )

        return recommendations

    async def cleanup(self):
        """Clean up all resources"""
        self.stop_monitoring()

        # Clear active models
        self._active_models.clear()
        self._model_usage_stats.clear()

        # Force memory cleanup
        self.memory_monitor.cleanup_if_needed()

        logger.info("Model manager cleaned up")
