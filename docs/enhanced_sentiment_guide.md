# Enhanced Sentiment Analysis Guide

## Overview

The Enhanced Sentiment Analysis system provides cost-effective ML capabilities while maintaining high accuracy. It includes efficient transformer models, domain adaptation, comprehensive cost monitoring, and advanced optimization strategies.

## 🚀 Key Features

### 1. Cost-Effective ML Models
- **4-bit and 8-bit quantization** for reduced memory usage
- **Multiple model sizes** from ultra-small to high-accuracy
- **Local inference** with no cloud API costs
- **Intelligent caching** for repeated analyses

### 2. Domain Adaptation
- **Few-shot learning** with minimal training data
- **Custom business domain models**
- **Incremental training** capabilities
- **Cost-controlled fine-tuning**

### 3. Comprehensive Cost Monitoring
- **Real-time resource tracking**
- **Budget limits and alerts**
- **Usage analytics and reporting**
- **Optimization recommendations**

### 4. Advanced Optimization
- **Model caching and compression**
- **Memory management**
- **Batch processing optimization**
- **Performance monitoring**

## 💰 Cost-Effective Model Options

### Ultra-Low Cost (`ultra_low`)
- **Memory**: ~50MB
- **Speed**: Very fast
- **Use case**: Bulk processing, preliminary analysis
- **Cost**: ~$0.0001 per 1000 requests

### Low Cost (`low`)
- **Memory**: ~150MB
- **Speed**: Fast
- **Use case**: General sentiment analysis
- **Cost**: ~$0.0005 per 1000 requests

### Balanced (`balanced`)
- **Memory**: ~300MB
- **Speed**: Medium
- **Use case**: Production applications
- **Cost**: ~$0.001 per 1000 requests

### High Accuracy (`high_accuracy`)
- **Memory**: ~500MB
- **Speed**: Slower
- **Use case**: Critical analysis, benchmarking
- **Cost**: ~$0.005 per 1000 requests

## 🛠 Quick Start

### Basic Usage

```python
from app.services.sentiment.enhanced_engine import create_enhanced_sentiment_engine

# Create engine with balanced cost preference
engine = await create_enhanced_sentiment_engine(
    cost_preference="balanced",
    budget_limit=10.0  # $10 monthly budget
)

# Analyze single text
result = await engine.analyze_cost_effective(
    text="This product is amazing!",
    cost_preference="low"
)

print(f"Sentiment: {result.polarity.value}")
print(f"Confidence: {result.confidence:.3f}")
```

### Batch Processing

```python
# Optimized batch processing
texts = [
    "Great product!",
    "Poor quality.",
    "Average experience."
]

batch_result = await engine.batch_analyze_optimized(
    texts=texts,
    cost_preference="ultra_low"  # Use cheapest model for bulk
)

print(f"Processed {len(batch_result.results)} texts")
print(f"Average confidence: {batch_result.average_confidence:.3f}")
```

### Domain Adaptation

```python
# Train domain-specific model
training_texts = ["Product review 1", "Product review 2", ...]
training_labels = ["positive", "negative", ...]

training_results = await engine.train_domain_model(
    domain_name="product_reviews",
    training_texts=training_texts,
    training_labels=training_labels,
    cost_limit=1.0  # Max $1 for training
)

# Use domain model
result = await engine.analyze_with_domain_model(
    text="This product is terrible!",
    domain_name="product_reviews"
)
```

## 📊 Cost Monitoring

### Real-time Monitoring

```python
# Get cost report
cost_report = engine.get_cost_report("month")

print(f"Monthly cost: ${cost_report['cost_summary']['total_cost']:.4f}")
print(f"Budget used: {cost_report['cost_summary']['budget_used_percent']:.1f}%")

# Get optimization recommendations
for rec in cost_report["optimization_recommendations"]:
    print(f"• {rec}")
```

### Budget Optimization

```python
# Optimize for specific budget
optimization = await engine.optimize_for_cost(target_budget=5.0)

for rec in optimization["recommendations"]:
    print(f"• {rec}")
```

## 🎯 Business Context Analysis

```python
from app.services.sentiment.core.models import AnalysisContext

# Business context analysis
context = AnalysisContext(
    brand="YourBrand",
    industry="technology",
    source="social_media",
    competitive_context=True
)

result = await engine.analyze_cost_effective(
    text="YourBrand vs Competitor - clear winner!",
    context=context,
    cost_preference="balanced"
)

print(f"Context type: {result.context_type.value}")
```

## ⚙️ Configuration

### Environment Variables

```bash
# Cost preferences
SENTIMENT_DEFAULT_COST_PREFERENCE=balanced
SENTIMENT_BUDGET_LIMIT=10.0

# Model optimization
SENTIMENT_ENABLE_QUANTIZATION=true
SENTIMENT_CACHE_DIR=./sentiment_cache
SENTIMENT_MAX_CACHE_SIZE_GB=5.0

# Training settings
SENTIMENT_TRAINING_DEVICE=auto
SENTIMENT_MAX_TRAINING_HOURS=2.0
```

### Custom Configuration

```python
from app.services.sentiment.enhanced_engine import EnhancedSentimentEngine

engine = EnhancedSentimentEngine(
    enable_cost_monitoring=True,
    budget_limit=15.0,
    enable_model_optimization=True,
    cache_dir="custom_cache"
)
```

## 📈 Performance Optimization

### Memory Optimization
- Use **quantized models** for memory-constrained environments
- Enable **model caching** for repeated analyses
- Configure **batch sizes** based on available memory

### Speed Optimization
- Use **smaller models** for non-critical analyses
- Enable **batch processing** for multiple texts
- Configure **concurrent processing** limits

### Cost Optimization
- Set appropriate **budget limits**
- Monitor **usage patterns** and optimize accordingly
- Use **domain models** for specialized use cases

## 🔧 Advanced Features

### Custom Model Training

```python
# Estimate training cost first
cost_estimate = engine.domain_adapter.estimate_training_cost(
    num_samples=1000,
    num_epochs=3
)

print(f"Estimated cost: ${cost_estimate['estimated_gpu_cost']:.4f}")
print(f"Estimated time: {cost_estimate['estimated_hours']:.2f} hours")

# Train if cost is acceptable
if cost_estimate['estimated_gpu_cost'] < 2.0:
    results = await engine.train_domain_model(...)
```

### Model Comparison

```python
# Compare different models
providers = ["ultra_low", "low", "balanced", "high_accuracy"]
results = {}

for provider in providers:
    result = await engine.analyze_cost_effective(
        text="Test text",
        cost_preference=provider
    )

    results[provider] = {
        "confidence": result.confidence,
        "processing_time": result.processing_time,
        "memory_usage": result.metadata.get("cost_metrics", {}).get("memory_usage_mb", 0)
    }
```

### Resource Monitoring

```python
# Get system statistics
capabilities = engine.get_available_capabilities()

print(f"Available models: {len(capabilities['cost_effective_models'])}")
print(f"Domain models: {capabilities['domain_models']}")
print(f"Budget limit: ${capabilities['budget_limit']}")

# Get detailed system stats
if engine.model_manager:
    stats = engine.model_manager.get_system_stats()
    print(f"Memory usage: {stats['memory_usage']}")
    print(f"Active models: {stats['active_models']}")
```

## 💡 Best Practices

### Cost Management
1. **Set realistic budgets** based on usage patterns
2. **Monitor costs regularly** and adjust strategies
3. **Use appropriate model sizes** for different use cases
4. **Leverage caching** for repeated analyses

### Model Selection
1. **Ultra-low cost**: Bulk processing, initial filtering
2. **Low cost**: General production use
3. **Balanced**: High-volume production applications
4. **High accuracy**: Critical decisions, benchmarking

### Domain Adaptation
1. **Start with small datasets** (50-100 samples)
2. **Validate training cost** before starting
3. **Use domain models** for specialized contexts
4. **Monitor domain model performance**

### Performance Optimization
1. **Batch similar requests** together
2. **Cache frequently analyzed content**
3. **Monitor memory usage** and adjust accordingly
4. **Use quantization** in memory-constrained environments

## 📊 Example Use Cases

### E-commerce Reviews
```python
# Train domain model for product reviews
domain_results = await engine.train_domain_model(
    domain_name="ecommerce_reviews",
    training_texts=product_review_texts,
    training_labels=sentiment_labels,
    cost_limit=0.5  # 30 minutes max
)

# Use for production analysis
result = await engine.analyze_with_domain_model(
    text="This product exceeded my expectations!",
    domain_name="ecommerce_reviews"
)
```

### Social Media Monitoring
```python
# Bulk analysis with cost optimization
social_posts = [...]  # List of social media posts

batch_result = await engine.batch_analyze_optimized(
    texts=social_posts,
    cost_preference="ultra_low",  # Minimize cost for bulk
    brand="YourBrand"
)

# Generate insights
sentiment_dist = batch_result.get_sentiment_distribution()
```

### Competitive Intelligence
```python
# Analyze competitive mentions
context = AnalysisContext(
    brand="YourBrand",
    competitive_context=True,
    source="news"
)

result = await engine.analyze_cost_effective(
    text="YourBrand outperforms Competitor in latest review",
    context=context,
    cost_preference="high_accuracy"  # Use best model for competitive analysis
)
```

## 🔍 Troubleshooting

### Common Issues

**High Memory Usage**
- Enable quantization: `quantization="4bit"`
- Reduce batch size: `batch_size=4`
- Clear model cache: `engine.model_manager.cache.clear_cache()`

**Slow Performance**
- Use smaller models: `cost_preference="ultra_low"`
- Enable compilation: `compile=True` in optimization config
- Increase batch size for bulk processing

**Budget Exceeded**
- Check current usage: `engine.get_cost_report("month")`
- Optimize model selection: `await engine.optimize_for_cost(target_budget)`
- Use domain models for specialized tasks

**Training Failures**
- Validate data format and labels
- Check available memory and disk space
- Reduce training parameters (epochs, batch size)

### Monitoring and Debugging

```python
# Enable detailed logging
import logging
logging.getLogger("app.services.sentiment").setLevel(logging.DEBUG)

# Get system diagnostics
stats = engine.model_manager.get_system_stats()
print(f"Memory usage: {stats['memory_usage']}")
print(f"Optimization recommendations: {stats['optimization_recommendations']}")

# Export detailed cost report
report_file = engine.cost_monitor.export_report("json")
print(f"Detailed report saved to: {report_file}")
```

## 🚀 Future Enhancements

The enhanced sentiment analysis system is designed for extensibility:

- **Additional model providers** (cloud APIs, custom models)
- **Advanced training techniques** (transfer learning, meta-learning)
- **Real-time streaming analysis**
- **Multi-language support**
- **Advanced cost optimization** (auto-scaling, spot instances)

---

For more examples and detailed API documentation, see the [examples directory](../examples/) and the source code documentation.