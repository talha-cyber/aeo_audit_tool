# Question Engine Reference

## Overview

The Question Engine is the core component responsible for generating strategic questions that will be used to audit brand visibility across AI platforms. It orchestrates multiple question providers to create a comprehensive, prioritized list of questions for competitive intelligence audits.

## Simple Explanation

### What It Does
Think of the Question Engine as a smart question generator for brand research. When you want to understand how your brand compares to competitors on AI platforms like ChatGPT or Claude, you need to ask the right questions. The Question Engine automatically creates these questions for you.

### How It Works
1. **Takes Input**: You provide your brand name, competitor names, industry, and product type
2. **Generates Questions**: Creates different types of questions using two methods:
   - **Template-based**: Uses pre-written question patterns
   - **AI-powered**: Uses artificial intelligence to create fresh, contextual questions
3. **Prioritizes**: Ranks questions by importance (comparisons and pricing get highest priority)
4. **Returns Results**: Gives you a final list of the most valuable questions to ask

### Example
If you're auditing "Slack" against competitors like "Microsoft Teams" and "Discord" in the "Communication" industry:

**Template questions might include:**
- "Slack vs Microsoft Teams: which is better?"
- "How much does Slack cost?"
- "Does Slack integrate with Salesforce?"

**AI-generated questions might include:**
- "Which team communication tool has the best security features for enterprise customers?"
- "What are the main differences between Slack's free tier and Microsoft Teams' free offering?"

## Technical Architecture

### Core Components

#### 1. QuestionEngine (Orchestrator)
```python
class QuestionEngine:
    def __init__(self, providers: Optional[List[QuestionProvider]] = None)
    async def generate_questions(...) -> List[Question]
    def prioritize_questions(...) -> List[Question]
```

**Responsibilities:**
- Orchestrates multiple question providers
- Runs providers concurrently using `asyncio.gather()`
- Merges and deduplicates results
- Applies priority scoring and limits output

**Key Methods:**
- `generate_questions()`: Main entry point that coordinates the entire process
- `_safe_generate()`: Wrapper that handles provider failures gracefully
- `prioritize_questions()`: Implements scoring algorithm for question ranking

#### 2. Provider Architecture

The system uses a modular provider pattern with two implementations:

##### Template Provider
```python
class TemplateProvider(QuestionProvider):
    name = "template_provider"
    can_handle(ctx) -> bool  # Always returns True
    async generate(ctx) -> ProviderResult
```

**Question Categories Generated:**
- **Comparison**: Direct brand comparisons (`"{brand_a} vs {brand_b}: which is better?"`)
- **Pricing**: Cost and tier questions (`"How much does {brand} cost?"`)
- **Integration**: Ecosystem compatibility (`"Does {brand} integrate with Salesforce?"`)
- **Security/Compliance**: Standards adherence (`"Is {brand} SOC 2 compliant?"`)
- **Implementation**: Migration and onboarding (`"How long does {brand} take to implement?"`)
- **Features**: Capability questions (`"Top features of {brand}?"`)
- **Reviews**: Social proof (`"{brand} reviews and ratings?"`)
- **Geography**: Regional availability (`"Is {brand} available in the EU?"`)

**Question Generation Logic:**
1. Uses predefined `LegacyQuestionTemplate` objects with category and variations
2. Applies string formatting with context variables (`{client_brand}`, `{competitor}`, `{industry}`)
3. Includes industry-specific knowledge via `IndustryKnowledge` module
4. Supports localization (English/German patterns)
5. Assigns priority scores based on question type

##### Dynamic Provider (AI-Powered)
```python
class DynamicProvider(QuestionProvider):
    name = "dynamic_provider"
    can_handle(ctx) -> bool  # Returns settings.DYNAMIC_Q_ENABLED
    async generate(ctx) -> ProviderResult
```

**Pipeline Components:**

1. **TrendsAdapter**: Fetches seed keywords from Google Trends/Reddit (with fallback seeds)
2. **PromptBuilder**: Creates deterministic LLM prompts with specific distribution requirements:
   - ≥30% pairwise brand comparisons
   - ≥20% pricing/tier questions mentioning brands
   - ≥15% integration questions with specific ecosystems
   - ≥10% security/compliance (SOC 2, ISO 27001, GDPR)
   - ≥10% implementation/migration/onboarding
   - Remaining: features, reviews, geography
3. **LLMClient**: Async OpenAI API client with:
   - Semaphore-based concurrency limiting
   - Exponential backoff retry logic (max 3 attempts)
   - 30-second global timeout
4. **PostProcessor**: Cleans and validates LLM output:
   - Deduplicates questions
   - Filters out questions >15 words
   - Tags with `category: "dynamic"`
5. **CacheManager**: Redis-backed caching with TTL

**Cache Key Schema:**
```
dynamic_q:{industry}:{product_type}:{md5(sorted(competitors))[:8]}:{YYYY-MM-DD}
```

#### 3. Data Models

##### QuestionContext
```python
class QuestionContext(BaseModel):
    client_brand: str
    competitors: List[str]
    industry: str
    product_type: str
    audit_run_id: uuid.UUID
    language: str = "en"
    market: Optional[str] = None
```

##### Question
```python
class Question(BaseModel):
    id: uuid.UUID
    question_text: str
    category: str
    provider: str
    priority_score: float = 0.0
    cost: Optional[float] = None
    tokens: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = {}
```

##### ProviderResult
```python
class ProviderResult(BaseModel):
    questions: List[Question]
    metadata: Dict[str, Any]
```

### Execution Flow

#### 1. Question Generation Process
```
Input (client_brand, competitors, industry, product_type, audit_run_id)
    ↓
Create QuestionContext
    ↓
Filter enabled providers (provider.can_handle(ctx))
    ↓
Concurrent execution: asyncio.gather(*[_safe_generate(p, ctx) for p in enabled_providers])
    ↓
    ├─ TemplateProvider
    │  ├─ Load base templates
    │  ├─ Apply industry knowledge
    │  ├─ Generate variations with string formatting
    │  └─ Return ProviderResult
    │
    └─ DynamicProvider
       ├─ Check cache (Redis lookup)
       ├─ If cache miss:
       │  ├─ TrendsAdapter.fetch_seeds()
       │  ├─ PromptBuilder.build() 
       │  ├─ LLMClient.generate_questions()
       │  └─ PostProcessor.process()
       └─ Cache result and return ProviderResult
    ↓
Merge all questions from successful providers
    ↓
Apply prioritization algorithm
    ↓
Return final question list (limited to max_questions)
```

#### 2. Prioritization Algorithm

**Priority Weights:**
```python
priority_weights = {
    # Decision-stage (highest priority)
    "comparison": 10,
    "pricing": 9,
    "integrations": 9,
    "security_compliance": 9,
    "implementation_migration": 8,
    "roi_tco": 8,
    "support_reliability": 8,
    
    # Discovery/consideration
    "alternatives": 7,
    "reviews": 7,
    "industry_specific": 7,
    "features": 6,
    "geography": 6,
    
    # Provider categories
    "dynamic": 8,
    "template": 5,
    "recommendation": 9,
}
```

**Process:**
1. Apply base scores from priority weights
2. Use `sub_category` from metadata when available for fine-grained scoring
3. Sort by priority_score (descending)
4. Deduplicate based on normalized question text
5. Truncate to `max_questions` limit

### Configuration & Controls

#### Settings
- `DYNAMIC_Q_ENABLED`: Feature flag for AI-powered questions
- `LLM_MODEL`: OpenAI model for dynamic generation
- `LLM_CONCURRENCY`: Concurrent API request limit
- `DYNAMIC_Q_CACHE_TTL`: Cache expiration time
- `DYNAMIC_Q_MAX`: Maximum questions from dynamic provider

#### Observability
- **Prometheus Metrics:**
  - `provider_calls_total`: Counter by provider name
  - `provider_failures_total`: Counter by provider name  
  - `provider_latency_seconds`: Histogram by provider name

#### Error Handling
- Provider failures are isolated using `_safe_generate()` wrapper
- Failed providers return empty results without stopping the process
- Comprehensive logging with structured data (audit_run_id, provider names, error details)
- Graceful degradation: system continues with working providers

### Performance Characteristics

#### Concurrency
- Providers execute in parallel using `asyncio.gather()`
- Dynamic provider implements connection pooling and rate limiting
- Template provider uses thread pool for CPU-bound operations

#### Caching
- Dynamic questions cached for 24 hours by default
- Cache key includes all context variables to ensure relevance
- Redis backend with configurable TTL

#### Scalability
- Stateless design allows horizontal scaling
- Provider pattern enables easy addition of new question sources
- Configurable limits prevent resource exhaustion

### Integration Points

#### Audit Processor Integration
```python
class AuditProcessor:
    async def _generate_questions(self, audit_run, context):
        raw_questions = await self.question_engine.generate_questions(
            client_brand=context["client"]["name"],
            competitors=context["client"].get("competitors", []),
            industry=context["client"]["industry"],
            product_type=context["client"].get("product_type"),
            audit_run_id=uuid.UUID(audit_run.id),
            language=context.get("language", "en"),
            max_questions=context.get("max_questions", 100),
        )
```

#### Database Persistence
Questions are stored in the `questions` table with relationships to audit runs and subsequent AI platform responses.

### Future Extensions

The provider architecture supports easy extension:
- **Specialized Providers**: Industry-specific question generators
- **User-Generated**: Custom question templates from users
- **ML-Enhanced**: Question effectiveness tracking and optimization
- **Multi-Language**: Localized question generation for global markets
- **Real-Time**: Live trend-based question adaptation

### Security Considerations

- **API Key Management**: OpenAI keys managed through secure configuration
- **Input Validation**: All user inputs validated through Pydantic models
- **Rate Limiting**: Built-in concurrency controls prevent API abuse
- **Logging**: Structured logging excludes sensitive information
- **Caching**: Redis access controlled through secure connection settings
