from prometheus_client import Counter, Histogram

# --- Provider Metrics ---

# Counter for tracking the number of times each provider's generate method is called.
# Labels:
# - provider_name: The name of the question provider (e.g., "template_provider").
PROVIDER_CALLS_TOTAL = Counter(
    "provider_calls_total",
    "Total number of calls to each question provider.",
    ["provider_name"],
)

# Counter for tracking failures in each provider. This should be incremented in the
# _safe_generate wrapper or within a provider if it has internal error handling.
# Labels:
# - provider_name: The name of the question provider.
PROVIDER_FAILURES_TOTAL = Counter(
    "provider_failures_total",
    "Total number of failures for each question provider.",
    ["provider_name"],
)

# Histogram for tracking the latency of each provider's generate method.
# This is useful for identifying slow providers and performance bottlenecks.
# The buckets are configured to capture a range of latencies, from very fast
# to several seconds.
# Labels:
# - provider_name: The name of the question provider.
PROVIDER_LATENCY_SECONDS = Histogram(
    "provider_latency_seconds",
    "Latency of the generate() method for each provider.",
    ["provider_name"],
    buckets=[0.1, 0.5, 1, 2.5, 5, 10, 30, 60],
)

# --- Cache Metrics ---

# Counter for tracking cache hits and misses.
# This helps in understanding the effectiveness of the cache.
# Labels:
# - cache_name: A name for the cache (e.g., "dynamic_question_cache").
# - result: "hit" or "miss".
CACHE_ACCESS_TOTAL = Counter(
    "cache_access_total",
    "Total number of cache hits and misses.",
    ["cache_name", "result"],
)

# --- LLM Metrics ---

# Counter for tracking OpenAI API calls.
# This can be used to monitor costs and usage.
# Labels:
# - model: The name of the model being called (e.g., "gpt-3.5-turbo").
LLM_CALLS_TOTAL = Counter(
    "llm_calls_total", "Total number of calls to the LLM.", ["model"]
)

# Counter for tracking LLM API errors.
# This is useful for identifying issues with the LLM provider.
# Labels:
# - model: The name of the model.
# - error_type: The type of error (e.g., "RateLimitError", "APIError").
LLM_FAILURES_TOTAL = Counter(
    "llm_failures_total",
    "Total number of failures when calling the LLM.",
    ["model", "error_type"],
)

# Counter for tracking the total cost of LLM calls.
# This is crucial for monitoring and managing expenses.
# Labels:
# - model: The name of the model.
LLM_COST_TOTAL = Counter(
    "llm_cost_total",
    "Total cost of LLM calls.",
    ["model"],
)
