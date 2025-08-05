import asyncio
from typing import Dict

from fastapi import APIRouter, Depends, Response

from app.services.question_engine import QuestionEngine

router = APIRouter()


def get_question_engine():
    """
    Dependency to get a QuestionEngine instance.

    In a real application, this might be a more complex dependency,
    e.g., managed by a DI container.
    """
    return QuestionEngine()


@router.get(
    "/health",
    tags=["providers"],
    summary="Get the health status of all question providers",
)
async def health_check(
    response: Response,
    engine: QuestionEngine = Depends(get_question_engine),
) -> Dict[str, Dict[str, str]]:
    """
    Checks the health of all registered question providers.

    This endpoint iterates through each provider, calls its `health_check` method,
    and returns a consolidated report. If any provider is unhealthy, the overall
    HTTP status code will be 503 Service Unavailable.

    Returns:
        A dictionary containing the health status of each provider.
    """
    provider_health_checks = [p.health_check() for p in engine.providers]
    results = await asyncio.gather(*provider_health_checks, return_exceptions=True)

    provider_statuses = {}
    is_overall_healthy = True

    for provider, result in zip(engine.providers, results):
        if isinstance(result, Exception) or not result:
            is_overall_healthy = False
            status = "error"
        else:
            status = "ok"
        provider_statuses[provider.name] = {"status": status}

    if not is_overall_healthy:
        response.status_code = 503

    return provider_statuses
