"""Provider-agnostic LLM router for question engine v2."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft7Validator

from app.core.config import settings
from app.services.ai_platforms.base import BasePlatform
from app.services.ai_platforms.registry import PlatformRegistry
from app.utils.logger import get_logger

logger = get_logger(__name__)

_API_KEY_LOOKUP: Dict[str, str] = {
    "openai": settings.OPENAI_API_KEY,
    "anthropic": settings.ANTHROPIC_API_KEY,
    "perplexity": settings.PERPLEXITY_API_KEY,
    "google_ai": settings.GOOGLE_AI_API_KEY,
}


class LLMRouter:
    """Lazy router that maps provider keys to configured LLM clients."""

    def __init__(self, default_provider: str = "openai") -> None:
        self.default_provider = default_provider
        self._clients: Dict[str, BasePlatform] = {}

    async def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_attempts: int = 2,
        **options: Any,
    ) -> Dict[str, Any]:
        """Execute a structured generation request against the configured provider.

        The default max_attempts=2 ensures a follow-up retry with a schema reminder when
        the first response fails validation.
        """
        provider_key, model_hint = self._resolve_provider_target(provider, model)
        client = await self._get_client(provider_key)

        validator = Draft7Validator(schema)
        attempts = 0
        last_error: Optional[str] = None

        while attempts < max_attempts:
            attempts += 1
            effective_prompt = prompt
            if attempts > 1 and last_error:
                effective_prompt = (
                    f"{prompt}\n\nPay close attention to the JSON schema above. "
                    f"Your previous response failed validation: {last_error}"
                )

            payload = self._build_payload(
                prompt=effective_prompt,
                schema=schema,
                model=model_hint,
                options=options,
            )

            response = await client.safe_query(**payload)
            if not response.get("success"):
                last_error = response.get("error") or "provider call failed"
                logger.warning(
                    "LLMRouter call failed",
                    provider=provider_key,
                    error=last_error,
                    attempt=attempts,
                )
                continue

            raw = response.get("response") or {}
            parsed, validation_error = self._parse_and_validate_raw(raw, validator)
            if parsed is not None:
                return parsed

            last_error = validation_error or "validation error"
            logger.warning(
                "LLMRouter schema validation failed",
                provider=provider_key,
                error=last_error,
                attempt=attempts,
            )

        raise ValueError(
            f"Structured completion failed for provider '{provider_key}': {last_error}"
        )

    def _build_payload(
        self,
        *,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct payload for platform client safe_query call."""
        payload: Dict[str, Any] = {
            "question": prompt,
            "metadata": {
                "prompt_type": "json_schema",
                "schema": schema,
            },
        }
        if model:
            payload["model"] = model
        payload.update(options)
        return payload

    def _parse_and_validate_raw(
        self,
        raw_response: Dict[str, Any],
        validator: Draft7Validator,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Extract JSON from platform response and validate against schema."""
        candidate: Optional[Dict[str, Any]] = None

        # Preferred: provider already returns parsed JSON structure via `json` key.
        if isinstance(raw_response, dict) and "json" in raw_response:
            candidate = raw_response["json"]  # type: ignore[assignment]
        elif isinstance(raw_response, dict) and "content" in raw_response:
            content = raw_response["content"]
            if isinstance(content, str):
                try:
                    candidate = json.loads(content)
                except json.JSONDecodeError as exc:  # pragma: no cover simple branch
                    return None, f"invalid json: {exc}"

        if candidate is None:
            return None, "missing json payload"

        errors = sorted(validator.iter_errors(candidate), key=lambda e: e.path)
        if errors:
            messages = [f"{'/'.join(map(str, err.path))}: {err.message}" for err in errors]
            return None, "; ".join(messages)

        return candidate, None

    async def close(self) -> None:
        """Dispose any instantiated platform clients."""
        for client in self._clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception:  # pragma: no cover - best effort cleanup
                logger.warning(
                    "LLM client close failed", provider=client.platform_name, exc_info=True
                )
        self._clients.clear()

    async def _get_client(self, provider_key: str) -> BasePlatform:
        if provider_key in self._clients:
            return self._clients[provider_key]

        api_key = _API_KEY_LOOKUP.get(provider_key)
        if not api_key:
            raise ValueError(f"API key for provider '{provider_key}' not configured")

        if not PlatformRegistry.is_platform_available(provider_key):
            raise ValueError(f"Provider '{provider_key}' is not registered")

        client = PlatformRegistry.create_platform(provider_key, api_key, config={})
        await client.__aenter__()
        self._clients[provider_key] = client
        return client

    def _resolve_provider_target(self, provider: Optional[str], model: Optional[str]) -> Tuple[str, Optional[str]]:
        if provider:
            if ":" in provider:
                provider_key, model_hint = provider.split(":", 1)
                return provider_key, model_hint
            return provider, model
        if model and ":" in model:
            provider_key, model_hint = model.split(":", 1)
            return provider_key, model_hint
        return self.default_provider, model


__all__ = ["LLMRouter"]
