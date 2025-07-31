from typing import Any, Dict, List, TypedDict, cast

import openai

from app.core.config import settings
from app.services.ai_platforms.base import BasePlatform


class _Message(TypedDict):
    content: str


class _Choice(TypedDict):
    message: _Message


class _OpenAIResponse(TypedDict):
    choices: List[_Choice]


class OpenAIPlatform(BasePlatform):
    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def query(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": question}],
            max_tokens=kwargs.get("max_tokens", 500),
            temperature=kwargs.get("temperature", 0.1),
        )
        return response.model_dump()  # type: ignore

    def extract_text_response(self, raw_response: Dict[str, Any]) -> str:
        typed_response = cast(_OpenAIResponse, raw_response)
        return typed_response["choices"][0]["message"]["content"]
