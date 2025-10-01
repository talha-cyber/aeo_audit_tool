"""Utility helpers shared across report generation modules."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from xml.sax.saxutils import escape


def extract_question_context(response: Any) -> Dict[str, Optional[str]]:
    """Derive question metadata associated with a response.

    Args:
        response: Response ORM object with optional relationship/metadata.

    Returns:
        Dictionary containing question-related context (id, text, provider, etc.).
    """

    question = getattr(response, "question", None)
    metadata = getattr(response, "response_metadata", None) or {}
    raw_response = getattr(response, "raw_response", None)

    question_text: Optional[str] = None
    provider: Optional[str] = None
    category: Optional[str] = None
    question_type: Optional[str] = None
    question_id: Optional[str] = None

    if question is not None:
        question_text = getattr(question, "question_text", None)
        provider = getattr(question, "provider", None)
        category = getattr(question, "category", None)
        question_type = getattr(question, "question_type", None)
        question_id = getattr(question, "id", None)

    if not question_text:
        question_text = metadata.get("question_preview") or metadata.get("prompt")

    if not provider:
        provider = metadata.get("provider")

    if not category:
        category = metadata.get("category")

    if not question_type:
        question_type = metadata.get("question_type")

    if not question_id:
        question_id = metadata.get("question_id")

    # Attempt to derive question text from raw response payloads when available.
    if not question_text and isinstance(raw_response, dict):
        question_text = raw_response.get("prompt")

        if not question_text:
            messages = raw_response.get("messages")
            if isinstance(messages, list):
                for message in reversed(messages):
                    if isinstance(message, dict) and message.get("role") == "user":
                        question_text = message.get("content")
                        if isinstance(question_text, list):
                            # Some providers return rich message payloads (e.g., content list)
                            question_text = "\n".join(
                                fragment.get("text", "")
                                for fragment in question_text
                                if isinstance(fragment, dict)
                            )
                        break

    return {
        "question_id": question_id,
        "question": question_text,
        "provider": provider,
        "category": category,
        "question_type": question_type,
    }


def group_records(
    records: List[Dict[str, Any]],
    group_by: str,
    grouping_rules: Optional[Dict[str, Callable[[Dict[str, Any]], str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group records deterministically for appendix-style sections."""

    default_rules: Dict[str, Callable[[Dict[str, Any]], str]] = {
        "provider": lambda item: item.get("provider") or "Unknown Provider",
        "question_type": lambda item: item.get("question_type") or "Uncategorized",
        "category": lambda item: item.get("category") or "Uncategorized",
    }

    rules = {**default_rules, **(grouping_rules or {})}
    key_fn = rules.get(group_by, lambda item: item.get(group_by) or "Other")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        key = key_fn(record) or "Other"
        grouped.setdefault(key, []).append(record)

    sorted_group_keys = sorted(grouped.keys(), key=lambda value: value.lower())
    sorted_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for key in sorted_group_keys:
        sorted_grouped[key] = sorted(
            grouped[key], key=lambda item: (item.get("question") or "").lower()
        )

    return sorted_grouped


def format_block_text(text: Optional[str]) -> str:
    """Escape text for block usage while preserving basic line breaks."""

    if not text:
        return ""

    escaped = escape(text)
    return escaped.replace("\n", "<br/>")


def format_inline_text(text: Optional[str]) -> str:
    """Escape text intended for inline usage."""

    if text is None:
        return ""

    return escape(text)
