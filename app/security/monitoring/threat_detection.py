from __future__ import annotations

import re

SUSPICIOUS_PATTERNS = [
    re.compile(r"\b(select|union|drop|insert|update)\b.*--", re.IGNORECASE),
    re.compile(r"<script[\s>]", re.IGNORECASE),
]


def is_suspicious(payload: str) -> bool:
    return any(p.search(payload) for p in SUSPICIOUS_PATTERNS)
