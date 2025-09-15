from __future__ import annotations

import html
import re
from typing import Optional

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(value: Optional[str], *, max_length: int = 2000) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    v = CONTROL_CHARS.sub("", v)
    v = html.escape(v)
    if len(v) > max_length:
        v = v[:max_length]
    return v
