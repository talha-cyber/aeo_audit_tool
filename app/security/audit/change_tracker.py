from __future__ import annotations

from typing import Any, Dict


def diff_dict(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    changed = {}
    keys = set(old) | set(new)
    for k in keys:
        if old.get(k) != new.get(k):
            changed[k] = {"old": old.get(k), "new": new.get(k)}
    return changed
