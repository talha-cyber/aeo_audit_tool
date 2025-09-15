from __future__ import annotations

from typing import Any, Dict, Tuple, Type

from pydantic import BaseModel, ValidationError


def validate_schema(data: Dict[str, Any], model: Type[BaseModel]) -> Tuple[bool, Any]:
    try:
        obj = model.model_validate(data)
        return True, obj
    except ValidationError as e:
        return False, e.errors()
