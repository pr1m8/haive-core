import inspect
import json
from typing import Any

from pydantic import BaseModel


def ensure_json_serializable(obj: Any) -> Any:
    """Ensure object is JSON serializable, converting non-serializable objects."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        if isinstance(obj, BaseModel):
            return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
        if inspect.isfunction(obj) or callable(obj):
            # Handle function objects by converting to string representation
            if hasattr(obj, "__name__"):
                return f"<function {obj.__name__}>"
            return "<callable object>"
        if isinstance(obj, dict):
            return {k: ensure_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list) or isinstance(obj, tuple):
            return [ensure_json_serializable(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return ensure_json_serializable(vars(obj))
        if hasattr(obj, "__str__"):
            return str(obj)
        return "Unserializable Object"
