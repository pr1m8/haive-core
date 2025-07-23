"""Common utilities for the graph system."""

from haive.core.graph.common.field_utils import (
    extract_base_field,
    extract_field,
    extract_fields_from_function,
    get_field_value,
    get_last_message_content,
)
from haive.core.graph.common.references import CallableReference, TypeReference
from haive.core.graph.common.serialization import (
    ensure_serializable,
    from_json,
    to_json,
)

__all__ = [
    "CallableReference",
    "TypeReference",
    "ensure_serializable",
    "extract_base_field",
    "extract_field",
    "extract_fields_from_function",
    "from_json",
    "get_field_value",
    "get_last_message_content",
    "to_json",
]
