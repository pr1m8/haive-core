"""Node Schema Composer - Flexible I/O configuration for graph nodes."""

from haive.core.graph.node.composer.extract_functions import (
    ExtractFunctions,
    extract_conditional,
    extract_functions,
    extract_messages_content,
    extract_multi_field,
    extract_simple_field,
    extract_typed,
    extract_with_path,
    extract_with_projection,
)
from haive.core.graph.node.composer.field_mapping import FieldMapping
from haive.core.graph.node.composer.path_resolver import PathResolver
from haive.core.graph.node.composer.protocols import (
    ExtractFunction,
    TransformFunction,
    UpdateFunction,
)
from haive.core.graph.node.composer.update_functions import (
    UpdateFunctions,
    update_conditional,
    update_functions,
    update_hierarchical,
    update_messages_append,
    update_multi_field,
    update_simple_field,
    update_type_aware,
    update_with_path,
    update_with_transform,
)

__all__ = [
    "FieldMapping",
    "PathResolver",
    "ExtractFunction",
    "UpdateFunction",
    "TransformFunction",
    "ExtractFunctions",
    "extract_functions",
    "extract_simple_field",
    "extract_with_path",
    "extract_with_projection",
    "extract_messages_content",
    "extract_conditional",
    "extract_multi_field",
    "extract_typed",
    "UpdateFunctions",
    "update_functions",
    "update_simple_field",
    "update_with_path",
    "update_messages_append",
    "update_type_aware",
    "update_conditional",
    "update_multi_field",
    "update_with_transform",
    "update_hierarchical",
]
