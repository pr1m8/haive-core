"""Haive Schema System.

This package provides a powerful foundation for dynamic state management in AI agents
and workflows. It extends Pydantic's model system with features specifically designed
for graph-based AI workflows, including field sharing between graphs, reducer functions
for state updates, and engine I/O tracking.

The schema system enables fully dynamic and serializable state schemas that can be
composed, modified, and extended at runtime, making it ideal for complex agent
architectures and nested workflows.

Key Components:
    - StateSchema: Base class that extends Pydantic models with sharing, reducers,
      and I/O tracking
    - SchemaComposer: Utility for building schemas from components dynamically
    - StateSchemaManager: Tool for manipulating schemas at runtime
    - MultiAgentStateSchema: Enhanced schema for multi-agent architectures
    - AgentSchemaComposer: Schema composer specialized for agent architectures
    - FieldDefinition: Representation of field type, default, and metadata
    - FieldExtractor: Utility for extracting fields from various sources
    - Field Utilities: Common functions for field manipulation

For detailed usage examples, see the haive_schema_system_readme.md file.
"""

# Import agent schema composer
from haive.core.schema.agent_schema_composer import AgentSchemaComposer, BuildMode

# Import core components for easy access
from haive.core.schema.field_definition import FieldDefinition
from haive.core.schema.field_extractor import FieldExtractor
from haive.core.schema.field_utils import (
    create_annotated_field,
    create_field,
    extract_type_metadata,
    get_common_reducers,
    infer_field_type,
    resolve_reducer,
)
from haive.core.schema.multi_agent_state_schema import (
    MultiAgentSchemaComposer,
    MultiAgentStateSchema,
)

# Import prebuilt states
from haive.core.schema.prebuilt.basic_agent_state import BasicAgentState

# Import preserve messages reducer
from haive.core.schema.preserve_messages_reducer import preserve_messages_reducer
from haive.core.schema.schema_composer import SchemaComposer
from haive.core.schema.schema_manager import StateSchemaManager
from haive.core.schema.state_schema import StateSchema
from haive.core.schema.ui import SchemaUI

__all__ = [
    "FieldDefinition",
    "FieldExtractor",
    "create_field",
    "create_annotated_field",
    "extract_type_metadata",
    "infer_field_type",
    "get_common_reducers",
    "resolve_reducer",
    "SchemaComposer",
    "StateSchemaManager",
    "StateSchema",
    "MultiAgentStateSchema",
    "MultiAgentSchemaComposer",
    "AgentSchemaComposer",
    "BuildMode",
    "preserve_messages_reducer",
    "SchemaUI",
]
