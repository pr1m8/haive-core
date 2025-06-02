"""
Base models for component discovery.

Defines Pydantic models for representing discovered components and their metadata.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, computed_field


class ComponentType(str, Enum):
    """Types of components that can be discovered."""

    TOOL = "tool"
    TOOLKIT = "toolkit"
    RETRIEVER = "retriever"
    DOCUMENT_LOADER = "document_loader"
    DOCUMENT_TRANSFORMER = "document_transformer"
    TEXT_SPLITTER = "text_splitter"
    EMBEDDING = "embedding"
    LLM = "llm"
    AGENT = "agent"


class ParameterInfo(BaseModel):
    """Information about a method parameter."""

    name: str = Field(..., description="Parameter name")
    type_hint: str = Field(default="Any", description="Type hint as string")
    default_value: Any = Field(default=..., description="Default value if any")
    is_required: bool = Field(
        default=True, description="Whether the parameter is required"
    )
    description: Optional[str] = Field(
        default=None, description="Parameter description from docstring"
    )


class MethodInfo(BaseModel):
    """Information about a component method."""

    name: str = Field(..., description="Method name")
    parameters: Dict[str, ParameterInfo] = Field(
        default_factory=dict, description="Method parameters"
    )
    return_type: str = Field(default="Any", description="Return type as string")
    docstring: str = Field(default="", description="Method docstring")
    is_async: bool = Field(default=False, description="Whether the method is async")
    source_code: Optional[str] = Field(
        default=None, description="Source code of the method"
    )
    signature_str: Optional[str] = Field(
        default=None, description="String representation of signature"
    )


class ComponentMetadata(BaseModel):
    """Metadata about a discovered component."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this component",
    )
    name: str = Field(..., description="Name of the component")
    display_name: Optional[str] = Field(
        default=None, description="Human-readable display name"
    )
    component_type: ComponentType = Field(..., description="Type of component")
    module_path: str = Field(..., description="Full module path")
    class_name: str = Field(..., description="Class name")
    description: str = Field(default="", description="Component description")

    # Schema information
    schema_json: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON schema if available"
    )
    is_serializable: bool = Field(
        default=False, description="Whether schema is serializable"
    )
    forced_serializable: bool = Field(
        default=False, description="Whether serialization was forced"
    )

    # Methods
    methods: Dict[str, MethodInfo] = Field(
        default_factory=dict, description="Component methods"
    )

    # Environment variables
    env_vars: Set[str] = Field(
        default_factory=set, description="Required environment variables"
    )
    env_vars_found: Set[str] = Field(
        default_factory=set, description="Environment variables that are set"
    )
    env_vars_missing: Set[str] = Field(
        default_factory=set, description="Environment variables that are missing"
    )

    # Categorization
    category: str = Field(default="general", description="Component category")
    tags: List[str] = Field(default_factory=list, description="Component tags")

    # Source information
    source_code: Optional[str] = Field(
        default=None, description="Source code if available"
    )
    parent_classes: List[str] = Field(
        default_factory=list, description="Parent class names"
    )

    # Usage information
    requires_api_key: bool = Field(
        default=False, description="Whether component requires API keys"
    )
    toolkit_name: Optional[str] = Field(
        default=None, description="Name of toolkit if part of one"
    )
    key_parameters: List[str] = Field(
        default_factory=list, description="Key parameters for this component"
    )

    # Document loader specific
    loader_methods: Dict[str, bool] = Field(
        default_factory=dict, description="Available loader methods"
    )

    # Text splitter specific
    splitter_methods: Dict[str, bool] = Field(
        default_factory=dict, description="Available splitter methods"
    )

    # Retriever specific
    retriever_methods: Dict[str, bool] = Field(
        default_factory=dict, description="Available retriever methods"
    )

    # Metadata
    error: Optional[str] = Field(
        default=None, description="Error if any during discovery"
    )
    discovered_at: datetime = Field(
        default_factory=datetime.now, description="Time of discovery"
    )

    @computed_field
    def key(self) -> str:
        """Get a unique key combining type and name."""
        return f"{self.component_type}:{self.name}"

    def model_post_init(self, __context):
        """Initialize after model creation."""
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()


class ComponentCollection(BaseModel):
    """Collection of component metadata."""

    components: Dict[str, ComponentMetadata] = Field(
        default_factory=dict, description="Components by ID"
    )
    discovery_timestamp: datetime = Field(
        default_factory=datetime.now, description="Time of discovery"
    )

    def add(self, component: ComponentMetadata) -> None:
        """Add a component to the collection."""
        self.components[component.id] = component

    def get(self, component_id: str) -> Optional[ComponentMetadata]:
        """Get a component by ID."""
        return self.components.get(component_id)

    def get_by_key(self, component_key: str) -> Optional[ComponentMetadata]:
        """Get a component by type:name key."""
        for component in self.components.values():
            if component.key == component_key:
                return component
        return None

    def list_by_type(self, component_type: ComponentType) -> List[ComponentMetadata]:
        """List all components of a specific type."""
        return [
            c for c in self.components.values() if c.component_type == component_type
        ]

    def list_by_category(self, category: str) -> List[ComponentMetadata]:
        """List all components in a specific category."""
        return [c for c in self.components.values() if c.category == category]

    def list_by_tag(self, tag: str) -> List[ComponentMetadata]:
        """List all components with a specific tag."""
        return [c for c in self.components.values() if tag in c.tags]

    def count_by_type(self) -> Dict[ComponentType, int]:
        """Count components by type."""
        counts = {t: 0 for t in ComponentType}
        for component in self.components.values():
            counts[component.component_type] += 1
        return counts
