"""Field Registry for standardized field definitions across Haive.

This module provides a centralized registry of commonly used field definitions
that can be referenced by nodes, engines, and schema composers. This ensures
consistency and allows for selective state schema composition.

Key benefits:
- Standardized field definitions across the framework
- Selective inclusion in state schemas (only what's needed)
- Type safety with proper generics
- Token counting integration for messages
- Backwards compatibility
"""

from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from haive.core.schema.field_definition import FieldDefinition

# Type variables for generics
T = TypeVar("T")


class StandardFields:
    """Registry of standard field definitions used across Haive.

    Each field is defined with:
    - name: field name in snake_case
    - type: proper Python type annotation
    - description: human-readable description
    - default: default value or factory
    - metadata: additional field metadata
    """

    # ========================================================================
    # CONVERSATION & MESSAGING FIELDS
    # ========================================================================

    @classmethod
    def messages(cls, use_enhanced: bool = True) -> FieldDefinition:
        """Standard messages field for conversation history.

        Args:
            use_enhanced: Whether to use the enhanced MessageList with token counting and metadata
        """
        if use_enhanced:
            from haive.core.schema.prebuilt.messages.messages_state import MessageList

            # Use the enhanced MessageList with all the token counting and
            # metadata features
            field_type = MessageList
            metadata = {
                "token_counting": True,
                "engine_attribution": True,
                "reducer": "add_messages",
                "shared": True,
            }
            default_factory = MessageList
        else:
            # Basic message list for backwards compatibility - use AnyMessage
            from langchain_core.messages import AnyMessage

            field_type = list[AnyMessage]
            metadata = {"reducer": "add_messages", "shared": True}
            default_factory = list

        # Note: shared is already in metadata, so don't pass it twice
        # Extract reducer_name from metadata if needed
        metadata.pop("reducer", None)

        return FieldDefinition(
            name="messages",
            field_type=field_type,
            default_factory=default_factory,
            description="Conversation message history with enhanced features",
            **metadata,
        )

    @classmethod
    def ai_message(cls) -> FieldDefinition:
        """Single AI message output field."""
        from langchain_core.messages import AIMessage

        return FieldDefinition(
            name="ai_message",
            field_type=Optional[AIMessage],
            default=None,
            description="Generated AI response message",
        )

    @classmethod
    def human_message(cls) -> FieldDefinition:
        """Single human message input field."""
        from langchain_core.messages import HumanMessage

        return FieldDefinition(
            name="human_message",
            field_type=Optional[HumanMessage],
            default=None,
            description="Human input message",
        )

    # ========================================================================
    # CONTEXT & RETRIEVAL FIELDS
    # ========================================================================

    @classmethod
    def context(cls) -> FieldDefinition:
        """Retrieved context documents."""
        return FieldDefinition(
            name="context",
            field_type=list[str],
            default_factory=list,
            description="Retrieved document contexts",
            reducer_name="extend",
        )

    @classmethod
    def query(cls) -> FieldDefinition:
        """User query string."""
        return FieldDefinition(
            name="query",
            field_type=str,
            default="",
            description="User search/question query",
        )

    @classmethod
    def documents(cls) -> FieldDefinition:
        """Retrieved documents with metadata."""
        from langchain_core.documents import Document

        return FieldDefinition(
            name="documents",
            field_type=list[Document],
            default_factory=list,
            description="Retrieved documents with metadata",
            reducer_name="extend",
        )

    # ========================================================================
    # PLANNING & REASONING FIELDS
    # ========================================================================

    @classmethod
    def plan_steps(cls) -> FieldDefinition:
        """Generated plan steps."""
        return FieldDefinition(
            name="plan_steps",
            field_type=list[str],
            default_factory=list,
            description="Generated planning steps",
            reducer_name="extend",
        )

    @classmethod
    def thoughts(cls) -> FieldDefinition:
        """Agent reasoning thoughts."""
        return FieldDefinition(
            name="thoughts",
            field_type=str,
            default="",
            description="Agent internal reasoning/thoughts",
        )

    @classmethod
    def observations(cls) -> FieldDefinition:
        """Agent observations from tools/environment."""
        return FieldDefinition(
            name="observations",
            field_type=list[str],
            default_factory=list,
            description="Observations from tool executions",
            reducer_name="extend",
        )

    # ========================================================================
    # ENGINE & TOOL FIELDS
    # ========================================================================

    @classmethod
    def engine_name(cls) -> FieldDefinition:
        """Name of the engine being used."""
        return FieldDefinition(
            name="engine_name",
            field_type=str,
            default="",
            description="Name of the active engine",
        )

    @classmethod
    def tool_routes(cls) -> FieldDefinition:
        """Tool routing configuration."""
        return FieldDefinition(
            name="tool_routes",
            field_type=dict[str, str],
            default_factory=dict,
            description="Tool name to route mapping",
        )

    @classmethod
    def available_nodes(cls) -> FieldDefinition:
        """Available graph nodes."""
        return FieldDefinition(
            name="available_nodes",
            field_type=list[str],
            default_factory=list,
            description="List of available graph nodes",
        )

    # ========================================================================
    # STRUCTURED OUTPUT FIELDS
    # ========================================================================

    @classmethod
    def structured_output(
        cls, model_class: type[BaseModel], field_name: str | None = None
    ) -> FieldDefinition:
        """Create a structured output field for a Pydantic model.

        Args:
            model_class: The Pydantic model class
            field_name: Optional custom field name (defaults to snake_case model name)
        """
        if not field_name:
            from haive.core.schema.field_utils import create_field_name_from_model

            field_name = create_field_name_from_model(model_class)

        return FieldDefinition(
            name=field_name,
            field_type=Optional[model_class],
            default=None,
            description=f"Structured output of type {model_class.__name__}",
            structured_model=model_class.__name__,
        )


class FieldRegistry:
    """Dynamic field registry for custom field definitions.

    This complements StandardFields by allowing registration of custom field definitions
    at runtime.
    """

    _registry: dict[str, FieldDefinition] = {}

    @classmethod
    def register(cls, field_def: FieldDefinition) -> None:
        """Register a custom field definition."""
        cls._registry[field_def.name] = field_def

    @classmethod
    def get(cls, name: str) -> FieldDefinition | None:
        """Get a registered field definition."""
        return cls._registry.get(name)

    @classmethod
    def list_fields(cls) -> list[str]:
        """List all registered field names."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        cls._registry.clear()


# Convenience function for getting standard fields
def get_standard_field(name: str, **kwargs) -> FieldDefinition | None:
    """Get a standard field definition by name.

    Args:
        name: Standard field name (e.g., 'messages', 'context', 'query')
        **kwargs: Additional arguments passed to the field method

    Returns:
        FieldDefinition or None if field not found
    """
    method = getattr(StandardFields, name, None)
    if method and callable(method):
        return method(**kwargs)
    return None


# Export commonly used field sets
class CommonFieldSets:
    """Pre-defined sets of fields for common use cases."""

    LLM_BASIC = [StandardFields.messages(use_enhanced=True)]
    LLM_WITH_CONTEXT = [
        StandardFields.messages(use_enhanced=True),
        StandardFields.context(),
    ]
    RAG_INPUT = [StandardFields.query(), StandardFields.messages(use_enhanced=True)]
    RAG_OUTPUT = [
        StandardFields.context(),
        StandardFields.documents(),
        StandardFields.ai_message(),
    ]
    PLANNER_INPUT = [
        StandardFields.messages(use_enhanced=True),
        StandardFields.context(),
    ]
    PLANNER_OUTPUT = [StandardFields.plan_steps(), StandardFields.thoughts()]


# Prebuilt state schemas registry
class PrebuiltStates:
    """Registry of prebuilt state schemas for common use cases.

    Hierarchy:
    - MessagesState (basic, no tokens)
    - MessagesStateWithTokenUsage (with token tracking)
      - LLMState (single engine + tokens + thresholds)
        - ToolState (tools + LLM features)
    """

    @classmethod
    def messages_with_tokens(cls) -> Any:
        """Get MessagesStateWithTokenUsage for token-aware conversations."""
        from haive.core.schema.prebuilt.messages.messages_with_token_usage import (
            MessagesStateWithTokenUsage,
        )

        return MessagesStateWithTokenUsage

    @classmethod
    def llm_state(cls) -> Any:
        """Get LLMState for single-engine LLM agents with token tracking and model
        awareness.
        """
        from haive.core.schema.prebuilt.llm_state import LLMState

        return LLMState

    @classmethod
    def tool_state(cls) -> Any:
        """Get ToolState for tool-using agents with LLM features, tools, and token
        tracking.
        """
        from haive.core.schema.prebuilt.tool_state import ToolState

        return ToolState

    @classmethod
    def base_messages_state(cls) -> Any:
        """Get basic MessagesState without token tracking."""
        from haive.core.schema.prebuilt.messages_state import MessagesState

        return MessagesState
