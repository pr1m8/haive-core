"""Prebuilt state schemas for common agent patterns.

This module provides ready-to-use state schemas for various agent architectures:

- BasicAgentState: Simple agent state with messages and context
- MessagesState: Conversation management with LangChain integration
- ToolState: Extended MessagesState with tool management
- MultiAgentStateSchema: State for multi-agent architectures
- MessagesStateWithTokenUsage: MessagesState with token tracking

The messages submodule provides additional functionality:
- TokenUsage: Token tracking and cost calculation
- TokenUsageMixin: Mixin for adding token tracking to any schema
- Enhanced message utilities (if available)
"""

from haive.core.schema.multi_agent_state_schema import MultiAgentStateSchema

# from haive.core.schema.prebuilt.basic_agent_state import BasicAgentState
from haive.core.schema.prebuilt.dynamic_activation_state import DynamicActivationState
from haive.core.schema.prebuilt.llm_state import LLMState

# Import messages module components
from haive.core.schema.prebuilt.messages import (
    MessagesStateWithTokenUsage,
    TokenUsage,
    TokenUsageMixin,
    aggregate_token_usage,
    calculate_token_cost,
    extract_token_usage_from_message,
)
from haive.core.schema.prebuilt.messages_state import MessagesState
from haive.core.schema.prebuilt.meta_state import MetaStateSchema
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState
from haive.core.schema.prebuilt.query_state import (
    QueryComplexity,
    QueryIntent,
    QueryMetrics,
    QueryProcessingConfig,
    QueryProcessingState,
    QueryResult,
    QueryState,
    QueryType,
    RetrievalStrategy,
)
from haive.core.schema.prebuilt.tool_state import ToolState

# Document state components are imported lazily to avoid triggering document system auto-registry
# from haive.core.schema.prebuilt.document_state import (
#     DocumentEngineInputSchema,
#     DocumentEngineOutputSchema,
#     DocumentState,
# )


# Convenient aliases
TokenAwareState = MessagesStateWithTokenUsage  # Shorter name
TokenToolState = ToolState  # Makes it clear it has token tracking
AgentState = LLMState  # Generic agent state with single engine

# Lazy loading for document components to avoid auto-registry initialization
_DOCUMENT_COMPONENTS = {
    "DocumentState",
    "DocumentEngineInputSchema",
    "DocumentEngineOutputSchema",
}


def __getattr__(name: str):
    """Lazy loading for document state components."""
    if name in _DOCUMENT_COMPONENTS:
        from haive.core.schema.prebuilt.document_state import (
            DocumentEngineInputSchema,
            DocumentEngineOutputSchema,
            DocumentState,
        )

        return locals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "AgentState",
    "DocumentEngineInputSchema",
    "DocumentEngineOutputSchema",
    # Document and query schemas - lazy loaded via __getattr__
    "DocumentState",
    # Core prebuilt schemas
    # "BasicAgentState",
    "DynamicActivationState",
    "LLMState",
    "MessagesState",
    "MessagesStateWithTokenUsage",
    "MetaStateSchema",
    "MultiAgentState",
    "MultiAgentStateSchema",
    "QueryComplexity",
    "QueryIntent",
    "QueryMetrics",
    "QueryProcessingConfig",
    "QueryProcessingState",
    "QueryResult",
    "QueryState",
    "QueryType",
    "RetrievalStrategy",
    # Aliases
    "TokenAwareState",
    "TokenToolState",
    # Token usage components
    "TokenUsage",
    "TokenUsageMixin",
    "ToolState",
    "aggregate_token_usage",
    "calculate_token_cost",
    # Token usage utilities
    "extract_token_usage_from_message",
]
