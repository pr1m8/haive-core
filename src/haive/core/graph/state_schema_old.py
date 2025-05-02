# src/haive/core/schema/StateSchema.py

import json
import logging
from collections.abc import Sequence
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Optional,
    TypeVar,
)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import Command, Send
from pydantic import BaseModel, ValidationError

from haive.core.graph.state.StateSchemaManager import (
    StateSchemaManager as SchemaManager,
)

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generics
T = TypeVar("T", bound=BaseModel)


class StateSchema(Generic[T]):
    """A specialized schema for LangGraph states with enhanced features:
    - Built-in standard fields for agent state (messages, memory, etc.)
    - Type-safe state manipulation methods
    - Helper methods for command generation and routing
    - Serialization/deserialization utilities
    - Integration with document stores and memory
    """

    # Registry of predefined schemas
    _schema_registry: ClassVar[dict[str, type[BaseModel]]] = {}

    def __init__(
        self,
        schema_data: dict[str, Any] | type[BaseModel] | SchemaManager | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        include_standard_fields: bool = True,
    ):
        """Initialize a new StateSchema.

        Args:
            schema_data: Base schema definition (dict, model, or SchemaManager)
            name: Name for the schema
            config: Configuration options
            include_standard_fields: Whether to include standard agent fields
        """
        self.name = name or "AgentState"
        self.config = config or {}

        # Initialize the schema manager
        if isinstance(schema_data, SchemaManager):
            self.schema_manager = schema_data
        else:
            self.schema_manager = SchemaManager(
                data=schema_data, name=self.name, config=self.config
            )

        # Add standard fields if requested
        if include_standard_fields:
            self._add_standard_fields()

        # The model will be lazily created when needed
        self._model = None

    def _add_standard_fields(self):
        """Add standard fields used in most agent states."""
        # Add messages field if not present
        if not self.schema_manager.has_field("messages"):
            self.schema_manager.add_field(
                "messages", Annotated[Sequence[BaseMessage], add_messages], default=[]
            )
            logger.debug("Added standard messages field")

        # Add error tracking field
        if not self.schema_manager.has_field("error"):
            self.schema_manager.add_field("error", Optional[str], default=None)
            logger.debug("Added standard error field")

        # Add step tracking fields
        if not self.schema_manager.has_field("current_step"):
            self.schema_manager.add_field("current_step", int, default=0)
            logger.debug("Added standard current_step field")

        if not self.schema_manager.has_field("max_steps"):
            self.schema_manager.add_field("max_steps", int, default=10)
            logger.debug("Added standard max_steps field")

        # Add memory field for state persistence
        if not self.schema_manager.has_field("memory"):
            self.schema_manager.add_field(
                "memory", dict[str, Any], default_factory=dict
            )
            logger.debug("Added standard memory field")

    @property
    def model(self) -> type[BaseModel]:
        """Get the Pydantic model for this schema."""
        if self._model is None:
            self._model = self.schema_manager.get_model()
        return self._model

    def add_field(
        self, name: str, field_type: type, default: Any = None, required: bool = False
    ) -> "StateSchema":
        """Add a field to the schema."""
        self.schema_manager.add_field(name, field_type, default, required)
        # Reset model since schema changed
        self._model = None
        return self

    def add_document_field(self, name: str = "documents") -> "StateSchema":
        """Add a field for document storage."""
        self.schema_manager.add_field(name, list[dict[str, Any]], default_factory=list)
        logger.debug(f"Added document field: {name}")
        # Reset model since schema changed
        self._model = None
        return self

    def add_tool_fields(self) -> "StateSchema":
        """Add fields for tool usage."""
        # Field for tool results
        if not self.schema_manager.has_field("tool_results"):
            self.schema_manager.add_field(
                "tool_results", list[dict[str, Any]], default_factory=list
            )
            logger.debug("Added tool_results field")

        # Field for available tools
        if not self.schema_manager.has_field("available_tools"):
            self.schema_manager.add_field(
                "available_tools", list[str], default_factory=list
            )
            logger.debug("Added available_tools field")

        # Field for tool calls
        if not self.schema_manager.has_field("tool_calls"):
            self.schema_manager.add_field(
                "tool_calls", list[dict[str, Any]], default_factory=list
            )
            logger.debug("Added tool_calls field")

        # Reset model since schema changed
        self._model = None
        return self

    def add_output_fields(self, parser_type: type | None = None) -> "StateSchema":
        """Add fields for output management."""
        # Add standard output field
        if not self.schema_manager.has_field("output"):
            self.schema_manager.add_field("output", str, default="")
            logger.debug("Added output field")

        # Add parsed output field if parser type provided
        if parser_type and not self.schema_manager.has_field("parsed_output"):
            self.schema_manager.add_field(
                "parsed_output", Optional[parser_type], default=None
            )
            logger.debug(f"Added parsed_output field with type {parser_type.__name__}")

        # Reset model since schema changed
        self._model = None
        return self

    def add_agent_fields(self) -> "StateSchema":
        """Add fields specific to agent operation."""
        # Add field for agent status
        if not self.schema_manager.has_field("status"):
            self.schema_manager.add_field("status", str, default="idle")
            logger.debug("Added status field")

        # Add field for agent config
        if not self.schema_manager.has_field("agent_config"):
            self.schema_manager.add_field(
                "agent_config", dict[str, Any], default_factory=dict
            )
            logger.debug("Added agent_config field")

        # Reset model since schema changed
        self._model = None
        return self

    def validate_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate a state dict against this schema."""
        try:
            model = self.model
            instance = model.model_validate(state)
            return instance.model_dump()
        except ValidationError as e:
            error_msg = f"State validation failed: {e}"
            logger.error(error_msg)
            # Return original with error field
            return {**state, "error": error_msg}

    # Command creation methods
    def create_command(
        self,
        state: dict[str, Any],
        updates: dict[str, Any] | None = None,
        goto: str | Send | None = None,
    ) -> Command:
        """Create a command for state updates and routing.

        Args:
            state: Current state
            updates: Fields to update in state
            goto: Where to route next

        Returns:
            Command for LangGraph
        """
        # Apply state updates
        updated_state = {**state}
        if updates:
            for key, value in updates.items():
                if key in state:
                    updated_state[key] = value

        # If goto is not specified, use END
        if goto is None:
            goto = END

        return Command(update=updated_state, goto=goto)

    def create_error_command(
        self, state: dict[str, Any], error_message: str, goto: str | Send | None = None
    ) -> Command:
        """Create an error command.

        Args:
            state: Current state
            error_message: Error message
            goto: Where to route (default: END)

        Returns:
            Command with error information
        """
        # Apply state updates with error
        updated_state = {**state, "error": error_message}

        # Add error as AI message if messages exists
        if "messages" in state:
            error_msg = AIMessage(content=f"Error: {error_message}")
            updated_state["messages"] = list(state["messages"]) + [error_msg]

        # If goto is not specified, use END
        if goto is None:
            goto = END

        return Command(update=updated_state, goto=goto)

    def create_message_command(
        self,
        state: dict[str, Any],
        message: str | BaseMessage,
        goto: str | Send | None = None,
    ) -> Command:
        """Create a command with a new message.

        Args:
            state: Current state
            message: Message to add (string or BaseMessage)
            goto: Where to route (default: END)

        Returns:
            Command with updated messages
        """
        # Ensure we have a BaseMessage
        if isinstance(message, str):
            message = AIMessage(content=message)

        # Apply state updates with new message
        updated_state = {**state}

        # Add message if messages field exists
        if "messages" in state:
            updated_state["messages"] = list(state["messages"]) + [message]

        # If goto is not specified, use END
        if goto is None:
            goto = END

        return Command(update=updated_state, goto=goto)

    # State manipulation methods
    def step_increment(self, state: dict[str, Any]) -> dict[str, Any]:
        """Increment the current step in state."""
        updated_state = {**state}
        if "current_step" in state:
            updated_state["current_step"] = state["current_step"] + 1
        return updated_state

    def add_to_memory(
        self, state: dict[str, Any], key: str, value: Any
    ) -> dict[str, Any]:
        """Add a value to the memory dict in state."""
        updated_state = {**state}
        if "memory" in state:
            memory = dict(state["memory"])
            memory[key] = value
            updated_state["memory"] = memory
        return updated_state

    def has_reached_max_steps(self, state: dict[str, Any]) -> bool:
        """Check if state has reached maximum steps."""
        if "current_step" in state and "max_steps" in state:
            return state["current_step"] >= state["max_steps"]
        return False

    def has_error(self, state: dict[str, Any]) -> bool:
        """Check if state has an error."""
        return "error" in state and state["error"] is not None

    def get_last_message(self, state: dict[str, Any]) -> BaseMessage | None:
        """Get the last message from state."""
        if state.get("messages"):
            return state["messages"][-1]
        return None

    def get_last_user_message(self, state: dict[str, Any]) -> str | None:
        """Get the content of the last user message."""
        if "messages" not in state:
            return None

        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage) or (
                hasattr(message, "type") and message.type == "human"
            ):
                return message.content
        return None

    def get_system_prompt(self, state: dict[str, Any]) -> str | None:
        """Get the system prompt from messages if present."""
        if "messages" not in state:
            return None

        for message in state["messages"]:
            if isinstance(message, SystemMessage) or (
                hasattr(message, "type") and message.type == "system"
            ):
                return message.content
        return None

    # Serialization methods
    def to_dict(self) -> dict[str, Any]:
        """Serialize the schema to a dictionary."""
        return {
            "name": self.name,
            "config": self.config,
            "fields": {
                name: str(field_type)
                for name, (field_type, _) in self.schema_manager.fields.items()
            },
        }

    def to_json(self) -> str:
        """Serialize the schema to JSON."""
        schema_dict = self.to_dict()
        return json.dumps(schema_dict, indent=2)

    # Graph integration
    def create_state_graph(self) -> StateGraph:
        """Create a StateGraph with this schema."""
        return StateGraph(self.model)

    def create_input_output_graph(
        self,
        input_model: type[BaseModel] | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> StateGraph:
        """Create a StateGraph with input and output schemas.

        Args:
            input_model: Optional input model
            output_model: Optional output model

        Returns:
            StateGraph with input and output schemas
        """
        if input_model and output_model:
            return StateGraph(self.model, input=input_model, output=output_model)
        if input_model:
            return StateGraph(self.model, input=input_model)
        if output_model:
            return StateGraph(self.model, output=output_model)
        return StateGraph(self.model)

    # Factory methods
    @classmethod
    def create_empty(cls, name: str | None = None) -> "StateSchema":
        """Create an empty state schema with standard fields."""
        return cls(name=name or "EmptyState")

    @classmethod
    def create_chat_schema(cls, name: str | None = None) -> "StateSchema":
        """Create a state schema for chat agents."""
        schema = cls(name=name or "ChatState")

        # Add chat-specific fields
        schema.add_field("chat_history", list[dict[str, Any]], default_factory=list)
        schema.add_field("user_id", Optional[str], default=None)
        schema.add_field("session_id", Optional[str], default=None)

        return schema

    @classmethod
    def create_tool_schema(cls, name: str | None = None) -> "StateSchema":
        """Create a state schema for tool-using agents."""
        schema = cls(name=name or "ToolState")

        # Add tool-specific fields
        schema.add_tool_fields()

        return schema

    @classmethod
    def create_rag_schema(cls, name: str | None = None) -> "StateSchema":
        """Create a state schema for retrieval-augmented generation."""
        schema = cls(name=name or "RAGState")

        # Add RAG-specific fields
        schema.add_document_field("documents")
        schema.add_field("context", str, default="")
        schema.add_field("query", str, default="")
        schema.add_field("search_results", list[dict[str, Any]], default_factory=list)

        return schema

    @classmethod
    def register_schema(cls, name: str, model: type[BaseModel]) -> None:
        """Register a predefined schema for reuse."""
        cls._schema_registry[name] = model
        logger.info(f"Registered schema: {name}")

    @classmethod
    def get_registered_schema(cls, name: str) -> type[BaseModel] | None:
        """Get a registered schema by name."""
        return cls._schema_registry.get(name)

    @classmethod
    def list_registered_schemas(cls) -> list[str]:
        """List all registered schema names."""
        return list(cls._schema_registry.keys())
