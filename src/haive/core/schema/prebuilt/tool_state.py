import operator
import uuid
from typing import Annotated, Any, ClassVar, Dict, List, Optional, Type, Union

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, BaseToolkit
from langgraph.graph import add_messages
from langgraph.types import Send
from pydantic import BaseModel, Field, model_validator

from haive.core.schema.prebuilt.messages_state import MessagesState
from haive.core.schema.schema_composer import SchemaComposer
from haive.core.schema.state_schema import StateSchema


class ToolState(MessagesState):
    """
    State schema for tool management and execution.
    Extends MessagesState with tool tracking, validation, and routing capabilities.
    """

    # Tool storage - actual tool instances
    tools: Dict[str, Type[BaseTool]] = Field(
        default_factory=dict, description="Available tools indexed by tool name"
    )

    # Tool schemas for validation - models/schemas defining tool parameters
    tool_schemas: Dict[str, Union[Type[BaseTool], Type[BaseModel]]] = Field(
        default_factory=dict,
        description="Tool schemas for validation indexed by tool name",
    )

    # Engine tracking
    engines: Dict[str, Any] = Field(
        default_factory=dict, description="Engines used in this workflow"
    )

    # Tool calls tracking with reducers
    tool_calls: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Current active tool calls"
    )

    validated_tool_calls: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Tool calls that have been validated"
    )

    invalid_tool_calls: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="Tool calls that failed validation with errors",
    )

    completed_tool_calls: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Tool calls that have been executed"
    )

    # Execution tracking
    validation_attempts: Dict[str, int] = Field(
        default_factory=dict, description="Track validation attempts by tool call ID"
    )

    execution_attempts: Dict[str, int] = Field(
        default_factory=dict, description="Track execution attempts by tool call ID"
    )

    # Limits
    max_validation_attempts: int = Field(
        default=3, description="Maximum validation attempts before giving up"
    )

    max_execution_attempts: int = Field(
        default=3, description="Maximum execution attempts before giving up"
    )

    # Routing field - replaces Command.goto
    __goto__: Optional[Union[str, List[str], Send, List[Send]]] = Field(
        default=None, description="Dynamic routing directive"
    )

    # Configuration for state schema
    __shared_fields__ = ["messages", "tools", "engines", "__goto__"]

    __serializable_reducers__ = {
        "messages": "add_messages",
        "tool_calls": "operator.add",
        "validated_tool_calls": "operator.add",
        "invalid_tool_calls": "operator.add",
        "completed_tool_calls": "operator.add",
    }

    __reducer_fields__ = {
        "messages": add_messages,
        "tool_calls": operator.add,
        "validated_tool_calls": operator.add,
        "invalid_tool_calls": operator.add,
        "completed_tool_calls": operator.add,
    }

    # Initialize tool system
    def initialize_tools(self, tools: List[Union[Type[BaseTool], BaseToolkit]]) -> None:
        """
        Initialize tool system with tools and schemas.

        Args:
            tools: List of tools or toolkits to initialize
        """
        # Process tools and toolkits
        for tool_item in tools:
            if isinstance(tool_item, BaseToolkit):
                # Get tools from toolkit
                toolkit_tools = tool_item.get_tools()
                for tool in toolkit_tools:
                    self._register_tool(tool)
            else:
                # Direct tool
                self._register_tool(tool_item)

    def _register_tool(self, tool: Union[Type[BaseTool], BaseTool]) -> None:
        """
        Register a single tool in the state.

        Args:
            tool: Tool to register
        """
        # Handle tool instance vs tool class
        if isinstance(tool, type):
            # Tool class - create instance
            tool_instance = tool()
            tool_class = tool
        else:
            # Tool instance
            tool_instance = tool
            tool_class = tool.__class__

        # Generate tool ID if needed
        tool_name = getattr(tool_instance, "name", None)
        if not tool_name:
            tool_name = f"tool_{uuid.uuid4().hex[:8]}"

        # Store tool and schema
        self.tools[tool_name] = tool_instance
        self.tool_schemas[tool_name] = tool_class

    def register_engine(self, engine: Any, engine_id: Optional[str] = None) -> str:
        """
        Register an engine in the state.

        Args:
            engine: Engine to register
            engine_id: Optional ID for the engine (generated if not provided)

        Returns:
            Engine ID
        """
        # Use provided ID or generate one
        if not engine_id:
            engine_id = getattr(engine, "id", f"engine_{uuid.uuid4().hex[:8]}")

        # Store engine
        self.engines[engine_id] = engine

        # Register any tools from the engine
        if hasattr(engine, "tools") and engine.tools:
            self.initialize_tools(engine.tools)

        return engine_id

    # Tool call management
    def extract_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the last AI message.

        Returns:
            List of tool call dictionaries
        """
        # Get the most recent AI message
        ai_message = self.get_last_ai_message()
        if not ai_message:
            return []

        # Check for tool_calls attribute
        extracted_calls = []
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            extracted_calls = ai_message.tool_calls

        # Check in additional_kwargs
        elif hasattr(
            ai_message, "additional_kwargs"
        ) and ai_message.additional_kwargs.get("tool_calls"):
            extracted_calls = ai_message.additional_kwargs["tool_calls"]

        # Normalize and add IDs if missing
        normalized_calls = []
        for call in extracted_calls:
            # Ensure it's a dictionary
            if isinstance(call, dict):
                call_dict = call
            else:
                # Convert to dictionary
                call_dict = {"name": getattr(call, "name", "unknown_tool")}
                if hasattr(call, "args"):
                    call_dict["args"] = call.args
                if hasattr(call, "id"):
                    call_dict["id"] = call.id

            # Ensure it has an ID
            if "id" not in call_dict:
                call_dict["id"] = f"call_{uuid.uuid4().hex[:8]}"

            normalized_calls.append(call_dict)

        return normalized_calls

    def update_tool_calls(self) -> None:
        """
        Update tool_calls from the last AI message.
        """
        new_calls = self.extract_tool_calls()
        if new_calls:
            self.tool_calls.extend(new_calls)

    def add_validated_call(self, tool_call: Dict[str, Any]) -> None:
        """
        Add a validated tool call.

        Args:
            tool_call: Validated tool call dictionary
        """
        # Add to validated calls
        self.validated_tool_calls.append(tool_call)

        # Remove from pending calls if present
        self._remove_from_pending(tool_call)

    def add_invalid_call(self, tool_call: Dict[str, Any], error: str) -> None:
        """
        Add an invalid tool call with error.

        Args:
            tool_call: Invalid tool call dictionary
            error: Error message
        """
        # Add error to the call
        tool_call_with_error = tool_call.copy()
        tool_call_with_error["error"] = error

        # Add to invalid calls
        self.invalid_tool_calls.append(tool_call_with_error)

        # Remove from pending calls if present
        self._remove_from_pending(tool_call)

        # Track validation attempt
        call_id = tool_call.get("id", "unknown")
        self.validation_attempts[call_id] = self.validation_attempts.get(call_id, 0) + 1

    def add_completed_call(self, tool_call: Dict[str, Any], result: Any) -> None:
        """
        Add a completed tool call with result.

        Args:
            tool_call: Executed tool call dictionary
            result: Execution result
        """
        # Add result to the call
        completed_call = tool_call.copy()
        completed_call["result"] = result

        # Add to completed calls
        self.completed_tool_calls.append(completed_call)

        # Remove from validated calls if present
        self._remove_from_validated(tool_call)

    def _remove_from_pending(self, tool_call: Dict[str, Any]) -> None:
        """Remove a tool call from pending list."""
        call_id = tool_call.get("id")
        if call_id:
            self.tool_calls = [
                call for call in self.tool_calls if call.get("id") != call_id
            ]

    def _remove_from_validated(self, tool_call: Dict[str, Any]) -> None:
        """Remove a tool call from validated list."""
        call_id = tool_call.get("id")
        if call_id:
            self.validated_tool_calls = [
                call for call in self.validated_tool_calls if call.get("id") != call_id
            ]

    def exceeded_validation_attempts(self, call_id: str) -> bool:
        """
        Check if validation attempts exceeded for a call.

        Args:
            call_id: Tool call ID

        Returns:
            True if max attempts exceeded
        """
        return self.validation_attempts.get(call_id, 0) >= self.max_validation_attempts

    def exceeded_execution_attempts(self, call_id: str) -> bool:
        """
        Check if execution attempts exceeded for a call.

        Args:
            call_id: Tool call ID

        Returns:
            True if max attempts exceeded
        """
        return self.execution_attempts.get(call_id, 0) >= self.max_execution_attempts

    # Routing helpers
    def route_to_validation(self) -> None:
        """Set routing to validation node."""
        self.__goto__ = "validation"

    def route_to_execution(self) -> None:
        """Set routing to tool execution node."""
        # If we have multiple validated calls, send them in parallel
        if len(self.validated_tool_calls) > 1:
            # Create Send objects for each tool call
            sends = [Send("tool_executor", call) for call in self.validated_tool_calls]
            self.__goto__ = sends
        elif len(self.validated_tool_calls) == 1:
            # Just route to execution
            self.__goto__ = "tool_executor"
        else:
            # No validated calls, go back to agent
            self.__goto__ = "agent"

    def route_to_agent(self) -> None:
        """Set routing back to agent."""
        self.__goto__ = "agent"

    def route_to_end(self) -> None:
        """Set routing to end."""
        self.__goto__ = "__end__"

    # Flow control
    def process_tool_flow(self) -> None:
        """
        Process the tool flow and set appropriate routing.
        """
        # Extract new tool calls
        self.update_tool_calls()

        # If we have pending tool calls, go to validation
        if self.tool_calls:
            self.route_to_validation()
        # If we have validated tool calls, go to execution
        elif self.validated_tool_calls:
            self.route_to_execution()
        # If we have completed or invalid calls, go to agent
        elif self.completed_tool_calls or self.invalid_tool_calls:
            self.route_to_agent()
        # Otherwise, we're done
        else:
            self.route_to_end()

    # Utility methods
    def has_pending_tools(self) -> bool:
        """Check if there are pending tool calls."""
        return len(self.tool_calls) > 0

    def has_validated_tools(self) -> bool:
        """Check if there are validated tool calls."""
        return len(self.validated_tool_calls) > 0

    def has_invalid_tools(self) -> bool:
        """Check if there are invalid tool calls."""
        return len(self.invalid_tool_calls) > 0

    def has_completed_tools(self) -> bool:
        """Check if there are completed tool calls."""
        return len(self.completed_tool_calls) > 0

    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(tool_name)

    def get_tool_schema_by_name(self, tool_name: str) -> Optional[Type]:
        """Get a tool schema by name."""
        return self.tool_schemas.get(tool_name)

    def clear_tool_status(self) -> None:
        """
        Clear all tool call tracking.
        """
        self.tool_calls = []
        self.validated_tool_calls = []
        self.invalid_tool_calls = []
        self.completed_tool_calls = []
        self.validation_attempts = {}
        self.execution_attempts = {}
