# src/haive/core/schema/prebuilt/tool_state.py

import operator
import uuid
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, Tool
from langgraph.graph import END, add_messages
from langgraph.types import Command, Send
from pydantic import BaseModel, Field, ValidationError, model_validator

from haive.core.schema.prebuilt.messages_state import MessagesState
from haive.core.schema.state_schema import StateSchema


class ToolState(MessagesState):
    """
    State schema for tool management and execution.
    Extends MessagesState with tool tracking, validation, and routing capabilities.
    """

    # Tool storage - actual tool instances with name indexing
    tools: Dict[str, Union[Type[BaseTool], StructuredTool, Tool, Callable]] = Field(
        default_factory=dict, description="Available tools indexed by tool name"
    )

    # Tool schemas for validation - defaults to tools unless explicitly provided
    tool_schemas: Dict[str, Union[Type[BaseTool], Type[BaseModel]]] = Field(
        default_factory=dict,
        description="Tool schemas for validation (defaults to tools unless explicitly provided)",
    )

    # Engine tracking (for integration with engine system)
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

    # Storage for Pydantic model instances created during tool execution
    model_instances: Dict[str, BaseModel] = Field(
        default_factory=dict, description="Pydantic model instances created by tools"
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

    # Version (for compatibility with different routing strategies)
    version: str = Field(default="v2", description="Version of tool handling to use")

    # Configuration for structured response (for tools that return structured data)
    response_format: Optional[Type[BaseModel]] = Field(
        default=None, description="Optional response format for structured outputs"
    )
    direct_parse_tools: Dict[str, Type[BaseModel]] = Field(
        default_factory=dict,
        description="Tools that should parse directly without execution",
    )
    # Configuration
    __shared_fields__ = ["messages", "tools", "engines"]

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
    output_schemas: Dict[str, Type[BaseModel]] = Field(
        default_factory=dict,
        description="Output schemas for structured parsing by tool name",
    )

    def register_direct_parse_model(
        self, model: Type[BaseModel], tool_name: Optional[str] = None
    ) -> None:
        """
        Register a model that should be directly parsed from LLM output
        without tool execution.

        Args:
            model: Pydantic model class
            tool_name: Optional tool name to associate with
        """
        model_name = model.__name__
        self.output_schemas[model_name] = model

        # Mark for direct parsing
        self.direct_parse_tools[model_name] = model

        # Store tool association if provided
        if tool_name:
            setattr(model, "tool_name", tool_name)

    # Initialize tool system
    def initialize_tools(
        self,
        tools: Sequence[
            Union[
                Type[BaseTool],
                StructuredTool,
                Tool,
                BaseToolkit,
                Callable,
                Type[BaseModel],
            ]
        ],
    ) -> None:
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

    def register_output_schema(self, schema_name: str, schema: Type[BaseModel]) -> None:
        """
        Register an output schema for parsing tool results.

        Args:
            schema_name: Name to identify this schema
            schema: Pydantic model class for parsing
        """
        self.output_schemas[schema_name] = schema

    def register_tool_with_output(
        self,
        tool: Union[BaseTool, StructuredTool],
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> None:
        """Register a tool with optional output schema for parsing."""
        # Register tool as normal
        self._register_tool(tool)

        # If output schema provided, store it
        if output_schema:
            tool_name = getattr(tool, "name", None)
            if tool_name:
                self.output_schemas[tool_name] = output_schema

    def register_validation_schema(
        self, tool_name: str, schema: Type[BaseModel]
    ) -> None:
        """
        Register an explicit validation schema for a tool.

        Args:
            tool_name: Name of the tool to associate with schema
            schema: Pydantic model to use for validation
        """
        self.tool_schemas[tool_name] = schema

    def _register_tool(
        self,
        tool: Union[Type[BaseTool], StructuredTool, Tool, Callable, Type[BaseModel]],
    ) -> None:
        """
        Register a single tool in the state.

        Args:
            tool: Tool to register
        """
        # Handle different tool types
        if isinstance(tool, type):
            # Tool class or BaseModel class
            if issubclass(tool, BaseModel):
                # This is a Pydantic model class for validation
                tool_name = getattr(tool, "__name__", f"model_{uuid.uuid4().hex[:8]}")
                self.tools[tool_name] = tool
                self.tool_schemas[tool_name] = tool
            elif issubclass(tool, BaseTool):
                # This is a tool class - create instance
                try:
                    tool_instance = tool()
                    tool_name = getattr(tool_instance, "name", None) or getattr(
                        tool, "__name__", None
                    )
                    if not tool_name:
                        tool_name = f"tool_{uuid.uuid4().hex[:8]}"

                    self.tools[tool_name] = tool

                    # For BaseTool classes, check if they have args_schema
                    args_schema = getattr(tool_instance, "args_schema", None)
                    if (
                        args_schema
                        and isinstance(args_schema, type)
                        and issubclass(args_schema, BaseModel)
                    ):
                        self.tool_schemas[tool_name] = args_schema
                    else:
                        # Default to using the tool itself as schema
                        self.tool_schemas[tool_name] = tool

                except Exception as e:
                    # If we can't instantiate, just store the class
                    tool_name = getattr(
                        tool, "__name__", f"tool_{uuid.uuid4().hex[:8]}"
                    )
                    self.tools[tool_name] = tool
                    self.tool_schemas[tool_name] = tool
        elif isinstance(tool, (StructuredTool, Tool)):
            # Tool instance
            tool_name = getattr(tool, "name", None)
            if not tool_name:
                tool_name = f"tool_{uuid.uuid4().hex[:8]}"

            self.tools[tool_name] = tool

            # Get schema if available
            schema = getattr(tool, "args_schema", None)
            if schema and isinstance(schema, type) and issubclass(schema, BaseModel):
                self.tool_schemas[tool_name] = schema
            else:
                # Default to using the tool itself
                self.tool_schemas[tool_name] = type(tool)
        elif callable(tool):
            # Function tool
            tool_name = getattr(tool, "__name__", f"func_{uuid.uuid4().hex[:8]}")
            self.tools[tool_name] = tool
            # No schema for callables by default
        else:
            # Unknown type
            tool_name = getattr(tool, "name", f"unknown_{uuid.uuid4().hex[:8]}")
            self.tools[tool_name] = tool

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
                call_dict = call.copy()
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

    def validate_tool_call(
        self, tool_call: Dict[str, Any]
    ) -> Union[Dict[str, Any], None]:
        """
        Validate a tool call against its schema.

        Args:
            tool_call: Tool call to validate

        Returns:
            Validated tool call or None if validation failed
        """
        # Get tool name and arguments
        tool_name = tool_call.get("name")
        args = tool_call.get("args", {})
        call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")

        # Ensure tool call has an ID
        if "id" not in tool_call:
            tool_call["id"] = call_id

        # Return immediately if no tool name
        if not tool_name:
            self.add_invalid_call(tool_call, "Missing tool name")
            return None

        # Check if tool exists
        if tool_name not in self.tools:
            self.add_invalid_call(tool_call, f"Unknown tool: {tool_name}")
            return None

        # Initialize validated call with copy of original
        validated_call = tool_call.copy()
        validated_call["validated"] = False

        # Get the tool instance or class
        tool = self.tools[tool_name]

        # Get the validation schema (explicit schema or tool itself)
        schema = self.tool_schemas.get(tool_name)

        # Check if we can get tool type information
        if hasattr(tool, "tool_type"):
            validated_call["tool_type"] = getattr(tool, "tool_type")

        # If no schema and we have a tool, just mark as validated
        if not schema:
            validated_call["validated"] = True
            return validated_call

        try:
            # Normalize arguments if they're a string (handle JSON strings)
            if isinstance(args, str):
                try:
                    import json

                    args_dict = json.loads(args)
                except json.JSONDecodeError:
                    args_dict = {"input": args}
            else:
                args_dict = args

            # Handle different schema types
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                # Pydantic model validation
                validated_model = schema(**args_dict)

                # Update validated call
                validated_call["args"] = validated_model.model_dump()
                validated_call["validated"] = True
                validated_call["model_type"] = schema.__name__

                # Store model instance for later reference
                self.model_instances[call_id] = validated_model

            elif isinstance(schema, type) and issubclass(schema, BaseTool):
                # Try to create a tool instance for validation
                try:
                    # Some tools may need args for initialization
                    tool_instance = schema()

                    # Check for args_schema
                    args_schema = getattr(tool_instance, "args_schema", None)
                    if (
                        args_schema
                        and isinstance(args_schema, type)
                        and issubclass(args_schema, BaseModel)
                    ):
                        # Validate with args schema
                        validated_model = args_schema(**args_dict)

                        # Update validated call
                        validated_call["args"] = validated_model.model_dump()
                        validated_call["validated"] = True
                        validated_call["model_type"] = args_schema.__name__

                        # Store model instance
                        self.model_instances[call_id] = validated_model
                    else:
                        # No schema but it's a valid tool type
                        validated_call["validated"] = True
                except Exception as e:
                    # Failed to instantiate the tool
                    self.add_invalid_call(tool_call, f"Failed to create tool: {str(e)}")
                    return None
            else:
                # For any other case, mark as validated
                validated_call["validated"] = True

            # Check for output schemas that match this tool
            for schema_name, output_schema in self.output_schemas.items():
                # Match by tool_name attribute
                if (
                    hasattr(output_schema, "tool_name")
                    and getattr(output_schema, "tool_name") == tool_name
                ):
                    validated_call["output_schema"] = schema_name
                    break

            # Check if this is a direct parse model (no execution needed)
            if "model_type" in validated_call:
                model_type = validated_call["model_type"]
                if model_type in self.direct_parse_tools:
                    validated_call["direct_parse"] = True
                    # Ensure output schema is set if not already
                    if "output_schema" not in validated_call:
                        validated_call["output_schema"] = model_type

            # Validation succeeded
            return validated_call

        except ValidationError as e:
            # Validation failed with a Pydantic error
            self.add_invalid_call(tool_call, str(e))
            return None
        except Exception as e:
            # Other unexpected error
            self.add_invalid_call(tool_call, f"Validation error: {str(e)}")
            return None

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

    def needs_validation(self) -> bool:
        """Check if there are tool calls that need validation."""
        return len(self.tool_calls) > 0

    def needs_execution(self) -> bool:
        """Check if there are validated tool calls that need execution."""
        return len(self.validated_tool_calls) > 0

    def has_structured_output_tools(self) -> bool:
        """Check if any of the validated tools produce structured output."""
        for call in self.validated_tool_calls:
            if "model_type" in call:
                return True
        return False

    def need_output_parsing(self) -> bool:
        """Check if any completed tools require output parsing."""
        for call in self.completed_tool_calls:
            if "model_type" in call:
                return True
        return False

    def should_continue(self) -> Union[str, List[Send]]:
        """
        Determine next steps based on the current state.
        Similar to the function in your example.

        Returns:
            Routing directive (node name or Send objects)
        """
        messages = self.messages

        if not messages:
            return END

        last_message = messages[-1]

        # If there's no AI message with tool calls, we're done
        if not isinstance(last_message, AIMessage) or not getattr(
            last_message, "tool_calls", None
        ):
            return (
                END if self.response_format is None else "generate_structured_response"
            )

        # Update tool calls
        self.update_tool_calls()

        # If we have pending calls, they need validation
        if self.needs_validation():
            return "validation"

        # If we have validated calls, they need execution
        if self.needs_execution():
            if self.version == "v1":
                return "tools"
            elif self.version == "v2":
                # Create Send objects for parallel execution
                return [
                    Send("tools", tool_call) for tool_call in self.validated_tool_calls
                ]

        # If we have completed calls that need parsing, go to parsing node
        if self.need_output_parsing():
            return "output_parsing"

        # Otherwise, we're done with tools, go back to agent
        return "agent" if self.invalid_tool_calls or self.completed_tool_calls else END

    def route_based_on_tool_state(self) -> Union[str, List[Union[str, Send]]]:
        """
        Make routing decision based on current tool state.

        Returns:
            Either a node name or Send objects
        """
        # Check if we need validation first
        if self.needs_validation():
            return "validation"

        # Check if we need execution
        if self.needs_execution():
            # For multiple tools, use parallel execution with Send
            if len(self.validated_tool_calls) > 1:
                return [
                    Send("tools", tool_call) for tool_call in self.validated_tool_calls
                ]
            # For single tool, just route to tools node
            elif len(self.validated_tool_calls) == 1:
                return "tools"

        # Check if we need output parsing
        if self.need_output_parsing():
            return "output_parsing"

        # Otherwise, go back to agent or end
        return "agent" if self.invalid_tool_calls or self.completed_tool_calls else END

    # Define validation node logic
    def validate_all_tool_calls(self) -> Command:
        """
        Validate all pending tool calls and prepare routing.

        Returns:
            Command with update and routing
        """
        # Make sure we have the latest tool calls
        self.update_tool_calls()

        # Validate each tool call
        for tool_call in list(self.tool_calls):
            validated_call = self.validate_tool_call(tool_call)
            if validated_call:
                self.add_validated_call(validated_call)

        # Determine next step
        next_step = self.route_based_on_tool_state()

        # Return Command with updated state and routing
        return Command(goto=next_step)

    # Output parsing node logic
    def parse_tool_outputs(self) -> Command:
        """
        Parse outputs from completed tools that produced structured data.

        Returns:
            Command with parsed models and routing
        """
        updates = {}

        # Process all completed tool calls
        for call in self.completed_tool_calls:
            if "model_type" in call and call.get("id") in self.model_instances:
                # Get the model instance
                model = self.model_instances[call["id"]]
                model_name = model.__class__.__name__

                # Add to updates
                updates[model_name] = model

        # Return Command with updates and go back to agent
        return Command(update=updates, goto="agent")
