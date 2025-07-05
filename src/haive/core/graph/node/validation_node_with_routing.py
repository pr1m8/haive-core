"""Validation node configuration with tool message updates and routing state."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from haive.core.graph.node.validation_node_config import ValidationNodeConfig
from haive.core.schema.prebuilt.tools.validation_state import (
    RouteRecommendation,
    ToolValidationResult,
    ValidationRoutingState,
    ValidationStateManager,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


class ValidationNodeWithRouting(ValidationNodeConfig):
    """Validation node that updates tool messages and provides routing state.

    Extends ValidationNodeConfig with:
    - Tool message status updates
    - Routing state generation for conditional branching
    - Validation result tracking
    - Integration with ToolStateWithValidation
    """

    # Configuration for routing and message updates
    update_tool_messages: bool = Field(
        default=True,
        description="Whether to update tool messages with validation results",
    )

    provide_routing_state: bool = Field(
        default=True,
        description="Whether to provide routing state for conditional branching",
    )

    validation_timeout: float = Field(
        default=30.0, description="Timeout for validation operations in seconds"
    )

    auto_correct_args: bool = Field(
        default=True, description="Whether to attempt automatic argument correction"
    )

    detailed_error_messages: bool = Field(
        default=True, description="Whether to provide detailed error messages"
    )

    # Routing behavior
    continue_on_partial_validation: bool = Field(
        default=True, description="Whether to continue if some tools pass validation"
    )

    retry_invalid_tools: bool = Field(
        default=False,
        description="Whether to automatically retry invalid tools with corrections",
    )

    def create_validation_function_with_routing(self) -> Callable:
        """Create validation function with message updates and routing."""

        def validation_node_with_routing(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()

            logger.info("[bold blue]Starting validation with routing[/bold blue]")

            # Create validation routing state
            routing_state = ValidationStateManager.create_routing_state()

            try:
                # Extract tool calls from the last AI message
                tool_calls = self._extract_tool_calls_from_state(state)

                if not tool_calls:
                    logger.info("No tool calls found for validation")
                    return self._handle_no_tool_calls(state, routing_state)

                logger.info(f"Validating {len(tool_calls)} tool calls")

                # Get tools and schemas for validation
                available_tools, validation_schemas = self._get_validation_resources(
                    state
                )

                # Validate each tool call
                for tool_call in tool_calls:
                    result = self._validate_single_tool_call(
                        tool_call, available_tools, validation_schemas, state
                    )
                    routing_state.add_validation_result(result)

                # Update validation duration
                routing_state.validation_duration = time.time() - start_time

                # Update state with validation results
                updated_state = self._update_state_with_validation_results(
                    state, routing_state
                )

                logger.info(
                    f"Validation complete: {routing_state.get_routing_summary()}"
                )

                return updated_state

            except Exception as e:
                logger.error(f"Validation failed with error: {e}")
                return self._handle_validation_error(state, routing_state, str(e))

        return validation_node_with_routing

    def _extract_tool_calls_from_state(
        self, state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from state messages."""
        messages = state.get(self.messages_key, [])
        if not messages:
            return []

        # Get the last AI message
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return []

        # Extract tool calls
        tool_calls = getattr(last_message, "tool_calls", None)
        if not tool_calls:
            return []

        return tool_calls

    def _get_validation_resources(
        self, state: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Get available tools and validation schemas."""
        # Get tools from engine or state
        available_tools = {}
        validation_schemas = {}

        if self.engine_name:
            # Get from specific engine
            engines = state.get("engines", {})
            if self.engine_name in engines:
                engine = engines[self.engine_name]
                if hasattr(engine, "get_tools"):
                    tools = engine.get_tools()
                    available_tools = {tool.name: tool for tool in tools}

                if hasattr(engine, "schemas"):
                    validation_schemas = getattr(engine, "schemas", {})
        else:
            # Get from state tools
            state_tools = state.get("tools", [])
            available_tools = {
                getattr(tool, "name", str(tool)): tool for tool in state_tools
            }

            # Get schemas from state
            validation_schemas = state.get("output_schemas", {})

        # Override with configured tools/schemas if provided
        if self.tools:
            config_tools = {
                getattr(tool, "name", str(tool)): tool for tool in self.tools
            }
            available_tools.update(config_tools)

        if self.schemas:
            validation_schemas.update(
                {
                    getattr(schema, "__name__", str(schema)): schema
                    for schema in self.schemas
                }
            )

        return available_tools, validation_schemas

    def _validate_single_tool_call(
        self,
        tool_call: Dict[str, Any],
        available_tools: Dict[str, Any],
        validation_schemas: Dict[str, Any],
        state: Dict[str, Any],
    ) -> ToolValidationResult:
        """Validate a single tool call and return detailed result."""

        # Extract tool call information
        tool_call_id = tool_call.get("id", f"call_{id(tool_call)}")
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})

        logger.debug(f"Validating tool call: {tool_name} with args: {tool_args}")

        # Check if tool exists
        if tool_name not in available_tools:
            return ValidationStateManager.create_validation_result(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                status=ValidationStatus.ERROR,
                route_recommendation=RouteRecommendation.AGENT,
                errors=[f"Tool '{tool_name}' not found in available tools"],
                target_node=self.agent_node,
            )

        tool = available_tools[tool_name]

        # Validate tool arguments
        validation_errors = []
        validation_warnings = []
        corrected_args = None

        try:
            # Basic argument validation
            validation_result = self._validate_tool_arguments(
                tool, tool_args, validation_schemas
            )

            validation_errors = validation_result.get("errors", [])
            validation_warnings = validation_result.get("warnings", [])
            corrected_args = validation_result.get("corrected_args")

        except Exception as e:
            validation_errors.append(f"Validation failed: {str(e)}")

        # Determine status and recommendation
        if validation_errors:
            if corrected_args and self.auto_correct_args:
                status = ValidationStatus.INVALID
                recommendation = RouteRecommendation.RETRY
                target_node = self._determine_target_node_for_tool(tool_name, state)
            else:
                status = ValidationStatus.INVALID
                recommendation = RouteRecommendation.AGENT
                target_node = self.agent_node
        else:
            status = ValidationStatus.VALID
            recommendation = RouteRecommendation.EXECUTE
            target_node = self._determine_target_node_for_tool(tool_name, state)

        return ValidationStateManager.create_validation_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            route_recommendation=recommendation,
            errors=validation_errors,
            warnings=validation_warnings,
            corrected_args=corrected_args,
            target_node=target_node,
            engine_name=self.engine_name,
            metadata={
                "validation_time": time.time(),
                "original_args": tool_args,
            },
        )

    def _validate_tool_arguments(
        self, tool: Any, args: Dict[str, Any], schemas: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate tool arguments and attempt corrections."""
        result = {"errors": [], "warnings": [], "corrected_args": None}

        # Check if tool has input schema
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                # Validate using Pydantic schema
                validated_args = tool.args_schema.model_validate(args)
                result["corrected_args"] = validated_args.model_dump()

            except Exception as e:
                result["errors"].append(f"Argument validation failed: {str(e)}")

                # Attempt auto-correction if enabled
                if self.auto_correct_args:
                    corrected = self._attempt_argument_correction(
                        tool.args_schema, args
                    )
                    if corrected:
                        result["corrected_args"] = corrected
                        result["warnings"].append("Arguments were auto-corrected")

        # Additional validation logic can be added here

        return result

    def _attempt_argument_correction(
        self, schema: Any, args: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Attempt to auto-correct invalid arguments."""
        try:
            # Simple correction strategies
            corrected = args.copy()

            # Get schema fields
            if hasattr(schema, "model_fields"):
                fields = schema.model_fields

                # Add missing required fields with defaults
                for field_name, field_info in fields.items():
                    if field_name not in corrected:
                        if (
                            hasattr(field_info, "default")
                            and field_info.default is not None
                        ):
                            corrected[field_name] = field_info.default
                        elif (
                            hasattr(field_info, "default_factory")
                            and field_info.default_factory
                        ):
                            corrected[field_name] = field_info.default_factory()

                # Type corrections
                for field_name, value in corrected.items():
                    if field_name in fields:
                        field_info = fields[field_name]
                        # Add type conversion logic here if needed

            # Validate corrected args
            validated = schema.model_validate(corrected)
            return validated.model_dump()

        except Exception:
            return None

    def _determine_target_node_for_tool(
        self, tool_name: str, state: Dict[str, Any]
    ) -> str:
        """Determine target node for a validated tool."""
        # Get tool route
        tool_routes = state.get("tool_routes", {})
        route = tool_routes.get(tool_name, "unknown")

        # Map route to node
        route_to_node = {
            "langchain_tool": self.tool_node,
            "function": self.tool_node,
            "pydantic_model": self.parser_node,
            "retriever": self.retriever_node,
        }

        # Check custom mappings first
        if route in self.custom_route_mappings:
            return self.custom_route_mappings[route]

        # Check direct node routes
        if route in self.direct_node_routes:
            return route

        # Use default mapping
        return route_to_node.get(route, self.tool_node)

    def _update_state_with_validation_results(
        self, state: Dict[str, Any], routing_state: ValidationRoutingState
    ) -> Dict[str, Any]:
        """Update state with validation results and routing information."""
        updated_state = state.copy()

        # Apply validation results to tool state with validation if available
        if hasattr(state, "apply_validation_results"):
            state.apply_validation_results(routing_state)
        else:
            # Add validation data to state dict
            updated_state["validation_state"] = routing_state
            updated_state["validation_routing"] = routing_state.get_routing_decision()

        # Update tool messages if enabled
        if self.update_tool_messages:
            updated_state = self._update_tool_messages_with_validation(
                updated_state, routing_state
            )

        # Add routing state if enabled
        if self.provide_routing_state:
            updated_state["branch_conditions"] = routing_state.get_routing_decision()
            updated_state["routing_data"] = {
                "should_continue": routing_state.should_continue_execution(),
                "should_return_to_agent": routing_state.should_return_to_agent(),
                "should_end": routing_state.should_end_processing(),
                "target_nodes": list(routing_state.target_nodes),
                "next_action": routing_state.next_action.value,
            }

        return updated_state

    def _update_tool_messages_with_validation(
        self, state: Dict[str, Any], routing_state: ValidationRoutingState
    ) -> Dict[str, Any]:
        """Update tool messages with validation status."""
        messages = state.get(self.messages_key, []).copy()

        # Find and update tool messages
        for i, message in enumerate(messages):
            if isinstance(message, ToolMessage):
                tool_call_id = getattr(message, "tool_call_id", None)
                if tool_call_id and tool_call_id in routing_state.tool_message_updates:
                    updates = routing_state.tool_message_updates[tool_call_id]

                    # Update message additional_kwargs
                    if not hasattr(message, "additional_kwargs"):
                        message.additional_kwargs = {}

                    message.additional_kwargs.update(updates)

                    # Update message in list
                    messages[i] = message

        # Update state with modified messages
        updated_state = state.copy()
        updated_state[self.messages_key] = messages

        return updated_state

    def _handle_no_tool_calls(
        self, state: Dict[str, Any], routing_state: ValidationRoutingState
    ) -> Dict[str, Any]:
        """Handle case when no tool calls are found."""
        # Set routing to end processing
        routing_state.next_action = RouteRecommendation.END

        updated_state = state.copy()
        if self.provide_routing_state:
            updated_state["routing_data"] = {
                "should_continue": False,
                "should_return_to_agent": False,
                "should_end": True,
                "target_nodes": [],
                "next_action": "end",
            }

        return updated_state

    def _handle_validation_error(
        self,
        state: Dict[str, Any],
        routing_state: ValidationRoutingState,
        error_message: str,
    ) -> Dict[str, Any]:
        """Handle validation errors."""
        # Set routing to return to agent
        routing_state.next_action = RouteRecommendation.AGENT
        routing_state.target_nodes.add(self.agent_node)

        updated_state = state.copy()
        updated_state["validation_error"] = error_message

        if self.provide_routing_state:
            updated_state["routing_data"] = {
                "should_continue": False,
                "should_return_to_agent": True,
                "should_end": False,
                "target_nodes": [self.agent_node],
                "next_action": "agent",
                "error": error_message,
            }

        return updated_state

    # Factory method for creating the node function
    def create_node_function(self) -> Callable:
        """Create the validation node function with routing."""
        return self.create_validation_function_with_routing()
