import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.prebuilt import ValidationNode
from langgraph.types import Command, Send
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin
from haive.core.graph.common.types import ConfigLike, StateLike
from haive.core.graph.node.base_config import NodeConfig
from haive.core.graph.node.types import NodeType

logger = logging.getLogger(__name__)
console = Console()


def has_tool_error(message: ToolMessage) -> bool:
    """
    Check if a ToolMessage contains an error.

    Args:
        message: The ToolMessage to check

    Returns:
        True if the message indicates a tool error, False otherwise
    """
    if not isinstance(message, ToolMessage):
        return False

    # Check for error content
    content = message.content
    if isinstance(content, str):
        # Look for common error indicators
        error_terms = ["error", "invalid", "failed", "exception", "validation error"]
        return any(term in content.lower() for term in error_terms)

    # If content is a dict, check for error keys
    if isinstance(content, dict):
        error_keys = ["error", "errors", "exception", "validation_error"]
        return any(key in content for key in error_keys)

    # Check additional_kwargs for error indicators
    if hasattr(message, "additional_kwargs") and message.additional_kwargs:
        if message.additional_kwargs.get("is_error"):
            return True

    return False


def get_tool_name(tool_call: Any) -> str:
    """
    Extract tool name from either ToolCall object or dictionary.

    Args:
        tool_call: A tool call object or dictionary

    Returns:
        The name of the tool
    """
    # Handle ToolCall object
    if hasattr(tool_call, "name"):
        return tool_call.name

    # Handle dictionary formats (OpenAI vs LangChain)
    if isinstance(tool_call, dict):
        if "name" in tool_call:
            return tool_call["name"]
        elif "function" in tool_call and "name" in tool_call["function"]:
            return tool_call["function"]["name"]
        elif (
            "type" in tool_call
            and tool_call["type"] == "function"
            and "function" in tool_call
        ):
            return tool_call["function"].get("name", "unknown_tool")

    # Fallback
    return "unknown_tool"


def get_tool_args(tool_call: Any) -> Dict[str, Any]:
    """
    Extract tool arguments from either ToolCall object or dictionary.

    Args:
        tool_call: A tool call object or dictionary

    Returns:
        Dictionary of tool arguments
    """
    # Handle ToolCall object
    if hasattr(tool_call, "args"):
        return tool_call.args

    # Handle dictionary formats
    if isinstance(tool_call, dict):
        if "args" in tool_call:
            return tool_call["args"]
        elif "arguments" in tool_call:
            # Try to parse arguments from string if needed
            args = tool_call["arguments"]
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except:
                    return {"raw_args": args}
            return args
        elif "function" in tool_call and "arguments" in tool_call["function"]:
            # Try to parse arguments from string if needed
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except:
                    return {"raw_args": args}
            return args

    # Fallback
    return {}


def get_tool_id(tool_call):
    """Extract tool ID from either ToolCall object or dictionary."""
    if hasattr(tool_call, "id"):
        return tool_call.id
    elif isinstance(tool_call, dict) and "id" in tool_call:
        return tool_call["id"]

    # Fallback
    return f"id_{id(tool_call)}"


class ValidationNodeConfig(NodeConfig, ToolRouteMixin):
    """
    Configuration for a validation node that routes tool calls to appropriate nodes.
    """

    node_type: NodeType = Field(
        default=NodeType.VALIDATION, description="The type of node"
    )
    name: str = Field(default="validation", description="The name of the node")
    messages_key: str = Field(
        default="messages", description="The key to use for the messages field"
    )
    schemas: List[Any] = Field(
        default_factory=list, description="The schemas to use for validation"
    )
    format_error: Optional[Callable] = Field(
        default=None, description="Custom formatter for validation errors"
    )

    # Node routing configuration - these should match actual nodes in your graph
    agent_node: str = Field(
        default="agent", description="Node to return to for validation errors"
    )
    tool_node: str = Field(
        default="tool_node", description="Node for executing standard tools"
    )
    parser_node: str = Field(
        default="parse_output", description="Node for parsing Pydantic models"
    )
    retriever_node: str = Field(
        default="retriever", description="Node for retriever tools"
    )

    tools: List[Any] = Field(
        default_factory=list, description="List of available tools"
    )
    # tool_routes now provided by ToolRouteMixin

    # Available nodes in the graph (for validation)
    available_nodes: List[str] = Field(
        default_factory=list, description="List of nodes available in the graph"
    )

    # Custom route mappings - can override default behavior
    custom_route_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom mappings from route names to node names",
    )

    # Direct node routes - routes that should skip standard mapping
    direct_node_routes: List[str] = Field(
        default_factory=list,
        description="Routes that map directly to node names without transformation",
    )

    def validate_node_exists(self, node_name: str) -> str:
        """
        Validate that a node exists in the graph, return fallback if not.

        Args:
            node_name: The node name to validate

        Returns:
            Valid node name or fallback to agent_node
        """
        logger.debug(f"Validating node exists: {node_name}")

        if not self.available_nodes:
            # If no available nodes specified, trust the provided name
            logger.debug("No available_nodes configured, trusting provided name")
            return node_name

        if node_name in self.available_nodes:
            logger.debug(f"Node {node_name} found in available_nodes")
            return node_name

        # Fallback logic
        logger.warning(
            f"Node '{node_name}' not found in graph. Available nodes: {self.available_nodes}"
        )

        # Try fallbacks in order of preference
        fallback_options = [self.agent_node, "agent", "START"]
        for fallback in fallback_options:
            if fallback in self.available_nodes:
                logger.info(f"Using fallback node: {fallback}")
                return fallback

        # If no fallbacks work, return the original name and let LangGraph handle the error
        logger.error(
            f"No valid fallback found. Returning original node name: {node_name}"
        )
        return node_name

    def _get_node_for_route(self, route: str) -> str:
        """Map tool route to appropriate node with validation."""
        logger.debug(f"Getting node for route: {route}")

        # First check custom route mappings
        if route in self.custom_route_mappings:
            target_node = self.custom_route_mappings[route]
            logger.debug(f"Found custom route mapping: {route} -> {target_node}")
            return self.validate_node_exists(target_node)

        # Check if this is a direct node route
        if route in self.direct_node_routes:
            logger.debug(f"Route {route} is configured as direct node route")
            return self.validate_node_exists(route)

        # Standard route mapping for tool types
        route_mapping = {
            "pydantic_model": self.parser_node,
            "langchain_tool": self.tool_node,
            "function": self.tool_node,
            "retriever": self.retriever_node,
            "unknown": self.tool_node,  # Default fallback
        }

        target_node = route_mapping.get(route, self.tool_node)
        logger.debug(f"Standard route mapping: {route} -> {target_node}")

        # Validate the node exists
        return self.validate_node_exists(target_node)

    def _sync_tools_and_schemas(self) -> None:
        """Sync tool routes from both tools and schemas."""
        logger.debug("Starting tool and schema synchronization")

        all_tools = []

        # Add schemas first (they take priority for routing)
        if self.schemas:
            all_tools.extend(self.schemas)
            logger.debug(f"Added {len(self.schemas)} schemas to all_tools")

        # Add tools (may override schema routes if same name)
        if self.tools:
            # Only add tools that aren't already in schemas
            schema_names = {
                getattr(s, "__name__", getattr(s, "name", str(s))) for s in self.schemas
            }
            added_tools = 0
            for tool in self.tools:
                tool_name = getattr(tool, "__name__", getattr(tool, "name", str(tool)))
                if tool_name not in schema_names:
                    all_tools.append(tool)
                    added_tools += 1
            logger.debug(f"Added {added_tools} additional tools (not in schemas)")

        # Sync routes using the mixin
        if all_tools:
            self.sync_tool_routes_from_tools(all_tools)
            logger.info(f"Synced routes for {len(all_tools)} total tools/schemas")

            # Log final tool routes
            logger.debug(f"Final tool_routes after sync: {self.tool_routes}")
        else:
            logger.warning("No tools or schemas to sync")

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Union[Command, List[Send], str]:
        """
        Validate and route tool calls to appropriate nodes.

        FIXED: Now properly adds ToolMessages to state after validation
        """
        logger.info("=" * 80)
        logger.info("VALIDATION NODE START")
        logger.info("=" * 80)

        # Sync tools and schemas to ensure routes are up to date
        self._sync_tools_and_schemas()

        # Debug initial state
        logger.debug(f"State type: {type(state)}")
        logger.debug(f"Has messages: {hasattr(state, 'messages')}")
        logger.debug(
            f"Message count: {len(state.messages) if hasattr(state, 'messages') and state.messages else 0}"
        )
        logger.debug(f"Config: {config}")
        logger.debug(f"Tools available: {len(self.tools)}")
        logger.debug(f"Schemas available: {len(self.schemas)}")
        logger.debug(f"Available nodes: {self.available_nodes}")
        logger.debug(f"Current tool_routes: {self.tool_routes}")
        logger.debug(f"Custom route mappings: {self.custom_route_mappings}")
        logger.debug(f"Direct node routes: {self.direct_node_routes}")

        # Build a mapping of tool names to actual tool classes/schemas
        tool_name_mapping = {}

        # First, check schemas (these are likely Pydantic models)
        for schema in self.schemas:
            if hasattr(schema, "__name__"):
                tool_name_mapping[schema.__name__] = schema
            # Also check for 'name' attribute
            if hasattr(schema, "name"):
                tool_name_mapping[schema.name] = schema

        # Then check tools (might override schemas if same name)
        for tool in self.tools:
            if hasattr(tool, "name"):
                tool_name_mapping[tool.name] = tool
            elif hasattr(tool, "__name__"):
                tool_name_mapping[tool.__name__] = tool

        logger.debug(f"Tool name mapping: {list(tool_name_mapping.keys())}")

        # Get messages from state
        if not hasattr(state, "messages") or not state.messages:
            logger.warning("No messages found in state")
            return "no_tool_calls"

        messages = state.messages
        logger.info(f"Found {len(messages)} messages in state")

        # Check for tool calls in the last message
        last_message = messages[-1]
        logger.debug(f"Last message type: {type(last_message).__name__}")

        if not isinstance(last_message, AIMessage):
            logger.warning("Last message is not an AIMessage")
            return "no_tool_calls"

        # Get tool calls - handle both attributes and additional_kwargs
        tool_calls = []
        logger.debug("Checking for tool calls...")

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls
            logger.info(f"Found {len(tool_calls)} tool calls in tool_calls attribute")
        elif (
            hasattr(last_message, "additional_kwargs")
            and "tool_calls" in last_message.additional_kwargs
        ):
            tool_calls = last_message.additional_kwargs["tool_calls"]
            logger.info(f"Found {len(tool_calls)} tool calls in additional_kwargs")

        if not tool_calls:
            logger.warning("No tool calls found in last message")
            return "no_tool_calls"

        # Log tool calls info
        logger.info("Tool calls found:")
        for i, tc in enumerate(tool_calls):
            name = get_tool_name(tc)
            args = get_tool_args(tc)
            logger.info(f"  [{i}] {name}: {args}")

        # Group tool calls by their destination
        tool_groups = {}
        all_valid = True
        error_messages = []
        validation_results = []

        # FIXED: Track all validated tool messages to add to state
        all_validated_tool_messages = []

        logger.info("-" * 60)
        logger.info("PROCESSING TOOL CALLS")
        logger.info("-" * 60)

        # Process each tool call individually
        for i, tool_call in enumerate(tool_calls):
            tool_name = get_tool_name(tool_call)
            logger.info(f"Processing tool {i+1}/{len(tool_calls)}: {tool_name}")

            # Create standardized tool call
            standardized_tool_call = tool_call
            if isinstance(tool_call, dict):
                logger.debug("Converting dict to ToolCall object")
                # Convert dict to ToolCall
                standardized_tool_call = ToolCall(
                    name=tool_name,
                    args=get_tool_args(tool_call),
                    id=get_tool_id(tool_call),
                )
            else:
                logger.debug("Tool call already standardized")

            # Check if tool should skip validation
            route = self.tool_routes.get(tool_name, "unknown")
            skip_validation = False

            # Check if this route is configured to skip validation
            if route in self.direct_node_routes:
                skip_validation = True
                logger.info(
                    f"Tool {tool_name} configured to skip validation (direct node route)"
                )

            if skip_validation:
                logger.info(f"Skipping validation for {tool_name}")

                # Direct routing without validation
                destination = self._get_node_for_route(route)

                logger.info(f"Direct routing: {tool_name} -> {destination}")

                if destination not in tool_groups:
                    tool_groups[destination] = []

                # Find the actual tool class
                tool_class = tool_name_mapping.get(tool_name)

                # Create a placeholder ToolMessage for consistency
                tool_message = ToolMessage(
                    content=f"Direct routing to {destination}",
                    name=tool_name,
                    tool_call_id=standardized_tool_call.id,
                )

                # FIXED: Add to tracked tool messages
                all_validated_tool_messages.append(tool_message)

                tool_groups[destination].append(
                    {
                        "tool_call": standardized_tool_call,
                        "tool_name": tool_name,
                        "tool_class": tool_class,
                        "route": route,
                        "tool_message": tool_message,
                        "skipped_validation": True,
                    }
                )

                validation_result = {
                    "tool_name": tool_name,
                    "has_error": False,
                    "route": route,
                    "destination": destination,
                    "skipped_validation": True,
                }

                logger.info(
                    f"Tool [{tool_name}] routed to [{destination}] (no validation)"
                )
                validation_results.append(validation_result)
                continue

            # Regular validation
            logger.info(f"Running validation for {tool_name}")

            # Create temporary message with just this tool call for validation
            temp_message = AIMessage(content="", tool_calls=[standardized_tool_call])
            temp_messages = messages[:-1] + [temp_message]

            logger.debug(
                f"Created temp message list with {len(temp_messages)} messages"
            )

            # Create validation node for this specific tool call
            validation_node = ValidationNode(
                schemas=self.schemas,
                format_error=self.format_error,
                name=f"validate_{tool_name}",
            )

            logger.debug(f"Running validation with {len(self.schemas)} schemas")

            # Run validation on this specific tool call
            validation_input = {"messages": temp_messages}
            try:
                result = validation_node.invoke(validation_input)
                validated_messages = result.get("messages", [])

                logger.info(
                    f"Validation completed - got {len(validated_messages)} messages back"
                )

                # Extract the ToolMessage from validation
                validated_tool_msg = None
                for msg in validated_messages:
                    if isinstance(msg, ToolMessage) and msg.name == tool_name:
                        validated_tool_msg = msg
                        logger.debug(f"Found validated ToolMessage for {tool_name}")
                        logger.debug(
                            f"  Tool message content: {validated_tool_msg.content}"
                        )
                        logger.debug(
                            f"  Tool message ID: {validated_tool_msg.tool_call_id}"
                        )
                        break

                # Check if this specific tool call has validation errors
                has_error = validated_tool_msg and has_tool_error(validated_tool_msg)

                validation_result = {
                    "tool_name": tool_name,
                    "has_error": has_error,
                    "message_count": len(validated_messages),
                }

                if has_error:
                    # This tool call has a validation error
                    logger.error(f"Validation error detected for {tool_name}")
                    all_valid = False
                    if validated_tool_msg:
                        error_messages.append(validated_tool_msg)
                        # FIXED: Still add error messages to tracked list
                        all_validated_tool_messages.append(validated_tool_msg)
                        logger.error(
                            f"Error in tool [{tool_name}]: {validated_tool_msg.content}"
                        )
                        validation_result["error_content"] = str(
                            validated_tool_msg.content
                        )[:100]
                else:
                    # This tool call is valid
                    if validated_tool_msg:
                        # FIXED: Add to tracked tool messages
                        all_validated_tool_messages.append(validated_tool_msg)
                        logger.info(
                            f"Added validated ToolMessage for {tool_name} to tracking list"
                        )

                    # Group by destination
                    destination = self._get_node_for_route(route)

                    logger.info(f"Tool {tool_name} is valid")
                    logger.info(f"Route: {route} -> Destination: {destination}")

                    if destination not in tool_groups:
                        tool_groups[destination] = []

                    # Find the actual tool/schema class
                    tool_class = tool_name_mapping.get(tool_name)
                    if not tool_class:
                        logger.warning(f"Could not find tool class for {tool_name}")
                        # As a fallback, check if it's in schemas or tools directly
                        for schema in self.schemas:
                            if (
                                getattr(schema, "__name__", None) == tool_name
                                or getattr(schema, "name", None) == tool_name
                            ):
                                tool_class = schema
                                break

                    tool_groups[destination].append(
                        {
                            "tool_call": standardized_tool_call,
                            "tool_name": tool_name,
                            "tool_class": tool_class,
                            "route": route,
                            "tool_message": validated_tool_msg,
                        }
                    )

                    validation_result.update(
                        {
                            "route": route,
                            "destination": destination,
                            "group_size": len(tool_groups[destination]),
                        }
                    )

                    logger.info(
                        f"Grouped tool [{tool_name}] with route [{route}] for destination [{destination}]"
                    )

                validation_results.append(validation_result)

            except Exception as e:
                # Handle unexpected errors during validation
                logger.exception(f"Error validating {tool_name}: {str(e)}")
                all_valid = False
                validation_results.append(
                    {
                        "tool_name": tool_name,
                        "has_error": True,
                        "error_content": f"Exception: {str(e)}",
                    }
                )

        # Display validation summary
        logger.info("-" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("-" * 60)

        for result in validation_results:
            status = "ERROR" if result["has_error"] else "VALID"
            if result.get("skipped_validation"):
                status = "SKIPPED"

            route = result.get("route", "N/A")
            dest = result.get("destination", "N/A")
            details = result.get(
                "error_content", f"Group size: {result.get('group_size', 'N/A')}"
            )

            logger.info(
                f"{result['tool_name']}: {status} | Route: {route} | Dest: {dest} | {details}"
            )

        # Log tool groups
        if tool_groups:
            logger.info(f"Tool Groups ({len(tool_groups)} destinations):")
            for dest, tools in tool_groups.items():
                tool_names = [t["tool_name"] for t in tools]
                logger.info(f"  {dest}: {tool_names}")

        # FIXED: Update state with ALL validated ToolMessages before routing
        logger.info("-" * 60)
        logger.info("UPDATING STATE WITH TOOL MESSAGES")
        logger.info("-" * 60)

        updated_messages = messages.copy()

        # Add all validated tool messages to state
        if all_validated_tool_messages:
            logger.info(
                f"Adding {len(all_validated_tool_messages)} ToolMessages to state"
            )
            for tool_msg in all_validated_tool_messages:
                logger.debug(
                    f"  Adding ToolMessage: {tool_msg.name} (ID: {tool_msg.tool_call_id})"
                )
            updated_messages.extend(all_validated_tool_messages)
        else:
            logger.warning("No validated tool messages to add to state!")

        # Create base update with the messages including ToolMessages
        base_update = {self.messages_key: updated_messages}

        # Handle routing based on validation results
        logger.info("-" * 60)
        logger.info("ROUTING DECISION")
        logger.info("-" * 60)

        if not all_valid and not tool_groups:
            # All tools had validation errors - route back to agent
            logger.warning("All tool calls failed validation, returning to agent")

            validated_agent_node = self.validate_node_exists(self.agent_node)
            return Command(update=base_update, goto=validated_agent_node)

        elif not all_valid and tool_groups:
            # Mixed results - execute valid tools but log warnings
            logger.warning(
                f"Mixed validation results - executing {len(tool_groups)} valid groups, some tools failed"
            )

        # If we have valid tools, determine routing strategy
        if len(tool_groups) == 1:
            # All valid tools go to the same destination - simple case
            destination = list(tool_groups.keys())[0]
            tools_for_dest = tool_groups[destination]

            logger.info(f"Single destination routing to: {destination}")
            logger.info(
                f"Tools for {destination}: {[t['tool_name'] for t in tools_for_dest]}"
            )

            if destination == self.parser_node:
                # For parser node, we need to send each tool individually
                logger.info("Creating individual sends for parser node")
                sends = []
                for tool_info in tools_for_dest:
                    if tool_info.get("tool_message") and tool_info.get("tool_class"):
                        send_obj = Send(
                            node=destination,
                            arg={
                                "tool_name": tool_info["tool_name"],
                                "tool": tool_info["tool_class"],
                                "tool_call": tool_info["tool_call"],
                                "tool_message": tool_info["tool_message"],
                                self.messages_key: updated_messages,  # Use updated messages
                            },
                        )
                        sends.append(send_obj)
                        logger.debug(
                            f"Created send for {tool_info['tool_name']} with tool class"
                        )
                    else:
                        if not tool_info.get("tool_message"):
                            logger.error(
                                f"No validated message found for {tool_info['tool_name']}"
                            )
                        if not tool_info.get("tool_class"):
                            logger.error(
                                f"No tool class found for {tool_info['tool_name']}"
                            )

                logger.info(f"Returning {len(sends)} Send objects for parser node")
                return sends

            else:
                # For other nodes, update state and goto
                logger.info(f"Returning Command with goto={destination}")
                return Command(update=base_update, goto=destination)

        else:
            # Multiple destinations - need to send to each
            logger.info(f"Multi-destination routing to {len(tool_groups)} destinations")
            sends = []

            for destination, tools_for_dest in tool_groups.items():
                logger.info(f"Processing destination: {destination}")

                if destination == self.parser_node:
                    # Parser node needs individual sends
                    logger.debug(
                        f"Creating individual sends for parser ({len(tools_for_dest)} tools)"
                    )
                    for tool_info in tools_for_dest:
                        if tool_info.get("tool_message") and tool_info.get(
                            "tool_class"
                        ):
                            send_obj = Send(
                                node=destination,
                                arg={
                                    "tool_name": tool_info["tool_name"],
                                    "tool": tool_info["tool_class"],
                                    "tool_call": tool_info["tool_call"],
                                    "tool_message": tool_info["tool_message"],
                                    self.messages_key: updated_messages,  # Use updated messages
                                },
                            )
                            sends.append(send_obj)
                            logger.debug(
                                f"Added send for {tool_info['tool_name']} with tool class"
                            )
                else:
                    # For other nodes, send with updated messages
                    send_obj = Send(
                        node=destination,
                        arg={self.messages_key: updated_messages},
                    )
                    sends.append(send_obj)
                    logger.debug(f"Added send to {destination}")

            logger.info(
                f"Returning {len(sends)} Send objects for multiple destinations"
            )
            return sends

        # Fallback case - no tool calls to route
        logger.warning("No valid tool calls to route - returning END")
        return END
