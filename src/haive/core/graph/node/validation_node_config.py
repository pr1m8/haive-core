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

    def validate_node_exists(self, node_name: str) -> str:
        """
        Validate that a node exists in the graph, return fallback if not.

        Args:
            node_name: The node name to validate

        Returns:
            Valid node name or fallback to agent_node
        """
        if not self.available_nodes:
            # If no available nodes specified, trust the provided name
            return node_name

        if node_name in self.available_nodes:
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
        # First check if this is a direct agent node route (for transfer tools)
        if route.startswith("agent_") or route in ["research", "math", "writer"]:
            # This is a direct agent route - validate it exists and return
            return self.validate_node_exists(route)

        # Standard route mapping for other tool types
        route_mapping = {
            "pydantic_model": self.parser_node,
            "langchain_tool": self.tool_node,
            "function": self.tool_node,
            "retriever": self.retriever_node,
            "unknown": self.tool_node,  # Default fallback
        }

        target_node = route_mapping.get(route, self.tool_node)
        # Validate the node exists
        return self.validate_node_exists(target_node)

    def _sync_tools_and_schemas(self) -> None:
        """Sync tool routes from both tools and schemas."""
        all_tools = []

        # Add schemas first (they take priority for routing)
        if self.schemas:
            all_tools.extend(self.schemas)

        # Add tools (may override schema routes if same name)
        if self.tools:
            # Only add tools that aren't already in schemas
            schema_names = {
                getattr(s, "__name__", getattr(s, "name", str(s))) for s in self.schemas
            }
            for tool in self.tools:
                tool_name = getattr(tool, "__name__", getattr(tool, "name", str(tool)))
                if tool_name not in schema_names:
                    all_tools.append(tool)

        # Sync routes using the mixin but override for transfer tools
        if all_tools:
            self.sync_tool_routes_from_tools(all_tools)

            # Override routing for transfer tools - they should route to agent nodes directly
            transfer_routes = {}
            for tool_name, route in self.tool_routes.items():
                if tool_name.startswith("transfer_to_"):
                    # Extract the agent name from the transfer tool name
                    agent_name = tool_name.replace("transfer_to_", "")
                    agent_node_name = f"agent_{agent_name}"

                    # Check if this agent node exists in our available nodes
                    if self.available_nodes and agent_node_name in self.available_nodes:
                        transfer_routes[tool_name] = f"agent_{agent_name}"
                        console.print(
                            f"[green]Mapped transfer tool {tool_name} -> {agent_node_name}[/green]"
                        )
                    else:
                        # Fallback to the agent name itself
                        transfer_routes[tool_name] = agent_name
                        console.print(
                            f"[yellow]Mapped transfer tool {tool_name} -> {agent_name} (fallback)[/yellow]"
                        )

            # Update the routes for transfer tools
            if transfer_routes:
                self.tool_routes.update(transfer_routes)
                console.print(
                    f"[cyan]Updated {len(transfer_routes)} transfer tool routes[/cyan]"
                )

            console.print(
                f"[cyan]ValidationNodeConfig synced routes for {len(all_tools)} tools[/cyan]"
            )
        else:
            console.print(
                "[yellow]ValidationNodeConfig: No tools or schemas to sync[/yellow]"
            )

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Union[Command, List[Send], str]:
        """
        Validate and route tool calls to appropriate nodes.
        """
        console.print("\n[bold blue]===== VALIDATION NODE START =====[/bold blue]")

        # Sync tools and schemas to ensure routes are up to date
        self._sync_tools_and_schemas()

        # Debug initial state
        debug_panel = Panel(
            f"State type: {type(state)}\n"
            f"Has messages: {hasattr(state, 'messages')}\n"
            f"Message count: {len(state.messages) if hasattr(state, 'messages') and state.messages else 0}\n"
            f"Config: {config}\n"
            f"Tools available: {len(self.tools)}\n"
            f"Schemas available: {len(self.schemas)}\n"  # Added schemas count
            f"Available nodes: {self.available_nodes}\n"
            f"Current tool_routes: {self.tool_routes}",
            title="Validation Node Debug Info",
            border_style="blue",
        )
        console.print(debug_panel)

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

        console.print(
            f"[cyan]Tool name mapping: {list(tool_name_mapping.keys())}[/cyan]"
        )

        # Get messages from state
        if not hasattr(state, "messages") or not state.messages:
            console.print("[red]❌ No messages found in state[/red]")
            logger.warning("No messages found in state")
            return "no_tool_calls"

        messages = state.messages
        console.print(f"[green]✓ Found {len(messages)} messages[/green]")

        # Check for tool calls in the last message
        last_message = messages[-1]
        console.print(f"[cyan]Last message type: {type(last_message).__name__}[/cyan]")

        if not isinstance(last_message, AIMessage):
            console.print("[red]❌ Last message is not an AIMessage[/red]")
            logger.warning("Last message is not an AIMessage")
            return "no_tool_calls"

        # Get tool calls - handle both attributes and additional_kwargs
        tool_calls = []
        console.print("[yellow]Checking for tool calls...[/yellow]")

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls
            console.print(
                f"[green]✓ Found {len(tool_calls)} tool calls in tool_calls attribute[/green]"
            )
        elif (
            hasattr(last_message, "additional_kwargs")
            and "tool_calls" in last_message.additional_kwargs
        ):
            tool_calls = last_message.additional_kwargs["tool_calls"]
            console.print(
                f"[green]✓ Found {len(tool_calls)} tool calls in additional_kwargs[/green]"
            )

        if not tool_calls:
            console.print("[red]❌ No tool calls found in last message[/red]")
            logger.warning("No tool calls found in last message")
            return "no_tool_calls"

        # Display tool calls info
        from rich.table import Table

        tool_table = Table(title="Tool Calls Found")
        tool_table.add_column("Index", style="cyan")
        tool_table.add_column("Name", style="green")
        tool_table.add_column("Type", style="yellow")
        tool_table.add_column("Args Preview", style="dim")

        for i, tc in enumerate(tool_calls):
            name = get_tool_name(tc)
            args = get_tool_args(tc)
            args_preview = str(args)[:50] + "..." if len(str(args)) > 50 else str(args)
            tool_table.add_row(str(i), name, str(type(tc).__name__), args_preview)

        console.print(tool_table)

        logger.info(f"Processing {len(tool_calls)} tool calls")

        # Group tool calls by their destination
        tool_groups = {}
        all_valid = True
        error_messages = []
        validation_results = []

        console.print("\n[bold yellow]===== PROCESSING TOOL CALLS =====[/bold yellow]")

        # Process each tool call individually
        for i, tool_call in enumerate(tool_calls):
            tool_name = get_tool_name(tool_call)
            console.print(
                f"\n[bold cyan]--- Processing Tool {i+1}/{len(tool_calls)}: {tool_name} ---[/bold cyan]"
            )

            logger.info(f"Processing tool call: {tool_name}")

            # Create standardized tool call
            standardized_tool_call = tool_call
            if isinstance(tool_call, dict):
                console.print("[yellow]Converting dict to ToolCall object[/yellow]")
                # Convert dict to ToolCall
                standardized_tool_call = ToolCall(
                    name=tool_name,
                    args=get_tool_args(tool_call),
                    id=get_tool_id(tool_call),
                )
            else:
                console.print("[green]Tool call already standardized[/green]")

            # Check if this is a transfer tool that should skip validation
            route = self.tool_routes.get(tool_name, "unknown")
            is_transfer_tool = tool_name.startswith("transfer_to_")

            if is_transfer_tool:
                console.print(
                    f"[magenta]🚀 Transfer tool detected - skipping validation[/magenta]"
                )

                # Transfer tools skip validation and go directly to destination
                destination = self._get_node_for_route(route)

                console.print(f"[green]✓ Transfer tool {tool_name}[/green]")
                console.print(
                    f"[cyan]Route: {route} -> Destination: {destination}[/cyan]"
                )

                if destination not in tool_groups:
                    tool_groups[destination] = []

                # Find the actual tool class
                tool_class = tool_name_mapping.get(tool_name)

                # Create a mock validated message for transfer tools
                validated_messages = messages[:-1] + [
                    AIMessage(content="", tool_calls=[standardized_tool_call])
                ]

                tool_groups[destination].append(
                    {
                        "tool_call": standardized_tool_call,
                        "tool_name": tool_name,
                        "tool_class": tool_class,
                        "route": route,
                        "validated_messages": validated_messages,
                        "is_transfer": True,  # Mark as transfer tool
                    }
                )

                validation_result = {
                    "tool_name": tool_name,
                    "has_error": False,
                    "message_count": len(validated_messages),
                    "route": route,
                    "destination": destination,
                    "group_size": len(tool_groups[destination]),
                    "skipped_validation": True,
                }

                logger.info(
                    f"Transfer tool [{tool_name}] routed to [{destination}] (no validation)"
                )
                validation_results.append(validation_result)
                continue

            # Regular validation for non-transfer tools
            console.print(f"[yellow]Running validation for {tool_name}[/yellow]")

            # Create temporary message with just this tool call for validation
            temp_message = AIMessage(content="", tool_calls=[standardized_tool_call])
            temp_messages = messages[:-1] + [temp_message]

            console.print(
                f"[dim]Created temp message with {len(temp_messages)} total messages[/dim]"
            )

            # Create validation node for this specific tool call
            validation_node = ValidationNode(
                schemas=self.schemas,
                format_error=self.format_error,
                name=f"validate_{tool_name}",
            )

            console.print(
                f"[cyan]Running validation with {len(self.schemas)} schemas[/cyan]"
            )

            # Run validation on this specific tool call
            validation_input = {"messages": temp_messages}
            try:
                result = validation_node.invoke(validation_input)
                validated_messages = result.get("messages", [])

                console.print(
                    f"[green]Validation completed - got {len(validated_messages)} messages back[/green]"
                )

                # Check if this specific tool call has validation errors
                has_error = any(
                    isinstance(msg, ToolMessage)
                    and has_tool_error(msg)
                    and msg.name == tool_name
                    for msg in validated_messages
                )

                validation_result = {
                    "tool_name": tool_name,
                    "has_error": has_error,
                    "message_count": len(validated_messages),
                }

                if has_error:
                    # This tool call has a validation error
                    console.print(
                        f"[red]❌ Validation error detected for {tool_name}[/red]"
                    )
                    logger.error(f"Validation error in {tool_name}")
                    all_valid = False
                    error_message = next(
                        (
                            msg
                            for msg in validated_messages
                            if isinstance(msg, ToolMessage) and msg.name == tool_name
                        ),
                        None,
                    )
                    if error_message:
                        error_messages.append(error_message)
                        console.print(
                            Panel(
                                f"Error in tool [{tool_name}]: {error_message.content}",
                                title="Validation Error",
                                border_style="red",
                            )
                        )
                        validation_result["error_content"] = str(error_message.content)[
                            :100
                        ]
                else:
                    # This tool call is valid - group by destination
                    destination = self._get_node_for_route(route)

                    console.print(f"[green]✓ Tool {tool_name} is valid[/green]")
                    console.print(
                        f"[cyan]Route: {route} -> Destination: {destination}[/cyan]"
                    )

                    if destination not in tool_groups:
                        tool_groups[destination] = []

                    # Find the actual tool/schema class
                    tool_class = tool_name_mapping.get(tool_name)
                    if not tool_class:
                        console.print(
                            f"[yellow]Warning: Could not find tool class for {tool_name}[/yellow]"
                        )
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
                            "tool_class": tool_class,  # Store the actual tool class
                            "route": route,
                            "validated_messages": validated_messages,
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
                console.print(
                    f"[red]❌ Exception during validation of {tool_name}: {str(e)}[/red]"
                )
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
        console.print("\n[bold magenta]===== VALIDATION SUMMARY =====[/bold magenta]")

        summary_table = Table(title="Validation Results")
        summary_table.add_column("Tool", style="cyan")
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Route", style="yellow")
        summary_table.add_column("Destination", style="blue")
        summary_table.add_column("Details", style="dim")

        for result in validation_results:
            status = "❌ ERROR" if result["has_error"] else "✓ VALID"

            # Special status for transfer tools
            if result.get("skipped_validation"):
                status = "🚀 TRANSFER"

            route = result.get("route", "N/A")
            dest = result.get("destination", "N/A")

            # Special details for transfer tools
            if result.get("skipped_validation"):
                details = "Routed directly (no validation)"
            else:
                details = result.get(
                    "error_content", f"Group size: {result.get('group_size', 'N/A')}"
                )

            summary_table.add_row(result["tool_name"], status, route, dest, details)

        console.print(summary_table)

        # Display tool groups
        if tool_groups:
            console.print(
                f"\n[bold green]Tool Groups ({len(tool_groups)} destinations):[/bold green]"
            )
            for dest, tools in tool_groups.items():
                tool_names = [t["tool_name"] for t in tools]
                console.print(f"  {dest}: {tool_names}")

        # Handle routing based on validation results
        console.print("\n[bold yellow]===== ROUTING DECISION =====[/bold yellow]")

        if not all_valid and not tool_groups:
            # All tools had validation errors - route back to agent with error info
            console.print(
                "[red]🚨 All tools failed validation - returning to agent[/red]"
            )
            logger.warning("All tool calls failed validation, returning to agent")

            # Add error messages to state and route back to agent
            error_update = {self.messages_key: error_messages}

            validated_agent_node = self.validate_node_exists(self.agent_node)
            return Command(update=error_update, goto=validated_agent_node)

        elif not all_valid and tool_groups:
            # Mixed results - execute valid tools but log warnings
            console.print(
                f"[orange1]⚠️ Mixed results - executing {len(tool_groups)} valid groups, some tools failed[/orange1]"
            )
            logger.warning(
                f"Mixed validation results - executing {len(tool_groups)} valid groups, some tools failed"
            )

        # If we have valid tools, determine routing strategy
        if len(tool_groups) == 1:
            # All valid tools go to the same destination - simple case
            destination = list(tool_groups.keys())[0]
            tools_for_dest = tool_groups[destination]

            console.print(
                f"[green]✓ Single destination routing to: {destination}[/green]"
            )
            console.print(
                f"[cyan]Tools for {destination}: {[t['tool_name'] for t in tools_for_dest]}[/cyan]"
            )

            logger.info(f"All valid tools route to {destination}")

            if destination == self.parser_node:
                # For parser node, we need to send each tool individually
                console.print(
                    "[yellow]Creating individual sends for parser node[/yellow]"
                )
                sends = []
                for tool_info in tools_for_dest:
                    # Find the validated tool message
                    validated_tool_msg = next(
                        (
                            msg
                            for msg in tool_info["validated_messages"]
                            if isinstance(msg, ToolMessage)
                            and msg.name == tool_info["tool_name"]
                        ),
                        None,
                    )

                    console.print(
                        f"[cyan]  Tool class for {tool_info['tool_name']}: {tool_info.get('tool_class')}[/cyan]"
                    )

                    if validated_tool_msg and tool_info.get("tool_class"):
                        send_obj = Send(
                            node=destination,
                            arg={
                                "tool_name": tool_info["tool_name"],
                                "tool": tool_info[
                                    "tool_class"
                                ],  # Use the stored tool class
                                "tool_call": tool_info["tool_call"],
                                "tool_message": validated_tool_msg,
                                self.messages_key: tool_info["validated_messages"],
                            },
                        )
                        sends.append(send_obj)
                        console.print(
                            f"[green]  ✓ Created send for {tool_info['tool_name']} with tool class[/green]"
                        )
                    else:
                        if not validated_tool_msg:
                            console.print(
                                f"[red]  ❌ No validated message found for {tool_info['tool_name']}[/red]"
                            )
                        if not tool_info.get("tool_class"):
                            console.print(
                                f"[red]  ❌ No tool class found for {tool_info['tool_name']}[/red]"
                            )

                console.print(
                    f"[bold green]Returning {len(sends)} Send objects for parser node[/bold green]"
                )
                return sends

            else:
                # For tool_node or retriever_node, send all tools together
                console.print(
                    f"[yellow]Creating combined send for {destination}[/yellow]"
                )
                all_tool_calls = [t["tool_call"] for t in tools_for_dest]
                combined_message = AIMessage(content="", tool_calls=all_tool_calls)

                send_obj = Send(
                    node=destination,
                    arg={self.messages_key: messages[:-1] + [combined_message]},
                )

                console.print(
                    f"[green]✓ Created combined send with {len(all_tool_calls)} tool calls[/green]"
                )
                console.print(
                    f"[bold green]Returning 1 Send object for {destination}[/bold green]"
                )
                return [send_obj]

        else:
            # Multiple destinations - need to send to each
            console.print(
                f"[yellow]Multi-destination routing to {len(tool_groups)} destinations[/yellow]"
            )
            sends = []

            for destination, tools_for_dest in tool_groups.items():
                console.print(f"[cyan]Processing destination: {destination}[/cyan]")

                if destination == self.parser_node:
                    # Parser node needs individual sends
                    console.print(
                        f"[dim]  Creating individual sends for parser ({len(tools_for_dest)} tools)[/dim]"
                    )
                    for tool_info in tools_for_dest:
                        validated_tool_msg = next(
                            (
                                msg
                                for msg in tool_info["validated_messages"]
                                if isinstance(msg, ToolMessage)
                                and msg.name == tool_info["tool_name"]
                            ),
                            None,
                        )

                        console.print(
                            f"[cyan]  Tool class for {tool_info['tool_name']}: {tool_info.get('tool_class')}[/cyan]"
                        )

                        if validated_tool_msg and tool_info.get("tool_class"):
                            send_obj = Send(
                                node=destination,
                                arg={
                                    "tool_name": tool_info["tool_name"],
                                    "tool": tool_info[
                                        "tool_class"
                                    ],  # Use the stored tool class
                                    "tool_call": tool_info["tool_call"],
                                    "tool_message": validated_tool_msg,
                                    self.messages_key: tool_info["validated_messages"],
                                },
                            )
                            sends.append(send_obj)
                            console.print(
                                f"[green]    ✓ Added send for {tool_info['tool_name']} with tool class[/green]"
                            )
                else:
                    # Combine tools for this destination
                    console.print(
                        f"[dim]  Creating combined send for {destination} ({len(tools_for_dest)} tools)[/dim]"
                    )
                    tool_calls_for_dest = [t["tool_call"] for t in tools_for_dest]
                    combined_message = AIMessage(
                        content="", tool_calls=tool_calls_for_dest
                    )

                    send_obj = Send(
                        node=destination,
                        arg={self.messages_key: messages[:-1] + [combined_message]},
                    )
                    sends.append(send_obj)
                    console.print(
                        f"[green]    ✓ Added combined send with {len(tool_calls_for_dest)} tool calls[/green]"
                    )

            console.print(
                f"[bold green]Returning {len(sends)} Send objects for multiple destinations[/bold green]"
            )
            return sends

        # Fallback case - no tool calls to route
        console.print("[yellow]No valid tool calls to route - returning END[/yellow]")
        return END
