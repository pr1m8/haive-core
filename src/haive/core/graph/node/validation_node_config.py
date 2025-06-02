import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ValidationNode
from langgraph.types import Command, Send
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

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
                import json

                try:
                    return json.loads(args)
                except:
                    return {"raw_args": args}
            return args

    # Fallback
    return {}


# Helper function to safely get tool ID from various formats
def get_tool_id(tool_call):
    """Extract tool ID from either ToolCall object or dictionary."""
    if hasattr(tool_call, "id"):
        return tool_call.id
    elif isinstance(tool_call, dict) and "id" in tool_call:
        return tool_call["id"]

    # Fallback
    return f"id_{id(tool_call)}"


class ValidationNodeConfig(NodeConfig):
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
    agent_node: str = Field(
        default="agent_node", description="Node to return to for validation errors"
    )
    tool_node: str = Field(
        default="tool_node", description="Node for executing standard tools"
    )
    parser_node: str = Field(
        default="parse_output", description="Node for parsing Pydantic models"
    )
    retriever_node: str = Field(
        default="retriever_node", description="Node for retriever tools"
    )
    tools: List[Any] = Field(
        default_factory=list, description="List of available tools"
    )
    tool_routes: Dict[str, str] = Field(
        default_factory=dict, description="Mapping of tool names to routes"
    )

    def _sync_tool_routes(self):
        """Synchronize tool_routes with current tools."""
        new_routes = {}

        for i, tool in enumerate(self.tools):
            # Determine tool name
            if hasattr(tool, "name"):
                tool_name = tool.name
            elif isinstance(tool, type) and hasattr(tool, "__name__"):
                tool_name = tool.__name__
            elif hasattr(tool, "__name__"):
                tool_name = tool.__name__
            else:
                tool_name = f"tool_{i}"

            # Determine route/type
            if isinstance(tool, type) and issubclass(tool, BaseModel):
                route = "pydantic_model"
            elif isinstance(tool, BaseTool) or (
                isinstance(tool, type) and issubclass(tool, BaseTool)
            ):
                route = "langchain_tool"
            elif callable(tool):
                route = "function"
            else:
                route = "unknown"

            new_routes[tool_name] = route

        self.tool_routes = new_routes

    def _get_node_for_route(self, route: str) -> str:
        """Map tool route to appropriate node."""
        route_mapping = {
            "pydantic_model": self.parser_node,
            "langchain_tool": self.tool_node,
            "function": self.tool_node,
            "retriever": self.retriever_node,
            "unknown": self.tool_node,  # Default fallback
        }
        return route_mapping.get(route, self.tool_node)

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Union[Command, List[Send], str]:
        """
        Validate and route tool calls to appropriate nodes.
        """
        console.print("\n[bold blue]===== VALIDATION NODE START =====[/bold blue]")

        # Debug initial state
        debug_panel = Panel(
            f"State type: {type(state)}\n"
            f"Has messages: {hasattr(state, 'messages')}\n"
            f"Message count: {len(state.messages) if hasattr(state, 'messages') and state.messages else 0}\n"
            f"Config: {config}\n"
            f"Tools available: {len(self.tools)}\n"
            f"Current tool_routes: {self.tool_routes}",
            title="Validation Node Debug Info",
            border_style="blue",
        )
        console.print(debug_panel)

        # Sync tool routes with current tools
        console.print("[yellow]Syncing tool routes...[/yellow]")
        old_routes = self.tool_routes.copy()
        self._sync_tool_routes()

        if old_routes != self.tool_routes:
            console.print(
                f"[green]Tool routes updated:[/green] {old_routes} -> {self.tool_routes}"
            )
        else:
            console.print(f"[dim]Tool routes unchanged: {self.tool_routes}[/dim]")

        # Get messages from state
        messages = state.messages
        if not messages:
            console.print("[red]❌ No messages found in state[/red]")
            logger.warning("No messages found in state")
            return "no_tool_calls"

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

            logger.info(f"Validating tool call: {tool_name}")

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
                    route = self.tool_routes.get(tool_name, "function")
                    destination = self._get_node_for_route(route)

                    console.print(f"[green]✓ Tool {tool_name} is valid[/green]")
                    console.print(
                        f"[cyan]Route: {route} -> Destination: {destination}[/cyan]"
                    )

                    if destination not in tool_groups:
                        tool_groups[destination] = []

                    tool_groups[destination].append(
                        {
                            "tool_call": standardized_tool_call,
                            "tool_name": tool_name,
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
            route = result.get("route", "N/A")
            dest = result.get("destination", "N/A")
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
            # All tools had validation errors
            console.print(
                "[red]🚨 All tools failed validation - returning 'has_errors'[/red]"
            )
            logger.warning("All tool calls failed validation, returning error status")
            return "has_errors"

        elif not all_valid and tool_groups:
            # Mixed results - this is problematic for the conversation flow
            # We'll execute valid tools but note that some failed
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

                    if validated_tool_msg:
                        send_obj = Send(
                            node=destination,
                            arg={
                                "tool_name": tool_info["tool_name"],
                                "tool_call": tool_info["tool_call"],
                                "tool_message": validated_tool_msg,
                                self.messages_key: tool_info["validated_messages"],
                            },
                        )
                        sends.append(send_obj)
                        console.print(
                            f"[green]  ✓ Created send for {tool_info['tool_name']}[/green]"
                        )
                    else:
                        console.print(
                            f"[red]  ❌ No validated message found for {tool_info['tool_name']}[/red]"
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

                        if validated_tool_msg:
                            send_obj = Send(
                                node=destination,
                                arg={
                                    "tool_name": tool_info["tool_name"],
                                    "tool_call": tool_info["tool_call"],
                                    "tool_message": validated_tool_msg,
                                    self.messages_key: tool_info["validated_messages"],
                                },
                            )
                            sends.append(send_obj)
                            console.print(
                                f"[green]    ✓ Added send for {tool_info['tool_name']}[/green]"
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
            return Command(update={}, goto=sends)
