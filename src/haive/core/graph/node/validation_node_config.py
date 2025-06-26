# src/haive/core/graph/node/validation_node_config.py

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt import ValidationNode
from pydantic import BaseModel, Field

from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin
from haive.core.graph.common.types import ConfigLike, StateLike
from haive.core.graph.node.base_config import NodeConfig
from haive.core.graph.node.types import NodeType

# Configure logger with rich handler
logger = logging.getLogger(__name__)

# Add rich handler if not already present
if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    from rich.logging import RichHandler

    handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=True,
        enable_link_path=True,
        markup=True,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
    )
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def has_tool_error(message: ToolMessage) -> bool:
    """Check if a ToolMessage contains an error."""
    if not isinstance(message, ToolMessage):
        return False

    content = message.content

    # Check string content for error terms
    if isinstance(content, str):
        error_terms = ["error", "invalid", "failed", "exception", "validation error"]
        contains_error = any(term in content.lower() for term in error_terms)
        if contains_error:
            logger.debug(
                f"[bold yellow]Tool error detected in content:[/bold yellow] {content[:100]}..."
            )
        return contains_error

    # Check dict content for error keys
    if isinstance(content, dict):
        error_keys = ["error", "errors", "exception", "validation_error"]
        has_error_key = any(key in content for key in error_keys)
        if has_error_key:
            logger.debug(
                f"[bold yellow]Tool error detected in dict:[/bold yellow] {list(content.keys())}"
            )
        return has_error_key

    # Check additional kwargs
    if hasattr(message, "additional_kwargs") and message.additional_kwargs:
        if message.additional_kwargs.get("is_error"):
            logger.debug(
                "[bold yellow]Tool error detected in additional_kwargs[/bold yellow]"
            )
            return True

    return False


def get_tool_name(tool_call: Any) -> str:
    """Extract tool name from either ToolCall object or dictionary."""
    if hasattr(tool_call, "name"):
        return tool_call.name

    if isinstance(tool_call, dict):
        if "name" in tool_call:
            return tool_call["name"]
        elif "function" in tool_call and "name" in tool_call["function"]:
            return tool_call["function"]["name"]

    logger.warning(
        "[bold yellow]Could not extract tool name from tool_call[/bold yellow]"
    )
    return "unknown_tool"


def get_tool_args(tool_call: Any) -> Dict[str, Any]:
    """Extract tool arguments from either ToolCall object or dictionary."""
    if hasattr(tool_call, "args"):
        return tool_call.args

    if isinstance(tool_call, dict):
        if "args" in tool_call:
            return tool_call["args"]
        elif "arguments" in tool_call:
            args = tool_call["arguments"]
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except:
                    return {"raw_args": args}
            return args
        elif "function" in tool_call and "arguments" in tool_call["function"]:
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except:
                    return {"raw_args": args}
            return args

    logger.warning(
        "[bold yellow]Could not extract tool args from tool_call[/bold yellow]"
    )
    return {}


def get_tool_id(tool_call: Any) -> str:
    """Extract tool ID from either ToolCall object or dictionary."""
    if hasattr(tool_call, "id"):
        return tool_call.id
    elif isinstance(tool_call, dict) and "id" in tool_call:
        return tool_call["id"]

    # Generate a fallback ID
    fallback_id = f"tool_call_{id(tool_call)}"
    logger.debug(f"[bold cyan]Generated fallback tool ID:[/bold cyan] {fallback_id}")
    return fallback_id


class ValidationNodeConfig(NodeConfig, ToolRouteMixin):
    """Configuration for a validation node that routes tool calls to appropriate nodes."""

    node_type: NodeType = Field(
        default=NodeType.VALIDATION, description="The type of node"
    )
    name: str = Field(default="validation", description="The name of the node")
    messages_key: str = Field(
        default="messages", description="The key to use for the messages field"
    )

    # Engine name to get tools/schemas from
    engine_name: Optional[str] = Field(
        default=None, description="Name of the engine to get tools/schemas from"
    )

    # Override tools/schemas if needed
    schemas: List[Any] = Field(
        default_factory=list, description="The schemas to use for validation (override)"
    )
    tools: List[Any] = Field(
        default_factory=list, description="List of available tools (override)"
    )

    format_error: Optional[Callable] = Field(
        default=None, description="Custom formatter for validation errors"
    )

    # Node routing configuration
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
        default="retriever", description="Node for retriever tools"
    )

    # Available nodes in the graph (for validation)
    available_nodes: List[str] = Field(
        default_factory=list, description="List of nodes available in the graph"
    )

    # Custom route mappings
    custom_route_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom mappings from route names to node names",
    )

    # Direct node routes
    direct_node_routes: List[str] = Field(
        default_factory=list,
        description="Routes that map directly to node names without transformation",
    )

    def validate_node_exists(self, node_name: str) -> str:
        """Validate that a node exists in the graph."""
        logger.debug(f"[bold blue]Validating node exists:[/bold blue] {node_name}")

        if not self.available_nodes:
            logger.debug("  No available_nodes list, assuming node exists")
            return node_name

        if node_name in self.available_nodes:
            logger.debug(f"  [bold green]✓ Node exists[/bold green]")
            return node_name

        # Try fallbacks
        logger.warning(
            f"[bold yellow]Node '{node_name}' not found in available nodes[/bold yellow]"
        )
        fallback_options = [self.agent_node, "agent_node", "START"]

        for fallback in fallback_options:
            if fallback in self.available_nodes:
                logger.info(f"[bold cyan]Using fallback node:[/bold cyan] {fallback}")
                return fallback

        logger.error(
            f"[bold red]No valid fallback found, returning original:[/bold red] {node_name}"
        )
        return node_name

    def _get_node_for_route(self, route: str) -> str:
        """Map tool route to appropriate node."""
        logger.debug(f"[bold blue]Getting node for route:[/bold blue] {route}")

        # Check custom mappings first
        if route in self.custom_route_mappings:
            node = self.custom_route_mappings[route]
            logger.debug(f"  Found in custom mappings: {node}")
            return self.validate_node_exists(node)

        # Check if this is a direct node route
        if route in self.direct_node_routes:
            logger.debug(f"  Direct node route: {route}")
            return self.validate_node_exists(route)

        # Standard route mapping
        route_mapping = {
            "pydantic_model": self.parser_node,
            "langchain_tool": self.tool_node,
            "function": self.tool_node,
            "retriever": self.retriever_node,
            "unknown": self.tool_node,
        }

        target_node = route_mapping.get(route, self.tool_node)
        logger.debug(f"  Standard mapping: {route} -> {target_node}")
        return self.validate_node_exists(target_node)

    def _get_engine_from_state(self, state: StateLike) -> Optional[Any]:
        """Get engine from state.engines or registry."""
        logger.debug(f"[bold blue]Getting engine:[/bold blue] {self.engine_name}")

        if not self.engine_name:
            logger.debug("  No engine name specified")
            return None

        # FIRST: Try to get from state.engines using engine name
        if hasattr(state, "engines") and isinstance(state.engines, dict):
            logger.debug(f"  State has engines dict with {len(state.engines)} engines")
            logger.debug(f"  Available engine keys: {list(state.engines.keys())}")

            # Try exact name match
            engine = state.engines.get(self.engine_name)
            if engine:
                logger.info(
                    f"[bold green]✓ Found engine in state.engines:[/bold green] {self.engine_name}"
                )
                logger.debug(f"    Engine type: {type(engine).__name__}")
                logger.debug(f"    Engine has tools: {hasattr(engine, 'tools')}")
                logger.debug(
                    f"    Engine has tool_routes: {hasattr(engine, 'tool_routes')}"
                )
                return engine

            # Try to find by engine.name attribute
            for key, eng in state.engines.items():
                if hasattr(eng, "name") and eng.name == self.engine_name:
                    logger.info(
                        f"[bold green]✓ Found engine by name attribute:[/bold green] {self.engine_name} (key: {key})"
                    )
                    return eng

        # SECOND: Try to get from state directly (if engines are stored as attributes)
        if hasattr(state, self.engine_name):
            engine = getattr(state, self.engine_name)
            logger.info(
                f"[bold green]✓ Found engine as state attribute:[/bold green] {self.engine_name}"
            )
            return engine

        # LAST: Fallback to registry
        logger.debug("  Engine not found in state, trying registry...")
        try:
            from haive.core.engine.base import EngineRegistry

            registry = EngineRegistry.get_instance()
            engine = registry.find(self.engine_name)
            if engine:
                logger.info(
                    f"[bold green]✓ Found engine in EngineRegistry:[/bold green] {self.engine_name}"
                )
                return engine
            else:
                logger.warning(
                    f"[bold yellow]Engine not found in registry:[/bold yellow] {self.engine_name}"
                )
        except Exception as e:
            logger.error(f"[bold red]Error accessing EngineRegistry:[/bold red] {e}")

        logger.error(
            f"[bold red]Engine '{self.engine_name}' not found anywhere[/bold red]"
        )
        return None

    def _get_tools_and_schemas_from_engine(
        self, engine: Any
    ) -> tuple[List[Any], List[Any]]:
        """Extract tools and schemas from an engine."""
        logger.debug("[bold blue]Extracting tools and schemas from engine[/bold blue]")

        tools = []
        schemas = []

        # Log what the engine has
        logger.debug(
            f"  Engine attributes: {[attr for attr in dir(engine) if not attr.startswith('_')][:20]}..."
        )

        # Get tools
        if hasattr(engine, "tools") and engine.tools:
            tools.extend(engine.tools)
            logger.debug(f"  Found {len(engine.tools)} tools")
            for tool in engine.tools[:3]:  # Log first 3
                tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
                logger.debug(f"    - {tool_name}")

        # Get pydantic tools/schemas
        if hasattr(engine, "pydantic_tools") and engine.pydantic_tools:
            schemas.extend(engine.pydantic_tools)
            logger.debug(f"  Found {len(engine.pydantic_tools)} pydantic_tools")

        if hasattr(engine, "schemas") and engine.schemas:
            schemas.extend(engine.schemas)
            logger.debug(f"  Found {len(engine.schemas)} schemas")

        # Check for structured_output_model
        if (
            hasattr(engine, "structured_output_model")
            and engine.structured_output_model
        ):
            schemas.append(engine.structured_output_model)
            model_name = getattr(
                engine.structured_output_model,
                "__name__",
                str(engine.structured_output_model),
            )
            logger.debug(f"  Found structured_output_model: {model_name}")

        logger.info(
            f"[bold cyan]Total extracted:[/bold cyan] {len(tools)} tools, {len(schemas)} schemas"
        )
        return tools, schemas

    def _sync_tools_and_schemas(self, state: StateLike) -> tuple[List[Any], List[Any]]:
        """Sync tools and schemas from engine in state - ONLY from the specified engine."""
        logger.info("[bold magenta]Syncing tools and schemas from state[/bold magenta]")

        all_tools = []
        all_schemas = []

        # Get from engine if specified
        if self.engine_name:
            engine = self._get_engine_from_state(state)
            if engine:
                # IMPORTANT: Only use tools/schemas from THIS specific engine
                # Do not use tools from other engines in the state to prevent contamination
                logger.info(
                    f"[bold yellow]Using ONLY tools/schemas from engine: {self.engine_name}[/bold yellow]"
                )

                # Get tools and schemas ONLY from this specific engine
                engine_tools, engine_schemas = self._get_tools_and_schemas_from_engine(
                    engine
                )

                # Filter tools to only include those that actually belong to this engine
                # by checking the engine's original tool definitions
                original_engine_tools = []
                original_engine_schemas = []

                # Get the original tools from the engine - filter out contaminated tools
                if hasattr(engine, "tools") and engine.tools:
                    for tool in engine.tools:
                        tool_name = getattr(
                            tool, "name", getattr(tool, "__name__", str(tool))
                        )

                        # FILTER: Only include tools that are NOT Pydantic model classes
                        # Pydantic models (like Plan) should only be in structured_output_model, not tools
                        if hasattr(tool, "__bases__") and any(
                            "BaseModel" in str(base) for base in tool.__bases__
                        ):
                            logger.debug(
                                f"  Skipping Pydantic model '{tool_name}' from tools list (belongs in structured_output)"
                            )
                            continue

                        # FILTER: Only include actual callable tools
                        if callable(tool) or hasattr(tool, "invoke"):
                            original_engine_tools.append(tool)
                            logger.debug(f"  Including tool: {tool_name}")
                        else:
                            logger.debug(f"  Skipping non-callable tool: {tool_name}")

                # For schemas, only include structured_output_model if this engine has it
                if (
                    hasattr(engine, "structured_output_model")
                    and engine.structured_output_model
                ):
                    original_engine_schemas.append(engine.structured_output_model)
                    model_name = getattr(
                        engine.structured_output_model,
                        "__name__",
                        str(engine.structured_output_model),
                    )
                    logger.debug(f"  Including structured output model: {model_name}")

                # ADDITIONAL FILTER: Remove any non-tool items that might have been added incorrectly
                filtered_tools = []
                for tool in original_engine_tools:
                    # Skip if this looks like a Pydantic model class
                    if isinstance(tool, type) and hasattr(tool, "model_fields"):
                        logger.debug(
                            f"  Filtering out Pydantic model class from tools: {tool}"
                        )
                        continue
                    filtered_tools.append(tool)

                original_engine_tools = filtered_tools

                # Use filtered tools/schemas instead of all tools
                all_tools = original_engine_tools
                all_schemas = original_engine_schemas

                logger.info(
                    f"[bold green]✓ Filtered to {len(all_tools)} tools and {len(all_schemas)} schemas from engine[/bold green]"
                )

                # Get tool routes from engine, but filter them too
                if hasattr(engine, "tool_routes") and engine.tool_routes:
                    # Only copy routes for tools/schemas we actually have
                    filtered_routes = {}
                    tool_names = {
                        getattr(tool, "name", getattr(tool, "__name__", str(tool)))
                        for tool in all_tools
                    }
                    schema_names = {
                        getattr(schema, "__name__", str(schema))
                        for schema in all_schemas
                    }
                    all_names = tool_names | schema_names

                    for name, route in engine.tool_routes.items():
                        if name in all_names:
                            filtered_routes[name] = route

                    self.tool_routes = filtered_routes
                    logger.info(
                        f"[bold green]✓ Filtered tool routes to {len(self.tool_routes)} entries[/bold green]"
                    )
                    for name, route in list(self.tool_routes.items())[
                        :5
                    ]:  # Log first 5
                        logger.debug(f"    {name} -> {route}")
            else:
                logger.error(
                    f"[bold red]Could not find engine '{self.engine_name}'[/bold red]"
                )

        # Add any manual overrides (these take precedence)
        if self.schemas:
            all_schemas.extend(self.schemas)
            logger.debug(f"  Added {len(self.schemas)} override schemas")
        if self.tools:
            all_tools.extend(self.tools)
            logger.debug(f"  Added {len(self.tools)} override tools")

        # Sync routes if we don't have them from engine
        if not self.tool_routes and (all_tools or all_schemas):
            logger.info("[bold cyan]Syncing tool routes from tools/schemas[/bold cyan]")
            combined = all_schemas + all_tools  # Schemas first for priority
            self.sync_tool_routes_from_tools(combined)
            logger.debug(f"  Generated {len(self.tool_routes)} tool routes")
            for name, route in list(self.tool_routes.items())[:5]:  # Log first 5
                logger.debug(f"    {name} -> {route}")

        return all_tools, all_schemas

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Union[str, List[str]]:
        """
        Validate and route tool calls to appropriate nodes.

        Returns ONLY routing decisions - no state updates!
        """
        logger.info(
            "[bold magenta]=== ValidationNodeConfig Execution ===[/bold magenta]"
        )
        logger.debug(f"State type: {type(state).__name__}")
        logger.debug(f"Available nodes: {self.available_nodes}")

        # Log state contents for debugging
        if hasattr(state, "__dict__"):
            logger.debug(f"State attributes: {list(state.__dict__.keys())}")
        if hasattr(state, "engines"):
            logger.debug(
                f"State.engines keys: {list(state.engines.keys()) if isinstance(state.engines, dict) else 'Not a dict'}"
            )

        # Get tools and schemas from engine/state
        validation_tools, validation_schemas = self._sync_tools_and_schemas(state)

        # Get messages from state
        if not hasattr(state, self.messages_key):
            logger.error(
                f"[bold red]State missing messages key:[/bold red] {self.messages_key}"
            )
            return "has_errors"

        messages = getattr(state, self.messages_key, [])
        if not messages:
            logger.warning("[bold yellow]No messages found in state[/bold yellow]")
            return "no_tool_calls"

        logger.info(f"[bold cyan]Processing {len(messages)} messages[/bold cyan]")
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage):
            logger.debug(
                f"Last message is not AIMessage: {type(last_message).__name__}"
            )
            return "no_tool_calls"

        # Get tool calls
        tool_calls = []
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls = last_message.tool_calls
            logger.info(
                f"[bold green]✓ Found {len(tool_calls)} tool calls in message[/bold green]"
            )
        elif (
            hasattr(last_message, "additional_kwargs")
            and "tool_calls" in last_message.additional_kwargs
        ):
            tool_calls = last_message.additional_kwargs["tool_calls"]
            logger.info(
                f"[bold green]✓ Found {len(tool_calls)} tool calls in additional_kwargs[/bold green]"
            )

        if not tool_calls:
            logger.info("[bold yellow]No tool calls found[/bold yellow]")
            return "no_tool_calls"

        # Build tool name mapping for validation
        logger.debug("[bold blue]Building tool name mapping[/bold blue]")
        tool_name_mapping = {}

        for schema in validation_schemas:
            if hasattr(schema, "__name__"):
                tool_name_mapping[schema.__name__] = schema
                logger.debug(f"  Schema: {schema.__name__}")
            if hasattr(schema, "name"):
                tool_name_mapping[schema.name] = schema
                logger.debug(f"  Schema (by name): {schema.name}")

        for tool in validation_tools:
            if hasattr(tool, "name"):
                tool_name_mapping[tool.name] = tool
                logger.debug(f"  Tool: {tool.name}")
            elif hasattr(tool, "__name__"):
                tool_name_mapping[tool.__name__] = tool
                logger.debug(f"  Tool (by __name__): {tool.__name__}")

        logger.info(
            f"[bold cyan]Total tools/schemas available:[/bold cyan] {len(tool_name_mapping)}"
        )

        # Process tool calls and determine routing
        logger.info("[bold blue]Processing tool calls for routing[/bold blue]")
        destinations = set()
        has_errors = False

        for i, tool_call in enumerate(tool_calls):
            tool_name = get_tool_name(tool_call)
            logger.debug(f"\n[bold cyan]Tool call {i+1}:[/bold cyan] {tool_name}")

            # Check if tool exists in our mapping
            if tool_name not in tool_name_mapping:
                logger.error(
                    f"  [bold red]✗ Tool '{tool_name}' not found in available tools![/bold red]"
                )
                logger.debug(f"  Available tools: {list(tool_name_mapping.keys())}")
                has_errors = True
                continue

            # Get route for this tool
            route = self.tool_routes.get(tool_name, "unknown")
            logger.debug(f"  Route: {route}")

            # Determine destination node
            destination = self._get_node_for_route(route)
            logger.debug(f"  Destination: {destination}")

            # Check if should skip validation for direct routes
            if route in self.direct_node_routes:
                logger.debug(
                    f"  [bold green]✓ Direct route - skipping validation[/bold green]"
                )
                destinations.add(destination)
                continue

            # Validate the tool call
            logger.debug(f"  [bold blue]Running validation[/bold blue]")

            # Create temporary message for validation
            temp_message = AIMessage(content="", tool_calls=[tool_call])
            temp_messages = messages[:-1] + [temp_message]

            # Create validation node with both tools and schemas
            # ValidationNode expects a combined list of tools and schemas
            combined_schemas = list(validation_tools) + list(validation_schemas)
            validation_node = ValidationNode(
                schemas=combined_schemas,
                format_error=self.format_error,
            )

            try:
                # Run validation
                result = validation_node.invoke({"messages": temp_messages})
                validated_messages = result.get("messages", [])

                # Check for validation errors
                error_found = False
                for msg in validated_messages:
                    if isinstance(msg, ToolMessage) and msg.name == tool_name:
                        if has_tool_error(msg):
                            logger.warning(
                                f"  [bold red]✗ Validation error for {tool_name}[/bold red]"
                            )
                            error_found = True
                            has_errors = True
                        else:
                            logger.info(
                                f"  [bold green]✓ Validation passed[/bold green]"
                            )
                        break

                if not error_found:
                    destinations.add(destination)

            except Exception as e:
                logger.exception(
                    f"[bold red]Validation exception for {tool_name}:[/bold red] {e}"
                )
                has_errors = True

        # Determine routing based on results
        logger.info(f"\n[bold blue]Routing decision:[/bold blue]")
        logger.info(f"  Destinations: {destinations}")
        logger.info(f"  Has errors: {has_errors}")

        # If all validations failed, return to agent
        if has_errors and not destinations:
            logger.warning(
                f"[bold yellow]All validations failed - routing to:[/bold yellow] has_errors"
            )
            return "has_errors"

        # Route to appropriate destination(s)
        if not destinations:
            logger.warning("[bold yellow]No valid destinations found[/bold yellow]")
            return END

        # Convert destinations to list for consistent return type
        destinations_list = list(destinations)

        if len(destinations_list) == 1:
            destination = destinations_list[0]
            logger.info(f"[bold green]✓ Single destination:[/bold green] {destination}")
            return destination
        else:
            logger.info(
                f"[bold green]✓ Multiple destinations:[/bold green] {destinations_list}"
            )
            return destinations_list

        logger.info(
            "[bold magenta]=== ValidationNodeConfig Complete ===[/bold magenta]"
        )
