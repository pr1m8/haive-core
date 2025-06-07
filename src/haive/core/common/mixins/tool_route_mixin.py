"""
Common mixins for configuration classes.

Provides reusable functionality that can be mixed into various config classes
throughout the haive framework.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

logger = logging.getLogger(__name__)
console = Console()


class ToolRouteMixin(BaseModel):
    """
    Mixin for managing tool routes and converting configurations to tools.

    Provides functionality for:
    - Setting and managing tool routes
    - Base to_tool method that can be overridden
    - Tool metadata management
    - Route-based tool organization
    """

    # Tool routes mapping tool names to their types/destinations
    tool_routes: Dict[str, str] = Field(
        default_factory=dict, description="Mapping of tool names to their routes/types"
    )

    # Tool metadata for enhanced management
    tool_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Metadata for each tool"
    )

    def set_tool_route(
        self, tool_name: str, route: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "ToolRouteMixin":
        """
        Set a tool route with optional metadata.

        Args:
            tool_name: Name of the tool
            route: Route/type for the tool (e.g., 'retriever', 'pydantic_model', 'function')
            metadata: Optional metadata for the tool

        Returns:
            Self for chaining
        """
        self.tool_routes[tool_name] = route

        if metadata:
            if not hasattr(self, "tool_metadata"):
                self.tool_metadata = {}
            self.tool_metadata[tool_name] = metadata

        logger.debug(f"Set tool route: {tool_name} -> {route}")
        if metadata:
            logger.debug(f"  Metadata: {metadata}")

        return self

    def get_tool_route(self, tool_name: str) -> Optional[str]:
        """Get the route for a specific tool."""
        return self.tool_routes.get(tool_name)

    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific tool."""
        if not hasattr(self, "tool_metadata"):
            return None
        return self.tool_metadata.get(tool_name)

    def list_tools_by_route(self, route: str) -> List[str]:
        """Get all tool names for a specific route."""
        return [name for name, r in self.tool_routes.items() if r == route]

    def remove_tool_route(self, tool_name: str) -> "ToolRouteMixin":
        """Remove a tool route and its metadata."""
        if tool_name in self.tool_routes:
            del self.tool_routes[tool_name]

        if hasattr(self, "tool_metadata") and tool_name in self.tool_metadata:
            del self.tool_metadata[tool_name]

        logger.debug(f"Removed tool route: {tool_name}")
        return self

    def update_tool_routes(self, routes: Dict[str, str]) -> "ToolRouteMixin":
        """Update multiple tool routes at once."""
        self.tool_routes.update(routes)
        logger.debug(f"Updated tool routes: {routes}")
        return self

    def clear_tool_routes(self) -> "ToolRouteMixin":
        """Clear all tool routes and metadata."""
        self.tool_routes.clear()
        if hasattr(self, "tool_metadata"):
            self.tool_metadata.clear()
        logger.debug("Cleared all tool routes")
        return self

    def sync_tool_routes_from_tools(self, tools: List[Any]) -> "ToolRouteMixin":
        """
        Synchronize tool_routes with a list of tools.

        Args:
            tools: List of tools to analyze and create routes for

        Returns:
            Self for chaining
        """
        # Clear existing routes and start fresh
        self.clear_tool_routes()

        for i, tool in enumerate(tools):
            # Determine tool name
            if hasattr(tool, "name"):
                tool_name = tool.name
            elif isinstance(tool, type) and hasattr(tool, "__name__"):
                tool_name = tool.__name__
            elif hasattr(tool, "__name__"):
                tool_name = tool.__name__
            else:
                tool_name = f"tool_{i}"

            # Determine route/type and metadata
            metadata = {}
            if isinstance(tool, type) and issubclass(tool, BaseModel):
                route = "pydantic_model"
                metadata = {
                    "class_name": tool.__name__,
                    "module": getattr(tool, "__module__", "unknown"),
                    "tool_type": "pydantic_model",
                }
            elif hasattr(tool, "__class__") and "BaseTool" in str(
                tool.__class__.__mro__
            ):
                route = "langchain_tool"
                metadata = {
                    "tool_type": "BaseTool",
                    "is_instance": not isinstance(tool, type),
                }
            elif callable(tool):
                route = "function"
                metadata = {
                    "callable_type": type(tool).__name__,
                    "has_annotations": hasattr(tool, "__annotations__"),
                }
            else:
                route = "unknown"
                metadata = {"original_type": type(tool).__name__}

            # Set the route with metadata
            self.set_tool_route(tool_name, route, metadata)

        return self

    def to_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        route: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Convert this configuration to a tool.

        Base implementation that should be overridden by specific config types.
        For example, RetrieverConfig would override this to create retriever tools.

        Args:
            name: Tool name (defaults to config name if available)
            description: Tool description (defaults to config description if available)
            route: Tool route/type to set
            **kwargs: Additional kwargs for tool creation

        Returns:
            A tool that can be used with LLMs
        """
        # Get default name and description from config
        config_name = getattr(self, "name", None)
        config_description = getattr(self, "description", None)

        tool_name = name or config_name or f"tool_{id(self)}"
        tool_description = (
            description
            or config_description
            or f"Tool created from {type(self).__name__}"
        )

        # Set tool route if provided
        if route:
            self.set_tool_route(tool_name, route, kwargs.get("metadata"))

        # Call the specific implementation
        tool = self._create_tool_implementation(tool_name, tool_description, **kwargs)

        logger.debug(f"Created tool: {tool_name} ({type(tool).__name__})")
        return tool

    def _create_tool_implementation(self, name: str, description: str, **kwargs) -> Any:
        """
        Create the actual tool implementation.

        This should be overridden by specific config types:
        - RetrieverConfig: Create retriever tools
        - LLMConfig: Create LLM function tools
        - Other configs: Create appropriate tools
        """
        raise NotImplementedError(
            f"Tool creation not implemented for {type(self).__name__}. "
            f"Override _create_tool_implementation() to create tools from this config type."
        )

    def debug_tool_routes(self) -> "ToolRouteMixin":
        """Print debug information about tool routes."""
        console.print("\n" + "=" * 60)
        console.print("[bold blue]🛤️ TOOL ROUTES DEBUG[/bold blue]")
        console.print("=" * 60)

        if not self.tool_routes:
            console.print("[yellow]No tool routes configured[/yellow]")
            return self

        # Create a tree for routes
        routes_tree = Tree("🛤️ [cyan]Tool Routes[/cyan]")

        # Group by route type
        route_groups = {}
        for tool_name, route in self.tool_routes.items():
            if route not in route_groups:
                route_groups[route] = []
            route_groups[route].append(tool_name)

        for route, tools in route_groups.items():
            route_branch = routes_tree.add(
                f"[yellow]{route}[/yellow] ({len(tools)} tools)"
            )
            for tool_name in tools:
                metadata = self.get_tool_metadata(tool_name)
                metadata_str = f" [dim]({metadata})[/dim]" if metadata else ""
                route_branch.add(f"{tool_name}{metadata_str}")

        console.print(routes_tree)

        # Show metadata table if available
        if hasattr(self, "tool_metadata") and self.tool_metadata:
            metadata_table = Table(title="Tool Metadata", show_header=True)
            metadata_table.add_column("Tool", style="cyan")
            metadata_table.add_column("Metadata", style="yellow")

            for tool_name, metadata in self.tool_metadata.items():
                metadata_str = (
                    str(metadata)
                    if len(str(metadata)) < 50
                    else str(metadata)[:47] + "..."
                )
                metadata_table.add_row(tool_name, metadata_str)

            console.print(metadata_table)

        console.print("=" * 60 + "\n")
        return self
