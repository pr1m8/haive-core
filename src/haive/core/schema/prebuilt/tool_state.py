import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import Field, computed_field, model_validator

from haive.core.schema.prebuilt.messages_state import MessagesState

logger = logging.getLogger(__name__)


class ToolState(MessagesState):
    """
    State schema for tool-based agents with proper Pydantic v2 patterns.

    Inherits from MessagesState for message handling and adds tool-specific functionality.
    Follows the same tool routes pattern as AugLLMConfig.
    """

    # Tool-related fields - matching AugLLMConfig pattern
    tools: List[Any] = Field(default_factory=list, description="Available tools")
    tool_routes: Dict[str, str] = Field(
        default_factory=dict, description="Mapping of tool names to their routes/types"
    )
    name_attrs: List[str] = Field(
        default_factory=lambda: ["name", "__name__", "func_name"],
        description="Attributes to check for tool names",
    )
    content: Optional[str] = Field(default=None, description="Content field")
    output_schemas: Dict[str, Any] = Field(
        default_factory=dict, description="Output schemas for tools"
    )

    @model_validator(mode="after")
    def sync_tools_and_update_routes(self) -> "ToolState":
        """
        Sync tools from engines and update tool routes after model creation.

        This runs after the parent validator, so engines are already set up.
        """
        # Call parent validator first
        super().setup_engines_and_tools()

        # Now sync tool routes based on the current tools
        self._sync_tool_routes()

        return self

    def _sync_tool_routes(self) -> None:
        """Synchronize tool_routes with current tools - matches AugLLMConfig pattern."""
        new_routes = {}

        for i, tool in enumerate(self.tools):
            # Determine tool name
            tool_name = self._get_tool_name(tool, i)

            # Determine route/type - matching AugLLMConfig categorization
            route = self._get_tool_route(tool)

            new_routes[tool_name] = route
            logger.debug(f"Mapped tool '{tool_name}' to route '{route}'")

        self.tool_routes = new_routes

    def _get_tool_name(self, tool: Any, index: int) -> str:
        """Extract tool name from various possible attributes."""
        if hasattr(tool, "name"):
            return tool.name
        elif isinstance(tool, type) and hasattr(tool, "__name__"):
            return tool.__name__
        elif hasattr(tool, "__name__"):
            return tool.__name__
        else:
            return f"tool_{index}"

    def _get_tool_route(self, tool: Any) -> str:
        """Determine the route type for a tool."""
        if isinstance(tool, type) and self._is_basemodel_subclass(tool):
            return "pydantic_model"
        elif isinstance(tool, BaseTool) or (
            isinstance(tool, type) and issubclass(tool, BaseTool)
        ):
            return "langchain_tool"
        elif callable(tool):
            return "function"
        else:
            return "unknown"

    def _is_basemodel_subclass(self, tool: Any) -> bool:
        """Check if tool is a BaseModel subclass."""
        try:
            from pydantic import BaseModel

            return isinstance(tool, type) and issubclass(tool, BaseModel)
        except Exception:
            return False

    @computed_field
    @property
    def tool_types(self) -> Dict[str, str]:
        """
        Computed field for backward compatibility.

        Maps the new tool_routes to the old tool_types interface:
        - pydantic_model → parse_output
        - langchain_tool → tool_node
        - function → tool_node
        - unknown → tool_node
        """
        legacy_mapping = {
            "pydantic_model": "parse_output",
            "langchain_tool": "tool_node",
            "function": "tool_node",
            "unknown": "tool_node",
        }

        return {
            tool_name: legacy_mapping.get(route, "tool_node")
            for tool_name, route in self.tool_routes.items()
        }

    def add_tool(self, tool: Any, route: Optional[str] = None) -> None:
        """
        Add a tool and update tool routes - matches AugLLMConfig pattern.

        Args:
            tool: Tool to add
            route: Optional explicit route/type (pydantic_model, langchain_tool, function, unknown)
        """
        if tool not in self.tools:
            self.tools.append(tool)

            # Determine tool name
            tool_name = self._get_tool_name(tool, len(self.tools))

            # Set route - use explicit route if provided, otherwise auto-determine
            if route:
                self.tool_routes[tool_name] = route
            else:
                self.tool_routes[tool_name] = self._get_tool_route(tool)

            logger.debug(
                f"Added tool '{tool_name}' with route '{self.tool_routes[tool_name]}'"
            )

    def remove_tool(self, tool: Any) -> None:
        """
        Remove a tool and update tool routes.

        Args:
            tool: Tool to remove
        """
        if tool in self.tools:
            self.tools.remove(tool)

            # Find and remove from tool_routes
            tool_name = None
            for i, t in enumerate(self.tools):
                if t == tool:
                    tool_name = self._get_tool_name(t, i)
                    break

            if tool_name and tool_name in self.tool_routes:
                del self.tool_routes[tool_name]
                logger.debug(f"Removed tool '{tool_name}'")

            # Re-sync to ensure consistency
            self._sync_tool_routes()

    def get_tool_by_name(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by its name.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool if found, None otherwise
        """
        for i, tool in enumerate(self.tools):
            if self._get_tool_name(tool, i) == tool_name:
                return tool
        return None

    def get_tools_by_route(self, route: str) -> List[Any]:
        """
        Get all tools of a specific route type.

        Args:
            route: Route type to retrieve (pydantic_model, langchain_tool, function, unknown)

        Returns:
            List of tools of the specified route type
        """
        result = []
        for tool_name, tool_route in self.tool_routes.items():
            if tool_route == route:
                tool = self.get_tool_by_name(tool_name)
                if tool:
                    result.append(tool)
        return result

    def get_tools_by_type(self, tool_type: str) -> List[Any]:
        """
        Get all tools of a specific type (legacy interface).

        Args:
            tool_type: Type of tools to retrieve (parse_output, tool_node)

        Returns:
            List of tools of the specified type
        """
        # Convert legacy type to routes
        type_to_routes = {
            "parse_output": ["pydantic_model"],
            "tool_node": ["langchain_tool", "function", "unknown"],
        }

        routes = type_to_routes.get(tool_type, [])
        result = []
        for route in routes:
            result.extend(self.get_tools_by_route(route))
        return result

    def has_tool_route(self, route: str) -> bool:
        """
        Check if any tools of a specific route exist.

        Args:
            route: Route to check for (pydantic_model, langchain_tool, function, unknown)

        Returns:
            True if any tools of this route exist
        """
        return route in self.tool_routes.values()

    def has_tool_type(self, tool_type: str) -> bool:
        """
        Check if any tools of a specific type exist (legacy interface).

        Args:
            tool_type: Type to check for (parse_output, tool_node)

        Returns:
            True if any tools of this type exist
        """
        return tool_type in self.tool_types.values()

    def refresh_tool_routes(self) -> None:
        """
        Manually refresh tool routes if needed.
        """
        self._sync_tool_routes()

    def update_tool_types(self) -> None:
        """
        Legacy interface - now calls refresh_tool_routes.
        """
        self.refresh_tool_routes()

    def get_tool_route(self, tool_name: str) -> Optional[str]:
        """
        Get the route of a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool route if found (pydantic_model, langchain_tool, function, unknown), None otherwise
        """
        return self.tool_routes.get(tool_name)

    def get_tool_type(self, tool_name: str) -> Optional[str]:
        """
        Get the type of a specific tool (legacy interface).

        Args:
            tool_name: Name of the tool

        Returns:
            Tool type if found (parse_output, tool_node), None otherwise
        """
        return self.tool_types.get(tool_name)
