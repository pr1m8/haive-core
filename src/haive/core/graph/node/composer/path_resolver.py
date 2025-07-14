"""Path resolver for extracting values from objects using path notation.

This module provides path-based value extraction from objects, supporting
simple field access initially, with progressive enhancement for complex paths.
"""

from typing import Any, Optional


class PathResolver:
    """Resolver for extracting values from objects using path notation.

    This class handles path-based extraction from objects, starting with
    simple field access and progressively supporting more complex patterns.

    Phase 1: Simple field access (e.g., "messages", "temperature")
    Phase 2: Dot notation (e.g., "config.temperature")
    Phase 3: Array access (e.g., "messages[0]", "messages[-1]")
    Phase 4: Advanced patterns (wildcards, method calls, etc.)
    """

    def extract_value(self, obj: Any, path: str, default: Any = None) -> Any:
        """Extract value from object using path notation.

        Args:
            obj: Object to extract from (dict, Pydantic model, etc.)
            path: Path string (e.g., "messages", "config.temp")
            default: Default value if path not found or None

        Returns:
            Extracted value or default

        Examples:
            # Simple field access
            resolver = PathResolver()
            state = {"messages": ["hello"], "temp": 0.7}

            value = resolver.extract_value(state, "messages")
            # Returns: ["hello"]

            value = resolver.extract_value(state, "missing", default=[])
            # Returns: []
        """
        if not obj or not path:
            return default

        try:
            # Phase 1: Simple field access only
            # Handle both dict and object attribute access

            if hasattr(obj, path):
                # Object attribute access (Pydantic models, dataclasses)
                return getattr(obj, path, default)
            elif hasattr(obj, "__getitem__"):
                # Dict-like access
                return obj.get(path, default) if hasattr(obj, "get") else obj[path]
            else:
                return default

        except (KeyError, AttributeError, TypeError):
            return default
