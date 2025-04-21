"""Pattern registry for the Haive framework.

This module provides a registry for storing and retrieving graph patterns
and branch definitions, enabling reuse of common workflow structures.
"""

import importlib
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

from haive.core.graph.patterns.base import (
    BranchDefinition,
    GraphPattern,
    ParameterDefinition,
    PatternMetadata,
)

logger = logging.getLogger(__name__)


class GraphPatternRegistry:
    """Registry for graph patterns and branch definitions.
    
    This registry serves as a central repository for reusable workflow patterns
    and branch conditions that can be applied to graphs.
    """
    _instance = None

    @classmethod
    def get_instance(cls) -> "GraphPatternRegistry":
        """Get the singleton instance of the registry.
        
        Returns:
            The registry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the registry with empty collections."""
        self.patterns: dict[str, GraphPattern] = {}
        self.branches: dict[str, BranchDefinition] = {}
        self.pattern_categories: dict[str, list[str]] = {}
        self.branch_categories: dict[str, list[str]] = {}

    def register_pattern(
        self,
        pattern: GraphPattern | dict[str, Any] | Callable,
        name: str | None = None,
        pattern_type: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        **pattern_params
    ) -> GraphPattern:
        """Register a pattern directly without requiring decorator usage.
        
        Args:
            pattern: Pattern object, dict, or function to register
            name: Optional name override
            pattern_type: Optional type categorization
            description: Optional description
            metadata: Additional metadata
            **pattern_params: Default parameters for the pattern
            
        Returns:
            Registered GraphPattern
        """
        # Handle different input types
        if isinstance(pattern, GraphPattern):
            # Use directly with optional overrides
            pattern_obj = pattern

            # Override metadata if provided
            if name or pattern_type or description or metadata:
                # Create updated metadata
                metadata_dict = pattern_obj.metadata.model_dump() if hasattr(pattern_obj.metadata, "model_dump") else pattern_obj.metadata.dict()

                if name:
                    metadata_dict["name"] = name
                if pattern_type:
                    metadata_dict["pattern_type"] = pattern_type
                if description:
                    metadata_dict["description"] = description
                if metadata:
                    for key, value in metadata.items():
                        if key in metadata_dict:
                            metadata_dict[key] = value

                # Update pattern with new metadata
                pattern_obj.metadata = PatternMetadata(**metadata_dict)

        elif isinstance(pattern, dict):
            # Create from dictionary
            if "metadata" not in pattern:
                # Create metadata from parameters
                metadata_dict = {
                    "name": name or pattern.get("name", f"pattern_{len(self.patterns)}"),
                    "description": description or pattern.get("description", ""),
                    "pattern_type": pattern_type or pattern.get("pattern_type", "undefined"),
                    "parameters": pattern.get("parameters", {})
                }

                # Add additional metadata
                if metadata:
                    for key, value in metadata.items():
                        metadata_dict[key] = value

                pattern["metadata"] = PatternMetadata(**metadata_dict)

            # Create pattern object
            pattern_obj = GraphPattern(**pattern)

        elif callable(pattern):
            # Create from function
            if not name:
                name = getattr(pattern, "__name__", f"pattern_{len(self.patterns)}")

            if not description:
                description = getattr(pattern, "__doc__", "")

            if not pattern_type:
                pattern_type = "undefined"

            # Create metadata
            metadata_dict = {
                "name": name,
                "description": description,
                "pattern_type": pattern_type,
                "parameters": {}
            }

            # Extract parameter info from function signature
            sig = inspect.signature(pattern)
            for param_name, param in sig.parameters.items():
                if param_name == "graph" or param_name == "self":
                    continue

                param_type = "Any"
                if param.annotation != inspect.Parameter.empty:
                    param_type = str(param.annotation)

                default = None
                required = param.default == inspect.Parameter.empty
                if not required:
                    default = param.default

                # Add parameter definition
                metadata_dict["parameters"][param_name] = ParameterDefinition(
                    type=param_type,
                    default=default,
                    description=f"Parameter {param_name}",
                    required=required
                ).model_dump() if hasattr(ParameterDefinition, "model_dump") else ParameterDefinition(
                    type=param_type,
                    default=default,
                    description=f"Parameter {param_name}",
                    required=required
                ).dict()

            # Add additional metadata
            if metadata:
                for key, value in metadata.items():
                    metadata_dict[key] = value

            # Create pattern object
            pattern_obj = GraphPattern(
                metadata=PatternMetadata(**metadata_dict),
                apply_func=pattern
            )

        else:
            raise ValueError(f"Unsupported pattern type: {type(pattern)}")

        # Store in registry
        pattern_name = pattern_obj.metadata.name
        if pattern_name in self.patterns:
            logger.warning(f"Overwriting existing pattern: {pattern_name}")

        self.patterns[pattern_name] = pattern_obj

        # Track category
        category = pattern_obj.metadata.pattern_type
        if category not in self.pattern_categories:
            self.pattern_categories[category] = []
        if pattern_name not in self.pattern_categories[category]:
            self.pattern_categories[category].append(pattern_name)

        logger.info(f"Registered pattern: {pattern_name} (type: {category})")

        return pattern_obj

    def register_branch(
        self,
        branch: BranchDefinition | dict[str, Any] | Callable,
        name: str | None = None,
        condition_type: str | None = None,
        description: str | None = None,
        routes: dict[str, str] | None = None,
        default_route: str | None = None,
        **branch_params
    ) -> BranchDefinition:
        """Register a branch definition directly.
        
        Args:
            branch: Branch object, dict, or function to register
            name: Optional name override
            condition_type: Optional condition type categorization
            description: Optional description
            routes: Optional route mapping
            default_route: Optional default route
            **branch_params: Additional branch parameters
            
        Returns:
            Registered BranchDefinition
        """
        # Handle different input types
        if isinstance(branch, BranchDefinition):
            # Use directly with optional overrides
            branch_obj = branch

            # Override fields if provided
            if name:
                branch_obj.name = name
            if condition_type:
                branch_obj.condition_type = condition_type
            if description:
                branch_obj.description = description
            if routes:
                branch_obj.routes = routes
            if default_route:
                branch_obj.default_route = default_route

        elif isinstance(branch, dict):
            # Create from dictionary
            branch_dict = branch.copy()

            # Add overrides
            if name:
                branch_dict["name"] = name
            if condition_type:
                branch_dict["condition_type"] = condition_type
            if description:
                branch_dict["description"] = description
            if routes:
                branch_dict["routes"] = routes
            if default_route:
                branch_dict["default_route"] = default_route

            # Create branch object
            branch_obj = BranchDefinition(**branch_dict)

        elif callable(branch):
            # Create from function
            if not name:
                name = getattr(branch, "__name__", f"branch_{len(self.branches)}")

            if not description:
                description = getattr(branch, "__doc__", "")

            if not condition_type:
                condition_type = "undefined"

            if not routes:
                # Extract possible return values from docstring or function code
                routes = {}

            # Create branch definition
            branch_obj = BranchDefinition(
                name=name,
                description=description,
                condition_type=condition_type,
                routes=routes or {},
                default_route=default_route or "END",
                condition_func=branch
            )

            # Add parameters from function signature
            sig = inspect.signature(branch)
            parameters = {}

            for param_name, param in sig.parameters.items():
                if param_name == "state" or param_name == "self":
                    continue

                param_type = "Any"
                if param.annotation != inspect.Parameter.empty:
                    param_type = str(param.annotation)

                default = None
                required = param.default == inspect.Parameter.empty
                if not required:
                    default = param.default

                # Add parameter definition
                parameters[param_name] = ParameterDefinition(
                    type=param_type,
                    default=default,
                    description=f"Parameter {param_name}",
                    required=required
                )

            branch_obj.parameters = parameters

        else:
            raise ValueError(f"Unsupported branch type: {type(branch)}")

        # Store in registry
        branch_name = branch_obj.name
        if branch_name in self.branches:
            logger.warning(f"Overwriting existing branch: {branch_name}")

        self.branches[branch_name] = branch_obj

        # Track category
        category = branch_obj.condition_type
        if category not in self.branch_categories:
            self.branch_categories[category] = []
        if branch_name not in self.branch_categories[category]:
            self.branch_categories[category].append(branch_name)

        logger.info(f"Registered branch: {branch_name} (type: {category})")

        return branch_obj

    def get_pattern(self, name: str) -> GraphPattern | None:
        """Get a pattern by name.
        
        Args:
            name: Name of the pattern to retrieve
            
        Returns:
            Pattern if found, None otherwise
        """
        return self.patterns.get(name)

    def get_branch(self, name: str) -> BranchDefinition | None:
        """Get a branch by name.
        
        Args:
            name: Name of the branch to retrieve
            
        Returns:
            Branch if found, None otherwise
        """
        return self.branches.get(name)

    def list_patterns(self, pattern_type: str | None = None) -> list[str]:
        """List all pattern names, optionally filtered by type.
        
        Args:
            pattern_type: Optional type to filter by
            
        Returns:
            List of pattern names
        """
        if pattern_type:
            return self.pattern_categories.get(pattern_type, [])
        return list(self.patterns.keys())

    def list_branches(self, condition_type: str | None = None) -> list[str]:
        """List all branch names, optionally filtered by type.
        
        Args:
            condition_type: Optional type to filter by
            
        Returns:
            List of branch names
        """
        if condition_type:
            return self.branch_categories.get(condition_type, [])
        return list(self.branches.keys())

    def list_pattern_types(self) -> list[str]:
        """List all pattern types/categories.
        
        Returns:
            List of pattern types
        """
        return list(self.pattern_categories.keys())

    def list_branch_types(self) -> list[str]:
        """List all branch condition types/categories.
        
        Returns:
            List of branch condition types
        """
        return list(self.branch_categories.keys())

    def find_patterns(
        self,
        pattern_type: str | None = None,
        tags: list[str] | None = None,
        compatible_with: list[Any] | None = None,
        search_term: str | None = None
    ) -> list[GraphPattern]:
        """Find patterns matching specified criteria.
        
        Args:
            pattern_type: Filter by pattern type
            tags: Filter by tags (must have all specified tags)
            compatible_with: Filter by component compatibility
            search_term: Search in name/description
            
        Returns:
            List of matching patterns
        """
        results = []

        for pattern in self.patterns.values():
            # Filter by type
            if pattern_type and pattern.metadata.pattern_type != pattern_type:
                continue

            # Filter by tags
            if tags:
                if not all(tag in pattern.metadata.tags for tag in tags):
                    continue

            # Filter by search term
            if search_term:
                search_lower = search_term.lower()
                name_match = search_lower in pattern.metadata.name.lower()
                desc_match = search_lower in pattern.metadata.description.lower()
                if not (name_match or desc_match):
                    continue

            # Filter by component compatibility
            if compatible_with:
                missing = pattern.metadata.check_required_components(compatible_with)
                if missing:
                    continue

            # All filters passed
            results.append(pattern)

        return results

    def find_branches(
        self,
        condition_type: str | None = None,
        tags: list[str] | None = None,
        search_term: str | None = None
    ) -> list[BranchDefinition]:
        """Find branches matching specified criteria.
        
        Args:
            condition_type: Filter by condition type
            tags: Filter by tags (must have all specified tags)
            search_term: Search in name/description
            
        Returns:
            List of matching branches
        """
        results = []

        for branch in self.branches.values():
            # Filter by type
            if condition_type and branch.condition_type != condition_type:
                continue

            # Filter by tags
            if tags:
                if not all(tag in branch.tags for tag in tags):
                    continue

            # Filter by search term
            if search_term:
                search_lower = search_term.lower()
                name_match = search_lower in branch.name.lower()
                desc_match = search_lower in branch.description.lower()
                if not (name_match or desc_match):
                    continue

            # All filters passed
            results.append(branch)

        return results

    def get_pattern_documentation(self, pattern_name: str) -> dict[str, Any]:
        """Get structured documentation for a pattern.
        
        Args:
            pattern_name: Pattern to document
            
        Returns:
            Documentation dictionary
        """
        pattern = self.get_pattern(pattern_name)
        if not pattern:
            raise ValueError(f"Pattern not found: {pattern_name}")

        # Create documentation dictionary
        doc = {
            "name": pattern.metadata.name,
            "description": pattern.metadata.description,
            "version": pattern.metadata.version,
            "type": pattern.metadata.pattern_type,
            "tags": pattern.metadata.tags,
            "required_components": [],
            "parameters": {},
            "examples": pattern.metadata.examples
        }

        # Add component requirements
        for req in pattern.metadata.required_components:
            req_dict = {}
            req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
            doc["required_components"].append(req_dict)

        # Add parameters
        for name, param in pattern.metadata.parameters.items():
            param_dict = {}
            param_dict = param.model_dump() if hasattr(param, "model_dump") else param.dict()
            doc["parameters"][name] = param_dict

        return doc

    def get_branch_documentation(self, branch_name: str) -> dict[str, Any]:
        """Get structured documentation for a branch.
        
        Args:
            branch_name: Branch to document
            
        Returns:
            Documentation dictionary
        """
        branch = self.get_branch(branch_name)
        if not branch:
            raise ValueError(f"Branch not found: {branch_name}")

        # Create documentation dictionary
        doc = {
            "name": branch.name,
            "description": branch.description,
            "type": branch.condition_type,
            "version": branch.version,
            "tags": branch.tags,
            "routes": branch.routes,
            "default_route": branch.default_route,
            "parameters": {}
        }

        # Add parameters
        for name, param in branch.parameters.items():
            param_dict = {}
            param_dict = param.model_dump() if hasattr(param, "model_dump") else param.dict()
            doc["parameters"][name] = param_dict

        return doc

    def clear(self) -> None:
        """Clear all patterns and branches from the registry."""
        self.patterns = {}
        self.branches = {}
        self.pattern_categories = {}
        self.branch_categories = {}
        logger.info("Cleared pattern registry")

    def save_to_file(self, filename: str) -> None:
        """Save the registry contents to a file.
        
        Args:
            filename: Path to save the registry
        """
        data = {
            "patterns": {},
            "branches": {}
        }

        # Serialize patterns
        for name, pattern in self.patterns.items():
            data["patterns"][name] = pattern.to_dict()

        # Serialize branches
        for name, branch in self.branches.items():
            data["branches"][name] = branch.to_dict()

        # Save to file
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved pattern registry to {filename}")

    def load_from_file(self, filename: str, func_resolver: Callable | None = None) -> None:
        """Load registry contents from a file.
        
        Args:
            filename: Path to load the registry from
            func_resolver: Optional function to resolve function references
        """
        # Load from file
        with open(filename) as f:
            data = json.load(f)

        # Clear existing registry
        self.clear()

        # Load patterns
        for name, pattern_data in data.get("patterns", {}).items():
            try:
                pattern = GraphPattern.from_dict(pattern_data, func_resolver)
                self.patterns[name] = pattern

                # Track category
                category = pattern.metadata.pattern_type
                if category not in self.pattern_categories:
                    self.pattern_categories[category] = []
                self.pattern_categories[category].append(name)

            except Exception as e:
                logger.error(f"Error loading pattern {name}: {e}")

        # Load branches
        for name, branch_data in data.get("branches", {}).items():
            try:
                branch = BranchDefinition.from_dict(branch_data, func_resolver)
                self.branches[name] = branch

                # Track category
                category = branch.condition_type
                if category not in self.branch_categories:
                    self.branch_categories[category] = []
                self.branch_categories[category].append(name)

            except Exception as e:
                logger.error(f"Error loading branch {name}: {e}")

        logger.info(f"Loaded pattern registry from {filename}")

    def import_patterns_from_module(self, module_path: str) -> int:
        """Import patterns from a module.
        
        Args:
            module_path: Python module path to import from
            
        Returns:
            Number of patterns imported
        """
        try:
            module = importlib.import_module(module_path)
            count = 0

            # Look for pattern objects
            for name in dir(module):
                obj = getattr(module, name)

                if isinstance(obj, GraphPattern):
                    self.register_pattern(obj)
                    count += 1

                elif isinstance(obj, BranchDefinition):
                    self.register_branch(obj)
                    count += 1

            logger.info(f"Imported {count} patterns/branches from {module_path}")
            return count

        except ImportError as e:
            logger.error(f"Error importing module {module_path}: {e}")
            return 0


def register_pattern(
    name: str | None = None,
    pattern_type: str | None = None,
    description: str | None = None,
    **pattern_params
):
    """Decorator to register a function as a pattern.
    
    Args:
        name: Name for the pattern (defaults to function name)
        pattern_type: Type/category of pattern
        description: Description of the pattern (defaults to function docstring)
        **pattern_params: Additional pattern parameters
        
    Returns:
        Decorator function
    """
    def decorator(func):
        registry = GraphPatternRegistry.get_instance()
        registry.register_pattern(
            func,
            name=name,
            pattern_type=pattern_type,
            description=description,
            **pattern_params
        )
        return func
    return decorator


def register_branch(
    name: str | None = None,
    condition_type: str | None = None,
    routes: dict[str, str] | None = None,
    default_route: str | None = None,
    **branch_params
):
    """Decorator to register a function as a branch condition.
    
    Args:
        name: Name for the branch (defaults to function name)
        condition_type: Type/category of branch condition
        routes: Mapping of condition outputs to node names
        default_route: Default route if no condition matches
        **branch_params: Additional branch parameters
        
    Returns:
        Decorator function
    """
    def decorator(func):
        registry = GraphPatternRegistry.get_instance()
        registry.register_branch(
            func,
            name=name,
            condition_type=condition_type,
            routes=routes,
            default_route=default_route,
            **branch_params
        )
        return func
    return decorator
