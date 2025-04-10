# src/haive/core/registry/AbstractRegistry.py

import logging
from typing import Dict, List, Any, Optional, Type, TypeVar, Generic, Callable, Set, Union, Tuple
import inspect
import functools

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generics
T = TypeVar('T')
K = TypeVar('K')

class AbstractRegistry(Generic[K, T]):
    """
    Abstract registry pattern for managing components within the framework.
    
    This registry provides a central location for registering, retrieving, and managing
    components like tools, schemas, node functions, and graph patterns.
    
    Generic type parameters:
    - K: Key type for registry entries (usually str or Type)
    - T: Value type for registry entries (components being registered)
    """
    
    def __init__(self, name: str, description: Optional[str] = None):
        """
        Initialize a new registry.
        
        Args:
            name: Name of the registry
            description: Optional description of the registry purpose
        """
        self.name = name
        self.description = description or f"Registry for {name}"
        self._registry: Dict[K, T] = {}
        self._metadata: Dict[K, Dict[str, Any]] = {}
        self._tag_index: Dict[str, Set[K]] = {}
        self._dependencies: Dict[K, Set[K]] = {}
        self._listeners: List[Callable[[str, K, T], None]] = []
    
    def register(self, key: K, item: T, tags: Optional[List[str]] = None, 
                metadata: Optional[Dict[str, Any]] = None, 
                dependencies: Optional[List[K]] = None) -> T:
        """
        Register an item in the registry.
        
        Args:
            key: Unique key for the item
            item: The item to register
            tags: Optional list of tags for categorization
            metadata: Optional metadata about the item
            dependencies: Optional list of keys this item depends on
            
        Returns:
            The registered item (for decorator pattern)
        """
        if key in self._registry:
            logger.warning(f"Overwriting existing entry in {self.name} registry: {key}")
        
        # Store the item
        self._registry[key] = item
        
        # Store metadata
        self._metadata[key] = metadata or {}
        
        # Add to tag index
        if tags:
            for tag in tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(key)
        
        # Track dependencies
        if dependencies:
            self._dependencies[key] = set(dependencies)
            # Verify dependencies exist
            missing = [d for d in dependencies if d not in self._registry]
            if missing:
                logger.warning(f"Registering {key} with missing dependencies: {missing}")
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener("register", key, item)
            except Exception as e:
                logger.error(f"Error in registry listener: {e}")
        
        logger.info(f"Registered {key} in {self.name} registry")
        return item
    
    def get(self, key: K, default: Optional[T] = None) -> Optional[T]:
        """
        Get an item from the registry.
        
        Args:
            key: Key to look up
            default: Default value if key not found
            
        Returns:
            The registered item or default if not found
        """
        return self._registry.get(key, default)
    
    def get_metadata(self, key: K) -> Dict[str, Any]:
        """
        Get metadata for a registry item.
        
        Args:
            key: Key to look up
            
        Returns:
            Metadata dictionary for the item
        """
        return self._metadata.get(key, {})
    
    def get_all(self) -> Dict[K, T]:
        """
        Get all items in the registry.
        
        Returns:
            Dictionary of all registered items
        """
        return dict(self._registry)
    
    def get_by_tag(self, tag: str) -> Dict[K, T]:
        """
        Get all items with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            Dictionary of items with the specified tag
        """
        if tag not in self._tag_index:
            return {}
        
        return {k: self._registry[k] for k in self._tag_index[tag] if k in self._registry}
    
    def get_by_tags(self, tags: List[str], require_all: bool = False) -> Dict[K, T]:
        """
        Get items matching multiple tags.
        
        Args:
            tags: List of tags to filter by
            require_all: If True, items must have all tags; if False, any tag matches
            
        Returns:
            Dictionary of matching items
        """
        if not tags:
            return {}
        
        # Find keys matching the tag criteria
        if require_all:
            # Must match all tags
            matching_keys = None
            for tag in tags:
                tag_keys = self._tag_index.get(tag, set())
                if matching_keys is None:
                    matching_keys = tag_keys
                else:
                    matching_keys &= tag_keys  # Intersection
        else:
            # Match any tag
            matching_keys = set()
            for tag in tags:
                matching_keys |= self._tag_index.get(tag, set())  # Union
        
        # Return matching items
        if not matching_keys:
            return {}
        
        return {k: self._registry[k] for k in matching_keys if k in self._registry}
    
    def get_dependencies(self, key: K) -> Set[K]:
        """
        Get dependencies for a specific item.
        
        Args:
            key: Key to get dependencies for
            
        Returns:
            Set of keys this item depends on
        """
        return self._dependencies.get(key, set())
    
    def get_dependents(self, key: K) -> Set[K]:
        """
        Get items that depend on a specific item.
        
        Args:
            key: Key to find dependents for
            
        Returns:
            Set of keys that depend on this item
        """
        return {k for k, deps in self._dependencies.items() if key in deps}
    
    def has(self, key: K) -> bool:
        """
        Check if an item exists in the registry.
        
        Args:
            key: Key to check
            
        Returns:
            True if the key exists, False otherwise
        """
        return key in self._registry
    
    def remove(self, key: K) -> Optional[T]:
        """
        Remove an item from the registry.
        
        Args:
            key: Key to remove
            
        Returns:
            The removed item or None if not found
        """
        if key not in self._registry:
            return None
        
        # Get the item before removal
        item = self._registry[key]
        
        # Check for dependents
        dependents = self.get_dependents(key)
        if dependents:
            logger.warning(f"Removing {key} which has dependents: {dependents}")
        
        # Remove from registry
        del self._registry[key]
        
        # Remove metadata
        if key in self._metadata:
            del self._metadata[key]
        
        # Remove from tag index
        for tag, keys in list(self._tag_index.items()):
            if key in keys:
                keys.remove(key)
                # Clean up empty tag sets
                if not keys:
                    del self._tag_index[tag]
        
        # Remove from dependencies
        if key in self._dependencies:
            del self._dependencies[key]
        
        # Remove from other items' dependencies
        for k, deps in self._dependencies.items():
            if key in deps:
                deps.remove(key)
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener("remove", key, item)
            except Exception as e:
                logger.error(f"Error in registry listener: {e}")
        
        logger.info(f"Removed {key} from {self.name} registry")
        return item
    
    def update(self, key: K, item: T, update_metadata: Optional[Dict[str, Any]] = None,
              add_tags: Optional[List[str]] = None, remove_tags: Optional[List[str]] = None) -> Optional[T]:
        """
        Update an existing registry item.
        
        Args:
            key: Key to update
            item: New item value
            update_metadata: Metadata to update (will be merged with existing)
            add_tags: Tags to add
            remove_tags: Tags to remove
            
        Returns:
            The updated item or None if key not found
        """
        if key not in self._registry:
            logger.warning(f"Cannot update non-existent key: {key}")
            return None
        
        # Update the item
        old_item = self._registry[key]
        self._registry[key] = item
        
        # Update metadata if provided
        if update_metadata:
            if key not in self._metadata:
                self._metadata[key] = {}
            self._metadata[key].update(update_metadata)
        
        # Add new tags
        if add_tags:
            for tag in add_tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(key)
        
        # Remove tags
        if remove_tags:
            for tag in remove_tags:
                if tag in self._tag_index and key in self._tag_index[tag]:
                    self._tag_index[tag].remove(key)
                    # Clean up empty tag sets
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener("update", key, item)
            except Exception as e:
                logger.error(f"Error in registry listener: {e}")
        
        logger.info(f"Updated {key} in {self.name} registry")
        return item
    
    def clear(self) -> None:
        """Clear all items from the registry."""
        keys = list(self._registry.keys())
        for key in keys:
            self.remove(key)
        
        # Ensure everything is cleared
        self._registry.clear()
        self._metadata.clear()
        self._tag_index.clear()
        self._dependencies.clear()
        
        logger.info(f"Cleared all entries from {self.name} registry")
    
    def add_listener(self, listener: Callable[[str, K, T], None]) -> None:
        """
        Add a listener for registry events.
        
        Args:
            listener: Function that takes (event_type, key, item)
        """
        self._listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[str, K, T], None]) -> None:
        """
        Remove a registry event listener.
        
        Args:
            listener: The listener to remove
        """
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    # Decorator methods for easier registration
    def register_decorator(self, key: Optional[K] = None, tags: Optional[List[str]] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         dependencies: Optional[List[K]] = None) -> Callable[[T], T]:
        """
        Create a decorator for registering items.
        
        Args:
            key: Key for the item (defaults to item name if None)
            tags: Optional tags for the item
            metadata: Optional metadata
            dependencies: Optional dependencies
            
        Returns:
            Decorator function
        """
        def decorator(item: T) -> T:
            # Determine key if not provided
            nonlocal key
            actual_key = key
            
            if actual_key is None:
                # Try to get a name from the item
                if hasattr(item, "__name__"):
                    actual_key = item.__name__  # type: ignore
                elif hasattr(item, "name"):
                    actual_key = item.name  # type: ignore
                else:
                    raise ValueError("Key must be provided if item doesn't have a name")
            
            # Register the item
            return self.register(actual_key, item, tags, metadata, dependencies)
        
        return decorator
    
    # Inspection and introspection
    def describe(self, key: K) -> Dict[str, Any]:
        """
        Get full description of a registry item.
        
        Args:
            key: Key to describe
            
        Returns:
            Dictionary with item description
        """
        if not self.has(key):
            return {"error": f"Key {key} not found"}
        
        item = self.get(key)
        metadata = self.get_metadata(key)
        dependencies = list(self.get_dependencies(key))
        dependents = list(self.get_dependents(key))
        
        # Get tags for this item
        tags = [tag for tag, keys in self._tag_index.items() if key in keys]
        
        # Try to get additional info from the item itself
        item_info = {}
        if hasattr(item, "__doc__") and item.__doc__:
            item_info["doc"] = inspect.cleandoc(item.__doc__)
        
        if inspect.isfunction(item) or inspect.ismethod(item):
            # For functions, get signature
            try:
                sig = inspect.signature(item)
                item_info["signature"] = str(sig)
                item_info["parameters"] = {
                    name: {
                        "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                        "default": param.default if param.default != inspect.Parameter.empty else None,
                        "kind": str(param.kind)
                    }
                    for name, param in sig.parameters.items()
                }
            except Exception:
                pass
        
        # Build complete description
        return {
            "key": key,
            "type": type(item).__name__,
            "metadata": metadata,
            "tags": tags,
            "dependencies": dependencies,
            "dependents": dependents,
            "item_info": item_info
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            "name": self.name,
            "description": self.description,
            "item_count": len(self._registry),
            "tag_count": len(self._tag_index),
            "tags": list(self._tag_index.keys()),
            "items_with_dependencies": len(self._dependencies),
            "total_dependency_relationships": sum(len(deps) for deps in self._dependencies.values())
        }

# Specialized Registry Types
class ToolRegistry(AbstractRegistry[str, Any]):
    """Registry for tools and tool configurations."""
    
    def __init__(self):
        super().__init__(name="tools", description="Registry for agent tools")

class SchemaRegistry(AbstractRegistry[str, Any]):
    """Registry for schemas and schema patterns."""
    
    def __init__(self):
        super().__init__(name="schemas", description="Registry for state schemas")

class NodeRegistry(AbstractRegistry[str, Callable]):
    """Registry for node functions and patterns."""
    
    def __init__(self):
        super().__init__(name="nodes", description="Registry for graph nodes")

class GraphRegistry(AbstractRegistry[str, Any]):
    """Registry for graph patterns and templates."""
    
    def __init__(self):
        super().__init__(name="graphs", description="Registry for graph patterns")

# Global registry instances
tool_registry = ToolRegistry()
schema_registry = SchemaRegistry()
node_registry = NodeRegistry()
graph_registry = GraphRegistry()

# Registry manager for centralized access
class RegistryManager:
    """Central manager for all registries."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RegistryManager, cls).__new__(cls)
            cls._instance._registries = {
                "tools": tool_registry,
                "schemas": schema_registry,
                "nodes": node_registry,
                "graphs": graph_registry,
                # Additional registries can be added here
            }
        return cls._instance
    
    def get_registry(self, registry_name: str) -> Optional[AbstractRegistry]:
        """Get a registry by name."""
        return self._registries.get(registry_name)
    
    def add_registry(self, name: str, registry: AbstractRegistry) -> None:
        """Add a new registry to the manager."""
        if name in self._registries:
            logger.warning(f"Overwriting existing registry: {name}")
        self._registries[name] = registry
    
    def list_registries(self) -> List[str]:
        """List all available registries."""
        return list(self._registries.keys())

# Create global instance
registry_manager = RegistryManager()

# Helper decorators for common registrations
def register_tool(name: str, tags: Optional[List[str]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
    """Decorator to register a tool."""
    return tool_registry.register_decorator(name, tags, metadata)

def register_schema(name: str, tags: Optional[List[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None):
    """Decorator to register a schema."""
    return schema_registry.register_decorator(name, tags, metadata)

def register_node(name: str, tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None):
    """Decorator to register a node function."""
    return node_registry.register_decorator(name, tags, metadata)

def register_graph(name: str, tags: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
    """Decorator to register a graph pattern."""
    return graph_registry.register_decorator(name, tags, metadata)