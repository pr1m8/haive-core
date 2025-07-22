"""Dynamic Registry System for Component Activation.

This module provides a generic registry system for managing activatable components
(tools, agents, services) with full type safety and activation tracking.

Based on the Dynamic Activation Pattern:
@project_docs/active/patterns/dynamic_activation_pattern.md
"""

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

T = TypeVar("T")  # Generic type for registry items


class RegistryItem(BaseModel, Generic[T]):
    """Base class for registry items with activation state.

    This class wraps any component type with activation tracking,
    metadata, and usage statistics.

    Args:
        id: Unique identifier for the component
        name: Human-readable name for the component
        description: Brief description of component capabilities
        component: The actual component instance (tool, agent, etc.)
        is_active: Whether the component is currently active
        metadata: Additional metadata for the component
        activation_count: Number of times component has been activated
        last_activated: Timestamp of last activation

    Examples:
        Create a tool registry item::

            from langchain_core.tools import tool

            @tool
            def calculator(expression: str) -> float:
                '''Calculate mathematical expression.'''
                return eval(expression)

            item = RegistryItem(
                id="calc_001",
                name="calculator",
                description="Basic math operations",
                component=calculator,
                metadata={"category": "math"}
            )

        Create an agent registry item::

            from haive.agents.simple import SimpleAgent

            agent = SimpleAgent(name="helper")
            item = RegistryItem(
                id="agent_001",
                name="helper_agent",
                description="General purpose assistant",
                component=agent
            )
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the component",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable name for the component",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Brief description of component capabilities",
    )
    component: T = Field(
        ..., description="The actual component instance (tool, agent, etc.)"
    )
    is_active: bool = Field(
        default=False, description="Whether the component is currently active"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the component"
    )
    activation_count: int = Field(
        default=0, ge=0, description="Number of times component has been activated"
    )
    last_activated: datetime | None = Field(
        default=None, description="Timestamp of last activation"
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate ID format."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("ID must be alphanumeric with underscores/hyphens only")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name format."""
        if not v.strip():
            raise ValueError("Name cannot be empty or whitespace only")
        return v.strip()


class DynamicRegistry(BaseModel, Generic[T]):
    """Generic registry for managing activatable components.

    This registry provides a type-safe way to manage components that can be
    dynamically activated and deactivated. It supports limits on active
    components and tracks activation history.

    Args:
        items: Dictionary of component ID to RegistryItem
        active_items: Set of currently active component IDs
        max_active: Maximum number of components that can be active simultaneously
        activation_history: List of activation/deactivation events

    Examples:
        Create a tool registry::

            from langchain_core.tools import Tool

            registry = DynamicRegistry[Tool]()

            # Register a tool
            item = RegistryItem(
                id="calc",
                name="calculator",
                description="Math operations",
                component=calculator_tool
            )
            registry.register(item)

            # Activate the tool
            success = registry.activate("calc")
            assert success is True

            # Get active tools
            active_tools = registry.get_active_components()

        Create an agent registry with limits::

            from haive.agents.base import Agent

            registry = DynamicRegistry[Agent](max_active=3)

            # Register multiple agents
            for i, agent in enumerate(agents):
                item = RegistryItem(
                    id=f"agent_{i}",
                    name=f"Agent {i}",
                    description=f"Agent {i} description",
                    component=agent
                )
                registry.register(item)

            # Activate agents up to limit
            for i in range(5):  # Only first 3 will activate
                success = registry.activate(f"agent_{i}")
                print(f"Agent {i} activated: {success}")
    """

    items: dict[str, RegistryItem[T]] = Field(
        default_factory=dict, description="Dictionary of component ID to RegistryItem"
    )
    active_items: set[str] = Field(
        default_factory=set, description="Set of currently active component IDs"
    )
    max_active: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of components that can be active simultaneously",
    )
    activation_history: list[dict[str, Any]] = Field(
        default_factory=list, description="List of activation/deactivation events"
    )

    @field_validator("max_active")
    @classmethod
    def validate_max_active(cls, v: int | None) -> int | None:
        """Validate max_active is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("max_active must be positive")
        return v

    @model_validator(mode="after")
    @classmethod
    def validate_active_items(cls) -> "DynamicRegistry[T]":
        """Validate that active_items are consistent with items."""
        # Remove any active items that don't exist in items
        valid_active = {
            item_id for item_id in self.active_items if item_id in self.items
        }
        if len(valid_active) != len(self.active_items):
            self.active_items = valid_active

        # Update is_active status for all items
        for item_id, item in self.items.items():
            item.is_active = item_id in self.active_items

        return self

    def register(self, item: RegistryItem[T]) -> None:
        """Register a new component in the registry.

        Args:
            item: RegistryItem to register

        Raises:
            ValueError: If item ID already exists in registry

        Examples:
            Register a tool::

                item = RegistryItem(
                    id="my_tool",
                    name="My Tool",
                    description="Does useful things",
                    component=tool_instance
                )
                registry.register(item)

            Register with metadata::

                item = RegistryItem(
                    id="special_tool",
                    name="Special Tool",
                    description="Special functionality",
                    component=tool_instance,
                    metadata={
                        "category": "utility",
                        "version": "1.0.0",
                        "author": "Developer"
                    }
                )
                registry.register(item)
        """
        if item.id in self.items:
            raise ValueError(
                f"Component with ID '{item.id}' already exists in registry"
            )

        self.items[item.id] = item

        # Log registration
        self.activation_history.append(
            {
                "timestamp": datetime.now(),
                "action": "register",
                "component_id": item.id,
                "component_name": item.name,
                "details": {"registration": True},
            }
        )

    def activate(self, item_id: str) -> bool:
        """Activate a component by ID.

        Args:
            item_id: ID of component to activate

        Returns:
            True if activation succeeded, False otherwise

        Examples:
            Activate a component::

                success = registry.activate("my_tool")
                if success:
                    print("Tool activated successfully")
                else:
                    print("Failed to activate tool")

            Check activation status::

                registry.activate("tool_1")
                registry.activate("tool_2")

                active_count = len(registry.active_items)
                print(f"Active components: {active_count}")
        """
        if item_id not in self.items:
            return False

        # Check if already active
        if item_id in self.active_items:
            return True

        # Check max_active limit
        if self.max_active and len(self.active_items) >= self.max_active:
            return False

        # Activate the component
        item = self.items[item_id]
        item.is_active = True
        item.activation_count += 1
        item.last_activated = datetime.now()
        self.active_items.add(item_id)

        # Log activation
        self.activation_history.append(
            {
                "timestamp": item.last_activated,
                "action": "activate",
                "component_id": item_id,
                "component_name": item.name,
                "details": {
                    "activation_count": item.activation_count,
                    "total_active": len(self.active_items),
                },
            }
        )

        return True

    def deactivate(self, item_id: str) -> bool:
        """Deactivate a component by ID.

        Args:
            item_id: ID of component to deactivate

        Returns:
            True if deactivation succeeded, False otherwise

        Examples:
            Deactivate a component::

                success = registry.deactivate("my_tool")
                if success:
                    print("Tool deactivated successfully")

            Deactivate all components::

                active_ids = list(registry.active_items)
                for item_id in active_ids:
                    registry.deactivate(item_id)
        """
        if item_id not in self.items or item_id not in self.active_items:
            return False

        # Deactivate the component
        item = self.items[item_id]
        item.is_active = False
        self.active_items.remove(item_id)

        # Log deactivation
        self.activation_history.append(
            {
                "timestamp": datetime.now(),
                "action": "deactivate",
                "component_id": item_id,
                "component_name": item.name,
                "details": {"total_active": len(self.active_items)},
            }
        )

        return True

    def get_active_components(self) -> list[T]:
        """Get all active component instances.

        Returns:
            List of active component instances

        Examples:
            Get active tools::

                active_tools = registry.get_active_components()
                for tool in active_tools:
                    print(f"Active tool: {tool.name}")

            Use active components::

                active_agents = registry.get_active_components()
                for agent in active_agents:
                    result = await agent.arun("Hello")
                    print(f"Agent {agent.name} response: {result}")
        """
        return [self.items[item_id].component for item_id in self.active_items]

    def get_active_items(self) -> list[RegistryItem[T]]:
        """Get all active registry items.

        Returns:
            List of active RegistryItem instances

        Examples:
            Get active items with metadata::

                active_items = registry.get_active_items()
                for item in active_items:
                    print(f"Active: {item.name}")
                    print(f"Activated: {item.activation_count} times")
                    print(f"Last used: {item.last_activated}")
                    print(f"Metadata: {item.metadata}")
        """
        return [self.items[item_id] for item_id in self.active_items]

    def get_inactive_items(self) -> list[RegistryItem[T]]:
        """Get all inactive registry items.

        Returns:
            List of inactive RegistryItem instances

        Examples:
            Show available but inactive components::

                inactive_items = registry.get_inactive_items()
                print("Available components:")
                for item in inactive_items:
                    print(f"- {item.name}: {item.description}")
        """
        return [
            item
            for item_id, item in self.items.items()
            if item_id not in self.active_items
        ]

    def is_active(self, item_id: str) -> bool:
        """Check if a component is currently active.

        Args:
            item_id: ID of component to check

        Returns:
            True if component is active, False otherwise

        Examples:
            Check activation status::

                if registry.is_active("my_tool"):
                    print("Tool is ready to use")
                else:
                    print("Tool needs to be activated first")
        """
        return item_id in self.active_items

    def get_item(self, item_id: str) -> RegistryItem[T] | None:
        """Get a registry item by ID.

        Args:
            item_id: ID of item to retrieve

        Returns:
            RegistryItem if found, None otherwise

        Examples:
            Get item information::

                item = registry.get_item("my_tool")
                if item:
                    print(f"Tool: {item.name}")
                    print(f"Description: {item.description}")
                    print(f"Active: {item.is_active}")
                    print(f"Used: {item.activation_count} times")
        """
        return self.items.get(item_id)

    def get_component(self, item_id: str) -> T | None:
        """Get a component instance by ID.

        Args:
            item_id: ID of component to retrieve

        Returns:
            Component instance if found, None otherwise

        Examples:
            Get and use component::

                tool = registry.get_component("calculator")
                if tool:
                    result = tool.invoke({"expression": "2 + 2"})
                    print(f"Result: {result}")
        """
        item = self.get_item(item_id)
        return item.component if item else None

    def list_components(self) -> list[str]:
        """List all registered component IDs.

        Returns:
            List of all component IDs in the registry

        Examples:
            List all components::

                component_ids = registry.list_components()
                for comp_id in component_ids:
                    item = registry.get_item(comp_id)
                    status = "ACTIVE" if item.is_active else "INACTIVE"
                    print(f"{comp_id}: {item.name} [{status}]")
        """
        return list(self.items.keys())

    def clear_inactive(self) -> int:
        """Remove all inactive components from registry.

        Returns:
            Number of components removed

        Examples:
            Clean up registry::

                removed_count = registry.clear_inactive()
                print(f"Removed {removed_count} inactive components")

            Selective cleanup::

                # Only remove old inactive components
                cutoff_time = datetime.now() - timedelta(hours=24)
                removed = 0
                for item_id, item in list(registry.items.items()):
                    if (not item.is_active and
                        item.last_activated and
                        item.last_activated < cutoff_time):
                        del registry.items[item_id]
                        removed += 1
                print(f"Removed {removed} old components")
        """
        removed_count = 0
        items_to_remove = []

        for item_id, item in self.items.items():
            if not item.is_active:
                items_to_remove.append(item_id)

        for item_id in items_to_remove:
            del self.items[item_id]
            removed_count += 1

            # Log removal
            self.activation_history.append(
                {
                    "timestamp": datetime.now(),
                    "action": "remove",
                    "component_id": item_id,
                    "component_name": self.items.get(item_id, {}).get(
                        "name", "unknown"
                    ),
                    "details": {"reason": "inactive_cleanup"},
                }
            )

        return removed_count

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics

        Examples:
            Show registry status::

                stats = registry.get_stats()
                print(f"Total components: {stats['total_components']}")
                print(f"Active components: {stats['active_components']}")
                print(f"Utilization: {stats['utilization_rate']:.1%}")
                print(f"Most used: {stats['most_used_component']}")
        """
        total_components = len(self.items)
        active_components = len(self.active_items)

        # Find most used component
        most_used = None
        max_activations = 0
        for item in self.items.values():
            if item.activation_count > max_activations:
                max_activations = item.activation_count
                most_used = item.name

        return {
            "total_components": total_components,
            "active_components": active_components,
            "inactive_components": total_components - active_components,
            "utilization_rate": (
                active_components / total_components if total_components > 0 else 0
            ),
            "max_active_limit": self.max_active,
            "most_used_component": most_used,
            "most_activations": max_activations,
            "total_activations": sum(
                item.activation_count for item in self.items.values()
            ),
            "activation_events": len(self.activation_history),
        }
