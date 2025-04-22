# src/haive/core/registry/__init__.py

from haive.core.registry.base import AbstractRegistry
from haive.core.registry.memory import MemoryRegistry
from haive.core.registry.manager import RegistryManager

# Initialize registry types
RegistryManager.register_registry_type("memory", MemoryRegistry)

# Export primary classes
__all__ = ['AbstractRegistry', 'MemoryRegistry', 'RegistryManager']