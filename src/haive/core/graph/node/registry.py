# python# src/haive/core/graph/node/registry.py
# Remove NodeConfig import and add decorators at module level

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

from haive.core.graph.node.protocols import (
    CommandHandler,
    InputProcessor,
    NodeProcessor,
    OutputProcessor,
)
from haive.core.registry.base import AbstractRegistry

logger = logging.getLogger(__name__)


# Define decorators directly at module level
def register_node_processor(node_type):
    """Decorator to register node processors."""

    def decorator(cls):
        registry = NodeTypeRegistry.get_instance()
        registry.register_node_processor(node_type, cls())
        return cls

    return decorator


def register_command_handler(handler_type):
    """Decorator to register command handlers."""

    def decorator(cls):
        registry = NodeTypeRegistry.get_instance()
        registry.register_command_handler(handler_type, cls())
        return cls

    return decorator


def register_input_processor(processor_type):
    """Decorator to register input processors."""

    def decorator(cls):
        registry = NodeTypeRegistry.get_instance()
        registry.register_input_processor(processor_type, cls())
        return cls

    return decorator


def register_output_processor(processor_type):
    """Decorator to register output processors."""

    def decorator(cls):
        registry = NodeTypeRegistry.get_instance()
        registry.register_output_processor(processor_type, cls())
        return cls

    return decorator


class NodeTypeRegistry(AbstractRegistry[NodeProcessor]):
    """
    Registry for node types, processors, and handlers.

    This registry enables the extensible node factory system by allowing
    registration of processors for different node types, command handlers,
    and input/output processors.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "NodeTypeRegistry":
        """Get the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        self.node_processors: Dict[str, NodeProcessor] = {}
        self.command_handlers: Dict[str, CommandHandler] = {}
        self.input_processors: Dict[str, InputProcessor] = {}
        self.output_processors: Dict[str, OutputProcessor] = {}
        self.items: Dict[str, NodeProcessor] = {}  # Required by AbstractRegistry

    def register(self, item: NodeProcessor, id: Optional[str] = None) -> NodeProcessor:
        """
        Register an item in the registry.

        Args:
            item: Item to register
            id: Optional ID for the item

        Returns:
            The registered item
        """
        item_id = id or f"processor_{len(self.items)}"
        self.items[item_id] = item
        return item

    def register_node_processor(self, node_type: str, processor: NodeProcessor) -> None:
        """
        Register a processor for a specific node type.

        Args:
            node_type: Node type identifier
            processor: Processor implementation
        """
        self.node_processors[node_type] = processor
        # Also register in AbstractRegistry items
        self.register(processor, f"node_processor_{node_type}")
        logger.debug(f"Registered node processor for {node_type}")

    def register_command_handler(
        self, handler_type: str, handler: CommandHandler
    ) -> None:
        """
        Register a command handler.

        Args:
            handler_type: Handler type identifier
            handler: Handler implementation
        """
        self.command_handlers[handler_type] = handler
        logger.debug(f"Registered command handler for {handler_type}")

    def register_input_processor(
        self, processor_type: str, processor: InputProcessor
    ) -> None:
        """
        Register an input processor.

        Args:
            processor_type: Processor type identifier
            processor: Processor implementation
        """
        self.input_processors[processor_type] = processor
        logger.debug(f"Registered input processor for {processor_type}")

    def register_output_processor(
        self, processor_type: str, processor: OutputProcessor
    ) -> None:
        """
        Register an output processor.

        Args:
            processor_type: Processor type identifier
            processor: Processor implementation
        """
        self.output_processors[processor_type] = processor
        logger.debug(f"Registered output processor for {processor_type}")

    def get(self, item_type: Any, name: str) -> Optional[NodeProcessor]:
        """
        Get an item by type and name.

        Args:
            item_type: Type of item
            name: Name of item

        Returns:
            Item if found, None otherwise
        """
        # For node processors, use the specific dictionary
        if name in self.node_processors:
            return self.node_processors[name]

        # Otherwise check the general items
        return self.items.get(name)

    def find_by_id(self, id: str) -> Optional[NodeProcessor]:
        """
        Find an item by ID.

        Args:
            id: Item ID

        Returns:
            Item if found, None otherwise
        """
        return self.items.get(id)

    def list(self, item_type: Any = None) -> List[str]:
        """
        List all items of a type.

        Args:
            item_type: Optional type to filter by

        Returns:
            List of item names
        """
        if item_type == "node_processor":
            return list(self.node_processors.keys())
        elif item_type == "command_handler":
            return list(self.command_handlers.keys())
        elif item_type == "input_processor":
            return list(self.input_processors.keys())
        elif item_type == "output_processor":
            return list(self.output_processors.keys())
        else:
            return list(self.items.keys())

    def get_all(self, item_type: Any = None) -> Dict[str, NodeProcessor]:
        """
        Get all items of a type.

        Args:
            item_type: Optional type to filter by

        Returns:
            Dictionary of items
        """
        if item_type == "node_processor":
            return self.node_processors
        elif item_type == "command_handler":
            return cast(Dict[str, NodeProcessor], self.command_handlers)
        elif item_type == "input_processor":
            return cast(Dict[str, NodeProcessor], self.input_processors)
        elif item_type == "output_processor":
            return cast(Dict[str, NodeProcessor], self.output_processors)
        else:
            return self.items

    def clear(self) -> None:
        """Clear the registry."""
        self.node_processors.clear()
        self.command_handlers.clear()
        self.input_processors.clear()
        self.output_processors.clear()
        self.items.clear()

    def get_node_processor(self, node_type: str) -> Optional[NodeProcessor]:
        """
        Get a processor for a node type.

        Args:
            node_type: Node type identifier

        Returns:
            Processor if found, None otherwise
        """
        return self.node_processors.get(node_type)

    def get_command_handler(self, handler_type: str) -> Optional[CommandHandler]:
        """
        Get a command handler.

        Args:
            handler_type: Handler type identifier

        Returns:
            Handler if found, None otherwise
        """
        return self.command_handlers.get(handler_type)

    def get_input_processor(self, processor_type: str) -> Optional[InputProcessor]:
        """
        Get an input processor.

        Args:
            processor_type: Processor type identifier

        Returns:
            Processor if found, None otherwise
        """
        return self.input_processors.get(processor_type)

    def get_output_processor(self, processor_type: str) -> Optional[OutputProcessor]:
        """
        Get an output processor.

        Args:
            processor_type: Processor type identifier

        Returns:
            Processor if found, None otherwise
        """
        return self.output_processors.get(processor_type)

    def find_processor_for_engine(self, engine: Any) -> Optional[NodeProcessor]:
        """
        Find the first processor that can handle an engine.

        Args:
            engine: Engine to process

        Returns:
            Processor if found, None otherwise
        """
        for processor in self.node_processors.values():
            try:
                if processor.can_process(engine):
                    return processor
            except Exception as e:
                logger.debug(f"Error checking processor compatibility: {e}")
                continue
        return None

    def register_default_processors(self) -> None:
        """Register the default set of processors."""
        # Import processors directly using direct imports
        # instead of importing from a module that might cause circular imports
        try:
            # Create processor instances directly
            from haive.core.graph.node.handlers import (
                DirectInputProcessor,
                MappedInputProcessor,
                StandardCommandHandler,
                StandardOutputProcessor,
                StructuredOutputProcessor,
            )
            from haive.core.graph.node.processors import (
                AsyncInvokableNodeProcessor,
                AsyncNodeProcessor,
                CallableNodeProcessor,
                GenericNodeProcessor,
                InvokableNodeProcessor,
                MappingNodeProcessor,
            )

            # Register processors directly
            self.node_processors["invokable"] = InvokableNodeProcessor()
            self.node_processors["async_invokable"] = AsyncInvokableNodeProcessor()
            self.node_processors["callable"] = CallableNodeProcessor()
            self.node_processors["async"] = AsyncNodeProcessor()
            self.node_processors["mapping"] = MappingNodeProcessor()
            self.node_processors["generic"] = GenericNodeProcessor()

            # Register command handlers
            self.command_handlers["standard"] = StandardCommandHandler()

            # Register input processors
            self.input_processors["direct"] = DirectInputProcessor()
            self.input_processors["mapped"] = MappedInputProcessor()

            # Register output processors
            self.output_processors["standard"] = StandardOutputProcessor()
            self.output_processors["structured"] = StructuredOutputProcessor()

            logger.info("Default processors registered successfully")
        except Exception as e:
            logger.warning(f"Error registering processors: {str(e)}")
            logger.warning("Default processors not registered")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert registry to a serializable dictionary.

        Returns:
            Dictionary representation
        """
        # We can only serialize the registered types, not the actual objects
        serialized = {
            "node_processors": list(self.node_processors.keys()),
            "command_handlers": list(self.command_handlers.keys()),
            "input_processors": list(self.input_processors.keys()),
            "output_processors": list(self.output_processors.keys()),
        }
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeTypeRegistry":
        """
        Create registry from dictionary representation.

        Args:
            data: Dictionary representation

        Returns:
            Populated registry
        """
        registry = cls()

        # Register default processors to populate the registry
        registry.register_default_processors()

        return registry
