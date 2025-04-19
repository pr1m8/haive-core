# src/haive/core/engine/engine_base.py

"""Core engine abstractions for the Haive framework.

This module provides the base classes and abstractions for all engines in the Haive framework.
Engines are the core components that provide a consistent interface for creating and using
AI components like LLMs, retrievers, vector stores, etc.

Classes:
    EngineType: Enum of supported engine types
    Engine: Abstract base class for all engines
    InvokableEngine: Base class for engines that can be directly invoked
    NonInvokableEngine: Base class for utility engines
    EngineRegistry: Central registry for managing engines

Example:
    ```python
    class MyLLMEngine(InvokableEngine[str, str]):
        engine_type = EngineType.LLM
        
        def create_runnable(self, runnable_config: Optional[RunnableConfig] = None):
            # Create and return LLM instance
            pass
    ```
"""

from __future__ import annotations

import builtins
import inspect
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import (
    Any,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, create_model

logger = logging.getLogger(__name__)

# Type variables for generic relationships
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

class EngineType(str, Enum):
    """Types of engines the system can use."""
    LLM = "llm"
    VECTOR_STORE = "vector_store"
    RETRIEVER = "retriever"
    TOOL = "tool"
    EMBEDDINGS = "embeddings"
    AGENT = "agent"
    DOCUMENT_LOADER = "document_loader"
    DOCUMENT_TRANSFORMER = "document_transformer"
    DOCUMENT_SPLITTER = "document_splitter"

@runtime_checkable
class Invokable(Protocol):
    """Protocol for objects that can be invoked."""
    def invoke(self, input_data: Any, **kwargs) -> Any: ...

@runtime_checkable
class AsyncInvokable(Protocol):
    """Protocol for objects that can be invoked asynchronously."""
    async def ainvoke(self, input_data: Any, **kwargs) -> Any: ...

class Engine(ABC, BaseModel, Generic[TIn, TOut]):
    """Abstract base class for all engine configurations.
    
    Engines are the core components of the Haive framework, providing a consistent
    interface for creating and using AI components like LLMs, retrievers, vector stores, etc.
    """
    # Core identification fields
    id: str = Field(
        default_factory=lambda: f"{uuid.uuid4().hex}",
        description="Unique identifier for this engine instance"
    )
    name: str = Field(
        default_factory=lambda: f"engine_{uuid.uuid4().hex[:8]}",
        description="Unique name for this engine"
    )
    engine_type: EngineType = Field(
        description="Type of engine"
    )
    description: str | None = Field(
        default=None,
        description="Optional description of this engine"
    )

    # Schema definitions
    input_schema: type[BaseModel] | None = Field(
        default=None,
        description="Input schema for this engine",
        exclude=True
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description="Output schema for this engine",
        exclude=True
    )

    # Metadata for serialization and graph integration
    version: str = Field(
        default="1.0.0",
        description="Version of this engine configuration"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this engine"
    )

    # Configuration for model serialization
    model_config = ConfigDict(arbitrary_types_allowed = True)

    @abstractmethod
    def create_runnable(self, runnable_config: RunnableConfig | None = None) -> Any:
        """Create a runnable instance from this engine config.
        
        Args:
            runnable_config: Optional runtime configuration
            
        Returns:
            A runnable object (e.g., LLM, Retriever, etc.)
        """

    def instantiate(self, runnable_config: RunnableConfig | None = None) -> Any:
        """Instantiate the engine (alias for create_runnable).
        
        Args:
            runnable_config: Optional runtime configuration
            
        Returns:
            Instantiated engine
        """
        return self.create_runnable(runnable_config)

    def register(self) -> Engine:
        """Register this engine in the global registry.
        
        Returns:
            Self for chaining
        """
        return EngineRegistry.get_instance().register(self)

    @classmethod
    def get(cls, name: str) -> Engine | None:
        """Get an engine of this type from the registry.
        
        Args:
            name: Name of the engine to retrieve
            
        Returns:
            Engine instance if found, None otherwise
        """
        registry = EngineRegistry.get_instance()
        return registry.get(cls.engine_type, name)

    @classmethod
    def find_by_id(cls, id: str) -> Engine | None:
        """Find an engine by its unique ID.
        
        Args:
            id: Unique ID of the engine to find
            
        Returns:
            Engine instance if found, None otherwise
        """
        registry = EngineRegistry.get_instance()
        return registry.find_by_id(id)

    @classmethod
    def list(cls) -> builtins.list[str]:
        """List all registered engines of this type.
        
        Returns:
            List of engine names
        """
        registry = EngineRegistry.get_instance()
        return registry.list(cls.engine_type)

    # In Engine.apply_runnable_config method

    def apply_runnable_config(self, runnable_config: RunnableConfig | None = None) -> dict[str, Any]:
        """Extract parameters from runnable_config relevant to this engine.
        
        Args:
            runnable_config: Runtime configuration to extract from
            
        Returns:
            Dictionary of relevant parameters
        """
        # Return empty dict if no config provided
        if not runnable_config or "configurable" not in runnable_config:
            return {}

        configurable = runnable_config["configurable"]
        params = {}

        # Engine identification - check for direct references using id or name
        if "engine_configs" in configurable:
            # Try by engine ID first (most specific)
            if self.id in configurable["engine_configs"]:
                params.update(configurable["engine_configs"][self.id])

            # Try by engine name if no id match
            elif self.name in configurable["engine_configs"]:
                params.update(configurable["engine_configs"][self.name])

            # Try by engine type if available (least specific)
            engine_type_key = f"{self.engine_type.value}_config"
            if engine_type_key in configurable["engine_configs"]:
                # Don't override existing params
                for k, v in configurable["engine_configs"][engine_type_key].items():
                    if k not in params:
                        params[k] = v

        # Extract parameters that match fields from this engine
        # This makes parameter extraction generic and based on the engine's fields
        fields = self.model_fields if hasattr(self, "model_fields") else getattr(self, "__fields__", {})
        for field_name in fields:
            if field_name in configurable and field_name not in params:
                params[field_name] = configurable[field_name]

        return params
    def derive_input_schema(self) -> type[BaseModel]:
        """Derive input schema for this engine.
        
        Returns:
            Pydantic model representing the input schema
        """
        # Use provided schema if available
        if self.input_schema:
            return self.input_schema

        # Try to derive from class type hints
        hints = {}

        # Try direct generic type arguments first
        for base_cls in self.__class__.__orig_bases__:
            if get_origin(base_cls) is Generic:
                args = get_args(base_cls)
                if len(args) >= 1:
                    # Extract TIn from Generic[TIn, TOut]
                    in_type = args[0]
                    if in_type is not TIn:  # Not the generic parameter itself
                        if isinstance(in_type, type) and issubclass(in_type, BaseModel):
                            return in_type
                        if get_origin(in_type) is Union:
                            # Handle Union types
                            for arg in get_args(in_type):
                                if isinstance(arg, type) and issubclass(arg, BaseModel):
                                    return arg
                        # Create field type hints
                        hints["input"] = (in_type, ...)
                        break

        # If we couldn't derive from generics, use a default schema
        if not hints:
            from typing import Optional
            hints["input"] = (Optional[Any], Field(default=None))

        # Create and return the model
        return create_model(f"{self.__class__.__name__}Input", **hints)

    def derive_output_schema(self) -> type[BaseModel]:
        """Derive output schema for this engine.
        
        Returns:
            Pydantic model representing the output schema
        """
        # Use provided schema if available
        if self.output_schema:
            return self.output_schema

        # Check if engine has a structured output model (common in LLM engines)
        if hasattr(self, "structured_output_model") and self.structured_output_model:
            return self.structured_output_model

        # Check if get_schema_fields is implemented
        schema_fields = self.get_schema_fields()
        if schema_fields:
            # Use fields from get_schema_fields for output schema too
            return create_model(
                f"{self.__class__.__name__}Output",
                **{name: field_info for name, field_info in schema_fields.items()}
            )

        # Try to derive from Generic type arguments
        for base_cls in self.__class__.__orig_bases__:
            if get_origin(base_cls) is Generic:
                args = get_args(base_cls)
                if len(args) >= 2:
                    # Extract TOut from Generic[TIn, TOut]
                    out_type = args[1]
                    if out_type is not TOut:  # Not the generic parameter itself
                        if isinstance(out_type, type) and issubclass(out_type, BaseModel):
                            return out_type
                        if get_origin(out_type) is Union:
                            # Handle Union types
                            for arg in get_args(out_type):
                                if isinstance(arg, type) and issubclass(arg, BaseModel):
                                    return arg
                        # Create field type hints
                        hints = {"output": (out_type, ...)}
                        return create_model(f"{self.__class__.__name__}Output", **hints)

        # Default schema with output field
        hints = {"output": (Optional[Any], Field(default=None))}
        return create_model(f"{self.__class__.__name__}Output", **hints)

    def get_schema_fields(self) -> dict[str, tuple[type, Any]]:
        """Get schema fields for this engine.
        
        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        fields = {}

        # First check for engine-specific schema implementation
        if hasattr(self, "_get_schema_fields") and callable(self._get_schema_fields):
            engine_fields = self._get_schema_fields()
            if engine_fields:
                return engine_fields

        # Add input schema fields if available
        # Use existing schema to avoid recursion in derive_input_schema
        if self.input_schema:
            input_schema = self.input_schema
        else:
            # Try to derive from generic types directly
            input_schema = None
            for base_cls in self.__class__.__orig_bases__:
                if get_origin(base_cls) is Generic:
                    args = get_args(base_cls)
                    if len(args) >= 1:
                        in_type = args[0]
                        if in_type is not TIn and isinstance(in_type, type) and issubclass(in_type, BaseModel):
                            input_schema = in_type
                            break

            # If still no schema, create a simple default one
            if not input_schema:
                from typing import Optional
                fields["input"] = (Optional[Any], Field(default=None))
                return fields

        if input_schema:
            # Handle Pydantic v2
            if hasattr(input_schema, "model_fields"):
                for name, field_info in input_schema.model_fields.items():
                    # Skip internal fields
                    if name.startswith("__") or name in ["runnable_config"]:
                        continue
                    fields[name] = (field_info.annotation, field_info.default)
            # Handle Pydantic v1
            elif hasattr(input_schema, "__fields__"):
                for name, field_info in input_schema.__fields__.items():
                    # Skip internal fields
                    if name.startswith("__") or name in ["runnable_config"]:
                        continue
                    fields[name] = (field_info.type_, field_info.default)

        return fields

    def to_runnable_config(self, **runtime_params) -> RunnableConfig:
        """Convert engine configuration to a RunnableConfig.
        
        Args:
            **runtime_params: Additional runtime parameters
            
        Returns:
            RunnableConfig with engine configuration
        """
        from haive_core.config.runnable import RunnableConfigManager

        # Start with default config
        config = RunnableConfigManager.create(
            thread_id=runtime_params.pop("thread_id", None),
            user_id=runtime_params.pop("user_id", None)
        )

        # Extract serializable parameters from this engine
        engine_params = self.extract_params()

        # Add engine config by ID for precise targeting
        config = RunnableConfigManager.add_engine_config(
            config,
            self.id,
            **engine_params
        )

        # Also add by name for convenience
        config = RunnableConfigManager.add_engine_config(
            config,
            self.name,
            **engine_params
        )

        # Also add by engine type for type-based lookups
        config = RunnableConfigManager.add_engine_config(
            config,
            f"{self.engine_type.value}_config",
            **engine_params
        )

        # Add runtime params
        for key, value in runtime_params.items():
            config["configurable"][key] = value

        return config

    def extract_params(self) -> dict[str, Any]:
        """Extract parameters from this engine for serialization.
        
        Returns:
            Dictionary of engine parameters
        """
        params = {}

        # Get fields from the model
        fields = self.model_fields

        # Add all relevant fields
        for field_name in fields:
            # Skip fields that shouldn't be in params
            if field_name.startswith("_") or field_name in [
                "input_schema", "output_schema", "id", "name", "engine_type"
            ]:
                continue

            # Get the value if it exists
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                params[field_name] = value

        return params

    def to_dict(self) -> dict[str, Any]:
        """Convert engine to a dictionary suitable for serialization.
        
        Returns:
            Dictionary representation of the engine
        """
        # Convert to dictionary using Pydantic
        if hasattr(self, "model_dump"):
            # Pydantic v2
            data = self.model_dump(exclude={"input_schema", "output_schema"})
        else:
            # Pydantic v1
            data = self.dict(exclude={"input_schema", "output_schema"})

        # Add schema information if available
        # Use existing schema instead of trying to derive to avoid recursion
        if self.input_schema:
            data["input_schema_name"] = self.input_schema.__name__

        if self.output_schema:
            data["output_schema_name"] = self.output_schema.__name__

        # Add class information for type reconstruction
        data["engine_class"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Engine:
        """Create an engine from a dictionary.
        
        Args:
            data: Dictionary representation of the engine
            
        Returns:
            Instantiated engine
        """
        # Extract class information for dynamic loading
        engine_class_path = data.pop("engine_class", None)

        if engine_class_path:
            try:
                # Dynamically load the class
                module_name, class_name = engine_class_path.rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name])
                engine_cls = getattr(module, class_name)

                # Clean data before instantiation
                for field_name in ["input_schema_name", "output_schema_name"]:
                    if field_name in data:
                        data.pop(field_name)

                # Instantiate the correct class
                return engine_cls(**data)
            except (ImportError, AttributeError, ValueError) as e:
                logger.warning(f"Could not load engine class '{engine_class_path}': {e}")

        # Fallback to instantiating the base class
        return cls(**data)

    def to_json(self) -> str:
        """Convert engine to JSON string.
        
        Returns:
            JSON representation of the engine
        """
        from haive_core.utils.pydantic_utils import ensure_json_serializable
        data = self.to_dict()
        serializable_data = ensure_json_serializable(data)
        return json.dumps(serializable_data)

    @classmethod
    def from_json(cls, json_str: str) -> Engine:
        """Create an engine from a JSON string.
        
        Args:
            json_str: JSON representation of the engine
            
        Returns:
            Instantiated engine
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def as_node(self,
                command_goto: str | None = None,
                input_mapping: dict[str, str] | None = None,
                output_mapping: dict[str, str] | None = None) -> Callable:
        """Convert this engine to a node function that can be used in a graph.
        
        Args:
            command_goto: Optional next node to route to
            input_mapping: Optional mapping from state keys to engine input keys
            output_mapping: Optional mapping from engine output keys to state keys
            
        Returns:
            A callable node function
        """
        # Import here to avoid circular imports
        from haive_core.graph.NodeFactory import NodeFactory

        # Create a node function from this engine
        return NodeFactory.create_node_function(
            config=self,
            command_goto=command_goto,
            input_mapping=input_mapping,
            output_mapping=output_mapping
        )

    def add_to_graph(self,
                    graph_builder,
                    node_name: str | None = None,
                    command_goto: str | None = None,
                    input_mapping: dict[str, str] | None = None,
                    output_mapping: dict[str, str] | None = None) -> Any:
        """Add this engine as a node to a graph builder.
        
        Args:
            graph_builder: The graph builder to add to (DynamicGraph, StateGraphEditor, etc.)
            node_name: Optional node name (defaults to engine name)
            command_goto: Optional next node to route to
            input_mapping: Optional mapping from state keys to engine input keys
            output_mapping: Optional mapping from engine output keys to state keys
            
        Returns:
            The graph builder for chaining
        """
        # Use engine name if node_name not provided
        node_name = node_name or self.name

        # Add to graph
        if hasattr(graph_builder, "add_node"):
            graph_builder.add_node(
                name=node_name,
                config=self,
                command_goto=command_goto,
                input_mapping=input_mapping,
                output_mapping=output_mapping
            )

        return graph_builder

class InvokableEngine(Engine[TIn, TOut]):
    """Base class for engines that support direct invocation.
    
    These engines can be invoked directly with input data and return output data.
    Examples include LLMs, Retrievers, and VectorStores.
    """

    def invoke(self, input_data: TIn, runnable_config: RunnableConfig | None = None) -> TOut:
        """Invoke the engine with input data.
        
        Args:
            input_data: Input data for the engine
            runnable_config: Optional runtime configuration
            
        Returns:
            Output from the engine
        """
        # Create or get the runnable instance
        runnable = self.create_runnable(runnable_config)

        # Use langchain's invoke if available
        if isinstance(runnable, Invokable):
            return runnable.invoke(input_data, config=runnable_config)

        # Otherwise use fallback
        return self._fallback_invoke(runnable, input_data, runnable_config)

    def _fallback_invoke(self, runnable, input_data: TIn, runnable_config) -> TOut:
        """Fallback implementation for invoke when runnable doesn't support it.
        
        Args:
            runnable: The created runnable
            input_data: Input data
            runnable_config: Runtime configuration
            
        Returns:
            Output from the runnable
        """
        # If it's callable, just call it
        if callable(runnable):
            # Check if the callable accepts a config parameter
            sig = inspect.signature(runnable)
            if "config" in sig.parameters:
                return runnable(input_data, config=runnable_config)
            return runnable(input_data)

        # Otherwise, we don't know how to invoke it
        raise NotImplementedError(f"Engine {self.name} does not support invoke")

    async def ainvoke(self, input_data: TIn, runnable_config: RunnableConfig | None = None) -> TOut:
        """Asynchronously invoke the engine with input data.
        
        Args:
            input_data: Input data for the engine
            runnable_config: Optional runtime configuration
            
        Returns:
            Output from the engine
        """
        # Create or get the runnable instance
        runnable = self.create_runnable(runnable_config)

        # Use langchain's ainvoke if available
        if isinstance(runnable, AsyncInvokable):
            return await runnable.ainvoke(input_data, config=runnable_config)

        # Otherwise use fallback
        return await self._fallback_ainvoke(runnable, input_data, runnable_config)

    async def _fallback_ainvoke(self, runnable, input_data: TIn, runnable_config) -> TOut:
        """Fallback implementation for ainvoke when runnable doesn't support it.
        
        Args:
            runnable: The created runnable
            input_data: Input data
            runnable_config: Runtime configuration
            
        Returns:
            Output from the runnable
        """
        # Use asyncio to run in a thread
        import asyncio

        if callable(runnable):
            # Check if the callable accepts a config parameter
            sig = inspect.signature(runnable)
            if "config" in sig.parameters:
                return await asyncio.to_thread(runnable, input_data, config=runnable_config)
            return await asyncio.to_thread(runnable, input_data)

        # Otherwise, we don't know how to invoke it
        raise NotImplementedError(f"Engine {self.name} does not support ainvoke")

class NonInvokableEngine(Engine[TIn, TOut]):
    """Base class for engines that don't support direct invocation.
    
    These engines provide utility functionality but don't follow the invokable pattern.
    Examples include Embeddings engines and DocumentLoader engines.
    """

    def instantiate(self, runnable_config: RunnableConfig | None = None) -> Any:
        """Instantiate the engine.
        
        Args:
            runnable_config: Optional runtime configuration
            
        Returns:
            An instantiated engine
        """
        return self.create_runnable(runnable_config)

class EngineRegistry:
    """Central registry for all engines in the system.
    
    Provides methods for registering, retrieving, and listing engines.
    """
    _instance = None

    @classmethod
    def get_instance(cls) -> EngineRegistry:
        """Get singleton instance.
        
        Returns:
            The singleton EngineRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the registry with empty dictionaries for each engine type."""
        self.engines = {engine_type: {} for engine_type in EngineType}
        self.engine_ids = {}  # id -> engine mapping

    def register(self, engine: Engine) -> Engine:
        """Register an engine.
        
        Args:
            engine: Engine to register
            
        Returns:
            The registered engine (for chaining)
        """
        self.engines[engine.engine_type][engine.name] = engine
        self.engine_ids[engine.id] = engine
        logger.debug(f"Registered engine {engine.name} (id: {engine.id}) of type {engine.engine_type}")
        return engine

    def get(self, engine_type: EngineType, name: str) -> Engine | None:
        """Get an engine by type and name.
        
        Args:
            engine_type: Type of engine to retrieve
            name: Name of the engine to retrieve
            
        Returns:
            Engine if found, None otherwise
        """
        return self.engines[engine_type].get(name)

    def find_by_id(self, id: str) -> Engine | None:
        """Find an engine by its unique ID.
        
        Args:
            id: Unique ID of the engine to find
            
        Returns:
            Engine if found, None otherwise
        """
        return self.engine_ids.get(id)

    def find(self, name_or_id: str) -> Engine | None:
        """Find an engine by name or ID across all engine types.
        
        Args:
            name_or_id: Name or ID of the engine to find
            
        Returns:
            Engine if found, None otherwise
        """
        # Check ID first (faster lookup)
        if engine := self.engine_ids.get(name_or_id):
            return engine

        # Search through all engine types by name
        for engine_type in EngineType:
            if engine := self.get(engine_type, name_or_id):
                return engine

        return None

    def list(self, engine_type: EngineType) -> builtins.list[str]:
        """List all engines of a type.
        
        Args:
            engine_type: Type of engines to list
            
        Returns:
            List of engine names
        """
        return list(self.engines[engine_type].keys())

    def get_all(self, engine_type: EngineType) -> dict[str, Engine]:
        """Get all engines of a type.
        
        Args:
            engine_type: Type of engines to retrieve
            
        Returns:
            Dictionary mapping engine names to engines
        """
        return self.engines[engine_type]

    def clear(self):
        """Clear the registry (mainly for testing)."""
        self.engines = {engine_type: {} for engine_type in EngineType}
        self.engine_ids = {}

    def to_dict(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Convert registry to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the registry
        """
        result = {}

        for engine_type, engines in self.engines.items():
            result[engine_type] = {}
            for engine_name, engine in engines.items():
                result[engine_type][engine_name] = engine.to_dict()

        return result

    def from_dict(self, data: dict[str, dict[str, dict[str, Any]]]) -> None:
        """Load engines from a dictionary.
        
        Args:
            data: Dictionary representation of engines
        """
        for engine_type_str, engines_dict in data.items():
            try:
                engine_type = EngineType(engine_type_str)
                for engine_name, engine_data in engines_dict.items():
                    try:
                        engine = Engine.from_dict(engine_data)
                        self.register(engine)
                    except Exception as e:
                        logger.error(f"Error loading engine {engine_name}: {e}")
            except ValueError:
                logger.warning(f"Unknown engine type: {engine_type_str}")

    def to_json(self) -> str:
        """Convert registry to JSON string.
        
        Returns:
            JSON representation of the registry
        """
        from haive_core.utils.pydantic_utils import ensure_json_serializable
        data = self.to_dict()
        serializable_data = ensure_json_serializable(data)
        return json.dumps(serializable_data, indent=2)

    def from_json(self, json_str: str) -> None:
        """Load engines from a JSON string.
        
        Args:
            json_str: JSON representation of engines
        """
        data = json.loads(json_str)
        self.from_dict(data)
