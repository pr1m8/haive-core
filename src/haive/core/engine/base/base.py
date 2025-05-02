"""Core engine abstractions for the Haive framework.

This module provides the base classes and abstractions for all engines in the Haive framework.
Engines are the core components that provide a consistent interface for creating and using
AI components like LLMs, retrievers, vector stores, etc.

The Engine class is a configuration/factory class that produces runnable objects,
not an invokable itself. It standardizes how engines define their input and output
field requirements.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, create_model

# Import registry base
from haive.core.engine.base.protocols import AsyncInvokable, Invokable
from haive.core.engine.base.types import EngineType

logger = logging.getLogger(__name__)

# Type variables with constraints for input/output types
TIn = TypeVar("TIn", str, Dict[str, Any], BaseModel)
TOut = TypeVar("TOut", str, Dict[str, Any], BaseModel, List[Any])


class Engine(ABC, BaseModel, Generic[TIn, TOut]):
    """Abstract base class for all engine configurations.

    Engines are configuration/factory classes that create runtime objects.
    The Engine class itself is not invokable - it produces objects that may be invokable.

    All engine subclasses must implement:
    - get_input_fields() - defining required input fields
    - get_output_fields() - defining produced output fields
    - create_runnable() - creating the actual runtime object
    """

    # Core identification fields
    id: str = Field(
        default_factory=lambda: f"{uuid.uuid4().hex}",
        description="Unique identifier for this engine instance",
    )
    name: str = Field(
        default_factory=lambda: f"engine_{uuid.uuid4().hex[:8]}",
        description="Name of this engine instance",
    )
    engine_type: EngineType = Field(description="Type of engine")
    description: Optional[str] = Field(
        default=None, description="Optional description of this engine"
    )

    # Schema definitions - used only if explicitly provided
    input_schema: Optional[Type[BaseModel]] = Field(
        default=None, description="Input schema for this engine", exclude=True
    )
    output_schema: Optional[Type[BaseModel]] = Field(
        default=None, description="Output schema for this engine", exclude=True
    )

    # Metadata for serialization
    version: str = Field(
        default="1.0.0", description="Version of this engine configuration"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for this engine"
    )

    # Configuration for model serialization
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Return input field definitions as field_name -> (type, default) pairs.
        Must be implemented by all engine subclasses.

        This can represent BaseModel fields, dictionary keys, or primitive inputs.

        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        pass

    @abstractmethod
    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Return output field definitions as field_name -> (type, default) pairs.
        Must be implemented by all engine subclasses.

        This can represent BaseModel fields, dictionary keys, or primitive outputs.

        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        pass

    @abstractmethod
    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> Any:
        """
        Create a runtime object from this engine config.

        The returned object may implement Invokable/AsyncInvokable,
        but the Engine itself is just a factory/configuration class.

        Args:
            runnable_config: Optional runtime configuration

        Returns:
            A runtime object (e.g., LLM, Retriever, etc.)
        """
        pass

    def get_schema_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Get combined schema fields for this engine.

        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        # Combine input and output fields
        fields = {}
        fields.update(self.get_input_fields())
        fields.update(self.get_output_fields())
        return fields

    def derive_input_schema(self) -> Type[BaseModel]:
        """
        Derive input schema for this engine.

        Returns:
            Pydantic model representing the input schema
        """
        # Use provided schema if available
        if self.input_schema:
            return self.input_schema

        # Get schema fields from abstract method
        fields = self.get_input_fields()

        # Create and return input schema model
        return create_model(f"{self.__class__.__name__}Input", **fields)

    def derive_output_schema(self) -> Type[BaseModel]:
        """
        Derive output schema for this engine.

        Returns:
            Pydantic model representing the output schema
        """
        # Use provided schema if available
        if self.output_schema:
            return self.output_schema

        # Get schema fields from abstract method
        fields = self.get_output_fields()

        # Create and return output schema model
        return create_model(f"{self.__class__.__name__}Output", **fields)

    def instantiate(self, runnable_config: Optional[RunnableConfig] = None) -> Any:
        """
        Instantiate the engine (alias for create_runnable).

        Args:
            runnable_config: Optional runtime configuration

        Returns:
            Instantiated runtime object
        """
        return self.create_runnable(runnable_config)

    def register(self) -> Engine:
        """
        Register this engine in the global registry.

        Returns:
            Self for chaining
        """

        return EngineRegistry.get_instance().register(self)

    def apply_runnable_config(
        self, runnable_config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Extract parameters from runnable_config relevant to this engine.

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
        fields = self.model_fields
        for field_name in fields:
            if field_name in configurable and field_name not in params:
                params[field_name] = configurable[field_name]

        return params

    def extract_params(self) -> Dict[str, Any]:
        """
        Extract parameters from this engine for serialization.

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
                "input_schema",
                "output_schema",
                "id",
                "name",
                "engine_type",
            ]:
                continue

            # Get the value if it exists
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                params[field_name] = value

        return params

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert engine to a dictionary suitable for serialization.

        Returns:
            Dictionary representation of the engine
        """
        # Convert to dictionary using Pydantic
        data = self.model_dump(exclude={"input_schema", "output_schema"})

        # Add schema information if available
        if self.input_schema:
            data["input_schema_name"] = self.input_schema.__name__

        if self.output_schema:
            data["output_schema_name"] = self.output_schema.__name__

        # Add class information for type reconstruction
        data["engine_class"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Engine:
        """
        Create an engine from a dictionary.

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
                logger.warning(
                    f"Could not load engine class '{engine_class_path}': {e}"
                )

        # Fallback to instantiating the base class
        return cls(**data)

    def to_json(self) -> str:
        """
        Convert engine to JSON string.

        Returns:
            JSON representation of the engine
        """
        from haive.core.utils.pydantic_utils import ensure_json_serializable

        data = self.to_dict()
        serializable_data = ensure_json_serializable(data)
        return json.dumps(serializable_data)

    @classmethod
    def from_json(cls, json_str: str) -> Engine:
        """
        Create an engine from a JSON string.

        Args:
            json_str: JSON representation of the engine

        Returns:
            Instantiated engine
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def with_config_overrides(self, overrides: Dict[str, Any]) -> "Engine":
        """
        Create a new engine with configuration overrides.

        Args:
            overrides: Configuration overrides to apply

        Returns:
            New engine instance with overrides applied
        """
        # Create a copy of this engine
        config = self.model_dump()

        # Apply overrides
        for key, value in overrides.items():
            if key in config:
                config[key] = value

        # Create new instance
        return self.__class__.model_validate(config)


class InvokableEngine(Engine[TIn, TOut]):
    """
    Base class for engines that create invokable runtime objects.

    The engine itself is not directly invokable, but provides convenience methods
    that create the runtime object and invoke it.

    Examples include LLMs, Retrievers, and VectorStores.
    """

    def invoke(
        self, input_data: TIn, runnable_config: Optional[RunnableConfig] = None
    ) -> TOut:
        """
        Convenience method to create and invoke the runnable with input data.

        This is a shortcut for:
            runnable = self.create_runnable(runnable_config)
            return runnable.invoke(input_data)

        Args:
            input_data: Input data for the runnable
            runnable_config: Optional runtime configuration

        Returns:
            Output from the runnable
        """
        # Create runnable
        runnable = self.create_runnable(runnable_config)

        # Use direct invokable interface if available
        if isinstance(runnable, Invokable):
            return runnable.invoke(input_data, config=runnable_config)

        # Call with config if supported
        if callable(runnable):
            try:
                import inspect

                sig = inspect.signature(runnable)
                if "config" in sig.parameters:
                    return runnable(input_data, config=runnable_config)
                return runnable(input_data)
            except Exception as e:
                logger.error(f"Error invoking runnable: {e}")
                raise

        raise NotImplementedError(
            f"Engine {self.name} creates a runnable that does not support invoke"
        )

    async def ainvoke(
        self, input_data: TIn, runnable_config: Optional[RunnableConfig] = None
    ) -> TOut:
        """
        Convenience method to create and asynchronously invoke the runnable.

        This is a shortcut for:
            runnable = self.create_runnable(runnable_config)
            return await runnable.ainvoke(input_data)

        Args:
            input_data: Input data for the runnable
            runnable_config: Optional runtime configuration

        Returns:
            Output from the runnable
        """
        # Create runnable
        runnable = self.create_runnable(runnable_config)

        # Use async invokable interface if available
        if isinstance(runnable, AsyncInvokable):
            return await runnable.ainvoke(input_data, config=runnable_config)

        # Use asyncio to run in thread if not async
        import asyncio

        if callable(runnable):
            try:
                import inspect

                sig = inspect.signature(runnable)
                if "config" in sig.parameters:
                    return await asyncio.to_thread(
                        runnable, input_data, config=runnable_config
                    )
                return await asyncio.to_thread(runnable, input_data)
            except Exception as e:
                logger.error(f"Error invoking runnable asynchronously: {e}")
                raise

        raise NotImplementedError(
            f"Engine {self.name} creates a runnable that does not support ainvoke"
        )


class NonInvokableEngine(Engine[TIn, TOut]):
    """
    Base class for engines that create non-invokable utility objects.

    Examples include Embeddings engines and DocumentLoader engines.
    """

    pass
