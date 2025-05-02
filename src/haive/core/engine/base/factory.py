# src/haive/core/engine/factory.py

from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from haive.core.engine.base.reference import ComponentRef

T = TypeVar("T")  # Type variable for the created component


class ComponentFactory(BaseModel, Generic[T]):
    """
    Factory for creating runtime components from engine configurations.

    This separates the serializable configuration from the
    non-serializable runtime components.
    """

    # Reference to the engine configuration
    engine_ref: ComponentRef = Field(
        description="Reference to the engine configuration"
    )

    # Runtime configuration
    runtime_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Runtime configuration"
    )

    # Private attribute for component cache
    _component: Optional[T] = PrivateAttr(default=None)

    # Configuration for model serialization
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create(self) -> T:
        """
        Create the runtime component.

        Returns:
            The created runtime component
        """
        # Return cached component if available
        if self._component is not None:
            return self._component

        # Resolve the engine
        engine = self.engine_ref.resolve()
        if not engine:
            raise ValueError(f"Failed to resolve engine reference: {self.engine_ref}")

        # Create the runtime component
        component = engine.create_runnable(self.runtime_config)

        # Cache the component
        self._component = component

        return component

    def invalidate_cache(self) -> None:
        """Invalidate the cached component."""
        self._component = None
        self.engine_ref.invalidate_cache()

    @classmethod
    def for_engine(
        cls, engine: Any, runtime_config: Optional[Dict[str, Any]] = None
    ) -> "ComponentFactory[T]":
        """
        Create a factory for an engine.

        Args:
            engine: Engine configuration
            runtime_config: Optional runtime configuration

        Returns:
            Factory for the engine
        """
        return cls(
            engine_ref=ComponentRef.from_engine(engine), runtime_config=runtime_config
        )
