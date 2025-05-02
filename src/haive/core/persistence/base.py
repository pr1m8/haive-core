from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

from pydantic import BaseModel, Field

from haive.core.persistence.types import (
    CheckpointerMode,
    CheckpointerType,
    CheckpointStorageMode,
)

# Type variable for different connection types
T = TypeVar("T")


class CheckpointerConfig(BaseModel, ABC, Generic[T]):
    """
    Base configuration for checkpoint persistence.

    This abstract base class defines the interface for all checkpointer
    configurations in the Haive framework. Implementations must provide
    concrete methods for creating actual checkpointer instances.
    """

    type: CheckpointerType = Field(description="Type of checkpointer to use")
    mode: CheckpointerMode = Field(
        default=CheckpointerMode.SYNC,
        description="Operational mode - synchronous or asynchronous",
    )
    storage_mode: CheckpointStorageMode = Field(
        default=CheckpointStorageMode.FULL,
        description="Storage mode - full history or shallow (latest only)",
    )
    setup_needed: bool = Field(
        default=True, description="Whether tables need to be setup on first use"
    )

    class Config:
        arbitrary_types_allowed = True

    def is_async_mode(self) -> bool:
        """
        Check if operating in async mode.

        Returns:
            True if async mode, False otherwise
        """
        return self.mode == CheckpointerMode.ASYNC

    @abstractmethod
    def create_checkpointer(self) -> Any:
        """
        Create a synchronous checkpointer instance based on this configuration.

        Returns:
            A configured checkpointer instance
        """
        pass

    @abstractmethod
    async def create_async_checkpointer(self) -> Any:
        """
        Create an asynchronous checkpointer instance.

        Returns:
            A configured async checkpointer instance
        """
        pass

    @abstractmethod
    async def initialize_async_checkpointer(self) -> Any:
        """
        Initialize an async checkpointer with proper resource management.

        This method should return an async context manager that handles
        the lifecycle of any resources needed by the checkpointer.

        Returns:
            An async context manager or the checkpointer itself
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        # Use Pydantic v2 serialization if available
        if hasattr(self, "model_dump"):
            # Exclude SecretStr fields for security
            return self.model_dump(exclude={"db_pass"})

        # Fall back to Pydantic v1
        return self.dict(exclude={"db_pass"})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointerConfig":
        """
        Create a configuration from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Instantiated checkpointer configuration
        """
        # Ensure type is specified
        if "type" not in data:
            raise ValueError("Checkpointer type must be specified")

        # Get appropriate subclass based on type
        checkpointer_type = data["type"]
        if checkpointer_type == CheckpointerType.MEMORY:
            from haive.core.persistence.memory import MemoryCheckpointerConfig

            return MemoryCheckpointerConfig(**data)
        elif checkpointer_type == CheckpointerType.POSTGRES:
            from haive.core.persistence.postgres_config import (
                PostgresCheckpointerConfig,
            )

            return PostgresCheckpointerConfig(**data)
        else:
            raise ValueError(f"Unsupported checkpointer type: {checkpointer_type}")
