from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Generic, TypeVar, Union, Type
from pydantic import BaseModel, Field, SecretStr

from haive.core.persistence.types import (
    CheckpointerType, 
    CheckpointerMode,
    CheckpointStorageMode, 
    ConnectionOptions
)

# Type variable for different connection types
T = TypeVar('T')

class CheckpointerConfig(BaseModel, ABC, Generic[T]):
    """
    Base configuration for checkpoint persistence.
    
    This abstract base class defines the interface for all checkpointer
    configurations in the Haive framework. Implementations must provide
    concrete methods for creating actual checkpointer instances.
    """
    type: CheckpointerType = Field(
        description="Type of checkpointer to use"
    )
    mode: CheckpointerMode = Field(
        default=CheckpointerMode.SYNC,
        description="Operational mode - synchronous or asynchronous"
    )
    storage_mode: CheckpointStorageMode = Field(
        default=CheckpointStorageMode.FULL,
        description="Storage mode - full history or shallow (latest only)"
    )
    setup_needed: bool = Field(
        default=True,
        description="Whether tables need to be setup on first use"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def create_checkpointer(self) -> Any:
        """
        Create a checkpointer instance based on this configuration.
        
        Returns:
            A configured checkpointer instance
        """
        pass
    
    @abstractmethod
    def create_async_checkpointer(self) -> Any:
        """
        Create an asynchronous checkpointer instance.
        
        Returns:
            A configured async checkpointer instance
        """
        pass