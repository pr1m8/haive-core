# src/haive/core/engine/agent/persistence/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from .types import CheckpointerType

class CheckpointerConfig(BaseModel, ABC):
    """
    Base configuration for agent state persistence.
    
    This abstract class defines the interface that all checkpointer
    configurations must implement.
    """
    type: CheckpointerType
    setup_needed: bool = Field(default=True, description="Whether to initialize storage on first use")
    
    @abstractmethod
    def create_checkpointer(self) -> Any:
        """
        Create a checkpointer instance based on this configuration.
        
        Returns:
            A checkpointer instance compatible with LangGraph
        """
        pass
    
    def register_thread(self, thread_id: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a thread in the persistence system.
        
        Args:
            thread_id: Unique identifier for the thread
            name: Optional human-readable name for the thread
            metadata: Optional metadata to associate with the thread
        """
        pass
    
    def put_checkpoint(self, config: Dict[str, Any], data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a checkpoint in the persistence system.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            data: The checkpoint data to store
            metadata: Optional metadata to associate with the checkpoint
            
        Returns:
            Updated config with checkpoint_id
        """
        pass
    
    def get_checkpoint(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a checkpoint from the persistence system.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            
        Returns:
            The checkpoint data if found, None otherwise
        """
        pass
    
    def list_checkpoints(self, config: Dict[str, Any], limit: Optional[int] = None) -> List[Tuple[Dict[str, Any], Any]]:
        """
        List checkpoints for a thread.
        
        Args:
            config: Configuration with thread_id
            limit: Optional maximum number of checkpoints to return
            
        Returns:
            List of (config, checkpoint) tuples
        """
        return []
    
    def close(self) -> None:
        """
        Close any resources associated with this checkpointer.
        
        This method should be called when the checkpointer is no longer needed.
        """
        pass