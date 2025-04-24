# src/haive/core/persistence/memory.py

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from .types import CheckpointerType, CheckpointTuple
from .base import CheckpointerConfig

logger = logging.getLogger(__name__)

class MemoryCheckpointerConfig(CheckpointerConfig):
    """
    Configuration for in-memory checkpointing.
    
    This class configures a simple in-memory checkpoint saver,
    useful for development, testing, or simple applications that don't need
    persistence between restarts.
    """
    type: CheckpointerType = CheckpointerType.memory
    
    # No additional configuration parameters needed for memory checkpointer
    setup_needed: bool = Field(default=False, description="No setup needed for memory checkpointer")
    
    # Internal state (not serialized)
    checkpointer: Optional[Any] = Field(default=None, exclude=True)
    
    def create_checkpointer(self) -> Any:
        """
        Create an in-memory checkpointer.
        
        Returns:
            A MemorySaver instance
        """
        try:
            # Import here to avoid circular imports
            from langgraph.checkpoint.memory import MemorySaver
            
            if self._checkpointer is None:
                logger.debug("Creating new in-memory checkpointer")
                self._checkpointer = MemorySaver()
            
            return self._checkpointer
            
        except ImportError:
            logger.error("Failed to import MemorySaver from langgraph")
            raise
    
    def register_thread(self, thread_id: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a thread in the in-memory store.
        
        For memory checkpointer, this is a no-op since threads are implicitly created.
        
        Args:
            thread_id: The thread ID to register
            name: Optional thread name
            metadata: Optional metadata dict
        """
        logger.debug(f"Thread registration not needed for memory checkpointer: {thread_id}")
        # No explicit registration needed for memory checkpointer
        pass
    
    def put_checkpoint(self, config: Dict[str, Any], data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a checkpoint in memory.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            data: The checkpoint data to store
            metadata: Optional metadata to associate with the checkpoint
            
        Returns:
            Updated config with checkpoint_id
        """
        checkpointer = self.create_checkpointer()
        
        # Structure data correctly for MemorySaver
        checkpoint_data = {
            "id": config["configurable"].get("checkpoint_id", ""),  # Will be auto-generated if empty
            "channel_values": data if isinstance(data, dict) else {"root": data}
        }
        
        # Prepare metadata
        checkpoint_metadata = metadata or {}
        
        # Empty channel versions (required by newer API)
        channel_versions = {}
        
        # Store the checkpoint
        try:
            updated_config = checkpointer.put(
                config,
                checkpoint_data,
                checkpoint_metadata,
                channel_versions
            )
            return updated_config
        except Exception as e:
            logger.error(f"Error storing checkpoint in memory: {e}")
            return config
    
    def get_checkpoint(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a checkpoint from memory.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            
        Returns:
            The checkpoint data if found, None otherwise
        """
        checkpointer = self.create_checkpointer()
        
        try:
            checkpoint = checkpointer.get(config)
            if checkpoint is None:
                return None
                
            # Return the channel_values as that's the agent state
            return checkpoint.get("channel_values", {})
        except Exception as e:
            logger.error(f"Error retrieving checkpoint from memory: {e}")
            return None
    
    def get_checkpoint_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """
        Retrieve a checkpoint tuple from memory.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            
        Returns:
            CheckpointTuple if found, None otherwise
        """
        checkpointer = self.create_checkpointer()
        
        try:
            result = checkpointer.get_tuple(config)
            if result is None:
                return None
                
            return CheckpointTuple(
                config=result.config,
                checkpoint=result.checkpoint,
                metadata=result.metadata,
                parent_config=result.parent_config,
                pending_writes=result.pending_writes
            )
        except Exception as e:
            logger.error(f"Error retrieving checkpoint tuple from memory: {e}")
            return None
    
    def list_checkpoints(self, config: Dict[str, Any], limit: Optional[int] = None) -> List[CheckpointTuple]:
        """
        List checkpoints from memory.
        
        Args:
            config: Configuration with thread_id
            limit: Optional maximum number of checkpoints to return
            
        Returns:
            List of CheckpointTuple objects
        """
        checkpointer = self.create_checkpointer()
        
        try:
            # Get checkpoint tuples from the checkpointer
            results = list(checkpointer.list(config, limit=limit))
            
            # Convert to our CheckpointTuple type
            return [
                CheckpointTuple(
                    config=result.config,
                    checkpoint=result.checkpoint,
                    metadata=result.metadata,
                    parent_config=result.parent_config,
                    pending_writes=result.pending_writes
                ) 
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error listing checkpoints from memory: {e}")
            return []
    
    def delete_thread(self, thread_id: str) -> None:
        """
        Delete all checkpoints for a thread from memory.
        
        Args:
            thread_id: Identifier for the thread to delete
        """
        checkpointer = self.create_checkpointer()
        
        try:
            checkpointer.delete_thread(thread_id)
            logger.info(f"Deleted thread {thread_id} from memory")
        except Exception as e:
            logger.error(f"Error deleting thread from memory: {e}")
    
    def close(self) -> None:
        """Close the checkpointer (no-op for memory checkpointer)."""
        # Memory checkpointer doesn't need closing
        self._checkpointer = None