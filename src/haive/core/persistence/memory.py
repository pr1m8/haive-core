# src/haive/core/engine/agent/persistence/memory.py

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from .types import CheckpointerType
from .base import CheckpointerConfig

# Try to import LangGraph's memory saver
try:
    from langgraph.checkpoint.memory import MemorySaver
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryCheckpointerConfig(CheckpointerConfig):
    """
    Configuration for in-memory state persistence.
    
    This provides a simple in-memory implementation that stores
    checkpoints in memory, which is useful for testing and
    development but doesn't persist data across restarts.
    """
    type: CheckpointerType = CheckpointerType.memory
    
    # Set to store registered threads with metadata
    _threads: Dict[str, Dict[str, Any]] = Field(default_factory=dict, exclude=True)
    
    # Dictionary to store checkpoints by thread_id and checkpoint_id
    _checkpoints: Dict[str, Dict[str, Dict[str, Any]]] = Field(default_factory=dict, exclude=True)
    
    # Internal state
    _checkpointer: Optional[Any] = Field(default=None, exclude=True)
    
    def create_checkpointer(self) -> Any:
        """
        Create a memory-based checkpointer.
        
        Returns:
            A MemorySaver instance from LangGraph
        """
        if not self._checkpointer:
            if MEMORY_AVAILABLE:
                self._checkpointer = MemorySaver()
            else:
                # Simple fallback if LangGraph is not available
                self._checkpointer = self  # Use self as the checkpointer
        
        return self._checkpointer
    
    def register_thread(self, thread_id: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a thread in memory.
        
        Args:
            thread_id: Unique identifier for the thread
            name: Optional human-readable name for the thread
            metadata: Optional metadata to associate with the thread
        """
        if thread_id not in self._threads:
            self._threads[thread_id] = {
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {},
                "name": name
            }
            logger.info(f"Thread {thread_id} registered in memory")
            
            # Initialize checkpoint storage for this thread
            if thread_id not in self._checkpoints:
                self._checkpoints[thread_id] = {}
        else:
            logger.debug(f"Thread {thread_id} already exists in memory")
    
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
        # For LangGraph MemorySaver
        if MEMORY_AVAILABLE and isinstance(self._checkpointer, MemorySaver):
            # Create a checkpoint structure with correct fields
            checkpoint_data = {
                "id": config["configurable"].get("checkpoint_id", str(uuid.uuid4())),
                "channel_values": data
            }
            
            # Get versions parameter required by newer versions
            try:
                # Try get method signature to determine needed parameters
                import inspect
                sig = inspect.signature(self._checkpointer.put)
                if "new_versions" in sig.parameters:
                    # New API
                    result = self._checkpointer.put(
                        config, 
                        checkpoint_data, 
                        metadata or {}, 
                        {}  # Empty versions dict
                    )
                else:
                    # Old API
                    result = self._checkpointer.put(config, checkpoint_data)
                return result
            except Exception as e:
                logger.error(f"Error storing checkpoint: {e}")
                return config
        
        # Simple in-memory implementation
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        # Generate a new checkpoint ID if none was provided
        if not checkpoint_id:
            checkpoint_id = str(uuid.uuid4())
            
        # Make sure the thread exists
        if thread_id not in self._threads:
            self.register_thread(thread_id)
            
        # Store the checkpoint
        self._checkpoints[thread_id][checkpoint_id] = {
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Return updated config with checkpoint_id
        updated_config = config.copy()
        updated_config["configurable"]["checkpoint_id"] = checkpoint_id
        return updated_config
    
    def get_checkpoint(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a checkpoint from memory.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            
        Returns:
            The checkpoint data if found, None otherwise
        """
        # For LangGraph MemorySaver
        if MEMORY_AVAILABLE and isinstance(self._checkpointer, MemorySaver):
            return self._checkpointer.get(config)
            
        # Simple in-memory implementation
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        # If no specific checkpoint was requested, return the latest
        if not checkpoint_id and thread_id in self._checkpoints:
            checkpoints = self._checkpoints[thread_id]
            if not checkpoints:
                return None
                
            # Find the latest checkpoint by timestamp
            latest = None
            latest_timestamp = None
            
            for cp_id, cp_data in checkpoints.items():
                if latest_timestamp is None or cp_data["timestamp"] > latest_timestamp:
                    latest = cp_data["data"]
                    latest_timestamp = cp_data["timestamp"]
                    
            return latest
            
        # Return the specific checkpoint if it exists
        if thread_id in self._checkpoints and checkpoint_id in self._checkpoints[thread_id]:
            return self._checkpoints[thread_id][checkpoint_id]["data"]
            
        return None
    
    def list_checkpoints(self, config: Dict[str, Any], limit: Optional[int] = None) -> List[Tuple[Dict[str, Any], Any]]:
        """
        List checkpoints for a thread.
        
        Args:
            config: Configuration with thread_id
            limit: Optional maximum number of checkpoints to return
            
        Returns:
            List of (config, checkpoint) tuples
        """
        # For LangGraph MemorySaver
        if MEMORY_AVAILABLE and isinstance(self._checkpointer, MemorySaver):
            try:
                checkpoint_tuples = list(self._checkpointer.list(config, limit=limit))
                return [(cp.config, cp.checkpoint) for cp in checkpoint_tuples]
            except Exception as e:
                logger.error(f"Error listing checkpoints: {e}")
                return []
        
        # Simple in-memory implementation
        thread_id = config["configurable"]["thread_id"]
        
        if thread_id not in self._checkpoints:
            return []
            
        # Sort checkpoints by timestamp (newest first)
        checkpoints = []
        for cp_id, cp_data in self._checkpoints[thread_id].items():
            checkpoints.append((
                {"configurable": {"thread_id": thread_id, "checkpoint_id": cp_id}},
                cp_data["data"]
            ))
            
        # Sort by timestamp
        checkpoints.sort(key=lambda x: self._checkpoints[thread_id][x[0]["configurable"]["checkpoint_id"]]["timestamp"], reverse=True)
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            checkpoints = checkpoints[:limit]
            
        return checkpoints
    
    def close(self) -> None:
        """No-op for memory checkpointer since there are no resources to close."""
        pass