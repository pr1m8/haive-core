from typing import Optional, Dict, Any, Union
from pydantic import Field
import logging

from haive.core.persistence.base import CheckpointerConfig
from haive.core.persistence.types import (
    CheckpointerType, 
    CheckpointerMode,
    CheckpointStorageMode
)

logger = logging.getLogger(__name__)

class MemoryCheckpointerConfig(CheckpointerConfig[Dict[str, Any]]):
    """
    Configuration for in-memory checkpoint persistence.
    
    This implementation provides a simple non-persistent memory-based
    checkpointer suitable for development and testing.
    """
    type: CheckpointerType = CheckpointerType.MEMORY
    mode: CheckpointerMode = Field(
        default=CheckpointerMode.SYNC,
        description="Memory checkpointer supports both sync and async modes"
    )
    storage_mode: CheckpointStorageMode = Field(
        default=CheckpointStorageMode.FULL,
        description="Storage mode - memory checkpointer always stores full history"
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
    
    def create_checkpointer(self) -> Any:
        """
        Create a synchronous memory checkpointer.
        
        Returns:
            MemorySaver instance
        """
        try:
            from langgraph.checkpoint.memory import MemorySaver
            
            # Create checkpointer
            checkpointer = MemorySaver()
            
            logger.info("Memory checkpointer created successfully")
            return checkpointer
            
        except Exception as e:
            logger.error(f"Failed to create memory checkpointer: {e}")
            raise RuntimeError(f"Failed to create memory checkpointer: {e}")
    
    async def create_async_checkpointer(self) -> Any:
        """
        Create an asynchronous memory checkpointer.
        
        For memory checkpointers, we just return the synchronous version
        since it's thread-safe and can be used in async contexts.
        
        Returns:
            MemorySaver instance
        """
        # Force async mode
        self.mode = CheckpointerMode.ASYNC
        
        # Return the regular memory saver - it's thread-safe for async use
        return self.create_checkpointer()
    
    async def initialize_async_checkpointer(self) -> Any:
        """
        Initialize an async checkpointer.
        
        For memory checkpointers, we simply return the checkpointer directly
        as there are no resources to manage with an async context.
        
        Returns:
            MemorySaver instance
        """
        # Simply create and return
        return await self.create_async_checkpointer()