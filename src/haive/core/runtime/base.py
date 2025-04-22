# src/haive/core/runtime/base.py

from typing import Generic, TypeVar, Any, Optional, Dict
from abc import ABC, abstractmethod

from langchain_core.runnables import RunnableConfig
from haive.core.engine.engine_base import Engine

# Engine config type
EC = TypeVar('EC', bound=Engine)

class RuntimeComponent(Generic[EC]):
    """Base class for runtime components built from engine configs."""
    
    def __init__(self, config: EC, **kwargs):
        """Initialize with engine configuration.
        
        Args:
            config: Engine configuration
            **kwargs: Additional parameters
        """
        self.config = config
        self.initialize(**kwargs)
    
    def initialize(self, **kwargs):
        """Initialize the component.
        
        This method can be overridden by subclasses.
        
        Args:
            **kwargs: Additional parameters
        """
        pass
    
    @abstractmethod
    def invoke(self, input_data: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        """Invoke the component.
        
        Args:
            input_data: Input data
            config: Optional runtime configuration
            **kwargs: Additional parameters
            
        Returns:
            Component output
        """
        pass
    
    async def ainvoke(self, input_data: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        """Asynchronously invoke the component.
        
        By default, this calls invoke in a thread. Override for true async implementation.
        
        Args:
            input_data: Input data
            config: Optional runtime configuration
            **kwargs: Additional parameters
            
        Returns:
            Component output
        """
        import asyncio
        return await asyncio.to_thread(self.invoke, input_data, config, **kwargs)