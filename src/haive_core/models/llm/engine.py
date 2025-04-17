"""LLM engine implementation.

This module provides the LLM engine implementation that builds on the core engine
structure. It handles model initialization, text generation, and cleanup.
"""

from __future__ import annotations
from typing import Any, Optional

from haive_core.engine.base import CoreEngine, EngineConfig
from haive_core.models.llm.base import LLMConfig

class LLMEngine(CoreEngine[LLMConfig]):
    """LLM engine implementation.
    
    This class provides methods for:
    - Initializing an LLM model from src.configuration
    - Generating text from prompts
    - Streaming text generation
    - Cleaning up resources
    
    Example:
        >>> config = LLMConfig(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     api_key="sk-..."
        ... )
        >>> engine = LLMEngine(config)
        >>> response = engine.generate("Hello, world!")
    """
    def __init__(self, config: LLMConfig):
        """Create a new LLM engine.
        
        Args:
            config (LLMConfig): Configuration for the LLM model
        """
        super().__init__(config)
    
    def initialize(self) -> None:
        """Initialize the LLM model if not already initialized."""
        if self._model is None:
            self._model = self.config.instantiate_llm()
    
    def cleanup(self) -> None:
        """Clean up resources used by the LLM model."""
        self._model = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
        """
        self.initialize()
        response = self._model.invoke(prompt)
        return response.content
    
    def generate_stream(self, prompt: str, **kwargs):
        """Generate text from a prompt in streaming mode.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            str: Generated text chunks
        """
        self.initialize()
        for chunk in self._model.stream(prompt):
            yield chunk.content
    
    def get_model(self) -> Optional[Any]:
        """Get the underlying LLM model.
        
        Returns:
            Optional[Any]: The LLM model instance, or None if not initialized
        """
        return self._model 