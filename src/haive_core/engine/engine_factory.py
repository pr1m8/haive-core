# src/haive/core/engine_factory.py

from typing import Any, Dict, List, Optional, Type, Union
import importlib

from haive.core.engine.engine_dep import Engine, LLMEngine, RunnableEngine, AgentEngine
from haive_core.models.llm.base import LLMConfig
from haive_core.engine.aug_llm import AugLLMConfig


def create_engine(config: Any) -> Engine:
    """
    Create an appropriate engine from a configuration object.
    
    Args:
        config: Configuration object (LLMConfig, AugLLMConfig, AgentConfig)
        
    Returns:
        An engine instance
    """
    # Import here to avoid circular imports
    from haive_core.models.llm.base import LLMConfig
    from haive_core.engine.aug_llm import AugLLMConfig
    
    if isinstance(config, LLMConfig):
        return LLMEngine.from_config(config)
    if isinstance(config, AugLLMConfig):
        return RunnableEngine.from_config(config)
    if hasattr(config, 'build_agent'):
        # Check if we're dealing with an AgentConfig by looking for build_agent method
        # We don't import AgentConfig directly to avoid circular dependency
        agent = config.build_agent()
        return AgentEngine(agent)
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")