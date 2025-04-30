"""
Utility functions for working with AugLLMConfig instances.

Provides helper functions for composing, managing, and utilizing LLM runnables.
"""

import logging
from typing import Any, Dict, List, Optional, Union, TypeVar, cast

from haive.core.engine.aug_llm.config import AugLLMConfig

# Type for runnable configs
T = TypeVar('T')

from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)


def compose_runnable(aug_llm_config: AugLLMConfig, runnable_config: Optional[RunnableConfig] = None) -> Runnable:
    """
    Compose a runnable from an AugLLMConfig.
    
    Args:
        aug_llm_config: Configuration for the LLM chain
        runnable_config: Optional runtime configuration
        
    Returns:
        A runnable LLM chain
    """
    try:
        return aug_llm_config.create_runnable(runnable_config)
    except Exception as e:
        logger.error(f"Error composing runnable: {e}")
        raise


def create_runnables_dict(runnables: Union[List[AugLLMConfig], Dict[str, AugLLMConfig], AugLLMConfig]) -> Dict[str, AugLLMConfig]:
    """
    Create a dictionary mapping names to runnable configs.
    
    Args:
        runnables: List of AugLLMConfig objects or dictionary of name->AugLLMConfig
        
    Returns:
        Dictionary mapping names to configs
    """
    # Handle list input
    if isinstance(runnables, list):
        return {runnable.name: runnable for runnable in runnables}
        
    # Handle dictionary input
    if isinstance(runnables, dict):
        return runnables
        
    # Handle single runnable
    if hasattr(runnables, "name"):
        return {runnables.name: runnables}
        
    # Fallback
    logger.warning(f"Unsupported input type for create_runnables_dict: {type(runnables)}")
    return {}


def compose_runnables_from_dict(
    runnables: Dict[str, Union[AugLLMConfig, Any]],
    runnable_config: Optional[RunnableConfig] = None
) -> Dict[str, Runnable]:
    """
    Compose and return a dictionary of runnables from configs.
    
    Args:
        runnables: Dictionary mapping names to configs
        runnable_config: Optional runtime configuration
        
    Returns:
        Dictionary mapping names to runnables
    """
    result = {}
    for key, config in runnables.items():
        try:
            if hasattr(config, "create_runnable"):
                result[key] = compose_runnable(config, runnable_config)
            else:
                logger.warning(f"Config for {key} does not have create_runnable method")
        except Exception as e:
            logger.error(f"Error composing runnable for {key}: {e}")
            
    return result


def chain_runnables(
    runnables: List[Runnable], 
    override_config: Optional[Dict[str, Any]] = None
) -> Runnable:
    """
    Chain multiple runnables together with optional configuration.
    
    Args:
        runnables: List of runnables to chain
        override_config: Optional configuration to apply to the chained runnable
        
    Returns:
        Chained runnable
    """
    if not runnables:
        raise ValueError("No runnables provided to chain")
        
    if len(runnables) == 1:
        return runnables[0]
        
    from langchain_core.runnables import chain
    chained = chain(*runnables)
    
    if override_config:
        chained = chained.with_config(**override_config)
        
    return chained


def merge_configs(base_config: Optional[RunnableConfig], override_config: Optional[RunnableConfig]) -> RunnableConfig:
    """
    Merge two runnable configs, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    # Handle None cases
    if not base_config:
        return override_config or {}
    if not override_config:
        return base_config
        
    # Copy base config to avoid modifying the original
    import copy
    result = copy.deepcopy(base_config)
    
    # Ensure configurable section exists
    if "configurable" not in result:
        result["configurable"] = {}
        
    # Merge configurable section
    if "configurable" in override_config:
        for key, value in override_config["configurable"].items():
            # Special handling for engine_configs section
            if key == "engine_configs" and "engine_configs" in result["configurable"]:
                if not result["configurable"]["engine_configs"]:
                    result["configurable"]["engine_configs"] = {}
                    
                # Merge engine configs
                for engine_name, engine_config in override_config["configurable"]["engine_configs"].items():
                    if engine_name in result["configurable"]["engine_configs"]:
                        # Update existing engine config
                        result["configurable"]["engine_configs"][engine_name].update(engine_config)
                    else:
                        # Add new engine config
                        result["configurable"]["engine_configs"][engine_name] = engine_config
            else:
                # Regular key override
                result["configurable"][key] = value
                
    # Handle other top-level sections
    for key, value in override_config.items():
        if key != "configurable":
            result[key] = value
            
    return result