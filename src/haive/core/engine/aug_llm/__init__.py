"""
AugLLM module for creating enhanced LLM chains.

This module provides a configuration system for building LLM chains with
prompts, tools, output parsers, and more.
"""

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.engine.aug_llm.factory import AugLLMFactory
from haive.core.engine.aug_llm.utils import (
    compose_runnable, 
    create_runnables_dict, 
    compose_runnables_from_dict,
    chain_runnables,
    merge_configs
)

__all__ = [
    "AugLLMConfig",
    "AugLLMFactory",
    "compose_runnable",
    "create_runnables_dict",
    "compose_runnables_from_dict",
    "chain_runnables",
    "merge_configs"
]