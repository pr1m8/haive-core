"""Contract adapters for existing engines.

This module provides adapters that add contracts to existing engines
without modifying their implementation.
"""

from haive.core.contracts.adapters.aug_llm_adapter import (
    ContractualAugLLMConfig,
    AugLLMContract,
)

__all__ = [
    "ContractualAugLLMConfig",
    "AugLLMContract",
]