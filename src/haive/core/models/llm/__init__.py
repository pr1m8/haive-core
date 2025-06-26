"""
Haive LLM Module.

This module provides comprehensive abstractions and implementations for working with
Large Language Models (LLMs) from various providers. It includes configuration classes,
provider-specific implementations, and utilities for model metadata.

The module supports a wide range of LLM providers including OpenAI, Anthropic, Google,
Azure, Mistral, and many others, with a consistent interface for configuration and use.

Key Components:
    - Base Classes: Abstract base classes for LLM configurations
    - Provider Types: Enumeration of supported LLM providers
    - Provider Implementations: Provider-specific configuration classes
    - Metadata: Utilities for accessing model capabilities and context windows

Typical usage example:
    ```python
    from haive.core.models.llm.base import OpenAILLMConfig

    # Configure an LLM
    config = OpenAILLMConfig(
        model="gpt-4",
        cache_enabled=True
    )

    # Instantiate the LLM
    llm = config.instantiate()

    # Generate text
    response = llm.generate("Explain quantum computing")
    ```
"""

from haive.core.models.llm.base import (
    AI21LLMConfig,
    AlephAlphaLLMConfig,
    AnthropicLLMConfig,
    AzureLLMConfig,
    CohereLLMConfig,
    DeepSeekLLMConfig,
    FireworksAILLMConfig,
    GeminiLLMConfig,
    GooseAILLMConfig,
    GroqLLMConfig,
    HuggingFaceLLMConfig,
    LLMConfig,
    MistralLLMConfig,
    MosaicMLLLMConfig,
    NLPCloudLLMConfig,
    OpenAILLMConfig,
    OpenLMLLMConfig,
    PerplexityLLMConfig,
    PetalsLLMConfig,
    ReplicateLLMConfig,
    TogetherAILLMConfig,
    VertexAILLMConfig,
)
from haive.core.models.llm.provider_types import LLMProvider

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "AzureLLMConfig",
    "OpenAILLMConfig",
    "AnthropicLLMConfig",
    "GeminiLLMConfig",
    "DeepSeekLLMConfig",
    "MistralLLMConfig",
    "GroqLLMConfig",
    "CohereLLMConfig",
    "TogetherAILLMConfig",
    "FireworksAILLMConfig",
    "PerplexityLLMConfig",
    "HuggingFaceLLMConfig",
    "AI21LLMConfig",
    "AlephAlphaLLMConfig",
    "GooseAILLMConfig",
    "MosaicMLLLMConfig",
    "NLPCloudLLMConfig",
    "OpenLMLLMConfig",
    "PetalsLLMConfig",
    "ReplicateLLMConfig",
    "VertexAILLMConfig",
]
