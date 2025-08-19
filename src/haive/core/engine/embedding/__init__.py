"""Embedding engine module for the Haive framework.

This module provides comprehensive embedding functionality with support for
multiple providers including OpenAI, Azure OpenAI, HuggingFace, Cohere,
Google Vertex AI, Ollama, and more.

The module follows a factory pattern where embedding configurations are created
through provider-specific config classes, then instantiated to create the actual
embedding instances. All providers are lazily loaded to minimize startup time.

Attributes:
    BaseEmbeddingConfig: Base class for all embedding configurations.
    EmbeddingConfigFactory: Factory for creating embedding configurations.
    EmbeddingType: Enum of supported embedding providers.
    create_embedding_config: Factory function for creating configurations.
    providers: Lazily loaded providers module.

Examples:
    Basic usage with OpenAI embeddings::

        from haive.core.engine.embedding import BaseEmbeddingConfig
        from haive.core.engine.embedding.providers import OpenAIEmbeddingConfig

        # Create configuration
        config = OpenAIEmbeddingConfig(
            name="my_embeddings",
            model="text-embedding-3-large"
        )

        # Instantiate embeddings
        embeddings = config.instantiate()

        # Embed text
        vectors = embeddings.embed_documents(["Hello world", "How are you?"])
        query_vector = embeddings.embed_query("Hello")

    Configuration discovery and dynamic provider selection::

        from haive.core.engine.embedding import BaseEmbeddingConfig

        # List all registered providers
        available_providers = BaseEmbeddingConfig.list_registered_types()
        print(f"Available providers: {available_providers}")

        # Get specific provider class
        provider_class = BaseEmbeddingConfig.get_config_class("OpenAI")
        config = provider_class(name="dynamic_embeddings")

    Using the factory function for simplified creation::

        from haive.core.engine.embedding import create_embedding_config

        config = create_embedding_config(
            provider="OpenAI",
            model="text-embedding-3-large",
            name="my_embeddings"
        )

        embeddings = config.instantiate()

Note:
    All provider classes are lazily loaded through the `providers` attribute
    to avoid import overhead during module initialization. This allows the
    framework to start quickly even when many providers are available.
"""

# Lazy loading - providers imported on first use to avoid startup overhead
from haive.core.engine.embedding.base import BaseEmbeddingConfig
from haive.core.engine.embedding.config import EmbeddingConfigFactory, create_embedding_config
from haive.core.engine.embedding.types import EmbeddingType


def __getattr__(name: str):
    """Lazy load providers to avoid import-time registration overhead.

    This function enables lazy loading of the providers module, which contains
    all embedding provider configurations. By deferring the import until first
    access, we significantly reduce module initialization time.

    Args:
        name: The attribute name being accessed.

    Returns:
        The requested module or attribute. Currently supports:
        - "providers": The providers module containing all embedding configs.

    Raises:
        AttributeError: If the requested attribute doesn't exist.

    Note:
        The imported module is cached in globals() to avoid repeated imports
        on subsequent accesses.
    """
    if name == "providers":
        import importlib

        # Use importlib.import_module to avoid __getattr__ recursion
        providers_module = importlib.import_module(f"{__name__}.providers")
        # Cache in globals to avoid re-importing
        globals()["providers"] = providers_module
        return providers_module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseEmbeddingConfig",
    "EmbeddingConfigFactory",
    "EmbeddingType",
    "create_embedding_config",
    "providers",
]
