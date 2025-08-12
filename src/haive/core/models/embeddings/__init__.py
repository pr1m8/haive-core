"""Haive Embeddings Module.

This module provides comprehensive abstractions and implementations for working with
text embedding models from various providers. Embeddings are vector representations
of text that capture semantic meaning, enabling similarity search, clustering,
and other NLP applications.

The module supports a wide range of embedding providers with a consistent interface
for configuration and use.

Supported Cloud Providers:
    - Azure OpenAI: Microsoft's hosted OpenAI embedding models
    - OpenAI: Direct OpenAI embedding models
    - Cohere: Specialized embedding models from Cohere
    - Jina AI: Jina AI embedding models
    - Google Vertex AI: Google Cloud's machine learning platform
    - AWS Bedrock: Amazon's foundation model service
    - Cloudflare Workers AI: Cloudflare's AI model hosting
    - Voyage AI: Specialized embedding models from Voyage AI
    - Anyscale: Anyscale embedding models

Supported Local/Self-hosted Providers:
    - HuggingFace: Open-source embedding models from the HuggingFace model hub
    - SentenceTransformers: Efficient sentence embedding models
    - FastEmbed: Lightweight embedding models optimized for CPU
    - Ollama: Local embedding models via Ollama
    - LlamaCpp: Local embedding models via llama.cpp

Key Components:
    - Base Classes: Abstract base classes for embedding configurations
    - Provider Types: Enumeration of supported embedding providers
    - Provider Implementations: Provider-specific configuration classes
    - Factory Functions: Simplified creation of embedding instances
    - Security: Secure handling of API keys with environment variable resolution
    - Caching: Efficient caching of embeddings for performance optimization

Typical usage example::

    from haive.core.models.embeddings import create_embeddings, OpenAIEmbeddingConfig

    # Configure an embedding model
    config = OpenAIEmbeddingConfig(
        model="text-embedding-3-small"
    )

    # Create the embeddings
    embeddings = create_embeddings(config)

    # Generate embeddings
    doc_vectors = embeddings.embed_documents(["Document text"])
    query_vector = embeddings.embed_query("Query text")
"""

from haive.core.models.embeddings.base import (  # Base classes; Cloud providers; Local providers; Factory function
    AnyscaleEmbeddingConfig,
    AzureEmbeddingConfig,
    BaseEmbeddingConfig,
    BedrockEmbeddingConfig,
    CloudflareEmbeddingConfig,
    CohereEmbeddingConfig,
    FastEmbedEmbeddingConfig,
    HuggingFaceEmbeddingConfig,
    JinaEmbeddingConfig,
    LlamaCppEmbeddingConfig,
    OllamaEmbeddingConfig,
    OpenAIEmbeddingConfig,
    SecureConfigMixin,
    SentenceTransformerEmbeddingConfig,
    VertexAIEmbeddingConfig,
    VoyageAIEmbeddingConfig,
    create_embeddings,
)

# Import enum and individual values for backward compatibility
from haive.core.models.embeddings.provider_types import EmbeddingProvider

# Create individual constants for backward compatibility
ANYSCALE = EmbeddingProvider.ANYSCALE
AZURE = EmbeddingProvider.AZURE
BEDROCK = EmbeddingProvider.BEDROCK
CLOUDFLARE = EmbeddingProvider.CLOUDFLARE
COHERE = EmbeddingProvider.COHERE
FASTEMBED = EmbeddingProvider.FASTEMBED
HUGGINGFACE = EmbeddingProvider.HUGGINGFACE
JINA = EmbeddingProvider.JINA
LLAMACPP = EmbeddingProvider.LLAMACPP
NOVITA = EmbeddingProvider.NOVITA
OLLAMA = EmbeddingProvider.OLLAMA
OPENAI = EmbeddingProvider.OPENAI
SENTENCE_TRANSFORMERS = EmbeddingProvider.SENTENCE_TRANSFORMERS
VERTEXAI = EmbeddingProvider.VERTEXAI
VOYAGEAI = EmbeddingProvider.VOYAGEAI

# Import test functions if available
try:
    from haive.core.models.embeddings.test_embeddings import (
        TestEmbeddingProviders,
        test_config_classes_exist,
        test_factory_function,
        test_provider_enum_values,
    )
except ImportError:
    # Test functions not available
    def test_config_classes_exist():
        """Placeholder test function."""

    def test_factory_function():
        """Placeholder test function."""

    def test_provider_enum_values():
        """Placeholder test function."""

    class TestEmbeddingProviders:
        """Placeholder test class."""


__all__ = [
    "ANYSCALE",
    "AZURE",
    "BEDROCK",
    "CLOUDFLARE",
    "COHERE",
    "FASTEMBED",
    "HUGGINGFACE",
    "JINA",
    "LLAMACPP",
    "NOVITA",
    "OLLAMA",
    "OPENAI",
    "SENTENCE_TRANSFORMERS",
    "VERTEXAI",
    "VOYAGEAI",
    "AnyscaleEmbeddingConfig",
    "AzureEmbeddingConfig",
    "BaseEmbeddingConfig",
    "BedrockEmbeddingConfig",
    "CloudflareEmbeddingConfig",
    "CohereEmbeddingConfig",
    "EmbeddingProvider",
    "FastEmbedEmbeddingConfig",
    "HuggingFaceEmbeddingConfig",
    "JinaEmbeddingConfig",
    "LlamaCppEmbeddingConfig",
    "OllamaEmbeddingConfig",
    "OpenAIEmbeddingConfig",
    "SecureConfigMixin",
    "SentenceTransformerEmbeddingConfig",
    "TestEmbeddingProviders",
    "VertexAIEmbeddingConfig",
    "VoyageAIEmbeddingConfig",
    "create_embeddings",
    "test_config_classes_exist",
    "test_factory_function",
    "test_provider_enum_values",
]
