"""Core models module for the Haive framework.

This module provides a comprehensive set of model abstractions and implementations
for working with large language models (LLMs), embeddings, retrievers, and vector
stores. The models are designed to be modular, extensible, and optimized for
use within the Haive agent ecosystem.

Key Components:
    - LLM: Large Language Model abstractions and provider implementations
    - Embeddings: Text embedding models for vector representations with multiple providers
    - Retrievers: Components for retrieving relevant information with various strategies
    - VectorStores: Storage systems for embedding vectors with similarity search capabilities
    - Metadata: Utilities for working with model metadata and configuration

Provider Architecture:
    Each model type follows a consistent provider pattern:

    1. **Provider Enums**: Define available providers (e.g., LLMProvider, EmbeddingProvider)
    2. **Base Config Classes**: Abstract configuration classes with common functionality
    3. **Provider Configs**: Specific configuration classes for each provider
    4. **Factory Functions**: Utility functions for creating instances

    The provider enums (LLMProvider, EmbeddingProvider, VectorStoreProvider, RetrieverType)
    serve as the "engines" that drive the selection and instantiation of specific implementations.

Supported Providers:
    **LLM Providers**:
        - OpenAI (GPT-4, GPT-3.5-turbo, etc.)
        - Anthropic (Claude family)
        - Azure OpenAI
        - Google (Gemini, PaLM)
        - AWS Bedrock
        - Mistral AI
        - Local models via Ollama, HuggingFace

    **Embedding Providers**:
        - OpenAI (text-embedding-3-small, text-embedding-3-large)
        - Azure OpenAI embeddings
        - HuggingFace Transformers
        - Sentence Transformers
        - Cohere embeddings
        - Local embeddings via FastEmbed, Ollama

    **Vector Store Providers**:
        - Chroma (local and server modes)
        - Pinecone (managed cloud service)
        - FAISS (Facebook AI Similarity Search)
        - Qdrant (vector search engine)
        - Weaviate (open-source vector database)
        - In-memory stores for development

    **Retriever Types**:
        - Vector store retrievers (semantic search)
        - Hybrid retrievers (combining multiple strategies)
        - Time-weighted retrievers (considering document recency)
        - Multi-query retrievers (expanding queries)
        - Ensemble retrievers (combining multiple retrievers)

Typical usage examples:
    Basic LLM configuration::

        from haive.core.models import LLMConfig, LLMProvider

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.7
        )
        llm = config.instantiate()

    Embedding model setup::

        from haive.core.models import OpenAIEmbeddingConfig

        embeddings = OpenAIEmbeddingConfig(
            model="text-embedding-3-small"
        ).instantiate()

    Vector store with custom embeddings::

        from haive.core.models import VectorStoreConfig, VectorStoreProvider

        vs_config = VectorStoreConfig(
            provider=VectorStoreProvider.Chroma,
            embedding_config=embeddings_config,
            collection_name="documents"
        )
        vectorstore = vs_config.instantiate()

    Complete RAG retriever setup::

        from haive.core.models import RetrieverConfig, RetrieverType

        retriever_config = RetrieverConfig(
            type=RetrieverType.VECTOR_STORE,
            vectorstore_config=vs_config,
            search_kwargs={"k": 5, "score_threshold": 0.7}
        )
        retriever = retriever_config.instantiate()

Performance Considerations:
    - Lazy Loading: Heavy dependencies (numpy, torch) are loaded only when needed
    - Connection Pooling: Automatic connection management for cloud providers
    - Caching: Built-in caching for embeddings and frequently accessed models
    - Async Support: Asynchronous operations for better concurrency
    - Resource Management: Proper cleanup and resource management

Advanced Features:
    - Rate Limiting: Built-in rate limiting for API-based providers
    - Retry Logic: Automatic retry with exponential backoff for transient failures
    - Monitoring: Integration with observability tools for performance tracking
    - Security: Secure handling of API keys with environment variable resolution
    - Validation: Comprehensive configuration validation with clear error messages

.. autosummary::
   :toctree: generated/

   LLMConfig
   LLMProvider
   LLMFactory
   OpenAIEmbeddingConfig
   EmbeddingProvider
   VectorStoreConfig
   VectorStoreProvider
   RetrieverConfig
   RetrieverType
   ModelMetadata
   MetadataMixin
"""

# Embeddings module imports - with most commonly used configs
from haive.core.models.embeddings import (
    BaseEmbeddingConfig,
    EmbeddingProvider,
    HuggingFaceEmbeddingConfig,
    OpenAIEmbeddingConfig,
    create_embeddings,
)

# LLM module imports - comprehensive provider support
from haive.core.models.llm import (
    LLMConfig,
    LLMFactory,
    LLMProvider,
    create_llm,
)
from haive.core.models.llm import get_available_providers as get_llm_providers
from haive.core.models.llm import (
    get_provider_models,
)

# Core metadata utilities
from haive.core.models.metadata import ModelMetadata
from haive.core.models.metadata_mixin import ModelMetadataMixin as MetadataMixin

# Retriever imports - configuration and types
from haive.core.models.retriever import (
    RetrieverConfig,
    RetrieverType,
    VectorStoreRetrieverConfig,
)

# Vector store imports - core configuration and providers
from haive.core.models.vectorstore import (
    VectorStoreConfig,
    VectorStoreProvider,
)

# Note: Submodules are available as imports but heavy dependencies
# (torch, transformers, etc.) are only loaded when actually instantiated

__all__ = [
    # Metadata utilities
    "ModelMetadata",
    "MetadataMixin",
    # LLM components
    "LLMConfig",
    "LLMFactory",
    "LLMProvider",
    "create_llm",
    "get_llm_providers",
    "get_provider_models",
    # Embedding components
    "BaseEmbeddingConfig",
    "EmbeddingProvider",
    "HuggingFaceEmbeddingConfig",
    "OpenAIEmbeddingConfig",
    "create_embeddings",
    # Vector store components
    "VectorStoreConfig",
    "VectorStoreProvider",
    # Retriever components
    "RetrieverConfig",
    "RetrieverType",
    "VectorStoreRetrieverConfig",
]
