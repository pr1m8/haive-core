"""Haive Vector Store Module.

This module provides comprehensive abstractions and implementations for working with
vector stores in the Haive framework. Vector stores are specialized databases optimized
for storing and retrieving high-dimensional vectors, typically used for similarity
search in RAG (Retrieval-Augmented Generation) applications.

Vector stores enable efficient semantic search by storing document embeddings and
providing fast similarity-based retrieval. They are essential components for
building RAG systems, recommendation engines, and other applications that require
similarity search over large document collections.

Supported Vector Store Providers:
    - Chroma: Open-source vector database with local and server modes
    - Pinecone: Managed vector database service
    - Weaviate: Open-source vector search engine
    - Qdrant: Vector similarity search engine
    - FAISS: Facebook AI Similarity Search (local/in-memory)
    - Milvus: Open-source vector database
    - OpenSearch: Elasticsearch-based vector search
    - Redis: Redis with vector search capabilities
    - Supabase: PostgreSQL with vector extensions (pgvector)
    - MongoDB Atlas: MongoDB with vector search
    - LanceDB: Serverless vector database
    - Marqo: Tensor search engine

Key Components:
    - Base Classes: Abstract base classes for vector store configurations
    - Provider Types: Enumeration of supported vector store providers
    - Configuration: Provider-specific configuration classes with validation
    - Factory Functions: Simplified creation of vector store instances
    - Security: Secure handling of connection strings and API keys
    - Performance: Connection pooling and caching optimizations

Typical usage example:
    ```python
    from haive.core.models.vectorstore import VectorStoreConfig, VectorStoreProvider

    # Configure a vector store
    config = VectorStoreConfig(
        provider=VectorStoreProvider.CHROMA,
        collection_name="documents",
        persist_directory="./chroma_db",
        embedding_function=embedding_config
    )

    # Create the vector store
    vectorstore = config.instantiate()

    # Add documents
    vectorstore.add_texts(["Document content"], metadatas=[{"source": "doc1"}])

    # Search for similar documents
    results = vectorstore.similarity_search("query text", k=5)
    ```

Architecture:
    Vector stores in Haive follow a consistent configuration pattern:

    1. Provider Selection: Choose from supported vector store providers
    2. Configuration: Set provider-specific parameters (endpoints, credentials, etc.)
    3. Instantiation: Create the actual vector store instance
    4. Operations: Add documents, search, update, delete operations

    All vector stores support the same core operations for consistency,
    while provider-specific features are available through configuration.

Performance Considerations:
    - Index Type: Different providers support different index types (HNSW, IVF, etc.)
    - Batch Operations: Use batch operations for better performance when adding many documents
    - Connection Pooling: Configured automatically for cloud providers
    - Caching: In-memory caching for frequently accessed embeddings

    See provider-specific documentation for optimization guidelines.

Examples:
    Local vector store for development::

        config = VectorStoreConfig(
            provider=VectorStoreProvider.CHROMA,
            persist_directory="./local_db"
        )

    Cloud vector store for production::

        config = VectorStoreConfig(
            provider=VectorStoreProvider.PINECONE,
            api_key_env_var="PINECONE_API_KEY",
            environment="us-west1-gcp",
            index_name="production-index"
        )

    Vector store with custom embeddings::

        from haive.core.models.embeddings import OpenAIEmbeddingConfig

        embedding_config = OpenAIEmbeddingConfig(model="text-embedding-3-small")

        config = VectorStoreConfig(
            provider=VectorStoreProvider.WEAVIATE,
            url="http://localhost:8080",
            embedding_config=embedding_config
        )

.. autosummary::
   :toctree: generated/

   VectorStoreConfig
   VectorStoreProvider
"""

from haive.core.models.vectorstore.base import VectorStoreConfig, VectorStoreProvider

# Create individual constants for backward compatibility
Chroma = VectorStoreProvider.Chroma
FAISS = VectorStoreProvider.FAISS
InMemory = VectorStoreProvider.InMemory
Milvus = VectorStoreProvider.Milvus
Pinecone = VectorStoreProvider.Pinecone
Qdrant = VectorStoreProvider.Qdrant
Weaviate = VectorStoreProvider.Weaviate
Zilliz = VectorStoreProvider.Zilliz

# Import factory functions if they exist
try:
    from haive.core.models.vectorstore.factory import (
        add_document,
        create_retriever,
        create_retriever_from_documents,
        create_vectorstore,
        create_vs_config_from_documents,
        create_vs_from_documents,
    )
except ImportError:
    # Factory functions not available
    def add_document(*args, **kwargs):
        """Placeholder function."""
        pass

    def create_retriever(*args, **kwargs):
        """Placeholder function."""
        pass

    def create_retriever_from_documents(*args, **kwargs):
        """Placeholder function."""
        pass

    def create_vectorstore(*args, **kwargs):
        """Placeholder function."""
        pass

    def create_vs_config_from_documents(*args, **kwargs):
        """Placeholder function."""
        pass

    def create_vs_from_documents(*args, **kwargs):
        """Placeholder function."""
        pass


__all__ = [
    "Chroma",
    "FAISS",
    "InMemory",
    "Milvus",
    "Pinecone",
    "Qdrant",
    "VectorStoreConfig",
    "VectorStoreProvider",
    "Weaviate",
    "Zilliz",
    "add_document",
    "create_retriever",
    "create_retriever_from_documents",
    "create_vectorstore",
    "create_vs_config_from_documents",
    "create_vs_from_documents",
]
