"""Vector store provider implementations for the Haive framework.

This module provides comprehensive vector store functionality with support for
over 25 different vector database backends including cloud-managed services,
open-source databases, and specialized search engines. All providers follow
a consistent configuration interface through BaseVectorStoreConfig.

The module uses automatic registration where each provider configuration class
extends BaseVectorStoreConfig and registers itself with the type system through
decorators. This enables dynamic discovery and instantiation of vector stores.

Supported Vector Store Categories:
    - **Cloud/Managed Services**: Pinecone, Weaviate, Qdrant Cloud, Supabase
    - **Open Source Databases**: Chroma, FAISS, Milvus, LanceDB
    - **Search Engines**: Elasticsearch, OpenSearch, TypeSense
    - **Database Extensions**: PostgreSQL (pgvector), Redis, MongoDB Atlas
    - **Graph Databases**: Neo4j with vector support
    - **Development/Testing**: InMemory, Fake stores

Key Features:
    - Automatic registration through decorators
    - Dynamic loading and discovery
    - Consistent configuration interface
    - Support for metadata filtering and similarity search
    - Integration with various embedding providers
    - Scalable from development to production

Available Providers:
    - **Amazon OpenSearch**: AWS managed OpenSearch with vector capabilities
    - **Annoy**: Spotify's approximate nearest neighbor library
    - **Azure AI Search**: Microsoft Azure cognitive search with vectors
    - **Cassandra**: Apache Cassandra with vector search extensions
    - **Chroma**: Popular open-source embedding database
    - **ClickHouse**: Analytical database with vector search
    - **DocArray**: Document-oriented vector storage
    - **Elasticsearch**: Enterprise search with dense/sparse vectors
    - **FAISS**: Facebook's efficient similarity search library
    - **InMemory**: Development and testing vector store
    - **LanceDB**: Modern columnar vector database
    - **Marqo**: Tensor-based search and recommendation engine
    - **Milvus**: Open-source vector database for AI applications
    - **MongoDB Atlas**: MongoDB with vector search capabilities
    - **Neo4j**: Graph database with vector similarity search
    - **OpenSearch**: Community-driven search and analytics
    - **Pinecone**: Managed vector database service
    - **PostgreSQL (pgvector)**: SQL database with vector extensions
    - **Qdrant**: Vector similarity search engine
    - **Redis**: In-memory database with vector search modules
    - **Scikit-learn**: ML library integration for vectors
    - **Supabase**: PostgreSQL-based backend with vector support
    - **Typesense**: Modern search engine with vector capabilities
    - **USearch**: High-performance similarity search
    - **Vectara**: Managed vector search platform
    - **Weaviate**: Open-source vector database
    - **Zilliz**: Cloud service for Milvus vector database

Examples:
    Basic Chroma vector store setup::

        from haive.core.engine.vectorstore.providers import ChromaVectorStoreConfig
        from haive.core.engine.embedding.providers import OpenAIEmbeddingConfig

        # Configure embeddings
        embeddings_config = OpenAIEmbeddingConfig(
            name="openai_embeddings",
            model="text-embedding-3-large"
        )

        # Configure vector store
        vector_config = ChromaVectorStoreConfig(
            name="chroma_store",
            collection_name="documents",
            embedding_config=embeddings_config,
            persist_directory="./chroma_db"
        )

        # Instantiate vector store
        vectorstore = vector_config.instantiate()

    Pinecone cloud vector store::

        from haive.core.engine.vectorstore.providers import PineconeVectorStoreConfig

        vector_config = PineconeVectorStoreConfig(
            name="pinecone_store",
            index_name="my-index",
            api_key="your-api-key",
            environment="us-west1-gcp-free"
        )

        vectorstore = vector_config.instantiate()

    PostgreSQL with pgvector extension::

        from haive.core.engine.vectorstore.providers import PGVectorStoreConfig

        vector_config = PGVectorStoreConfig(
            name="postgres_vectors",
            connection_string="postgresql://user:pass@localhost:5432/vectordb",
            collection_name="embeddings",
            embedding_config=embeddings_config
        )

    Configuration discovery and provider listing::

        from haive.core.engine.vectorstore import BaseVectorStoreConfig

        # List all registered vector store types
        available_stores = BaseVectorStoreConfig.list_registered_types()
        print(f"Available stores: {list(available_stores.keys())}")

        # Get specific provider class dynamically
        store_class = BaseVectorStoreConfig.get_config_class("Chroma")
        config = store_class(name="dynamic_store")

Note:
    All provider configurations are imported at module level to ensure proper
    registration with the base configuration system. This allows dynamic
    discovery and instantiation through the common interface.

    Vector stores automatically integrate with the embedding system and can
    be used for similarity search, document retrieval, and semantic analysis
    workflows throughout the Haive framework.
"""

# Import all vector store configs to register them
from haive.core.engine.vectorstore.providers.AmazonOpenSearchVectorStoreConfig import (
    AmazonOpenSearchVectorStoreConfig,
)

# Approximate nearest neighbor stores
from haive.core.engine.vectorstore.providers.AnnoyVectorStoreConfig import (
    AnnoyVectorStoreConfig,
)

# Cloud/managed vector stores
from haive.core.engine.vectorstore.providers.AzureSearchVectorStoreConfig import (
    AzureSearchVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.CassandraVectorStoreConfig import (
    CassandraVectorStoreConfig,
)

# Core open source vector stores
from haive.core.engine.vectorstore.providers.ChromaVectorStoreConfig import (
    ChromaVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.ClickHouseVectorStoreConfig import (
    ClickHouseVectorStoreConfig,
)

# Document-oriented vector stores
from haive.core.engine.vectorstore.providers.DocArrayVectorStoreConfig import (
    DocArrayVectorStoreConfig,
)

# Search engines
from haive.core.engine.vectorstore.providers.ElasticsearchVectorStoreConfig import (
    ElasticsearchVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.FAISSVectorStoreConfig import (
    FAISSVectorStoreConfig,
)

# Development and testing stores
from haive.core.engine.vectorstore.providers.InMemoryVectorStoreConfig import (
    InMemoryVectorStoreConfig,
)

# Columnar vector databases
from haive.core.engine.vectorstore.providers.LanceDBVectorStoreConfig import (
    LanceDBVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.MarqoVectorStoreConfig import (
    MarqoVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.MilvusVectorStoreConfig import (
    MilvusVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.MongoDBAtlasVectorStoreConfig import (
    MongoDBAtlasVectorStoreConfig,
)

# Graph databases with vector support
from haive.core.engine.vectorstore.providers.Neo4jVectorStoreConfig import (
    Neo4jVectorStoreConfig,
)

# Additional search engines with vector capabilities
from haive.core.engine.vectorstore.providers.OpenSearchVectorStoreConfig import (
    OpenSearchVectorStoreConfig,
)

# Database extensions
from haive.core.engine.vectorstore.providers.PGVectorStoreConfig import (
    PGVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.PineconeVectorStoreConfig import (
    PineconeVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.QdrantVectorStoreConfig import (
    QdrantVectorStoreConfig,
)

# In-memory databases
from haive.core.engine.vectorstore.providers.RedisVectorStoreConfig import (
    RedisVectorStoreConfig,
)

# ML integration stores
from haive.core.engine.vectorstore.providers.SKLearnVectorStoreConfig import (
    SKLearnVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.SupabaseVectorStoreConfig import (
    SupabaseVectorStoreConfig,
)

# Search engines with vector capabilities
from haive.core.engine.vectorstore.providers.TypesenseVectorStoreConfig import (
    TypesenseVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.USearchVectorStoreConfig import (
    USearchVectorStoreConfig,
)

# Managed vector search platforms
from haive.core.engine.vectorstore.providers.VectaraVectorStoreConfig import (
    VectaraVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.WeaviateVectorStoreConfig import (
    WeaviateVectorStoreConfig,
)
from haive.core.engine.vectorstore.providers.ZillizVectorStoreConfig import (
    ZillizVectorStoreConfig,
)

__all__ = [
    "AmazonOpenSearchVectorStoreConfig",
    # Approximate nearest neighbor stores
    "AnnoyVectorStoreConfig",
    "AzureSearchVectorStoreConfig",
    "CassandraVectorStoreConfig",
    # Core open source
    "ChromaVectorStoreConfig",
    "ClickHouseVectorStoreConfig",
    # Document-oriented vector stores
    "DocArrayVectorStoreConfig",
    # Search engines
    "ElasticsearchVectorStoreConfig",
    "FAISSVectorStoreConfig",
    # Development and testing stores
    "InMemoryVectorStoreConfig",
    # Columnar vector databases
    "LanceDBVectorStoreConfig",
    "MarqoVectorStoreConfig",
    "MilvusVectorStoreConfig",
    "MongoDBAtlasVectorStoreConfig",
    # Graph databases with vector support
    "Neo4jVectorStoreConfig",
    # Additional search engines with vector capabilities
    "OpenSearchVectorStoreConfig",
    # Database extensions
    "PGVectorStoreConfig",
    # Cloud/managed
    "PineconeVectorStoreConfig",
    "QdrantVectorStoreConfig",
    # In-memory databases
    "RedisVectorStoreConfig",
    # ML integration stores
    "SKLearnVectorStoreConfig",
    "SupabaseVectorStoreConfig",
    # Search engines with vector capabilities
    "TypesenseVectorStoreConfig",
    "USearchVectorStoreConfig",
    # Managed vector search platforms
    "VectaraVectorStoreConfig",
    "WeaviateVectorStoreConfig",
    "ZillizVectorStoreConfig",
]
