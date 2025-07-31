"""Haive Retriever Module.

This module provides comprehensive abstractions and implementations for document
retrievers in the Haive framework. Retrievers are components that find and return
relevant documents or information based on queries, forming the "R" in RAG
(Retrieval-Augmented Generation) systems.

Retrievers bridge the gap between raw data sources and AI agents by providing
intelligent, context-aware document retrieval. They can work with various data
sources including vector stores, search engines, databases, and APIs.

Supported Retriever Types:
    - Vector Store Retrievers: Semantic search using vector embeddings
    - Search Engine Retrievers: Traditional keyword-based search
    - Hybrid Retrievers: Combine multiple retrieval strategies
    - Multi-Query Retrievers: Generate multiple queries for better coverage
    - Parent Document Retrievers: Retrieve full documents from partial matches
    - Time-Weighted Retrievers: Consider document freshness and relevance
    - Self-Query Retrievers: Parse natural language queries into structured filters
    - Ensemble Retrievers: Combine multiple retrievers with weighted scoring

Key Components:
    - Base Classes: Abstract base classes for retriever configurations
    - Retriever Types: Enumeration of supported retriever patterns
    - Configuration: Type-specific configuration classes with validation
    - Factory Functions: Simplified creation of retriever instances
    - Filtering: Advanced filtering capabilities for retrieval results
    - Ranking: Sophisticated ranking and re-ranking algorithms

Typical usage example:
    ```python
    from haive.core.models.retriever import RetrieverConfig, RetrieverType
    from haive.core.models.vectorstore import VectorStoreConfig

    # Configure a vector store retriever
    vectorstore_config = VectorStoreConfig(provider="chroma")

    retriever_config = RetrieverConfig(
        type=RetrieverType.VECTORSTORE,
        vectorstore_config=vectorstore_config,
        search_type="similarity",
        search_kwargs={"k": 10, "score_threshold": 0.7}
    )

    # Create the retriever
    retriever = retriever_config.instantiate()

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents("What is machine learning?")

    # Async retrieval
    docs = await retriever.aget_relevant_documents("AI applications")
    ```

Architecture:
    Retrievers in Haive follow a layered architecture:

    1. Query Processing: Parse and enhance input queries
    2. Source Interaction: Connect to and query data sources
    3. Result Processing: Filter, rank, and format retrieved documents
    4. Caching: Cache results for performance optimization

    This architecture ensures consistent behavior across different retriever types
    while allowing for provider-specific optimizations.

Advanced Features:
    - Multi-Modal Retrieval: Support for text, images, and other media types
    - Contextual Retrieval: Consider conversation history and user context
    - Adaptive Retrieval: Learn from user feedback to improve results
    - Batch Retrieval: Efficient processing of multiple queries
    - Streaming Results: Stream results for large result sets
    - Fallback Strategies: Handle failures gracefully with backup retrieval methods

Performance Optimizations:
    - Query Caching: Cache frequently requested queries
    - Result Caching: Cache retrieval results with TTL
    - Connection Pooling: Reuse connections for better performance
    - Async Operations: Non-blocking retrieval for better throughput
    - Batch Processing: Process multiple queries efficiently
    - Index Optimization: Maintain optimal search indices

Examples:
    Simple vector store retriever::

        config = RetrieverConfig(
            type=RetrieverType.VECTORSTORE,
            vectorstore_config=chroma_config,
            search_kwargs={"k": 5}
        )

    Multi-query retriever for better coverage::

        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            base_retriever_config=base_config,
            llm_config=llm_config,
            query_count=3
        )

    Hybrid retriever combining multiple strategies::

        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            retrievers=[vector_config, keyword_config],
            weights=[0.7, 0.3],
            c=60  # RRF parameter
        )

    Time-weighted retriever for recent content::

        config = RetrieverConfig(
            type=RetrieverType.TIME_WEIGHTED,
            vectorstore_config=base_config,
            decay_rate=-0.01,
            k=10
        )

.. autosummary::
   :toctree: generated/

   RetrieverConfig
   RetrieverType
   VectorStoreRetrieverConfig
"""

from haive.core.models.retriever.base import RetrieverConfig, RetrieverType
from haive.core.models.retriever.vectorstore_retriever import VectorStoreRetrieverConfig

__all__ = [
    "RetrieverConfig",
    "RetrieverType",
    "VectorStoreRetrieverConfig",
]
