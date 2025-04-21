"""Types module for the Haive retriever engine.

This module defines the core types and enumerations used throughout the retriever
engine implementation. It provides a structured way to identify and categorize
different retriever implementations available in the Haive framework.

The main component of this module is the RetrieverType enum, which defines all
supported retriever implementations and their categories. This helps in organizing
and selecting appropriate retriever strategies for different use cases.

Example:
    ```python
    from haive.core.engine.retriever.types import RetrieverType

    # Use specific retriever types
    vector_store_type = RetrieverType.VECTOR_STORE
    ensemble_type = RetrieverType.ENSEMBLE

    # Check retriever type
    if retriever_type == RetrieverType.MULTI_QUERY:
        print("Using multi-query retrieval strategy")
    ```
"""

from enum import Enum


class RetrieverType(str, Enum):
    """Enumeration of supported retriever types in the Haive framework.

    This enum defines all available retriever implementations that can be used for document
    retrieval. The retrievers are categorized into different groups based on their
    functionality and approach to document retrieval.

    Categories:
        Base Vector Store Retrievers:
            Simple retrievers that directly use vector stores for similarity search.

        Advanced Retrieval Strategies:
            Sophisticated retrieval methods that enhance basic vector similarity search:
            - TIME_WEIGHTED: Considers document recency in retrieval
            - MULTI_QUERY: Generates multiple queries for better coverage
            - MULTI_VECTOR: Uses multiple vector representations per document
            - PARENT_DOCUMENT: Retrieves parent documents based on child matches
            - SELF_QUERY: Automatically generates structured queries
            - CONTEXTUAL_COMPRESSION: Compresses retrieved documents contextually
            - REPHRASE_QUERY: Reformulates queries for better results

        Ensemble Methods:
            Combine multiple retrievers for improved performance:
            - MERGER: Combines results from multiple retrievers
            - ENSEMBLE: Weighted combination of multiple retrievers

        Sparse Retrievers:
            Traditional information retrieval methods:
            - SPARSE: Generic sparse retrieval
            - KNN: K-Nearest Neighbors
            - TFIDF: Term Frequency-Inverse Document Frequency
            - BM25: Best Match 25 algorithm
            - SVM: Support Vector Machine based retrieval

        Specific Implementations:
            Retrievers for specific vector store backends:
            - ELASTICSEARCH: Elasticsearch-based retrieval
            - FAISS: Facebook AI Similarity Search
            - IN_MEMORY: In-memory vector store
            - PINECONE: Pinecone vector database
            - QDRANT: Qdrant vector database

    Example:
        ```python
        from haive.core.engine.retriever.types import RetrieverType
        from haive.core.engine.retriever import create_retriever_config

        # Create a vector store retriever config
        config = create_retriever_config(
            retriever_type=RetrieverType.VECTOR_STORE,
            name="my_retriever",
            vector_store_config=vs_config
        )

        # Create an ensemble retriever config
        ensemble_config = create_retriever_config(
            retriever_type=RetrieverType.ENSEMBLE,
            name="ensemble_retriever",
            retrievers=[config1, config2]
        )
        ```
    """
    # Base vector store retrievers
    VECTOR_STORE = "VectorStoreRetriever"

    # Advanced retrieval strategies
    TIME_WEIGHTED = "TimeWeightedVectorStoreRetriever"
    MULTI_QUERY = "MultiQueryRetriever"
    MULTI_VECTOR = "MultiVectorRetriever"
    PARENT_DOCUMENT = "ParentDocumentRetriever"
    SELF_QUERY = "SelfQueryRetriever"
    CONTEXTUAL_COMPRESSION = "ContextualCompressionRetriever"
    REPHRASE_QUERY = "RePhraseQueryRetriever"

    # Ensemble methods
    MERGER = "MergerRetriever"
    ENSEMBLE = "EnsembleRetriever"

    # Sparse retrievers
    SPARSE = "SparseRetriever"
    KNN = "KNNRetriever"
    TFIDF = "TFIDFRetriever"
    BM25 = "BM25Retriever"
    SVM = "SVMRetriever"

    # Specific implementations
    ELASTICSEARCH = "ElasticsearchRetriever"
    FAISS = "FAISSRetriever"
    IN_MEMORY = "InMemoryRetriever"
    PINECONE = "PineconeRetriever"
    QDRANT = "QdrantRetriever"
