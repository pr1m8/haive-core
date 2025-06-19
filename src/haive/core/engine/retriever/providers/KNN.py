# src/haive/core/engine/retriever/knn.py

"""KNN Retriever implementation for the Haive framework.

This module provides a configuration class for the KNN retriever,
which uses K-Nearest Neighbors for document retrieval based on embeddings.

The KNN (K-Nearest Neighbors) retriever works by:
1. Converting documents to embeddings using the specified embedding model
2. Converting the query to an embedding using the same model
3. Finding the k documents whose embeddings are closest to the query embedding
   using Euclidean distance (L2 norm)
4. Returning these documents ordered by similarity

This is a simple but effective approach for semantic search that doesn't require
maintaining a specialized vector database, making it useful for small to medium-sized
document collections or for prototyping.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from pydantic import Field, model_validator

from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.engine.retriever.types import RetrieverType
from haive.core.models.embeddings.base import BaseEmbeddingConfig


@BaseRetrieverConfig.register(RetrieverType.KNN)
class KNNRetrieverConfig(BaseRetrieverConfig):
    """Configuration for KNN retriever.

    This retriever uses K-Nearest Neighbors for document retrieval based on embeddings.
    It calculates similarity using Euclidean distance (L2 norm) and returns the most
    similar documents. Unlike more complex vector stores, KNN doesn't require building
    specialized indices, making it a good choice for smaller document collections or
    when simplicity is preferred over maximum performance.

    The KNN retriever is particularly useful when:
    - You have a small to medium-sized document collection
    - You want a simple semantic search without a vector database
    - You need a pure-Python solution with minimal dependencies
    - You're prototyping before implementing a more scalable solution

    Attributes:
        embeddings_config (BaseEmbeddingConfig): Configuration for the embedding model
            that will be used to generate vector representations of documents and queries.
            This is a required field.
        documents (List[Document]): List of Document objects to build the retrieval index
            from. These documents will be embedded and used for KNN search.
        k (int): Number of nearest neighbor documents to retrieve for each query. Default is 4.
        relevancy_threshold (Optional[float]): If provided, only documents with a
            similarity score above this threshold will be returned. This allows filtering
            out less relevant documents even if they are in the top k.

    Example:
        ```python
        from haive.core.engine.retriever.knn import KNNRetrieverConfig
        from haive.core.models.embeddings.base import OpenAIEmbeddingConfig
        from langchain_core.documents import Document

        # Create sample documents
        documents = [
            Document(page_content="Neural networks are used for deep learning"),
            Document(page_content="K-nearest neighbors is a simple algorithm"),
            Document(page_content="Embeddings represent text as vectors"),
            Document(page_content="Vector similarity enables semantic search"),
            Document(page_content="Machine learning uses statistical models")
        ]

        # Create KNN retriever configuration
        config = KNNRetrieverConfig(
            name="ml_docs_retriever",
            embeddings_config=OpenAIEmbeddingConfig(
                model="text-embedding-3-small"
            ),
            documents=documents,
            k=2,
            relevancy_threshold=0.7  # Only return documents above this similarity score
        )

        # Instantiate the retriever
        retriever = config.instantiate()

        # Retrieve documents related to vectors
        results = retriever.get_relevant_documents("How do vector embeddings work?")
        # Should return documents about embeddings and vector similarity
        ```
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.KNN, description="The type of retriever"
    )

    embeddings_config: BaseEmbeddingConfig = Field(
        ..., description="Configuration for the embedding model"
    )

    documents: List[Document] = Field(
        default_factory=list, description="Documents to retrieve from"
    )

    k: int = Field(default=4, description="Number of documents to retrieve")

    relevancy_threshold: Optional[float] = Field(
        default=None, description="Threshold for relevancy filtering"
    )

    @model_validator(mode="after")
    def validate_config(self):
        """Validate that embeddings_config is provided.

        This validator ensures that an embedding configuration is provided, which is
        essential for the KNN retriever to function properly. The embeddings are used
        to convert documents and queries into vector representations that can be
        compared using Euclidean distance.

        Returns:
            KNNRetrieverConfig: The validated configuration instance

        Raises:
            ValueError: If embeddings_config is not provided or is invalid
        """
        if not self.embeddings_config:
            raise ValueError("embeddings_config is required for KNNRetriever")
        return self

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for KNN retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each input parameter.

        The KNNRetriever accepts the following inputs:
            - query: The text query to search for in the document collection using
                KNN over the embedded vectors.
            - k: Optional override for the number of documents to retrieve (overrides
                the default k value specified in the configuration)
        """
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for KNN retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each output parameter.

        The KNNRetriever produces the following outputs:
            - documents: A list of Document objects retrieved using KNN search over
                the embedded vectors, ranked by their similarity to the query embedding.
                If a relevancy_threshold was set, only documents scoring above the
                threshold will be included.
        """
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create a KNN retriever from this configuration.

        This method instantiates a KNNRetriever from LangChain Community, which uses
        scikit-learn's NearestNeighbors implementation to find the most similar documents
        to a query. The method first instantiates the embedding model specified in
        embeddings_config, which will be used to convert documents and queries to vector
        representations.

        Returns:
            KNNRetriever: An instantiated KNN retriever ready to perform document
                retrieval using K-Nearest Neighbors search.

        Raises:
            ImportError: If scikit-learn or numpy are not installed. These dependencies
                are required for the KNN implementation.

        Example:
            ```python
            # Create a configuration with embedding model and documents
            config = KNNRetrieverConfig(
                name="knn_retriever",
                embeddings_config=OpenAIEmbeddingConfig(model="text-embedding-3-small"),
                documents=my_documents,
                k=5
            )

            # Instantiate the retriever
            retriever = config.instantiate()

            # Use the retriever to find semantically similar documents
            results = retriever.get_relevant_documents("What is machine learning?")
            ```
        """
        try:
            from langchain_community.retrievers import KNNRetriever
        except ImportError:
            raise ImportError(
                "KNNRetriever requires scikit-learn and numpy. Install with: pip install scikit-learn numpy"
            )

        # Create embeddings instance
        embeddings = self.embeddings_config.instantiate()

        # Create and return the KNN retriever
        return KNNRetriever.from_documents(
            documents=self.documents,
            embeddings=embeddings,
            k=self.k,
            relevancy_threshold=self.relevancy_threshold,
        )
