# src/haive/core/engine/retriever/svm.py

"""SVM Retriever implementation for the Haive framework.

This module provides a configuration class for the SVM (Support Vector Machine) retriever,
which uses SVM for document similarity ranking.

The SVM retriever works by:
1. Converting documents to embeddings using the specified embedding model
2. Training a Support Vector Machine classifier with the query embedding as the positive example
3. Ranking documents based on their distance from the SVM decision boundary
4. Returning the top k documents closest to the decision boundary

This approach is particularly effective for scenarios where the distinction between relevant
and irrelevant documents needs to be learned from embeddings in a more sophisticated way than
simple cosine similarity.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from pydantic import Field, model_validator

from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.engine.retriever.types import RetrieverType
from haive.core.models.embeddings.base import BaseEmbeddingConfig


@BaseRetrieverConfig.register(RetrieverType.SVM)
class SVMRetrieverConfig(BaseRetrieverConfig):
    """Configuration for SVM retriever.

    This retriever uses Support Vector Machine for document retrieval based on embeddings.
    It's particularly useful for finding documents that are most similar to a query using
    SVM decision boundaries. Unlike simple vector similarity methods, the SVM approach can
    learn more complex relationships between queries and documents.

    The SVM retriever works by treating the query embedding as a positive example and
    generating synthetic negative examples. It then trains an SVM classifier that separates
    the positive from the negative examples. Documents are then ranked by their distance
    from the SVM decision boundary.

    Attributes:
        embeddings_config (BaseEmbeddingConfig): Configuration for the embedding model
            that will be used to generate vector representations of documents and queries.
            This is a required field.
        documents (List[Document]): List of Document objects to build the retrieval index
            from. These documents will be embedded and used to train the SVM classifier
            during query time.
        k (int): Number of documents to retrieve for each query. Default is 4.
        relevancy_threshold (Optional[float]): If provided, only documents with a
            relevancy score above this threshold will be returned. This allows filtering
            out less relevant documents even if they are in the top k.

    Example:
        ```python
        from haive.core.engine.retriever.svm import SVMRetrieverConfig
        from haive.core.models.embeddings.base import OpenAIEmbeddingConfig
        from langchain_core.documents import Document

        # Create sample documents
        documents = [
            Document(page_content="Machine learning algorithms require data preprocessing"),
            Document(page_content="Support vector machines can be used for classification"),
            Document(page_content="Neural networks are effective for image recognition"),
            Document(page_content="Natural language processing uses text embeddings"),
            Document(page_content="Decision trees are simple but powerful algorithms")
        ]

        # Create SVM retriever configuration
        config = SVMRetrieverConfig(
            name="ml_document_retriever",
            embeddings_config=OpenAIEmbeddingConfig(
                model="text-embedding-3-small"
            ),
            documents=documents,
            k=2,
            relevancy_threshold=0.6  # Only return documents above this relevance score
        )

        # Instantiate the retriever
        retriever = config.instantiate()

        # Retrieve documents related to SVM
        results = retriever.get_relevant_documents("How do support vector machines work?")
        # Should return documents about SVMs ranked by relevance
        ```
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.SVM, description="The type of retriever"
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
        """Validate that required configurations are provided.

        This validator ensures that an embedding configuration is provided, which is
        essential for the SVM retriever to function properly. The embeddings are used
        to convert documents and queries into vector representations that the SVM
        classifier can operate on.

        Returns:
            SVMRetrieverConfig: The validated configuration instance

        Raises:
            ValueError: If embeddings_config is not provided or is invalid
        """
        if not self.embeddings_config:
            raise ValueError("embeddings_config is required for SVMRetriever")
        return self

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for SVM retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each input parameter.

        The SVMRetriever accepts the following inputs:
            - query: The text query to search for in the document collection
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
        """Return output field definitions for SVM retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each output parameter.

        The SVMRetriever produces the following outputs:
            - documents: A list of Document objects retrieved based on SVM classification,
                ordered by their distance from the SVM decision boundary (most relevant first).
                If a relevancy_threshold was set, only documents scoring above the threshold
                will be included.
        """
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create an SVM retriever instance based on this configuration.

        This method instantiates an SVMRetriever from LangChain Community, which uses
        a Support Vector Machine classifier to identify and rank relevant documents.
        The method first instantiates the embedding model specified in embeddings_config,
        which will be used to convert documents and queries to vector representations.

        Returns:
            SVMRetriever: An instantiated SVM retriever ready to perform document
                retrieval using Support Vector Machine classification.

        Raises:
            ImportError: If scikit-learn is not installed. This dependency is required
                for the SVM classification algorithm.
            ValueError: If the configuration is invalid, such as missing required fields
                or incompatible parameter combinations.

        Example:
            ```python
            # Create a configuration with embedding model and documents
            config = SVMRetrieverConfig(
                name="svm_retriever",
                embeddings_config=OpenAIEmbeddingConfig(model="text-embedding-3-small"),
                documents=my_documents,
                k=5
            )

            # Instantiate the retriever
            retriever = config.instantiate()

            # Use the retriever to get documents most relevant to a query
            results = retriever.get_relevant_documents("What are machine learning algorithms?")
            ```
        """
        try:
            from langchain_community.retrievers import SVMRetriever
        except ImportError:
            raise ImportError(
                "SVMRetriever requires scikit-learn. Install with: pip install scikit-learn"
            )

        # Create embeddings instance
        embeddings = self.embeddings_config.instantiate()

        # Create and return the SVM retriever
        return SVMRetriever.from_documents(
            documents=self.documents,
            embeddings=embeddings,
            k=self.k,
            relevancy_threshold=self.relevancy_threshold,
        )
