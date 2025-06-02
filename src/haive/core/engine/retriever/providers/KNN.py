# src/haive/core/engine/retriever/knn.py

"""KNN Retriever implementation for the Haive framework.

This module provides a configuration class for the KNN retriever,
which uses K-Nearest Neighbors for document retrieval based on embeddings.
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
    It calculates similarity using L2 norm and returns the most similar documents.

    Attributes:
        embeddings_config: Configuration for the embedding model
        documents: List of documents to retrieve from
        k: Number of documents to retrieve
        relevancy_threshold: Optional threshold for relevancy filtering

    Example:
        ```python
        from haive.core.engine.retriever.knn import KNNRetrieverConfig
        from haive.core.models.embeddings.base import HuggingFaceEmbeddingConfig

        config = KNNRetrieverConfig(
            name="knn_retriever",
            embeddings_config=HuggingFaceEmbeddingConfig(
                model="sentence-transformers/all-MiniLM-L6-v2"
            ),
            documents=documents,
            k=4
        )

        retriever = config.instantiate()
        docs = retriever.get_relevant_documents("query")
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
        """Validate that embeddings_config is provided."""
        if not self.embeddings_config:
            raise ValueError("embeddings_config is required for KNNRetriever")
        return self

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for KNN retriever."""
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for KNN retriever."""
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create a KNN retriever from this configuration.

        Returns:
            Instantiated KNN retriever

        Raises:
            ImportError: If required dependencies are not installed
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
