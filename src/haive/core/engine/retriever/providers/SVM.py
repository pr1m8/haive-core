# src/haive/core/engine/retriever/svm.py

"""SVM Retriever implementation for the Haive framework.

This module provides a configuration class for the SVM (Support Vector Machine) retriever,
which uses SVM for document similarity ranking.
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
    SVM decision boundaries.

    Attributes:
        embeddings_config: Configuration for the embedding model
        documents: List of documents to retrieve from
        k: Number of documents to retrieve
        relevancy_threshold: Optional threshold for relevancy filtering

    Example:
        ```python
        from haive.core.engine.retriever.svm import SVMRetrieverConfig
        from haive.core.models.embeddings.base import HuggingFaceEmbeddingConfig

        config = SVMRetrieverConfig(
            name="svm_retriever",
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
        """Validate that required configurations are provided."""
        if not self.embeddings_config:
            raise ValueError("embeddings_config is required for SVMRetriever")
        return self

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for SVM retriever."""
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for SVM retriever."""
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create an SVM retriever instance based on this configuration.

        Returns:
            Instantiated SVM retriever

        Raises:
            ImportError: If scikit-learn is not installed
            ValueError: If configuration is invalid
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
