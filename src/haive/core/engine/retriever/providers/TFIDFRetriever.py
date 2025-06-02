# src/haive/core/engine/retriever/tfidf.py

"""TF-IDF Retriever implementation for the Haive framework.

This module provides a configuration class for the TF-IDF retriever,
which uses Term Frequency-Inverse Document Frequency for document retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from pydantic import Field

from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.engine.retriever.types import RetrieverType


@BaseRetrieverConfig.register(RetrieverType.TFIDF)
class TFIDFRetrieverConfig(BaseRetrieverConfig):
    """Configuration for TF-IDF retriever.

    This retriever uses Term Frequency-Inverse Document Frequency for document retrieval,
    which is effective for text-based similarity search and traditional information retrieval.

    Attributes:
        documents: List of documents to retrieve from
        k: Number of documents to retrieve
        tfidf_params: Parameters for the TF-IDF vectorizer

    Example:
        ```python
        from haive.core.engine.retriever.tfidf import TFIDFRetrieverConfig

        config = TFIDFRetrieverConfig(
            name="tfidf_retriever",
            documents=documents,
            k=4,
            tfidf_params={"max_features": 1000, "stop_words": "english"}
        )

        retriever = config.instantiate()
        docs = retriever.get_relevant_documents("query")
        ```
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.TFIDF, description="The type of retriever"
    )

    documents: List[Document] = Field(
        default_factory=list, description="Documents to retrieve from"
    )

    k: int = Field(default=4, description="Number of documents to retrieve")

    tfidf_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters for the TF-IDF vectorizer"
    )

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for TF-IDF retriever."""
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for TF-IDF retriever."""
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create a TF-IDF retriever from this configuration.

        Returns:
            Instantiated TF-IDF retriever

        Raises:
            ImportError: If scikit-learn is not installed
        """
        try:
            from langchain_community.retrievers import TFIDFRetriever
        except ImportError:
            raise ImportError(
                "TFIDFRetriever requires scikit-learn. Install with: pip install scikit-learn"
            )

        return TFIDFRetriever.from_documents(
            documents=self.documents,
            k=self.k,
            tfidf_params=self.tfidf_params,
        )
