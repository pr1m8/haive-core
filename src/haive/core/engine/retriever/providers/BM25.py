# src/haive/core/engine/retriever/bm25.py

"""BM25 Retriever implementation for the Haive framework.

This module provides a configuration class for the BM25 retriever,
which uses the BM25 algorithm for keyword-based document retrieval.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from pydantic import Field

from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.engine.retriever.types import RetrieverType


def default_preprocessing_func(text: str) -> List[str]:
    """Default preprocessing function for BM25.

    Args:
        text: Input text to preprocess

    Returns:
        List of tokens
    """
    return text.split()


@BaseRetrieverConfig.register(RetrieverType.BM25)
class BM25RetrieverConfig(BaseRetrieverConfig):
    """Configuration for BM25 retriever.

    This retriever uses the BM25 algorithm for document retrieval,
    which is effective for keyword-based search and traditional information retrieval.

    Attributes:
        documents: List of documents to retrieve from
        k: Number of documents to retrieve
        bm25_params: Parameters for the BM25 algorithm
        preprocess_func: Function to preprocess text before BM25 vectorization

    Example:
        ```python
        from haive.core.engine.retriever.bm25 import BM25RetrieverConfig

        config = BM25RetrieverConfig(
            name="bm25_retriever",
            documents=documents,
            k=4,
            bm25_params={"k1": 1.2, "b": 0.75}
        )

        retriever = config.instantiate()
        docs = retriever.get_relevant_documents("query")
        ```
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.BM25, description="The type of retriever"
    )

    documents: List[Document] = Field(
        default_factory=list, description="Documents to retrieve from"
    )

    k: int = Field(default=4, description="Number of documents to retrieve")

    bm25_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters for the BM25 algorithm"
    )

    preprocess_func: Callable[[str], List[str]] = Field(
        default=default_preprocessing_func,
        description="Function to preprocess text before BM25 vectorization",
    )

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for BM25 retriever."""
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for BM25 retriever."""
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create a BM25 retriever from this configuration.

        Returns:
            Instantiated BM25 retriever

        Raises:
            ImportError: If rank_bm25 is not installed
        """
        try:
            from langchain_community.retrievers import BM25Retriever
        except ImportError:
            raise ImportError(
                "BM25Retriever requires rank_bm25. Install with: pip install rank-bm25"
            )

        return BM25Retriever.from_documents(
            documents=self.documents,
            k=self.k,
            bm25_params=self.bm25_params,
            preprocess_func=self.preprocess_func,
        )
