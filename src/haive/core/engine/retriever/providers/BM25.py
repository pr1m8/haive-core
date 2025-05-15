# src/haive/core/engine/retriever/bm25.py

from typing import Any, Callable, Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers.bm25 import default_preprocessing_func
from langchain_core.documents import Document
from pydantic import Field

from haive.core.engine.retriever import BaseRetrieverConfig, RetrieverType


@BaseRetrieverConfig.register(RetrieverType.BM25)
class BM25RetrieverConfig(BaseRetrieverConfig):
    """Configuration for BM25 retriever.

    This retriever uses the BM25 algorithm for document retrieval.
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

    preprocess_func: Callable = Field(
        default=default_preprocessing_func,
        description="Function to preprocess text before BM25 vectorization",
    )

    def instantiate(self) -> BM25Retriever:
        """Create a BM25 retriever from this configuration."""
        return BM25Retriever.from_documents(
            documents=self.documents,
            k=self.k,
            bm25_params=self.bm25_params,
            preprocess_func=self.preprocess_func,
        )
