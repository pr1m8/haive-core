# src/haive/core/engine/retriever/tfidf.py

from typing import Any, Dict, List, Optional

from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
from pydantic import Field

from haive.core.engine.retriever import BaseRetrieverConfig, RetrieverType


@BaseRetrieverConfig.register(RetrieverType.TFIDF)
class TFIDFRetrieverConfig(BaseRetrieverConfig):
    """Configuration for TF-IDF retriever.

    This retriever uses Term Frequency-Inverse Document Frequency for document retrieval.
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

    def instantiate(self) -> TFIDFRetriever:
        """Create a TF-IDF retriever from this configuration."""
        return TFIDFRetriever.from_documents(
            documents=self.documents, k=self.k, tfidf_params=self.tfidf_params
        )
