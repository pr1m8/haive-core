# src/haive/core/engine/retriever/knn.py

from typing import List, Optional

from langchain_community.retrievers import KNNRetriever
from langchain_core.documents import Document
from pydantic import Field, model_validator

from haive.core.engine.retriever import BaseRetrieverConfig, RetrieverType
from haive.core.models.embeddings.base import BaseEmbeddingConfig


@BaseRetrieverConfig.register(RetrieverType.KNN)
class KNNRetrieverConfig(BaseRetrieverConfig):
    """Configuration for KNN retriever.

    This retriever uses K-Nearest Neighbors for document retrieval.
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.KNN, description="The type of retriever"
    )

    embeddings_config: BaseEmbeddingConfig = Field(
        ..., description="Configuration for the embedding model"  # Required
    )

    documents: List[Document] = Field(
        default_factory=list, description="Documents to retrieve from"
    )

    k: int = Field(default=4, description="Number of documents to retrieve")

    relevancy_threshold: Optional[float] = Field(
        default=None, description="Threshold for relevancy"
    )

    @model_validator(mode="after")
    def validate_config(cls, values):
        """Validate that embeddings_config is provided."""
        if values.embeddings_config is None:
            raise ValueError("embeddings_config is required")
        return values

    def instantiate(self) -> KNNRetriever:
        """Create a KNN retriever from this configuration."""
        embeddings = self.embeddings_config.instantiate()
        return KNNRetriever.from_documents(
            documents=self.documents,
            embeddings=embeddings,
            k=self.k,
            relevancy_threshold=self.relevancy_threshold,
        )
