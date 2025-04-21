# retrievers/knn.py

from langchain_community.retrievers import KNNRetriever
from langchain_core.documents import Document
from pydantic import model_validator

from haive.core.models.embeddings.base import BaseEmbeddingConfig

from .base import BaseRetrieverConfig


class KNNRetrieverConfig(BaseRetrieverConfig):
    embeddings_config: BaseEmbeddingConfig
    documents: list[Document]
    k: int = 4
    relevancy_threshold: float | None = None
    @model_validator(mode="after")
    def validate_config(cls, data):
        if data.embeddings_config is None:
            raise ValueError("embeddings_config is required")
        return data
    def create(self) -> KNNRetriever:
        return KNNRetriever.from_documents(
            documents=self.documents,
            embeddings=self.embeddings_config.create(),
            k=self.k,
            relevancy_threshold=self.relevancy_threshold,
        )
