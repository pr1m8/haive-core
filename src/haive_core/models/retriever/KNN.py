# retrievers/knn.py
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from haive_core.models.embeddings.base import BaseEmbeddingConfig
from langchain_community.retrievers import KNNRetriever
from .base import BaseRetrieverConfig
from pydantic import model_validator

class KNNRetrieverConfig(BaseRetrieverConfig):
    embeddings_config: BaseEmbeddingConfig
    documents: List[Document]
    k: int = 4
    relevancy_threshold: Optional[float] = None
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
