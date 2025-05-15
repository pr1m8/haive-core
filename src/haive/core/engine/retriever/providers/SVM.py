# retrievers/svm.py

from typing import List

from langchain.retrievers import SVMRetriever
from langchain_core.documents import Document
from pydantic import Field

from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.models.embeddings.base import BaseEmbeddingConfig


class SVMRetrieverConfig(BaseRetrieverConfig):
    embeddings_config: BaseEmbeddingConfig
    documents: List[Document] = Field(default_factory=list)
    k: int = 4
    relevancy_threshold: float | None = None

    def create(self) -> SVMRetriever:
        return SVMRetriever.from_documents(
            documents=self.documents,
            embeddings_config=self.embeddings_config,
            k=self.k,
            relevancy_threshold=self.relevancy_threshold,
        )
