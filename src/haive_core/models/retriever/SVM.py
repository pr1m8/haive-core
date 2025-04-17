# retrievers/svm.py
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from haive_core.models.embeddings.base import BaseEmbeddingConfig
from .svm_retriever_impl import SVMRetriever
from .base import BaseRetrieverConfig


class SVMRetrieverConfig(BaseRetrieverConfig):
    embeddings_config: BaseEmbeddingConfig
    documents: List[Document]
    k: int = 4
    relevancy_threshold: Optional[float] = None

    def create(self) -> SVMRetriever:
        return SVMRetriever.from_documents(
            documents=self.documents,
            embeddings_config=self.embeddings_config,
            k=self.k,
            relevancy_threshold=self.relevancy_threshold,
        )
