from collections.abc import Callable

from langchain_community.retrievers import BM25Retriever, default_preprocessing_func
from langchain_core.documents import Document
from pydantic import Field

from .base import BaseRetrieverConfig  # shared interface

#from langchain_core.preprocessors import default_preprocessing_func


class BM25RetrieverConfig(BaseRetrieverConfig):
    documents: list[Document]
    k: int = 4
    bm25_params: dict | None = None
    preprocess_func: Callable | None = Field(default=default_preprocessing_func)

    def create(self) -> BM25Retriever:
        return BM25Retriever.from_documents(
            documents=self.documents,
            k=self.k,
            bm25_params=self.bm25_params,
            preprocess_func=self.preprocess_func,
        )
