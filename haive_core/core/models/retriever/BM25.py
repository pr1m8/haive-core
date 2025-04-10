from typing import List, Optional, Dict, Callable
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from .base import BaseRetrieverConfig  # shared interface
from langchain_community.retrievers import BM25Retriever,default_preprocessing_func
from langchain_core.documents import Document
#from langchain_core.preprocessors import default_preprocessing_func


class BM25RetrieverConfig(BaseRetrieverConfig):
    documents: List[Document]
    k: int = 4
    bm25_params: Optional[Dict] = None
    preprocess_func: Optional[Callable] = Field(default=default_preprocessing_func)

    def create(self) -> BM25Retriever:
        return BM25Retriever.from_documents(
            documents=self.documents,
            k=self.k,
            bm25_params=self.bm25_params,
            preprocess_func=self.preprocess_func,
        )