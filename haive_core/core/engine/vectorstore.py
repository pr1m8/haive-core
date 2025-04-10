from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from src.haive.core.models.embeddings.base import BaseEmbeddingConfig, HuggingFaceEmbeddingConfig
from src.haive.core.engine.base import Engine, EngineType
import logging

logger = logging.getLogger(__name__)

class VectorStoreProvider(str, Enum):
    """Enumeration of supported vector store providers."""
    CHROMA = "Chroma"
    FAISS = "FAISS"
    PINECONE = "Pinecone"
    WEAVIATE = "Weaviate"
    ZILLIZ = "Zilliz"
    MILVUS = "Milvus"
    QDRANT = "Qdrant"
    IN_MEMORY = "InMemory"


class VectorStoreConfig(Engine):
    """Configuration model for a vector store."""
    engine_type: EngineType = Field(default=EngineType.VECTOR_STORE)
    embedding_model: BaseEmbeddingConfig = Field(
        default=HuggingFaceEmbeddingConfig(model="sentence-transformers/all-mpnet-base-v2"),
        description="The embedding model to use for the vector store"
    )
    vector_store_provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.FAISS,
        description="The type of vector store to use"
    )
    vector_store_path: str = Field(default="vector_store", description="The path to the vector store")
    vector_store_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Optional kwargs for the vector store")
    documents: List[Document] = Field(default_factory=list, description="The raw documents to store")
    docstore_path: str = Field(default="docstore", description="Where to store raw and processed documents")
    @field_validator("engine_type")
    def validate_engine_type(cls, v):
        if v != EngineType.VECTOR_STORE:
            raise ValueError("engine_type must be VectorStore")
        return v
    def add_document(self, document: Document):
        """Add a single document to the vector store config."""
        self.documents.append(document)

    def create_runnable(self, async_mode: bool = False):
        """Create a vector store instance from this configuration (implements Engine interface)."""
        return self.create_vectorstore(async_mode)

    def create_vectorstore(self, async_mode: bool = False):
        """Create a vector store instance from this configuration."""
        # Dynamically select the backend
        if self.vector_store_provider == VectorStoreProvider.CHROMA:
            from langchain_community.vectorstores import Chroma
            vs = Chroma
        elif self.vector_store_provider == VectorStoreProvider.FAISS:
            from langchain_community.vectorstores import FAISS
            vs = FAISS
        elif self.vector_store_provider == VectorStoreProvider.PINECONE:
            from langchain_community.vectorstores import Pinecone
            vs = Pinecone
        elif self.vector_store_provider == VectorStoreProvider.WEAVIATE:
            from langchain_community.vectorstores import Weaviate
            vs = Weaviate
        elif self.vector_store_provider == VectorStoreProvider.ZILLIZ:
            from langchain_community.vectorstores import Zilliz
            vs = Zilliz
        elif self.vector_store_provider == VectorStoreProvider.MILVUS:
            from langchain_community.vectorstores import Milvus
            vs = Milvus
        elif self.vector_store_provider == VectorStoreProvider.QDRANT:
            from langchain_community.vectorstores import Qdrant
            vs = Qdrant
        elif self.vector_store_provider == VectorStoreProvider.IN_MEMORY:
            from langchain_core.vectorstores import InMemoryVectorStore
            vs = InMemoryVectorStore
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_provider}")

        # Get embedding model
        embedding = self.embedding_model.instantiate()

        # Instantiate the vector store with appropriate embedding model
        if async_mode:
            return vs.afrom_documents(
                self.documents,
                embedding,
                **self.vector_store_kwargs
            )
        else:
            return vs.from_documents(
                self.documents,
                embedding,
                **self.vector_store_kwargs
            )

    def create_retriever(self, async_mode: bool = False, **kwargs):
        """Create a retriever from the vector store."""
        vectorstore = self.create_vectorstore(async_mode=async_mode)
        return vectorstore.as_retriever(**kwargs)
    
    # Alias functions for backward compatibility
    def instantiate(self, async_mode: bool = False):
        """Alias for create_runnable."""
        return self.create_runnable(async_mode)
    
    def get_vectorstore(self, embedding=None, async_mode: bool = False):
        """Get the vector store with optional embedding override."""
        if embedding:
            # Save original embedding
            original_embedding = self.embedding_model
            # Set the provided embedding
            self.embedding_model = embedding
            # Create vector store
            result = self.create_vectorstore(async_mode)
            # Restore original embedding
            self.embedding_model = original_embedding
            return result
        else:
            return self.create_vectorstore(async_mode)
    
    def _create_retriever(self, **kwargs):
        """Private method for backward compatibility."""
        return self.create_retriever(**kwargs)

    @classmethod
    def create_vs_config_from_documents(cls, documents: List[Document], embedding_model: BaseEmbeddingConfig=None, **kwargs) -> "VectorStoreConfig":
        """Create a VectorStoreConfig from a list of documents."""
        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddingConfig(model="sentence-transformers/all-mpnet-base-v2")
        config = cls(documents=documents, embedding_model=embedding_model, **kwargs)
        return config
    
    @classmethod
    def create_vs_from_documents(cls, documents: List[Document], embedding_model: BaseEmbeddingConfig=None, **kwargs) -> VectorStore:
        """Create a VectorStore from a list of documents."""
        config = cls.create_vs_config_from_documents(documents, embedding_model, **kwargs)
        return config.create_vectorstore()


# Shorthand creator functions
def create_vectorstore(config: VectorStoreConfig, async_mode: bool = False):
    return config.create_vectorstore(async_mode=async_mode)

def create_retriever(config: VectorStoreConfig, async_mode: bool = False, **kwargs):
    return config.create_retriever(async_mode=async_mode, **kwargs)

def create_vs_config_from_documents(documents: List[Document], embedding_model: BaseEmbeddingConfig=None, **kwargs) -> VectorStoreConfig:
    return VectorStoreConfig.create_vs_config_from_documents(documents, embedding_model, **kwargs)

def create_vs_from_documents(documents: List[Document], embedding_model: BaseEmbeddingConfig=None, **kwargs) -> VectorStore:
    return VectorStoreConfig.create_vs_from_documents(documents, embedding_model, **kwargs)

def create_retriever_from_documents(documents: List[Document], embedding_model: BaseEmbeddingConfig=None, **kwargs) -> BaseRetriever:
    config = VectorStoreConfig.create_vs_config_from_documents(documents, embedding_model, **kwargs)
    return config.create_retriever(**kwargs)