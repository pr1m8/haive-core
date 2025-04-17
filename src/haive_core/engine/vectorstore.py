# src/haive/core/engine/vectorstore.py

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union, Tuple, Type
from enum import Enum
from pydantic import BaseModel, Field, field_validator, create_model, ConfigDict

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig

from haive_core.models.embeddings.base import BaseEmbeddingConfig, HuggingFaceEmbeddingConfig
from haive_core.engine.base import InvokableEngine, EngineType

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

class VectorStoreConfig(InvokableEngine[Union[str, Dict[str, Any]], List[Document]]):
    """
    Configuration model for a vector store engine.
    
    VectorStoreConfig provides a consistent interface for creating and using
    vector stores with embeddings.
    """
    engine_type: EngineType = Field(default=EngineType.VECTOR_STORE)
    
    # Core components
    embedding_model: BaseEmbeddingConfig = Field(
        default=HuggingFaceEmbeddingConfig(model="sentence-transformers/all-mpnet-base-v2"),
        description="The embedding model to use for the vector store"
    )
    vector_store_provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.FAISS,
        description="The type of vector store to use"
    )
    
    # Content and storage
    documents: List[Document] = Field(
        default_factory=list, 
        description="The raw documents to store"
    )
    vector_store_path: str = Field(
        default="vector_store", 
        description="The path to the vector store"
    )
    docstore_path: str = Field(
        default="docstore", 
        description="Where to store raw and processed documents"
    )
    
    # Search parameters
    k: int = Field(
        default=4, 
        description="Default number of documents to retrieve"
    )
    score_threshold: Optional[float] = Field(
        default=None, 
        description="Score threshold for similarity search"
    )
    search_type: str = Field(
        default="similarity", 
        description="Default search type (similarity, mmr, etc.)"
    )
    
    # Additional configuration
    vector_store_kwargs: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Optional kwargs for the vector store"
    )
        
    model_config = ConfigDict(arbitrary_types_allowed = True, )
    
    @field_validator("engine_type")
    def validate_engine_type(cls, v):
        if v != EngineType.VECTOR_STORE:
            raise ValueError("engine_type must be VECTOR_STORE")
        return v
    
    def add_document(self, document: Document) -> None:
        """
        Add a single document to the vector store config.
        
        Args:
            document: Document to add
        """
        self.documents.append(document)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to the vector store config.
        
        Args:
            documents: List of documents to add
        """
        self.documents.extend(documents)
    
    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> VectorStore:
        """
        Create a vector store instance with configuration applied.
        
        Args:
            runnable_config: Optional runtime configuration
            
        Returns:
            Instantiated vector store
        """
        # Extract config parameters
        params = self.apply_runnable_config(runnable_config)
        
        # Apply embedding model override if specified
        if "embedding_model" in params:
            return self.get_vectorstore(embedding=params["embedding_model"])
        
        # Create vector store with existing configuration
        return self.create_vectorstore()
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None, 
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
        search_type: Optional[str] = None,
        runnable_config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        """
        Perform similarity search with configurable parameters.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides default)
            score_threshold: Score threshold for filtering results
            filter: Optional filter for the search
            search_type: Search type (similarity, mmr, etc.)
            runnable_config: Optional runtime configuration
            
        Returns:
            List of retrieved documents
        """
        # Create vector store
        vectorstore = self.create_runnable(runnable_config)
        
        # Extract search parameters from config
        search_params = {}
        if runnable_config and "configurable" in runnable_config:
            configurable = runnable_config["configurable"]
            if "k" in configurable:
                search_params["k"] = configurable["k"]
            if "score_threshold" in configurable:
                search_params["score_threshold"] = configurable["score_threshold"]
            if "filter" in configurable:
                search_params["filter"] = configurable["filter"]
            if "search_type" in configurable:
                search_type = configurable["search_type"]
                
        # Override with explicit parameters if provided
        if k is not None:
            search_params["k"] = k
        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold
        if filter is not None:
            search_params["filter"] = filter
            
        # Use default k if not specified
        if "k" not in search_params:
            search_params["k"] = self.k
        
        # Use default search type if not specified
        search_type = search_type or self.search_type
        
        # Perform search based on search type
        if search_type == "similarity":
            return vectorstore.similarity_search(query, **search_params)
        elif search_type == "mmr":
            return vectorstore.max_marginal_relevance_search(query, **search_params)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
    
    def invoke(
        self, 
        input_data: Union[str, Dict[str, Any]], 
        runnable_config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        """
        Invoke the vector store with input data.
        
        Args:
            input_data: Query string or dictionary with search parameters
            runnable_config: Optional runtime configuration
            
        Returns:
            List of retrieved documents
        """
        # Handle different input formats
        if isinstance(input_data, str):
            query = input_data
            k = None
            filter = None
            score_threshold = None
            search_type = None
        elif isinstance(input_data, dict):
            query = input_data.get("query", "")
            k = input_data.get("k")
            filter = input_data.get("filter")
            score_threshold = input_data.get("score_threshold")
            search_type = input_data.get("search_type")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Perform search
        return self.similarity_search(
            query, 
            k=k, 
            filter=filter, 
            score_threshold=score_threshold,
            search_type=search_type,
            runnable_config=runnable_config
        )
    
    def apply_runnable_config(self, runnable_config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Extract parameters from runnable_config relevant to this vector store.
        
        Args:
            runnable_config: Runtime configuration
            
        Returns:
            Dictionary of relevant parameters
        """
        # Get common parameters
        params = super().apply_runnable_config(runnable_config)
        
        if runnable_config and "configurable" in runnable_config:
            configurable = runnable_config["configurable"]
            
            # Extract vector store specific parameters
            if "k" in configurable:
                params["k"] = configurable["k"]
            if "embedding_model" in configurable:
                params["embedding_model"] = configurable["embedding_model"]
            if "score_threshold" in configurable:
                params["score_threshold"] = configurable["score_threshold"]
            if "search_type" in configurable:
                params["search_type"] = configurable["search_type"]
                
        return params
    
    def create_vectorstore(self, async_mode: bool = False) -> VectorStore:
        """
        Create a vector store instance from this configuration.
        
        Args:
            async_mode: Whether to use async methods
            
        Returns:
            Instantiated vector store
        """
        # Dynamically select the backend
        vs_cls = self._get_vectorstore_class()
        
        # Get embedding model
        embedding = self.embedding_model.instantiate()
        
        # Check if we have documents
        if not self.documents:
            logger.warning(f"Creating empty vector store: {self.name}")
            
            # Use from_texts with empty list for providers that support it
            if hasattr(vs_cls, "from_texts") or hasattr(vs_cls, "afrom_texts"):
                if async_mode:
                    if hasattr(vs_cls, "afrom_texts"):
                        return vs_cls.afrom_texts(
                            [], 
                            embedding, 
                            **self.vector_store_kwargs
                        )
                else:
                    if hasattr(vs_cls, "from_texts"):
                        return vs_cls.from_texts(
                            [], 
                            embedding, 
                            **self.vector_store_kwargs
                        )
            
            # Fallback to empty constructor if available
            if hasattr(vs_cls, "__init__"):
                return vs_cls(embedding_function=embedding, **self.vector_store_kwargs)
            
            raise ValueError("Cannot create empty vector store with this provider")
        
        # Instantiate the vector store with provided documents
        if async_mode:
            return vs_cls.afrom_documents(
                self.documents,
                embedding,
                **self.vector_store_kwargs
            )
        else:
            return vs_cls.from_documents(
                self.documents,
                embedding,
                **self.vector_store_kwargs
            )
    
    def _get_vectorstore_class(self) -> Type[VectorStore]:
        """
        Get the vector store class based on provider.
        
        Returns:
            VectorStore class
        """
        if self.vector_store_provider == VectorStoreProvider.CHROMA:
            from langchain_community.vectorstores import Chroma
            return Chroma
        elif self.vector_store_provider == VectorStoreProvider.FAISS:
            from langchain_community.vectorstores import FAISS
            return FAISS
        elif self.vector_store_provider == VectorStoreProvider.PINECONE:
            from langchain_community.vectorstores import Pinecone
            return Pinecone
        elif self.vector_store_provider == VectorStoreProvider.WEAVIATE:
            from langchain_community.vectorstores import Weaviate
            return Weaviate
        elif self.vector_store_provider == VectorStoreProvider.ZILLIZ:
            from langchain_community.vectorstores import Zilliz
            return Zilliz
        elif self.vector_store_provider == VectorStoreProvider.MILVUS:
            from langchain_community.vectorstores import Milvus
            return Milvus
        elif self.vector_store_provider == VectorStoreProvider.QDRANT:
            from langchain_community.vectorstores import Qdrant
            return Qdrant
        elif self.vector_store_provider == VectorStoreProvider.IN_MEMORY:
            from langchain_core.vectorstores import InMemoryVectorStore
            return InMemoryVectorStore
        else:
            raise ValueError(f"Unsupported vector store provider: {self.vector_store_provider}")
    
    def create_retriever(
        self, 
        search_type: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseRetriever:
        """
        Create a retriever from the vector store.
        
        Args:
            search_type: Search type (similarity, mmr, etc.)
            search_kwargs: Search parameters
            **kwargs: Additional parameters for the retriever
            
        Returns:
            Configured retriever
        """
        # Create vector store
        vectorstore = self.create_vectorstore()
        
        # Set default search kwargs if not provided
        if search_kwargs is None:
            search_kwargs = {"k": self.k}
            if self.score_threshold is not None:
                search_kwargs["score_threshold"] = self.score_threshold
        
        # Use specified search type or default
        search_type = search_type or self.search_type
        
        # Create retriever
        return vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
            **kwargs
        )
    
    def derive_input_schema(self) -> Type[BaseModel]:
        """
        Derive input schema for this engine.
        
        Returns:
            Pydantic model for input schema
        """
        # Use provided schema if available
        if self.input_schema:
            return self.input_schema
        
        # Create a simple input schema
        return create_model(
            f"{self.__class__.__name__}Input",
            query=(str, ...),
            k=(Optional[int], None),
            filter=(Optional[Dict[str, Any]], None),
            score_threshold=(Optional[float], None),
            search_type=(Optional[str], None)
        )
    
    def derive_output_schema(self) -> Type[BaseModel]:
        """
        Derive output schema for this engine.
        
        Returns:
            Pydantic model for output schema
        """
        # Use provided schema if available
        if self.output_schema:
            return self.output_schema
        
        # Create output schema for documents
        return create_model(
            f"{self.__class__.__name__}Output",
            documents=(List[Document], ...)
        )
    
    def get_schema_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Get schema fields for this engine.
        
        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        from typing import Optional, Dict, List, Any
        
        fields = {
            "query": (str, ...),
            "k": (Optional[int], None),
            "filter": (Optional[Dict[str, Any]], None),
            "score_threshold": (Optional[float], None),
            "search_type": (Optional[str], None),
            "documents": (List[Document], [])
        }
        
        return fields
    
    def get_vectorstore(self, embedding=None, async_mode: bool = False) -> VectorStore:
        """
        Get the vector store with optional embedding override.
        
        Args:
            embedding: Optional embedding model override
            async_mode: Whether to use async methods
            
        Returns:
            Instantiated vector store
        """
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
    
    @classmethod
    def create_vs_config_from_documents(
        cls, 
        documents: List[Document], 
        embedding_model: Optional[BaseEmbeddingConfig] = None, 
        **kwargs
    ) -> "VectorStoreConfig":
        """
        Create a VectorStoreConfig from a list of documents.
        
        Args:
            documents: List of documents to include
            embedding_model: Optional embedding model configuration
            **kwargs: Additional parameters for the config
            
        Returns:
            Configured VectorStoreConfig
        """
        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddingConfig(model="sentence-transformers/all-mpnet-base-v2")
            
        return cls(documents=documents, embedding_model=embedding_model, **kwargs)
    
    @classmethod
    def create_vs_from_documents(
        cls, 
        documents: List[Document], 
        embedding_model: Optional[BaseEmbeddingConfig] = None, 
        **kwargs
    ) -> VectorStore:
        """
        Create a VectorStore from a list of documents.
        
        Args:
            documents: List of documents to include
            embedding_model: Optional embedding model configuration
            **kwargs: Additional parameters for the config
            
        Returns:
            Instantiated VectorStore
        """
        config = cls.create_vs_config_from_documents(documents, embedding_model, **kwargs)
        return config.create_vectorstore()


# Shorthand creator functions
def create_vectorstore(config: VectorStoreConfig, async_mode: bool = False) -> VectorStore:
    """
    Create a vector store from a configuration.
    
    Args:
        config: Vector store configuration
        async_mode: Whether to use async methods
        
    Returns:
        Instantiated vector store
    """
    return config.create_vectorstore(async_mode=async_mode)

def create_retriever(config: VectorStoreConfig, **kwargs) -> BaseRetriever:
    """
    Create a retriever from a vector store configuration.
    
    Args:
        config: Vector store configuration
        **kwargs: Additional parameters for the retriever
        
    Returns:
        Configured retriever
    """
    return config.create_retriever(**kwargs)

def create_vs_config_from_documents(
    documents: List[Document], 
    embedding_model: Optional[BaseEmbeddingConfig] = None, 
    **kwargs
) -> VectorStoreConfig:
    """
    Create a VectorStoreConfig from a list of documents.
    
    Args:
        documents: List of documents to include
        embedding_model: Optional embedding model configuration
        **kwargs: Additional parameters for the config
        
    Returns:
        Configured VectorStoreConfig
    """
    return VectorStoreConfig.create_vs_config_from_documents(documents, embedding_model, **kwargs)

def create_vs_from_documents(
    documents: List[Document], 
    embedding_model: Optional[BaseEmbeddingConfig] = None, 
    **kwargs
) -> VectorStore:
    """
    Create a VectorStore from a list of documents.
    
    Args:
        documents: List of documents to include
        embedding_model: Optional embedding model configuration
        **kwargs: Additional parameters for the config
        
    Returns:
        Instantiated VectorStore
    """
    return VectorStoreConfig.create_vs_from_documents(documents, embedding_model, **kwargs)

def create_retriever_from_documents(
    documents: List[Document], 
    embedding_model: Optional[BaseEmbeddingConfig] = None, 
    **kwargs
) -> BaseRetriever:
    """
    Create a retriever directly from documents.
    
    Args:
        documents: List of documents to include
        embedding_model: Optional embedding model configuration
        **kwargs: Additional parameters for the retriever
        
    Returns:
        Configured retriever
    """
    config = VectorStoreConfig.create_vs_config_from_documents(documents, embedding_model)
    return config.create_retriever(**kwargs)