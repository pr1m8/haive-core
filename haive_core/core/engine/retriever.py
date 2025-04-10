from enum import Enum
from pydantic import BaseModel, Field, field_validator  
from typing import Any, Dict, List, Optional, Union, ClassVar, Type
import importlib
import inspect
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.haive.core.engine.base import Engine, EngineType
from src.haive.core.engine.vectorstore import VectorStoreConfig

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  
class RetrieverType(str, Enum):
    """The type of retriever to use."""
    # Base vector store retrievers
    VECTOR_STORE = "VectorStoreRetriever"
    
    # Advanced retrieval strategies
    TIME_WEIGHTED = "TimeWeightedVectorStoreRetriever"
    MULTI_QUERY = "MultiQueryRetriever"
    MULTI_VECTOR = "MultiVectorRetriever"
    PARENT_DOCUMENT = "ParentDocumentRetriever"
    SELF_QUERY = "SelfQueryRetriever"
    CONTEXTUAL_COMPRESSION = "ContextualCompressionRetriever"
    REPHRASE_QUERY = "RePhraseQueryRetriever"
    
    # Ensemble methods
    MERGER = "MergerRetriever"
    ENSEMBLE = "EnsembleRetriever"
    
    # Sparse retrievers
    SPARSE = "SparseRetriever"
    KNN = "KNNRetriever"
    TFIDF = "TFIDFRetriever"
    BM25 = "BM25Retriever"
    SVM = "SVMRetriever"
    
    # Specific implementations
    ELASTICSEARCH = "ElasticsearchRetriever"
    FAISS = "FAISSRetriever"
    IN_MEMORY = "InMemoryRetriever"
    PINECONE = "PineconeRetriever"
    QDRANT = "QdrantRetriever"
    GOLDEN = "GoldenRetriever"
    
    # Graph-based retrievers
    NEO4J = "Neo4jRetriever"
    KNOWLEDGE_GRAPH = "KnowledgeGraphRetriever"


class RetrieverConfig(Engine):
    """Base configuration for all retrievers."""
    engine_type: EngineType = Field(default=EngineType.RETRIEVER)
    retriever_type: RetrieverType = Field(description="The type of retriever to use")
    description: Optional[str] = Field(default=None, description="Description of this retriever")
    
    # Common retriever parameters
    search_type: str = Field(default="similarity", description="Search type ('similarity', 'mmr', etc.)")
    search_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional search parameters")
    k: int = Field(default=4, description="Number of documents to retrieve")
    
    # Core functionality
    _registry: ClassVar[Dict[RetrieverType, Type['RetrieverConfig']]] = {}
    
    def create_runnable(self) -> BaseRetriever:
        """Create a retriever instance based on configuration (implements Engine interface)."""
        return self.instantiate()
        
    def instantiate(self) -> BaseRetriever:
        """Create the retriever instance based on configuration."""
        raise NotImplementedError("Subclasses must implement instantiate method")
    @field_validator("engine_type")
    def validate_engine_type(cls, v):
        if v != EngineType.RETRIEVER:
            raise ValueError("engine_type must be Retriever")
        return v
    @classmethod
    def register(cls, retriever_type: RetrieverType):
        """Decorator to register retriever config implementations."""
        def decorator(subclass):
            cls._registry[retriever_type] = subclass
            return subclass
        return decorator
    
    @classmethod
    def get_config_class(cls, retriever_type: RetrieverType) -> Type['RetrieverConfig']:
        """Get the appropriate config class for the retriever type."""
        if retriever_type not in cls._registry:
            logger.warning(f"No registered config for {retriever_type}, using base config")
            return cls
        return cls._registry[retriever_type]
    
    @classmethod
    def from_retriever_type(cls, retriever_type: RetrieverType, **kwargs) -> 'RetrieverConfig':
        """Create the appropriate config for the given retriever type."""
        config_class = cls.get_config_class(retriever_type)
        return config_class(retriever_type=retriever_type, **kwargs)
    

@RetrieverConfig.register(RetrieverType.VECTOR_STORE)
class VectorStoreRetrieverConfig(RetrieverConfig):
    """Configuration for a vector store retriever."""
    retriever_type: RetrieverType = Field(
        default=RetrieverType.VECTOR_STORE, 
        description="The type of retriever"
    )
    
    # Required vector store configuration
    vector_store_config: Union[VectorStoreConfig,Type[VectorStoreConfig]] = Field(
        ...,  # This makes it required
        description="Configuration for the vector store to retrieve from"
    )
    
    # Number of documents to retrieve
    k: int = Field(
        default=4, 
        ge=1,  # minimum of 1 document
        description="Number of documents to retrieve"
    )
    
    # Search configuration
    search_type: str = Field(
        default="similarity", 
        description="Search type: 'similarity', 'mmr', etc."
    )
    
    search_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional search parameters"
    )
    
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filter to apply to vector store search"
    )
    @field_validator("retriever_type",mode="after")
    def validate_retriever_type(cls, v):
        logging.info(f"Retriever type: {v}")
        if v != RetrieverType.VECTOR_STORE:
            raise ValueError("retriever_type must be VectorStore")
        return v
    def instantiate(self) -> BaseRetriever:
        """Create a VectorStoreRetriever instance based on this configuration."""
        try:
            # Create retriever
            retriever = self._create_retriever()
            
            logger.info(f"Created VectorStoreRetriever '{self.name}' with search_type={self.search_type}")
            return retriever
            
        except Exception as e:
            logger.error(f"Error instantiating VectorStoreRetriever: {str(e)}")
            raise ValueError(f"Failed to create VectorStoreRetriever: {str(e)}")
    
    def _create_retriever(self):
        """Create retriever from vector store."""
        try:
            # Prepare search kwargs
            search_kwargs = {"k": self.k}
            if self.search_kwargs:
                search_kwargs.update(self.search_kwargs)
            
            # Add filter if provided
            extra_kwargs = {}
            if self.filter:
                extra_kwargs["filter"] = self.filter
            
            # Create retriever
            return self.vector_store_config.create_retriever(
                search_type=self.search_type,
                search_kwargs=search_kwargs,
                **extra_kwargs
            )
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            raise ValueError(f"Failed to create retriever: {str(e)}")
    
    def get_retriever(self) -> BaseRetriever:
        """Helper method to create and return the retriever."""
        return self.instantiate()


def create_retriever_config(
    retriever_type: Union[RetrieverType, str],
    name: str,
    description: Optional[str] = None,
    vector_store_config: Optional[VectorStoreConfig] = None,
    llm_config: Optional[Any] = None,
    **kwargs
) -> RetrieverConfig:
    """
    Factory function to create appropriate retriever configuration.
    
    Args:
        retriever_type: Type of retriever to create
        name: Name identifier for this retriever
        description: Description of the retriever
        vector_store_config: Configuration for vector store (if needed)
        llm_config: Configuration for LLM (if needed)
        **kwargs: Additional parameters specific to retriever type
        
    Returns:
        Appropriate retriever configuration object
    """
    # Convert string to enum if needed
    if isinstance(retriever_type, str):
        retriever_type = RetrieverType(retriever_type)
    
    # Create the configuration with common parameters
    config_params = {
        "name": name,
        "description": description,
        **kwargs
    }
    
    # Add specific parameters based on retriever type
    if retriever_type in [
        RetrieverType.VECTOR_STORE, 
        RetrieverType.TIME_WEIGHTED,
        RetrieverType.MULTI_QUERY,
        RetrieverType.SELF_QUERY
    ]:
        if vector_store_config:
            config_params["vector_store_config"] = vector_store_config
    
    if retriever_type in [RetrieverType.MULTI_QUERY, RetrieverType.REPHRASE_QUERY]:
        if llm_config:
            config_params["llm_config"] = llm_config
    
    # Create and return the appropriate configuration
    return RetrieverConfig.from_retriever_type(retriever_type, **config_params)


# Helper function to create a VectorStoreRetriever directly from documents
def create_retriever_from_vectorstore(vector_store_config: VectorStoreConfig, **kwargs) -> BaseRetriever:
    """Create a retriever from a vector store configuration."""
    logging.info(f"Creating retriever from vector store configuration: {type(vector_store_config)}")
    logging.info(f"Vector store configuration: {isinstance(vector_store_config, VectorStoreConfig)}")   
    logging.info(f"Vector store configuration: {vector_store_config}")
    logging.info(f"Module of class: {vector_store_config.__class__.__module__}")
    retriever_config = VectorStoreRetrieverConfig(
        name=f"retriever_{vector_store_config.name}",
        vector_store_config=vector_store_config,
        **kwargs
    )
    return retriever_config.instantiate()