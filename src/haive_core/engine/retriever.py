# src/haive/core/engine/retriever.py

from typing import Any, Dict, List, Optional, Union, ClassVar, Type, Tuple
from pydantic import BaseModel, Field, field_validator, create_model, ConfigDict
from enum import Enum
import logging
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig

from haive_core.engine.base import InvokableEngine, EngineType
from haive_core.engine.vectorstore import VectorStoreConfig

logger = logging.getLogger(__name__)

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


class RetrieverConfig(InvokableEngine[Union[str, Dict[str, Any]], List[Document]]):
    """
    Base configuration for all retriever engines.
    
    Retrievers provide a consistent interface for retrieving relevant documents
    based on a query.
    """
    engine_type: EngineType = Field(default=EngineType.RETRIEVER)
    retriever_type: RetrieverType = Field(
        description="The type of retriever to use",
        default=RetrieverType.VECTOR_STORE
    )
    description: Optional[str] = Field(
        default=None, 
        description="Description of this retriever"
    )
    
    # Common retriever parameters
    search_type: str = Field(
        default="similarity", 
        description="Search type ('similarity', 'mmr', etc.)"
    )
    search_kwargs: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional search parameters"
    )
    k: int = Field(
        default=4, 
        description="Number of documents to retrieve"
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filter to apply to vector store search"
    )
    
    # Registry for retriever types
    _registry: ClassVar[Dict[RetrieverType, Type['RetrieverConfig']]] = {}
    
    model_config = ConfigDict(arbitrary_types_allowed = True, )
    
    @field_validator("engine_type")
    def validate_engine_type(cls, v):
        if v != EngineType.RETRIEVER:
            raise ValueError("engine_type must be RETRIEVER")
        return v
    
    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> BaseRetriever:
        """
        Create a retriever with configuration applied.
        
        Args:
            runnable_config: Optional runtime configuration
            
        Returns:
            Instantiated retriever
        """
        # Extract parameters from runnable_config
        params = self.apply_runnable_config(runnable_config)
        
        # Apply k parameter if specified
        if "k" in params:
            self.k = params["k"]
            
            # Update search_kwargs if it contains k
            if "k" in self.search_kwargs:
                self.search_kwargs["k"] = params["k"]
        
        # Apply filter parameter if specified
        if "filter" in params:
            self.filter = params["filter"]
            
            # Update search_kwargs if it contains filter
            if "filter" in self.search_kwargs:
                self.search_kwargs["filter"] = params["filter"]
                
        # Apply search_type if specified
        if "search_type" in params:
            self.search_type = params["search_type"]
        
        # Create the retriever with updated configuration
        return self.instantiate()
    
    def instantiate(self) -> BaseRetriever:
        """
        Create the retriever instance.
        
        Returns:
            Instantiated retriever
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement instantiate method")
    
    def apply_runnable_config(self, runnable_config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Extract parameters from runnable_config relevant to this retriever.
        
        Args:
            runnable_config: Runtime configuration
            
        Returns:
            Dictionary of relevant parameters
        """
        # Start with common parameters
        params = super().apply_runnable_config(runnable_config)
        
        if runnable_config and "configurable" in runnable_config:
            configurable = runnable_config["configurable"]
            
            # Extract retriever-specific parameters
            if "k" in configurable:
                params["k"] = configurable["k"]
            if "search_type" in configurable:
                params["search_type"] = configurable["search_type"]
            if "filter" in configurable:
                params["filter"] = configurable["filter"]
                
        return params
    
    def invoke(
        self, 
        input_data: Union[str, Dict[str, Any]], 
        runnable_config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        """
        Invoke the retriever with input data.
        
        Args:
            input_data: Query string or dictionary with query parameters
            runnable_config: Optional runtime configuration
            
        Returns:
            List of retrieved documents
        """
        # Create retriever with config
        retriever = self.create_runnable(runnable_config)
        
        # Handle different input formats
        if isinstance(input_data, str):
            query = input_data
        elif isinstance(input_data, dict):
            query = input_data.get("query", "")
            
            # Apply additional parameters if provided
            if "k" in input_data and hasattr(retriever, "k"):
                retriever.k = input_data["k"]
            if "filter" in input_data and hasattr(retriever, "search_kwargs"):
                retriever.search_kwargs["filter"] = input_data["filter"]
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Perform retrieval
        return retriever.get_relevant_documents(query)
    
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
            filter=(Optional[Dict[str, Any]], None)
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
            "documents": (List[Document], [])
        }
        
        return fields
    
    @classmethod
    def register(cls, retriever_type: RetrieverType):
        """
        Register a retriever config implementation.
        
        Args:
            retriever_type: Type of retriever to register
            
        Returns:
            Decorator function
        """
        def decorator(subclass):
            cls._registry[retriever_type] = subclass
            return subclass
        return decorator
    
    @classmethod
    def get_config_class(cls, retriever_type: RetrieverType) -> Type['RetrieverConfig']:
        """
        Get the appropriate config class for the retriever type.
        
        Args:
            retriever_type: Type of retriever
            
        Returns:
            Retriever config class
        """
        if retriever_type not in cls._registry:
            logger.warning(f"No registered config for {retriever_type}, using base config")
            return cls
        return cls._registry[retriever_type]
    
    @classmethod
    def from_retriever_type(cls, retriever_type: RetrieverType, **kwargs) -> 'RetrieverConfig':
        """
        Create the appropriate config for the given retriever type.
        
        Args:
            retriever_type: Type of retriever to create
            **kwargs: Additional parameters for the config
            
        Returns:
            Configured retriever config
        """
        config_class = cls.get_config_class(retriever_type)
        return config_class(retriever_type=retriever_type, **kwargs)


@RetrieverConfig.register(RetrieverType.VECTOR_STORE)
class VectorStoreRetrieverConfig(RetrieverConfig):
    """
    Configuration for a vector store retriever.
    
    VectorStoreRetriever wraps a vector store and provides a retrieval interface.
    """
    retriever_type: RetrieverType = Field(
        default=RetrieverType.VECTOR_STORE, 
        description="The type of retriever"
    )
    
    # Required vector store configuration
    vector_store_config: VectorStoreConfig = Field(
        ...,  # This makes it required
        description="Configuration for the vector store to retrieve from"
    )
    
    # Search configuration
    k: int = Field(
        default=4, 
        ge=1,  # minimum of 1 document
        description="Number of documents to retrieve"
    )
    
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
    
    @field_validator("retriever_type")
    def validate_retriever_type(cls, v):
        if v != RetrieverType.VECTOR_STORE:
            raise ValueError("retriever_type must be VectorStore")
        return v
    
    def instantiate(self) -> BaseRetriever:
        """
        Create a VectorStoreRetriever instance based on this configuration.
        
        Returns:
            Instantiated retriever
        """
        try:
            # Prepare search kwargs
            search_kwargs = {"k": self.k}
            if self.search_kwargs:
                search_kwargs.update(self.search_kwargs)
            
            # Add filter if provided
            extra_kwargs = {}
            if self.filter:
                search_kwargs["filter"] = self.filter
            
            # Create retriever
            return self.vector_store_config.create_retriever(
                search_type=self.search_type,
                search_kwargs=search_kwargs,
                **extra_kwargs
            )
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            raise ValueError(f"Failed to create retriever: {str(e)}") from e


# Convenience factory functions

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


def create_retriever_from_vectorstore(
    vector_store_config: VectorStoreConfig, 
    **kwargs
) -> BaseRetriever:
    """
    Create a retriever from a vector store configuration.
    
    Args:
        vector_store_config: Vector store configuration
        **kwargs: Additional parameters for the retriever
        
    Returns:
        Instantiated retriever
    """
    # Create retriever config
    retriever_config = VectorStoreRetrieverConfig(
        name=f"retriever_{vector_store_config.name}",
        vector_store_config=vector_store_config,
        **kwargs
    )
    
    # Instantiate retriever
    return retriever_config.instantiate()