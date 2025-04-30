# src/haive/core/engine/retriever.py

"""Retriever engine implementation for the Haive framework.

This module provides a flexible and extensible interface for document retrieval in the Haive framework.
It includes base classes and implementations for various retriever types, with a focus on vector
store-based retrieval.

The module supports different retriever types through a plugin architecture, allowing easy extension
with new retriever implementations while maintaining a consistent interface.

Classes:
    RetrieverConfig: Base configuration class for all retrievers
    VectorStoreRetrieverConfig: Configuration for vector store-based retrievers

Functions:
    create_retriever_config: Factory function for creating retriever configurations
    create_retriever_from_vectorstore: Helper to create a retriever from a vector store

Example:
    Basic usage of creating a vector store retriever:
    ```python
    from haive.core.engine.retriever import VectorStoreRetrieverConfig
    from haive.core.engine.vectorstore import VectorStoreConfig
    
    # Create vector store config
    vs_config = VectorStoreConfig(...)
    
    # Create retriever config
    retriever_config = VectorStoreRetrieverConfig(
        name="my_retriever",
        vector_store_config=vs_config,
        k=4
    )
    
    # Create and use the retriever
    retriever = retriever_config.instantiate()
    docs = retriever.get_relevant_documents("query")
    ```
"""

import logging
from typing import Any, ClassVar, Optional, Union, List, Dict, Type

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from haive.core.engine.base import EngineType, InvokableEngine
from haive.core.engine.retriever.types import RetrieverType
from haive.core.engine.vectorstore.vectorstore import VectorStoreConfig

logger = logging.getLogger(__name__)


# Define explicit input and output schemas
class RetrieverInput(BaseModel):
    """Schema for retriever input."""
    query: str = Field(description="Query string for retrieval")
    k: Optional[int] = Field(default=None, description="Number of documents to retrieve")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Filter criteria for retrieval")
    search_type: Optional[str] = Field(default=None, description="Type of search to perform")
    score_threshold: Optional[float] = Field(default=None, description="Minimum score threshold")

    model_config = ConfigDict(extra="allow")


class RetrieverOutput(BaseModel):
    """Schema for retriever output."""
    documents: List[Document] = Field(description="Retrieved documents")

    model_config = ConfigDict(extra="allow")


class BaseRetrieverConfig(InvokableEngine[Union[str, Dict[str, Any]], List[Document]]):
    """Base configuration for all retriever engines in the Haive framework.

    This class serves as the foundation for all retriever configurations, providing a consistent
    interface for document retrieval operations. It supports various retriever types through a
    plugin architecture and includes common parameters for search customization.

    Attributes:
        engine_type (EngineType): The type of engine (always RETRIEVER).
        retriever_type (RetrieverType): The specific type of retriever to use.
        description (Optional[str]): Optional description of the retriever.
        search_type (str): The type of search to perform ('similarity', 'mmr', etc.).
        search_kwargs (Dict[str, Any]): Additional search parameters.
        k (int): Number of documents to retrieve.
        filter (Optional[Dict[str, Any]]): Optional filter to apply to vector store search.
        _registry (ClassVar[Dict[RetrieverType, Type['RetrieverConfig']]]): Registry for retriever types.
        input_schema (Type[BaseModel]): Schema for retriever input.
        output_schema (Type[BaseModel]): Schema for retriever output.

    Example:
        ```python
        from haive.core.engine.retriever import RetrieverConfig, RetrieverType
        from haive.core.engine.vectorstore import VectorStoreConfig

        # Create a basic retriever config
        config = RetrieverConfig(
            name="my_retriever",
            retriever_type=RetrieverType.VECTOR_STORE,
            k=4,
            search_type="similarity"
        )

        # Create and use the retriever
        retriever = config.instantiate()
        docs = retriever.get_relevant_documents("query")
        ```
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

    # Explicitly set input and output schemas
    input_schema: Type[BaseModel] = Field(
        default=RetrieverInput,
        description="Input schema for this retriever"
    )
    output_schema: Type[BaseModel] = Field(
        default=RetrieverOutput,
        description="Output schema for this retriever"
    )

    # Registry for retriever types
    _registry: ClassVar[Dict[RetrieverType, Type["BaseRetrieverConfig"]]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("engine_type")
    def validate_engine_type(cls, v):
        if v != EngineType.RETRIEVER:
            raise ValueError("engine_type must be RETRIEVER")
        return v

    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> BaseRetriever:
        """Create a retriever with configuration applied.
        
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
        """Create the retriever instance.
        
        Returns:
            Instantiated retriever
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement instantiate method")

    def apply_runnable_config(self, runnable_config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Extract parameters from runnable_config relevant to this retriever.
        
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
        """Invoke the retriever with input data.
        
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
        return retriever.invoke(query)

    def derive_input_schema(self) -> Type[BaseModel]:
        """Return the input schema.
        
        Returns:
            Pydantic model for input schema
        """
        return self.input_schema

    def derive_output_schema(self) -> Type[BaseModel]:
        """Return the output schema.
        
        Returns:
            Pydantic model for output schema
        """
        return self.output_schema

    def get_schema_fields(self) -> Dict[str, tuple[Type, Any]]:
        """Get schema fields for this engine.
        
        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        from typing import Optional, List, Dict, Any

        # Use input schema fields
        input_fields = {}
        for name, field_info in self.input_schema.model_fields.items():
            input_fields[name] = (field_info.annotation, field_info.default)
            
        # Add output schema fields
        output_fields = {}
        for name, field_info in self.output_schema.model_fields.items():
            output_fields[name] = (field_info.annotation, field_info.default)
            
        # Combine fields
        return {**input_fields, **output_fields}

    @classmethod
    def register(cls, retriever_type: RetrieverType):
        """Register a retriever config implementation.
        
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
    def get_config_class(cls, retriever_type: RetrieverType) -> Type["BaseRetrieverConfig"]:
        """Get the appropriate config class for the retriever type.
        
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
    def from_retriever_type(cls, retriever_type: RetrieverType, **kwargs) -> "BaseRetrieverConfig":
        """Create the appropriate config for the given retriever type.
        
        Args:
            retriever_type: Type of retriever to create
            **kwargs: Additional parameters for the config
            
        Returns:
            Configured retriever config
        """
        config_class = cls.get_config_class(retriever_type)
        return config_class(retriever_type=retriever_type, **kwargs)


@BaseRetrieverConfig.register(RetrieverType.VECTOR_STORE)
class VectorStoreRetrieverConfig(BaseRetrieverConfig):
    """Configuration for a vector store-based retriever in the Haive framework.

    This class extends RetrieverConfig to provide specific configuration for vector store-based
    document retrieval. It integrates with various vector stores and supports customizable
    search parameters for efficient document retrieval.

    Attributes:
        vector_store_config (VectorStoreConfig): Configuration for the underlying vector store.
        k (int): Number of documents to retrieve (default: 4).
        search_type (str): Type of search to perform, e.g., 'similarity' or 'mmr' (default: 'similarity').
        search_kwargs (Dict[str, Any]): Additional parameters for the search operation.
        filter (Optional[Dict[str, Any]]): Optional metadata filter for the search.

    Example:
        ```python
        from haive.core.engine.retriever import VectorStoreRetrieverConfig
        from haive.core.engine.vectorstore import VectorStoreConfig

        # Create a vector store config
        vector_store_config = VectorStoreConfig(
            name="my_vectorstore",
            store_type="chroma",
            embedding_config={"model": "sentence-transformers/all-mpnet-base-v2"}
        )

        # Create a vector store retriever config
        retriever_config = VectorStoreRetrieverConfig(
            name="my_retriever",
            vector_store_config=vector_store_config,
            k=4,
            search_type="mmr",
            search_kwargs={"fetch_k": 20, "lambda_mult": 0.5}
        )

        # Create and use the retriever
        retriever = retriever_config.instantiate()
        docs = retriever.get_relevant_documents("query")
        ```
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
        """Create a VectorStoreRetriever instance based on this configuration.
        
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
) -> BaseRetrieverConfig:
    """Factory function to create appropriate retriever configuration.
    
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
    return BaseRetrieverConfig.from_retriever_type(retriever_type, **config_params)


def create_retriever_from_vectorstore(
    vector_store_config: VectorStoreConfig,
    **kwargs
) -> BaseRetriever:
    """Create a retriever from a vector store configuration.
    
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