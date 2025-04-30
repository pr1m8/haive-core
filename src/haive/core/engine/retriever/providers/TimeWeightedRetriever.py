# src/haive/core/engine/retriever/time_weighted.py

from typing import List, Optional, Dict, Any
from pydantic import Field

from langchain_core.documents import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever

from haive.core.engine.retriever import BaseRetrieverConfig, RetrieverType
from haive.core.engine.vectorstore import VectorStoreConfig

@BaseRetrieverConfig.register(RetrieverType.TIME_WEIGHTED)
class TimeWeightedRetrieverConfig(BaseRetrieverConfig):
    """Configuration for TimeWeighted retriever.
    
    This retriever combines embedding similarity with recency in retrieving documents.
    """
    retriever_type: RetrieverType = Field(
        default=RetrieverType.TIME_WEIGHTED,
        description="The type of retriever"
    )
    
    vector_store_config: VectorStoreConfig = Field(
        ...,  # Required
        description="Configuration for the vector store"
    )
    
    memory_stream: List[Document] = Field(
        default_factory=list,
        description="Memory stream of documents to search through"
    )
    
    decay_rate: float = Field(
        default=0.01,
        description="Exponential decay factor used as (1.0-decay_rate)**(hrs_passed)"
    )
    
    k: int = Field(
        default=4,
        description="Number of documents to retrieve"
    )
    
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: dict(k=100),
        description="Keyword arguments to pass to the vectorstore similarity search"
    )
    
    other_score_keys: List[str] = Field(
        default_factory=list,
        description="Other keys in the metadata to factor into the score"
    )
    
    default_salience: Optional[float] = Field(
        default=None,
        description="Salience to assign memories not retrieved from the vector store"
    )
    
    def instantiate(self) -> TimeWeightedVectorStoreRetriever:
        """Create a TimeWeighted retriever from this configuration."""
        # Create the vector store
        vectorstore = self.vector_store_config.instantiate()
        
        # Create the retriever
        return TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore,
            memory_stream=self.memory_stream,
            decay_rate=self.decay_rate,
            k=self.k,
            search_kwargs=self.search_kwargs,
            other_score_keys=self.other_score_keys,
            default_salience=self.default_salience
        )