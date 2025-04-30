# src/haive/core/engine/retriever/ensemble.py

from typing import List, Optional
from pydantic import Field, model_validator

from langchain.retrievers import EnsembleRetriever

from haive.core.engine.retriever import BaseRetrieverConfig, RetrieverType

@BaseRetrieverConfig.register(RetrieverType.ENSEMBLE)
class EnsembleRetrieverConfig(BaseRetrieverConfig):
    """Configuration for Ensemble retriever.
    
    This retriever combines results from multiple retrievers.
    """
    retriever_type: RetrieverType = Field(
        default=RetrieverType.ENSEMBLE,
        description="The type of retriever"
    )
    
    retrievers: List[BaseRetrieverConfig] = Field(
        ...,  # Required
        description="List of retriever configurations to ensemble"
    )
    
    weights: List[float] = Field(
        default=None,
        description="Weights for each retriever (defaults to equal weighting)"
    )
    
    c: int = Field(
        default=60,
        description="Constant added to the rank, controlling the balance between high and low ranks"
    )
    
    id_key: Optional[str] = Field(
        default=None,
        description="Key in document metadata used to determine unique documents"
    )
    
    @model_validator(mode="before")
    def set_weights(cls, values):
        """Set default weights if not provided."""
        if not values.get("weights") and "retrievers" in values:
            n_retrievers = len(values["retrievers"])
            values["weights"] = [1.0 / n_retrievers] * n_retrievers
        return values
    
    def instantiate(self) -> EnsembleRetriever:
        """Create an Ensemble retriever from this configuration."""
        # Instantiate all retrievers
        retrievers = [r.instantiate() for r in self.retrievers]
        
        # Create the ensemble retriever
        return EnsembleRetriever(
            retrievers=retrievers,
            weights=self.weights,
            c=self.c,
            id_key=self.id_key
        )