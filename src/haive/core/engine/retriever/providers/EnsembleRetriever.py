# src/haive/core/engine/retriever/ensemble.py

"""Ensemble Retriever implementation for the Haive framework.

This module provides a configuration class for the Ensemble retriever,
which combines results from multiple retrievers using weighted Reciprocal Rank Fusion.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from pydantic import Field, model_validator

from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.engine.retriever.types import RetrieverType


@BaseRetrieverConfig.register(RetrieverType.ENSEMBLE)
class EnsembleRetrieverConfig(BaseRetrieverConfig):
    """Configuration for Ensemble retriever.

    This retriever combines results from multiple retrievers using weighted
    Reciprocal Rank Fusion (RRF). It's useful for combining different retrieval
    strategies to improve overall recall and precision.

    Attributes:
        retrievers: List of retriever configurations to ensemble
        weights: Weights for each retriever (defaults to equal weighting)
        c: Constant added to rank for RRF calculation
        id_key: Key in document metadata for identifying unique documents

    Example:
        ```python
        from haive.core.engine.retriever.ensemble import EnsembleRetrieverConfig

        config = EnsembleRetrieverConfig(
            name="ensemble_retriever",
            retrievers=[bm25_config, vector_config, tfidf_config],
            weights=[0.4, 0.4, 0.2],
            c=60
        )

        retriever = config.instantiate()
        docs = retriever.get_relevant_documents("query")
        ```
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.ENSEMBLE, description="The type of retriever"
    )

    retrievers: List[BaseRetrieverConfig] = Field(
        ..., description="List of retriever configurations to ensemble"
    )

    weights: Optional[List[float]] = Field(
        default=None,
        description="Weights for each retriever (defaults to equal weighting)",
    )

    c: int = Field(
        default=60,
        description="Constant added to the rank, controlling the balance between high and low ranks",
    )

    id_key: Optional[str] = Field(
        default=None,
        description="Key in document metadata used to determine unique documents",
    )

    @model_validator(mode="after")
    def validate_and_set_weights(self):
        """Set default weights if not provided and validate configuration."""
        if not self.weights and self.retrievers:
            n_retrievers = len(self.retrievers)
            self.weights = [1.0 / n_retrievers] * n_retrievers

        if self.weights and len(self.weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")

        return self

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for Ensemble retriever."""
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for Ensemble retriever."""
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Ensemble retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create an Ensemble retriever from this configuration.

        Returns:
            Instantiated Ensemble retriever

        Raises:
            ImportError: If EnsembleRetriever is not available
            ValueError: If configuration is invalid
        """
        try:
            from langchain.retrievers import EnsembleRetriever
        except ImportError:
            raise ImportError(
                "EnsembleRetriever not available in current LangChain version"
            )

        # Instantiate all retrievers
        retrievers = [r.instantiate() for r in self.retrievers]

        # Create the ensemble retriever
        return EnsembleRetriever(
            retrievers=retrievers, weights=self.weights, c=self.c, id_key=self.id_key
        )
