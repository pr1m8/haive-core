# src/haive/core/engine/retriever/ensemble.py

"""Ensemble Retriever implementation for the Haive framework.

This module provides a configuration class for the Ensemble retriever,
which combines results from multiple retrievers using weighted Reciprocal Rank Fusion.

Reciprocal Rank Fusion (RRF) is a method for combining multiple ranked lists into a single
ranked list. It works by:
1. Assigning each document a score based on its rank in each retriever's results
2. Using the formula: score = sum(weight_i * 1/(rank + c)) for each retriever i
3. Ranking documents by their combined score from highest to lowest

The constant 'c' prevents documents that appear at the top of only one retriever's
results from dominating the ensemble. This approach is especially effective when
different retrieval methods have complementary strengths and weaknesses.
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
    strategies to improve overall recall and precision. The ensemble approach
    leverages the strengths of different retrieval methods while mitigating
    their individual weaknesses.

    Common ensemble combinations include:
    - Vector store + BM25/TF-IDF (semantic + keyword matching)
    - Multiple vector stores with different embedding models
    - Specialized retrievers for different document types or domains
    - Time-weighted + similarity-based retrievers

    Attributes:
        retrievers (List[BaseRetrieverConfig]): List of retriever configurations to
            ensemble. Each retriever will be instantiated and used in the ensemble.
        weights (Optional[List[float]]): Weight to assign to each retriever's results.
            If not provided, equal weights will be assigned to all retrievers.
            Weights do not need to sum to 1 as they will be used proportionally.
        c (int): Constant added to the rank in the RRF formula: 1/(rank + c).
            Higher values of c reduce the influence of high-ranked documents.
            Default is 60.
        id_key (Optional[str]): Key in document metadata used to determine unique
            documents when merging results. If not provided, documents will be
            treated as unique based on their content and metadata.

    Example:
        ```python
        from haive.core.engine.retriever.ensemble import EnsembleRetrieverConfig
        from haive.core.engine.retriever.tfidf import TFIDFRetrieverConfig
        from haive.core.engine.retriever.vectorstore import VectorStoreRetrieverConfig
        from haive.core.engine.vectorstore import VectorStoreConfig
        from haive.core.models.embeddings.base import OpenAIEmbeddingConfig

        # Create a vector store retriever config
        vector_config = VectorStoreRetrieverConfig(
            name="vector_retriever",
            vector_store_config=VectorStoreConfig(
                name="main_vectorstore",
                provider="FAISS",
                embedding_model="text-embedding-3-small"
            ),
            search_type="similarity",
            k=10
        )

        # Create a TF-IDF retriever config with the same documents
        tfidf_config = TFIDFRetrieverConfig(
            name="tfidf_retriever",
            documents=documents,  # Same documents as in vector store
            k=10
        )

        # Create an ensemble retriever that combines both approaches
        ensemble_config = EnsembleRetrieverConfig(
            name="hybrid_retriever",
            retrievers=[vector_config, tfidf_config],
            weights=[0.7, 0.3],  # Give more weight to vector similarity
            c=40  # Lower c value gives more influence to top-ranked documents
        )

        # Instantiate the retriever
        hybrid_retriever = ensemble_config.instantiate()

        # Use the ensemble retriever for queries that benefit from both
        # semantic similarity and keyword matching
        results = hybrid_retriever.get_relevant_documents(
            "What are the environmental impacts of renewable energy?"
        )
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
        """Set default weights if not provided and validate configuration.

        This validator performs two main functions:
        1. If weights are not provided, sets equal weights for all retrievers
        2. Validates that the number of weights matches the number of retrievers

        Equal weights means each retriever contributes equally to the final ranking.
        For example, with 3 retrievers, each would get a weight of 1/3 = 0.333...

        Returns:
            EnsembleRetrieverConfig: The validated configuration instance

        Raises:
            ValueError: If the number of weights doesn't match the number of retrievers
        """
        if not self.weights and self.retrievers:
            n_retrievers = len(self.retrievers)
            self.weights = [1.0 / n_retrievers] * n_retrievers

        if self.weights and len(self.weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")

        return self

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for Ensemble retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each input parameter.

        The EnsembleRetriever accepts the following inputs:
            - query: The text query to search for across all underlying retrievers
            - k: Optional override for the number of documents to retrieve in the
                final ensemble result
        """
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for Ensemble retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each output parameter.

        The EnsembleRetriever produces the following outputs:
            - documents: A list of Document objects retrieved and ranked by the
                ensemble of retrievers using Reciprocal Rank Fusion. The documents
                are ordered from most to least relevant based on the combined scores
                from all retrievers.
        """
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Ensemble retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create an Ensemble retriever from this configuration.

        This method instantiates an EnsembleRetriever from LangChain, which combines
        multiple retrievers using weighted Reciprocal Rank Fusion. The method first
        instantiates all the individual retrievers specified in the retrievers list,
        then creates the ensemble with the specified weights and parameters.

        Returns:
            EnsembleRetriever: An instantiated ensemble retriever that combines
                results from multiple retrievers using Reciprocal Rank Fusion.

        Raises:
            ImportError: If EnsembleRetriever is not available in the current
                LangChain version.
            ValueError: If the configuration is invalid, such as having mismatched
                numbers of retrievers and weights.

        Example:
            ```python
            # Create a configuration with multiple retrievers
            config = EnsembleRetrieverConfig(
                name="hybrid_search",
                retrievers=[vector_config, keyword_config],
                weights=[0.6, 0.4]
            )

            # Instantiate the ensemble retriever
            ensemble = config.instantiate()

            # Use the ensemble for retrieval
            results = ensemble.get_relevant_documents("What are neural networks?")
            ```
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
