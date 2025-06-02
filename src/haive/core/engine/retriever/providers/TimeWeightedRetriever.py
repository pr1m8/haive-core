# src/haive/core/engine/retriever/time_weighted.py

"""TimeWeighted Retriever implementation for the Haive framework.

This module provides a configuration class for the TimeWeighted retriever,
which combines embedding similarity with recency in retrieving documents.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from pydantic import Field

from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.engine.retriever.types import RetrieverType
from haive.core.engine.vectorstore import VectorStoreConfig


@BaseRetrieverConfig.register(RetrieverType.TIME_WEIGHTED)
class TimeWeightedRetrieverConfig(BaseRetrieverConfig):
    """Configuration for TimeWeighted retriever.

    This retriever combines embedding similarity with recency in retrieving documents.
    It's particularly useful for scenarios where document freshness matters,
    such as news retrieval or conversation history.

    Attributes:
        vector_store_config: Configuration for the vector store
        memory_stream: Memory stream of documents to search through
        decay_rate: Exponential decay factor for time-based scoring
        k: Number of documents to retrieve
        search_kwargs: Additional search parameters
        other_score_keys: Other metadata keys to factor into scoring
        default_salience: Default salience for non-retrieved documents

    Example:
        ```python
        from haive.core.engine.retriever.time_weighted import TimeWeightedRetrieverConfig
        from haive.core.engine.vectorstore import VectorStoreConfig

        vs_config = VectorStoreConfig(...)

        config = TimeWeightedRetrieverConfig(
            name="time_weighted_retriever",
            vector_store_config=vs_config,
            decay_rate=0.01,
            k=4
        )

        retriever = config.instantiate()
        docs = retriever.get_relevant_documents("query")
        ```
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.TIME_WEIGHTED, description="The type of retriever"
    )

    vector_store_config: VectorStoreConfig = Field(
        ..., description="Configuration for the vector store"
    )

    memory_stream: List[Document] = Field(
        default_factory=list, description="Memory stream of documents to search through"
    )

    decay_rate: float = Field(
        default=0.01,
        description="Exponential decay factor used as (1.0-decay_rate)**(hrs_passed)",
    )

    k: int = Field(default=4, description="Number of documents to retrieve")

    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: dict(k=100),
        description="Keyword arguments to pass to the vectorstore similarity search",
    )

    other_score_keys: List[str] = Field(
        default_factory=list,
        description="Other keys in the metadata to factor into the score",
    )

    default_salience: Optional[float] = Field(
        default=None,
        description="Salience to assign memories not retrieved from the vector store",
    )

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for TimeWeighted retriever."""
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for TimeWeighted retriever."""
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create a TimeWeighted retriever from this configuration.

        Returns:
            Instantiated TimeWeighted retriever

        Raises:
            ImportError: If TimeWeightedVectorStoreRetriever is not available
        """
        try:
            from langchain.retrievers import TimeWeightedVectorStoreRetriever
        except ImportError:
            raise ImportError(
                "TimeWeightedVectorStoreRetriever not available in current LangChain version"
            )

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
            default_salience=self.default_salience,
        )
