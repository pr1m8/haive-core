# src/haive/core/engine/retriever/time_weighted.py

"""TimeWeighted Retriever implementation for the Haive framework.

This module provides a configuration class for the TimeWeighted retriever,
which combines embedding similarity with recency in retrieving documents.
The retriever applies a time decay function to score documents based on both
their relevance to the query and their recency, making it particularly useful
for applications where information freshness is important.

The time decay is applied using an exponential decay function:
    score = similarity_score * (1.0 - decay_rate) ** hours_passed

This allows the retriever to prioritize more recent documents while still
considering semantic similarity to the query.
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
    such as news retrieval or conversation history. Documents are scored based on
    both their vector similarity to the query and how recently they were created,
    with newer documents receiving higher scores.

    The time-weighted scoring uses document creation timestamps stored in metadata.
    For each document, an exponential decay is applied based on the time elapsed
    since creation, and this is multiplied by the similarity score to produce a
    final ranking.

    Attributes:
        vector_store_config (VectorStoreConfig): Configuration for the vector store backend
            that will be used for embedding-based similarity search.
        memory_stream (List[Document]): List of Document objects to search through.
            These documents should have timestamps in their metadata.
        decay_rate (float): Exponential decay factor used in the formula:
            (1.0 - decay_rate) ** hours_passed. Higher values make the retriever
            prioritize recency more strongly. Default is 0.01.
        k (int): Number of documents to retrieve in the final result. Default is 4.
        search_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the
            vectorstore similarity search. By default, requests 100 candidates before
            applying time weighting.
        other_score_keys (List[str]): Other keys in the document metadata to factor
            into the final score calculation.
        default_salience (Optional[float]): Salience score to assign to documents
            not retrieved from the vector store. If None, these documents won't be
            considered.

    Example:
        ```python
        from haive.core.engine.retriever.time_weighted import TimeWeightedRetrieverConfig
        from haive.core.engine.vectorstore import VectorStoreConfig
        from datetime import datetime
        from langchain_core.documents import Document

        # Create vector store config
        vs_config = VectorStoreConfig(
            name="example_vectorstore",
            provider="FAISS",
            embedding_model="text-embedding-ada-002"
        )

        # Create documents with timestamps
        docs = [
            Document(
                page_content="Recent information about AI",
                metadata={"created_at": datetime.now().isoformat()}
            ),
            Document(
                page_content="Older information about AI",
                metadata={"created_at": "2023-01-01T00:00:00"}
            )
        ]

        # Create time-weighted retriever config
        config = TimeWeightedRetrieverConfig(
            name="time_weighted_retriever",
            vector_store_config=vs_config,
            memory_stream=docs,
            decay_rate=0.05,  # Stronger recency bias
            k=2
        )

        # Instantiate the retriever and use it
        retriever = config.instantiate()
        results = retriever.get_relevant_documents("latest AI developments")
        # Results will prioritize the more recent document if both are relevant
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
        """Return input field definitions for TimeWeighted retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each input parameter.

        The TimeWeightedRetriever accepts the following inputs:
            - query: The text query to search for
            - k: Optional override for the number of documents to retrieve
        """
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for TimeWeighted retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each output parameter.

        The TimeWeightedRetriever produces the following outputs:
            - documents: A list of Document objects retrieved and ranked by the retriever,
                ordered by their combined relevance and recency scores.
        """
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create a TimeWeighted retriever from this configuration.

        This method instantiates a TimeWeightedVectorStoreRetriever from LangChain
        using the configuration parameters defined in this class. It first creates
        the underlying vector store using the vector_store_config, then initializes
        the retriever with all the specified parameters.

        Returns:
            TimeWeightedVectorStoreRetriever: An instantiated TimeWeighted retriever
                ready to perform document retrieval with time decay.

        Raises:
            ImportError: If TimeWeightedVectorStoreRetriever is not available in the
                current LangChain version. This may happen with older versions of
                LangChain or if optional dependencies are not installed.

        Example:
            ```python
            config = TimeWeightedRetrieverConfig(
                name="time_weighted_retriever",
                vector_store_config=vs_config,
                decay_rate=0.01,
                k=4
            )

            retriever = config.instantiate()

            # Use the retriever to get relevant documents
            docs = retriever.get_relevant_documents("query")
            ```
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
