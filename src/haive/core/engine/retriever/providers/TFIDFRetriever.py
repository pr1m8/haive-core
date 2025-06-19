# src/haive/core/engine/retriever/tfidf.py

"""TF-IDF Retriever implementation for the Haive framework.

This module provides a configuration class for the TF-IDF retriever,
which uses Term Frequency-Inverse Document Frequency for document retrieval.

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to
evaluate how important a word is to a document in a collection of documents. This
importance increases proportionally to the number of times a word appears in the
document but is offset by the frequency of the word in the corpus.

Unlike embedding-based retrievers, TF-IDF:
1. Does not require pre-trained embedding models
2. Is computationally less expensive for small to medium document collections
3. Performs well for exact keyword matching scenarios
4. Relies solely on the statistical properties of the text
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from pydantic import Field

from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.engine.retriever.types import RetrieverType


@BaseRetrieverConfig.register(RetrieverType.TFIDF)
class TFIDFRetrieverConfig(BaseRetrieverConfig):
    """Configuration for TF-IDF retriever.

    This retriever uses Term Frequency-Inverse Document Frequency for document retrieval,
    which is effective for text-based similarity search and traditional information retrieval.
    TF-IDF calculates a numerical score that represents how relevant each document is to a
    query based on the terms they share, weighted by how unique those terms are across
    the entire document collection.

    The TF-IDF retriever is particularly useful when:
    - You don't need semantic understanding (just keyword matching)
    - You want a lightweight solution without dependencies on embedding models
    - You need interpretable search results (clear why documents match)
    - Processing a relatively small to medium collection of documents

    Attributes:
        documents (List[Document]): List of Document objects to build the TF-IDF index from
            and retrieve from during queries.
        k (int): Default number of documents to retrieve for each query. Default is 4.
        tfidf_params (Optional[Dict[str, Any]]): Optional parameters to configure the
            scikit-learn TfidfVectorizer. Common parameters include:
            - max_features: Maximum number of features to consider
            - min_df: Minimum document frequency for terms
            - max_df: Maximum document frequency for terms
            - stop_words: Language for stopword filtering or custom list
            - ngram_range: Range of n-values for n-grams to extract

    Example:
        ```python
        from haive.core.engine.retriever.tfidf import TFIDFRetrieverConfig
        from langchain_core.documents import Document

        # Create a list of documents
        documents = [
            Document(page_content="Python is a popular programming language"),
            Document(page_content="TF-IDF is used in information retrieval"),
            Document(page_content="Machine learning algorithms often use TF-IDF features"),
            Document(page_content="Python has libraries for natural language processing")
        ]

        # Create a TF-IDF retriever with custom parameters
        config = TFIDFRetrieverConfig(
            name="tfidf_retriever",
            documents=documents,
            k=2,  # Return top 2 matches
            tfidf_params={
                "max_features": 1000,  # Limit vocabulary size
                "stop_words": "english",  # Filter common English words
                "ngram_range": (1, 2)  # Include both unigrams and bigrams
            }
        )

        # Instantiate and use the retriever
        retriever = config.instantiate()
        results = retriever.get_relevant_documents("Python programming")
        # Would return the documents about Python, ranked by relevance
        ```
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.TFIDF, description="The type of retriever"
    )

    documents: List[Document] = Field(
        default_factory=list, description="Documents to retrieve from"
    )

    k: int = Field(default=4, description="Number of documents to retrieve")

    tfidf_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters for the TF-IDF vectorizer"
    )

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for TF-IDF retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each input parameter.

        The TFIDFRetriever accepts the following inputs:
            - query: The text query to match against the document collection
            - k: Optional override for the number of documents to retrieve (overrides
                the default k value specified in the configuration)
        """
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for TF-IDF retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each output parameter.

        The TFIDFRetriever produces the following outputs:
            - documents: A list of Document objects retrieved and ranked by TF-IDF similarity
                to the query, ordered from most to least relevant.
        """
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create a TF-IDF retriever from this configuration.

        This method instantiates a TFIDFRetriever from LangChain Community,
        which internally uses scikit-learn's TfidfVectorizer to build a TF-IDF
        matrix from the provided documents. The retriever converts queries into
        the same TF-IDF space and calculates cosine similarity to find the most
        relevant documents.

        Returns:
            TFIDFRetriever: An instantiated TF-IDF retriever ready to perform
                document retrieval based on term frequency-inverse document frequency.

        Raises:
            ImportError: If scikit-learn is not installed. This dependency is required
                for the TF-IDF vectorization and similarity calculations.

        Example:
            ```python
            # Create a configuration with documents
            config = TFIDFRetrieverConfig(
                name="tfidf_retriever",
                documents=my_documents,
                k=5
            )

            # Instantiate the retriever
            retriever = config.instantiate()

            # Use the retriever for document search
            results = retriever.get_relevant_documents("search query")
            ```
        """
        try:
            from langchain_community.retrievers import TFIDFRetriever
        except ImportError:
            raise ImportError(
                "TFIDFRetriever requires scikit-learn. Install with: pip install scikit-learn"
            )

        return TFIDFRetriever.from_documents(
            documents=self.documents,
            k=self.k,
            tfidf_params=self.tfidf_params,
        )
