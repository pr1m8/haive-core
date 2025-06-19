# src/haive/core/engine/retriever/bm25.py

"""BM25 Retriever implementation for the Haive framework.

This module provides a configuration class for the BM25 retriever,
which uses the BM25 algorithm for keyword-based document retrieval.

BM25 (Best Matching 25) is a ranking function used to estimate the relevance of
documents to a given search query. It's a probabilistic retrieval model that extends
the TF-IDF approach with better term frequency normalization and document length
normalization. The algorithm is widely used in search engines and information retrieval
systems.

Key features of BM25:
1. Considers both term frequency and inverse document frequency
2. Implements term frequency saturation (diminishing returns for repeated terms)
3. Normalizes for document length, giving more balanced results across short and long documents
4. Supports parameterization (k1 and b parameters) to tune the algorithm for specific corpora

Unlike embedding-based retrievers, BM25 operates on exact keyword matching rather than
semantic similarity, making it particularly effective for precise terminology searches.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from pydantic import Field

from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.engine.retriever.types import RetrieverType


def default_preprocessing_func(text: str) -> List[str]:
    """Default preprocessing function for BM25.

    This function provides basic tokenization by splitting text on whitespace.
    For more advanced preprocessing, consider using a custom function that handles:
    - Lowercasing
    - Punctuation removal
    - Stopword filtering
    - Stemming or lemmatization

    Args:
        text (str): Input text to preprocess and tokenize

    Returns:
        List[str]: List of tokens extracted from the input text

    Example:
        ```python
        tokens = default_preprocessing_func("Hello world!")
        # Returns: ["Hello", "world!"]
        ```
    """
    return text.split()


@BaseRetrieverConfig.register(RetrieverType.BM25)
class BM25RetrieverConfig(BaseRetrieverConfig):
    """Configuration for BM25 retriever.

    This retriever uses the BM25 algorithm for document retrieval,
    which is effective for keyword-based search and traditional information retrieval.
    BM25 is a bag-of-words retrieval function that ranks documents based on the
    frequency of query terms appearing in each document, with adjustments for
    document length and term saturation.

    BM25 is particularly well-suited for scenarios where:
    - Exact keyword matching is more important than semantic similarity
    - Query terms have specific technical meanings that should be preserved
    - You need an algorithm with transparent, interpretable scoring
    - You want a computationally efficient retrieval approach without embeddings

    Attributes:
        documents (List[Document]): List of Document objects to index and retrieve from.
            Each document's content will be tokenized according to the preprocess_func.
        k (int): Number of documents to retrieve for each query. Default is 4.
        bm25_params (Optional[Dict[str, Any]]): Parameters for fine-tuning the BM25 algorithm.
            The two main parameters are:
            - k1: Controls term frequency saturation (how much additional occurrences of a
                 term contribute to relevance). Default is 1.5, higher values increase the
                 impact of term frequency.
            - b: Controls document length normalization. Default is 0.75, with 0 disabling
                 length normalization and 1 fully normalizing for document length.
        preprocess_func (Callable[[str], List[str]]): Function to preprocess and tokenize
            text before BM25 indexing. The default simply splits on whitespace, but more
            sophisticated preprocessing can significantly improve retrieval quality.

    Example:
        ```python
        from haive.core.engine.retriever.bm25 import BM25RetrieverConfig
        from langchain_core.documents import Document
        import re
        import nltk
        from nltk.corpus import stopwords

        # Define a custom preprocessing function
        def preprocess_text(text: str) -> List[str]:
            # Convert to lowercase and tokenize
            tokens = re.findall(r'\b\w+\b', text.lower())
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stop_words]
            return tokens

        # Create documents
        documents = [
            Document(page_content="BM25 is a ranking function used by search engines"),
            Document(page_content="Information retrieval systems often use BM25"),
            Document(page_content="The algorithm was developed as an improvement over TF-IDF")
        ]

        # Create BM25 retriever with custom parameters
        config = BM25RetrieverConfig(
            name="bm25_search",
            documents=documents,
            k=2,
            bm25_params={"k1": 1.2, "b": 0.75},  # Fine-tune BM25 parameters
            preprocess_func=preprocess_text  # Use custom preprocessing
        )

        # Instantiate and use the retriever
        retriever = config.instantiate()
        results = retriever.get_relevant_documents("ranking algorithm search engines")
        ```
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.BM25, description="The type of retriever"
    )

    documents: List[Document] = Field(
        default_factory=list, description="Documents to retrieve from"
    )

    k: int = Field(default=4, description="Number of documents to retrieve")

    bm25_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters for the BM25 algorithm"
    )

    preprocess_func: Callable[[str], List[str]] = Field(
        default=default_preprocessing_func,
        description="Function to preprocess text before BM25 vectorization",
    )

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for BM25 retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each input parameter.

        The BM25Retriever accepts the following inputs:
            - query: The text query to match against the document collection using
                the BM25 algorithm. The query will be preprocessed using the same
                preprocessing function applied to the documents.
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
        """Return output field definitions for BM25 retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each output parameter.

        The BM25Retriever produces the following outputs:
            - documents: A list of Document objects retrieved using the BM25 algorithm,
                ranked by their BM25 score from highest to lowest (most to least relevant).
                The number of documents returned is determined by the k parameter.
        """
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    def instantiate(self):
        """Create a BM25 retriever from this configuration.

        This method instantiates a BM25Retriever from LangChain Community, which uses
        the rank_bm25 library to implement the BM25 algorithm. The method creates the
        retriever with the documents, parameters, and preprocessing function specified
        in this configuration.

        Returns:
            BM25Retriever: An instantiated BM25 retriever ready to perform document
                retrieval using the BM25 algorithm.

        Raises:
            ImportError: If rank_bm25 is not installed. This dependency is required
                for the BM25 implementation and can be installed with pip.

        Example:
            ```python
            # Create a configuration with documents
            config = BM25RetrieverConfig(
                name="keyword_search",
                documents=my_documents,
                k=5,
                bm25_params={"k1": 1.3, "b": 0.8}
            )

            # Instantiate the retriever
            retriever = config.instantiate()

            # Use the retriever for keyword-based search
            results = retriever.get_relevant_documents("important keywords to find")
            ```
        """
        try:
            from langchain_community.retrievers import BM25Retriever
        except ImportError:
            raise ImportError(
                "BM25Retriever requires rank_bm25. Install with: pip install rank-bm25"
            )

        return BM25Retriever.from_documents(
            documents=self.documents,
            k=self.k,
            bm25_params=self.bm25_params,
            preprocess_func=self.preprocess_func,
        )
