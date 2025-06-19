# src/haive/core/engine/retriever/multi_query.py

"""
MultiQuery Retriever implementation for the Haive framework.

This module provides a configuration class for the MultiQuery retriever, which
enhances document retrieval by generating multiple query variations using an LLM
and combining the results. This approach often improves recall by addressing the
"vocabulary mismatch" problem in information retrieval.

The MultiQueryRetriever works by:
1. Taking the user's original query
2. Using an LLM to generate multiple variations/reformulations of the query
3. Running each query variation against the base retriever
4. Combining and deduplicating the results to provide more comprehensive coverage

This retriever is particularly useful when:
- The user's query might use different vocabulary than the documents
- You want to improve recall without sacrificing precision too much
- The information need is complex and might benefit from multiple perspectives

The implementation integrates with LangChain's MultiQueryRetriever while providing
a consistent Haive configuration interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import Field, model_validator

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.retriever.retriever import BaseRetrieverConfig
from haive.core.engine.retriever.types import RetrieverType
from haive.core.engine.vectorstore import VectorStoreConfig

# Default prompt template
DEFAULT_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is 
to generate 3 different versions of the given user 
question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, 
your goal is to help the user overcome some of the limitations 
of distance-based similarity search. Provide these alternative 
questions separated by newlines. Original question: {question}"""

DEFAULT_PROMPT = PromptTemplate(
    input_variables=["question"], template=DEFAULT_QUERY_TEMPLATE
)


@BaseRetrieverConfig.register(RetrieverType.MULTI_QUERY)
class MultiQueryRetrieverConfig(BaseRetrieverConfig):
    """
    Configuration for MultiQuery retriever in the Haive framework.

    This retriever generates multiple query variations using an LLM and combines the results.
    It's particularly useful for improving recall by rephrasing queries to capture
    different aspects of the user's information need, addressing the vocabulary
    mismatch problem in information retrieval.

    The MultiQueryRetriever can work with any base retriever or directly with a
    vector store, providing flexibility in how it's integrated into your retrieval pipeline.

    Attributes:
        retriever_type (RetrieverType): The type of retriever (always MULTI_QUERY).
        retriever_config (Optional[BaseRetrieverConfig]): Base retriever configuration.
            Either this or vector_store_config must be provided.
        vector_store_config (Optional[VectorStoreConfig]): Vector store configuration.
            Alternative to retriever_config for creating the base retriever.
        llm_config (AugLLMConfig): LLM configuration for generating query variations.
            This LLM will be used to create multiple versions of the original query.
        prompt (PromptTemplate): Prompt template for generating query variations.
            The prompt should instruct the LLM to generate multiple alternative
            phrasings of the original query.
        include_original (bool): Whether to include the original query in the search.
            If True, both the original query and the generated variations will be used.

    Examples:
        >>> from haive.core.engine.retriever import MultiQueryRetrieverConfig
        >>> from haive.core.engine.aug_llm import AugLLMConfig
        >>> from haive.core.engine.vectorstore import VectorStoreConfig
        >>>
        >>> # Create an LLM config for query generation
        >>> llm_config = AugLLMConfig(
        ...     name="query_generator",
        ...     system_message="You are an expert at rephrasing search queries."
        ... )
        >>>
        >>> # Create a vector store config
        >>> vs_config = VectorStoreConfig(
        ...     name="document_store",
        ...     documents=[Document(page_content="Content...")]
        ... )
        >>>
        >>> # Create the multi-query retriever config
        >>> config = MultiQueryRetrieverConfig(
        ...     name="multi_query_retriever",
        ...     vector_store_config=vs_config,
        ...     llm_config=llm_config,
        ...     include_original=True
        ... )
        >>>
        >>> # Instantiate and use the retriever
        >>> retriever = config.instantiate()
        >>> docs = retriever.get_relevant_documents("What are the benefits of quantum computing?")
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.MULTI_QUERY, description="The type of retriever"
    )

    retriever_config: Optional[BaseRetrieverConfig] = Field(
        default=None, description="Base retriever configuration"
    )

    vector_store_config: Optional[VectorStoreConfig] = Field(
        default=None,
        description="Vector store configuration (alternative to retriever_config)",
    )

    llm_config: AugLLMConfig = Field(
        ..., description="LLM configuration for generating query variations"
    )

    prompt: PromptTemplate = Field(
        default=DEFAULT_PROMPT,
        description="Prompt template for generating query variations",
    )

    include_original: bool = Field(
        default=False, description="Whether to include the original query in the search"
    )

    @model_validator(mode="after")
    def validate_config(self):
        """Validate that at least one retriever source is provided."""
        if self.retriever_config is None and self.vector_store_config is None:
            raise ValueError(
                "Either retriever_config or vector_store_config must be provided"
            )
        return self

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for MultiQuery retriever."""
        return {
            "query": (str, Field(description="Query string for retrieval")),
            "k": (
                Optional[int],
                Field(default=None, description="Number of documents to retrieve"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for MultiQuery retriever."""
        return {
            "documents": (
                List[Document],
                Field(
                    default_factory=list, description="Multi-query retrieved documents"
                ),
            ),
        }

    def instantiate(self):
        """
        Create a MultiQuery retriever from this configuration.

        This method creates a MultiQueryRetriever instance based on the current
        configuration. It handles:
        1. Creating the base retriever (either from retriever_config or vector_store_config)
        2. Instantiating the LLM for query generation
        3. Setting up the MultiQueryRetriever with the appropriate parameters

        Returns:
            MultiQueryRetriever: Instantiated retriever ready for document retrieval.

        Raises:
            ImportError: If MultiQueryRetriever is not available in the current
                LangChain version.
            ValueError: If neither retriever_config nor vector_store_config is provided,
                making it impossible to create a base retriever.

        Examples:
            >>> config = MultiQueryRetrieverConfig(
            ...     name="multi_query_retriever",
            ...     vector_store_config=vs_config,
            ...     llm_config=llm_config
            ... )
            >>> retriever = config.instantiate()
            >>> # The MultiQueryRetriever will generate multiple variations of each query
            >>> # and combine the results from the base retriever
            >>> docs = retriever.get_relevant_documents("How does quantum computing work?")
        """
        try:
            from langchain.retrievers import MultiQueryRetriever
        except ImportError:
            raise ImportError(
                "MultiQueryRetriever not available in current LangChain version"
            )

        # Get the base retriever
        if self.retriever_config is not None:
            retriever = self.retriever_config.instantiate()
        elif self.vector_store_config is not None:
            retriever = self.vector_store_config.create_retriever()
        else:
            raise ValueError(
                "Either retriever_config or vector_store_config must be provided"
            )

        # Create the LLM
        llm = self.llm_config.instantiate()

        # Create the retriever
        return MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            prompt=self.prompt,
            include_original=self.include_original,
        )
