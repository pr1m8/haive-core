# src/haive/core/engine/retriever/multi_query.py

"""MultiQuery Retriever implementation for the Haive framework.

This module provides a configuration class for the MultiQuery retriever,
which generates multiple query variations using an LLM and combines the results.
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
    """Configuration for MultiQuery retriever.

    This retriever generates multiple query variations using an LLM and combines the results.
    It's particularly useful for improving recall by rephrasing queries to capture
    different aspects of the user's information need.

    Attributes:
        retriever_config: Base retriever configuration (optional)
        vector_store_config: Vector store configuration (alternative to retriever_config)
        llm_config: LLM configuration for generating query variations
        prompt: Prompt template for generating query variations
        include_original: Whether to include the original query in the search

    Example:
        ```python
        from haive.core.engine.retriever.multi_query import MultiQueryRetrieverConfig
        from haive.core.engine.aug_llm import AugLLMConfig
        from haive.core.engine.vectorstore import VectorStoreConfig

        llm_config = AugLLMConfig(...)
        vs_config = VectorStoreConfig(...)

        config = MultiQueryRetrieverConfig(
            name="multi_query_retriever",
            vector_store_config=vs_config,
            llm_config=llm_config,
            include_original=True
        )

        retriever = config.instantiate()
        docs = retriever.get_relevant_documents("query")
        ```
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
        """Create a MultiQuery retriever from this configuration.

        Returns:
            Instantiated MultiQuery retriever

        Raises:
            ImportError: If MultiQueryRetriever is not available
            ValueError: If configuration is invalid
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
