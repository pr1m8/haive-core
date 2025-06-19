# src/haive/core/engine/retriever/rephrase.py

"""RePhraseQuery Retriever implementation for the Haive framework.

This module provides a configuration class for the RePhraseQuery retriever,
which uses a Language Model to improve retrieval by rephrasing user queries
before passing them to the underlying retriever.

The RePhraseQuery retriever addresses common retrieval issues like:
1. Query-document vocabulary mismatch
2. Ambiguous or incomplete user queries
3. Domain-specific terminology differences
4. Query reformulation to better match document content

By leveraging an LLM to rephrase the original query, this retriever can often
significantly improve retrieval performance without changing the underlying
vector store or base retriever.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from langchain.retrievers import RePhraseQueryRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import Field, model_validator

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.retriever import BaseRetrieverConfig, RetrieverType
from haive.core.engine.vectorstore import VectorStoreConfig

# Default prompt template
DEFAULT_TEMPLATE = """You are an assistant tasked with taking a natural language \
query and converting it into a query for a vectorstore. \
Here is the user query: {question}"""
DEFAULT_PROMPT = PromptTemplate.from_template(DEFAULT_TEMPLATE)


@BaseRetrieverConfig.register(RetrieverType.REPHRASE_QUERY)
class RePhraseQueryRetrieverConfig(BaseRetrieverConfig):
    """Configuration for RePhraseQuery retriever.

    This retriever uses an LLM to rephrase queries before retrieving documents.
    It can significantly improve retrieval performance by transforming natural language
    queries into more effective search queries that better match the terminology and
    structure of the documents in the knowledge base.

    The retriever works by:
    1. Taking the original user query
    2. Sending it to an LLM with a prompt that asks the LLM to rephrase it
    3. Using the rephrased query with the underlying retriever
    4. Returning the documents retrieved by the underlying retriever

    This approach is particularly valuable when:
    - Users ask questions in colloquial language different from document terminology
    - Queries need expansion with related terms to improve recall
    - Domain-specific knowledge is required to formulate effective queries
    - Documents use technical terminology that users might not know

    Attributes:
        retriever_config (Optional[BaseRetrieverConfig]): Configuration for the base
            retriever to use. Either this or vector_store_config must be provided.
        vector_store_config (Optional[VectorStoreConfig]): Alternative to retriever_config,
            specifies a vector store to create a retriever from.
        llm_config (AugLLMConfig): Configuration for the LLM that will rephrase queries.
            This is a required field.
        prompt (PromptTemplate): Prompt template used to instruct the LLM how to rephrase
            the query. Defaults to a standard template focused on vectorstore retrieval.

    Example:
        ```python
        from haive.core.engine.retriever.rephrase import RePhraseQueryRetrieverConfig
        from haive.core.engine.vectorstore import VectorStoreConfig
        from haive.core.engine.aug_llm import AugLLMConfig
        from langchain_core.prompts import PromptTemplate

        # Create a custom prompt template for medical domain
        medical_prompt = PromptTemplate.from_template(
            "You are a medical terminology expert. Rephrase this patient question "
            "using proper medical terminology to improve retrieval from medical documents: "
            "{question}"
        )

        # Create configuration for vector store
        vs_config = VectorStoreConfig(
            name="medical_docs",
            provider="FAISS",
            embedding_model="text-embedding-3-small"
        )

        # Create configuration for LLM
        llm_config = AugLLMConfig(
            name="query_rephraser",
            model="gpt-4-turbo"
        )

        # Create the RePhraseQuery retriever configuration
        config = RePhraseQueryRetrieverConfig(
            name="medical_query_enhancer",
            vector_store_config=vs_config,
            llm_config=llm_config,
            prompt=medical_prompt
        )

        # Instantiate the retriever
        retriever = config.instantiate()

        # Use the retriever with natural language query
        results = retriever.get_relevant_documents(
            "Why does my stomach hurt after eating?"
        )
        # LLM might rephrase to: "What are the potential causes of postprandial
        # abdominal pain or gastric discomfort?"
        ```
    """

    retriever_type: RetrieverType = Field(
        default=RetrieverType.REPHRASE_QUERY, description="The type of retriever"
    )

    retriever_config: Optional[BaseRetrieverConfig] = Field(
        default=None, description="Base retriever configuration"
    )

    vector_store_config: Optional[VectorStoreConfig] = Field(
        default=None,
        description="Vector store configuration (alternative to retriever_config)",
    )

    llm_config: AugLLMConfig = Field(
        ..., description="LLM configuration for query rephrasing"  # Required
    )

    prompt: PromptTemplate = Field(
        default=DEFAULT_PROMPT, description="Prompt template for query rephrasing"
    )

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return input field definitions for RePhraseQuery retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each input parameter.

        The RePhraseQueryRetriever accepts the following inputs:
            - query: The original text query that will be rephrased by the LLM
                before being sent to the underlying retriever
        """
        return {
            "query": (
                str,
                Field(description="Query string for retrieval (will be rephrased)"),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """Return output field definitions for RePhraseQuery retriever.

        Returns:
            Dict[str, Tuple[Type, Any]]: Dictionary mapping field names to tuples of
                (type, field_definition) for each output parameter.

        The RePhraseQueryRetriever produces the following outputs:
            - documents: A list of Document objects retrieved using the LLM-rephrased query
        """
        return {
            "documents": (
                List[Document],
                Field(default_factory=list, description="Retrieved documents"),
            ),
        }

    @model_validator(mode="after")
    def validate_config(cls, values):
        """Validate that at least one retriever source is provided.

        This validator ensures that either retriever_config or vector_store_config
        is provided, as one of these is required to create the underlying retriever.
        Both cannot be None.

        Args:
            values: The configured values for the model

        Returns:
            The validated configuration instance

        Raises:
            ValueError: If neither retriever_config nor vector_store_config is provided
        """
        if values.retriever_config is None and values.vector_store_config is None:
            raise ValueError(
                "Either retriever_config or vector_store_config must be provided"
            )
        return values

    def instantiate(self) -> RePhraseQueryRetriever:
        """Create a RePhraseQuery retriever from this configuration.

        This method instantiates a RePhraseQueryRetriever from LangChain, which wraps
        a base retriever with an LLM-powered query rephrasing capability. The method
        first creates the underlying retriever, either from retriever_config or
        vector_store_config, then instantiates the LLM from llm_config.

        Returns:
            RePhraseQueryRetriever: An instantiated retriever that rephrases queries
                before passing them to the underlying retriever.

        Raises:
            ValueError: If neither retriever_config nor vector_store_config is provided

        Example:
            ```python
            config = RePhraseQueryRetrieverConfig(
                name="enhanced_retriever",
                vector_store_config=vector_store_config,
                llm_config=llm_config
            )

            retriever = config.instantiate()

            # The query will be rephrased by the LLM before retrieval
            results = retriever.get_relevant_documents("What's the best way to cook pasta?")
            ```
        """
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
        return RePhraseQueryRetriever.from_llm(
            retriever=retriever, llm=llm, prompt=self.prompt
        )
