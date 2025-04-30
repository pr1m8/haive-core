# src/haive/core/engine/retriever/rephrase.py

from typing import Optional, Union
from pydantic import Field, model_validator

from langchain_core.prompts import PromptTemplate
from langchain.retrievers import RePhraseQueryRetriever

from haive.core.engine.retriever import BaseRetrieverConfig, RetrieverType
from haive.core.engine.vectorstore import VectorStoreConfig
from haive.core.engine.aug_llm import AugLLMConfig

# Default prompt template
DEFAULT_TEMPLATE = """You are an assistant tasked with taking a natural language \
query and converting it into a query for a vectorstore. \
Here is the user query: {question}"""
DEFAULT_PROMPT = PromptTemplate.from_template(DEFAULT_TEMPLATE)

@BaseRetrieverConfig.register(RetrieverType.REPHRASE_QUERY)
class RePhraseQueryRetrieverConfig(BaseRetrieverConfig):
    """Configuration for RePhraseQuery retriever.
    
    This retriever uses an LLM to rephrase queries before retrieving documents.
    """
    retriever_type: RetrieverType = Field(
        default=RetrieverType.REPHRASE_QUERY,
        description="The type of retriever"
    )
    
    retriever_config: Optional[BaseRetrieverConfig] = Field(
        default=None,
        description="Base retriever configuration"
    )
    
    vector_store_config: Optional[VectorStoreConfig] = Field(
        default=None,
        description="Vector store configuration (alternative to retriever_config)"
    )
    
    llm_config: AugLLMConfig = Field(
        ...,  # Required
        description="LLM configuration for query rephrasing"
    )
    
    prompt: PromptTemplate = Field(
        default=DEFAULT_PROMPT,
        description="Prompt template for query rephrasing"
    )
    
    @model_validator(mode="after")
    def validate_config(cls, values):
        """Validate that at least one retriever source is provided."""
        if values.retriever_config is None and values.vector_store_config is None:
            raise ValueError("Either retriever_config or vector_store_config must be provided")
        return values
    
    def instantiate(self) -> RePhraseQueryRetriever:
        """Create a RePhraseQuery retriever from this configuration."""
        # Get the base retriever
        if self.retriever_config is not None:
            retriever = self.retriever_config.instantiate()
        elif self.vector_store_config is not None:
            retriever = self.vector_store_config.create_retriever()
        else:
            raise ValueError("Either retriever_config or vector_store_config must be provided")
        
        # Create the LLM
        llm = self.llm_config.instantiate()
        
        # Create the retriever
        return RePhraseQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            prompt=self.prompt
        )