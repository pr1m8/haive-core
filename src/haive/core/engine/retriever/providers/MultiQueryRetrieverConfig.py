# src/haive/core/engine/retriever/multi_query.py

from typing import Optional
from pydantic import Field, model_validator

from langchain_core.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever

from haive.core.engine.retriever import BaseRetrieverConfig, RetrieverType
from haive.core.engine.vectorstore import VectorStoreConfig
from haive.core.engine.aug_llm import AugLLMConfig

# Default prompt template
DEFAULT_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is 
to generate 3 different versions of the given user 
question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, 
your goal is to help the user overcome some of the limitations 
of distance-based similarity search. Provide these alternative 
questions separated by newlines. Original question: {question}"""

DEFAULT_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=DEFAULT_QUERY_TEMPLATE
)

@BaseRetrieverConfig.register(RetrieverType.MULTI_QUERY)
class MultiQueryRetrieverConfig(BaseRetrieverConfig):
    """Configuration for MultiQuery retriever.
    
    This retriever generates multiple query variations using an LLM and combines the results.
    """
    retriever_type: RetrieverType = Field(
        default=RetrieverType.MULTI_QUERY,
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
        description="LLM configuration for generating query variations"
    )
    
    prompt: PromptTemplate = Field(
        default=DEFAULT_PROMPT,
        description="Prompt template for generating query variations"
    )
    
    include_original: bool = Field(
        default=False,
        description="Whether to include the original query in the search"
    )
    
    @model_validator(mode="after")
    def validate_config(cls, values):
        """Validate that at least one retriever source is provided."""
        if values.retriever_config is None and values.vector_store_config is None:
            raise ValueError("Either retriever_config or vector_store_config must be provided")
        return values
    
    def instantiate(self) -> MultiQueryRetriever:
        """Create a MultiQuery retriever from this configuration."""
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
        return MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            prompt=self.prompt,
            include_original=self.include_original
        )