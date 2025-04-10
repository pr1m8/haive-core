from pydantic import BaseModel, Field, model_validator
from typing import Optional
from src.haive.core.models.llm.base import BaseLLMConfig
from src.haive.core.models.retriever.base import BaseRetrieverConfig
from src.haive.core.models.vectorstore.base import VectorStoreConfig
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import RePhraseQueryRetriever

DEFAULT_TEMPLATE = """You are an assistant tasked with taking a natural language \
query and converting it into a query for a vectorstore. \
Here is the user query: {question}"""
DEFAULT_PROMPT = PromptTemplate.from_template(DEFAULT_TEMPLATE)

class RePhraseQueryRetrieverConfig(BaseModel):
    retriever_config: Optional[BaseRetrieverConfig] = None
    vs_config: Optional[VectorStoreConfig] = None
    llm_config: BaseLLMConfig
    prompt: PromptTemplate = Field(default=DEFAULT_PROMPT)

    @classmethod
    @model_validator(mode="after")
    def validate_config(cls, data):
        # Ensure at least one of retriever_config or vs_config is provided
        if data.retriever_config is None and data.vs_config is None:
            raise ValueError("Either retriever_config or vs_config must be provided")
        
        # If no retriever_config is provided, create one from vs_config
        if data.retriever_config is None and data.vs_config is not None:
            # Here we assume that VectorStoreConfig has a method `create_retriever()` to create a retriever
            data.retriever_config = data.vs_config.create_retriever()  # Set retriever_config from vs_config
        
        return data
    
    def create(self) -> RePhraseQueryRetriever:
        # At this point, retriever_config is guaranteed to be set
        retriever = self.retriever_config.create()
        llm = self.llm_config.create()
        return RePhraseQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            prompt=self.prompt
        )
