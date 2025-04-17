from haive_core.models.retriever.base import BaseRetrieverConfig
from langchain.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from pydantic import Field
from typing import Optional
from haive_core.models.vectorstore.base import VectorStoreConfig
from haive_core.models.llm.base import BaseLLMConfig

DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is 
    to generate 3 different versions of the given user 
    question to retrieve relevant documents from a vector  database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines. Original question: {question}""",
)


class MultiQueryRetrieverConfig(BaseRetrieverConfig):
    retriever_config: Optional[BaseRetrieverConfig] = None
    vs_config: Optional[VectorStoreConfig] = None
    llm_config: BaseLLMConfig
    prompt: PromptTemplate = Field(default=DEFAULT_PROMPT)
