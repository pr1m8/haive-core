from haive.core.models.retriever.base import RetrieverConfig, RetrieverType
from haive.core.models.vectorstore.base import VectorStoreConfig
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from typing import Optional, Any, List, Dict
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@RetrieverConfig.register(RetrieverType.SELF_QUERY)
class SelfQueryRetrieverConfig(RetrieverConfig):
    """Configuration for self-query retrievers."""
    vector_store_config: Optional[VectorStoreConfig] = Field(
        default=None, description="Configuration for the vector store"
    )
    llm_config: Optional[Any] = Field(
        default=None, description="Configuration for the LLM"
    )
    document_contents: str = Field(
        default="", description="Description of document contents"
    )
    metadata_field_info: List[Dict[str, Any]] = Field(
        default_factory=list, description="Information about metadata fields"
    )
    
    def instantiate(self) -> BaseRetriever:
        """Create the self-query retriever."""
        from langchain.retrievers.self_query.base import SelfQueryRetriever
        from langchain.chains.query_constructor.base import AttributeInfo
        
        # Create the vector store
        vector_store = self.vector_store_config.create_vectorstore()
        
        # Create the LLM
        llm = self.llm_config.instantiate_llm()
        
        # Convert metadata field info
        metadata_field_info = [
            AttributeInfo(**field) for field in self.metadata_field_info
        ]
        
        # Create and return the retriever
        return SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vector_store,
            document_contents=self.document_contents,
            metadata_field_info=metadata_field_info,
            search_kwargs=self.search_kwargs
        )