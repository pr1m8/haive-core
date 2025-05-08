from typing import List

from langchain_community.document_loaders import DocumentTransformer
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class DocumentEngineInputSchema(BaseModel):
    """
    The input schema for the document engine.
    """

    documents: List[Document] = Field(description="The documents to process.")


class DocumentEngineOutputSchema(BaseModel):
    """
    The output schema for the document engine.
    """

    documents: List[Document] = Field(description="The processed documents.")
