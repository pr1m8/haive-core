from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from langchain_community.graphs import GraphDocument
from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator, model_validator

DocumentLike = TypeVar("DocumentLike", bound=Union[Document, GraphDocument])


class LoaderInputSchema(BaseModel):
    """
    Schema for the input of a loader.
    """

    source: Union[str, List[str]] = Field(description="The source of the data to load.")


class LoaderOutputSchema(BaseModel):
    """
    Schema for the output of a loader.
    """

    documents: List[DocumentLike] = Field(description="The loaded documents.")
