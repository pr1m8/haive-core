from abc import ABC, abstractmethod
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

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator, model_validator

from haive.core.engine.base import InvokableEngine
from haive.core.engine.document.base.schema import (
    DocumentEngineInputSchema,
    DocumentEngineOutputSchema,
)
from haive.core.engine.document.loaders.base.schema import LoaderInputSchema
from haive.core.engine.document.loaders.sources.types import SourceType

# from langchain_community.document_loaders import UnstructuredXMLLoader,JS


class LoaderConfig(
    ABC, InvokableEngine[DocumentEngineInputSchema, DocumentEngineOutputSchema]
):
    """
    Config for a loader.
    """

    # loader
    @abstractmethod
    def load(self, input: DocumentEngineInputSchema) -> DocumentEngineOutputSchema:
        """
        Load the data from the input.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def from_config(cls, config: BaseModel) -> "LoaderConfig":
        """
        Create a loader from a config.
        """
        return cls(**config.model_dump())

    @classmethod
    def from_dict(cls, config: Dict) -> "LoaderConfig":
        """
        Create a loader from a dict.
        """
        return cls(**config)

    # @classmethod

    @classmethod
    def create_runnable(cls, config: BaseModel) -> "LoaderConfig":
        """
        Create a runnable from a config.
        """
        return cls.from_config(config).load()
