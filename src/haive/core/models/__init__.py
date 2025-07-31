"""Core models module for the Haive framework.

This module provides a comprehensive set of model abstractions and implementations
for working with large language models (LLMs), embeddings, retrievers, and vector
stores. The models are designed to be modular, extensible, and optimized for
use within the Haive agent ecosystem.

Key Components:
    - LLM: Large Language Model abstractions and implementations
    - Embeddings: Text embedding models for vector representations
    - Retrievers: Components for retrieving relevant information
    - Vectorstores: Storage systems for embedding vectors with similarity search
    - Metadata: Utilities for working with model metadata

Typical usage example:
    ```python
    from haive.core.models.llm import BaseLLM
    from haive.core.models.embeddings import BaseEmbeddings
    from haive.core.models.retriever import BaseRetriever
    from haive.core.models.vectorstore import BaseVectorStore
    ```
"""

# Model metadata utilities

import importlib

from haive.core.models.metadata import ModelMetadata
from haive.core.models.metadata_mixin import ModelMetadataMixin as MetadataMixin

# Submodules are lazy-loaded to avoid heavy imports (embeddings trigger numpy/pandas)
# Use __getattr__ for lazy loading of submodules

_SUBMODULES = {"embeddings", "llm", "retriever", "vectorstore"}


def __getattr__(name):
    """Lazy load submodules to avoid heavy imports during module initialization.

    The embeddings module in particular triggers langchain_community imports
    which load numpy/pandas, adding 17+ seconds to import time.
    """
    if name in _SUBMODULES:
        return importlib.import_module(f"haive.core.models.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "MetadataMixin",
    "ModelMetadata",
    "embeddings",
    "llm",
    "retriever",
    "vectorstore",
]
