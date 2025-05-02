"""
This module contains the base class for retrievers and the different types of retrievers.
# TODO: add a base retriever class that can be used to create different types of retrievers.
-- consistnecy in naming, dynamic enum, type registry
-- create/instantiate consistency
-- vs/embed/docs.
-- naming.
"""

from .retriever import BaseRetrieverConfig, RetrieverType, VectorStoreRetrieverConfig

__all__ = ["BaseRetrieverConfig", "RetrieverType", "VectorStoreRetrieverConfig"]
