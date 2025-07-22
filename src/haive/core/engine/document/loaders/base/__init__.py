"""Base document loader classes and utilities for the Haive framework.

This module provides the foundational classes and utilities for all document
loaders in the Haive framework. It defines the base interfaces, common methods,
and schema definitions used throughout the document loading system.

The base loader system establishes consistent patterns for document loading,
error handling, metadata extraction, and result formatting across all loader
implementations.

Key Components:
    BaseDocumentLoader: Abstract base class for all document loaders
    SimpleDocumentLoader: Basic implementation for simple document loading
    TextDocumentLoader: Specialized loader for text documents
    LoaderInputSchema: Schema definition for loader input configuration
    LoaderOutputSchema: Schema definition for loader output results
    LoadMethod: Enumeration of document loading methods

Features:
    - Consistent loader interface definition
    - Common error handling patterns
    - Standardized metadata extraction
    - Result format normalization
    - Performance monitoring and logging
    - Extensible loader architecture

Examples:
    Creating a custom loader::

        from haive.core.engine.document.loaders.base import BaseDocumentLoader

        class CustomLoader(BaseDocumentLoader):
            def load(self, source: str) -> List[Document]:
                # Custom loading logic
                return self.process_documents(raw_docs)

    Using simple loader::

        from haive.core.engine.document.loaders.base import SimpleDocumentLoader

        loader = SimpleDocumentLoader()
        documents = loader.load("document.txt")

See Also:
    - Document loader implementations
    - Source-specific loaders
    - Loader registry system
"""

from haive.core.engine.document.loaders.base.base import (
    BaseDocumentLoader,
    SimpleDocumentLoader,
    TextDocumentLoader,
)
from haive.core.engine.document.loaders.base.methods import LoadMethod
from haive.core.engine.document.loaders.base.schema import (
    LoaderInputSchema,
    LoaderOutputSchema,
)

__all__ = [
    # Base Loader Classes
    "BaseDocumentLoader",
    "SimpleDocumentLoader",
    "TextDocumentLoader",
    # Utilities
    "LoadMethod",
    "LoaderInputSchema",
    "LoaderOutputSchema",
]
