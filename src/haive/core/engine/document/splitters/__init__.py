"""Document splitters module for the Haive framework.

This module provides various text splitting strategies for chunking documents
into smaller, manageable segments suitable for processing by LLMs, vector
databases, and other components that have size constraints.

The splitters subsystem supports multiple splitting algorithms optimized for
different document types and use cases, from simple character-based splitting
to sophisticated semantic and structural splitting.

Key Components:
    DocumentSplitterEngine: Main engine for splitting documents
    DocSplitterType: Enumeration of available splitter types
    Various text splitters from LangChain

Splitter Types:
    - RecursiveCharacterTextSplitter: Most versatile, tries multiple separators
    - CharacterTextSplitter: Simple character-based splitting
    - TokenTextSplitter: Split based on token count
    - SentenceTransformersTokenTextSplitter: Token splitting with sentence transformers
    - MarkdownTextSplitter: Preserves Markdown structure
    - HTMLHeaderTextSplitter: Splits HTML by headers
    - LatexTextSplitter: Preserves LaTeX structure
    - PythonCodeTextSplitter: Code-aware splitting for Python
    - RecursiveJsonSplitter: JSON structure-aware splitting
    - SpacyTextSplitter: Uses spaCy for linguistic splitting
    - NLTKTextSplitter: Uses NLTK for sentence splitting

Examples:
    Basic document splitting::

        from haive.core.engine.document.splitters import (
            DocumentSplitterEngine,
            DocSplitterType
        )

        # Create splitter engine
        splitter = DocumentSplitterEngine(
            splitter_type=DocSplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200
        )

        # Split documents
        chunks = splitter.invoke({"documents": documents})

    Code-aware splitting::

        from haive.core.engine.document.splitters import (
            DocumentSplitterEngine,
            DocSplitterType
        )

        # Python code splitter
        code_splitter = DocumentSplitterEngine(
            splitter_type=DocSplitterType.PYTHON_CODE,
            chunk_size=2000,
            chunk_overlap=100
        )

        code_chunks = code_splitter.invoke({"documents": python_docs})

    Markdown-aware splitting::

        # Markdown splitter preserves structure
        md_splitter = DocumentSplitterEngine(
            splitter_type=DocSplitterType.MARKDOWN,
            chunk_size=1500,
            strip_whitespace=True
        )

        md_chunks = md_splitter.invoke({"documents": markdown_docs})

See Also:
    - Document loader module for loading documents
    - Document transformer module for document transformation
    - LangChain text splitters documentation
"""

from haive.core.engine.document.splitters.base import (  # Core splitters; Token-based splitters; Language-specific splitters; NLP-based splitters; Types and utilities
    CharacterTextSplitter,
    ElementType,
    HeaderType,
    HTMLHeaderTextSplitter,
    KonlpyTextSplitter,
    Language,
    LatexTextSplitter,
    LineType,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    NLTKTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    SentenceTransformersTokenTextSplitter,
    SpacyTextSplitter,
    TextSplitter,
    Tokenizer,
    TokenTextSplitter,
    split_text_on_tokens,
)
from haive.core.engine.document.splitters.config import (
    DocSplitterType,
    SplitterConfig,
)
from haive.core.engine.document.splitters.engine import (
    DocSplitterInputSchema,
    DocSplitterOutputSchema,
    DocumentSplitterEngine,
)

__all__ = [
    # Main Engine
    "DocumentSplitterEngine",
    "DocSplitterInputSchema",
    "DocSplitterOutputSchema",
    # Configuration
    "DocSplitterType",
    "SplitterConfig",
    # Core Splitters
    "TextSplitter",
    "RecursiveCharacterTextSplitter",
    "CharacterTextSplitter",
    # Token-based Splitters
    "TokenTextSplitter",
    "SentenceTransformersTokenTextSplitter",
    # Language-specific Splitters
    "PythonCodeTextSplitter",
    "MarkdownTextSplitter",
    "MarkdownHeaderTextSplitter",
    "LatexTextSplitter",
    "HTMLHeaderTextSplitter",
    "RecursiveJsonSplitter",
    # NLP-based Splitters
    "SpacyTextSplitter",
    "NLTKTextSplitter",
    "KonlpyTextSplitter",
    # Types and Utilities
    "Language",
    "Tokenizer",
    "HeaderType",
    "LineType",
    "ElementType",
    "split_text_on_tokens",
]
