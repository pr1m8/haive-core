"""Text splitter base classes and imports for document processing."""

# Use try/except imports to handle missing dependencies gracefully during documentation builds
try:
    from langchain.document_loaders.base import Document
except ImportError:
    # Fallback for documentation builds
    from typing import Any
    class Document:
        page_content: str
        metadata: dict

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter as LangchainRecursiveCharacterTextSplitter
except ImportError:
    LangchainRecursiveCharacterTextSplitter = None

try:
    from haive.core.engine.base.base import InvokableEngine
except ImportError:
    # Fallback for documentation builds
    from typing import TypeVar, Generic, Any
    T = TypeVar('T')
    U = TypeVar('U')
    class InvokableEngine(Generic[T, U]):
        pass

# Import LangChain text splitters with error handling
try:
    from langchain_text_splitters import (
        Language,
        RecursiveCharacterTextSplitter,
        TextSplitter,
        Tokenizer,
        TokenTextSplitter,
    )
except ImportError:
    # Create placeholder classes for documentation builds
    class Language: pass
    class RecursiveCharacterTextSplitter: pass
    class TextSplitter: pass
    class Tokenizer: pass
    class TokenTextSplitter: pass
# Import specific splitter implementations with error handling
try:
    from langchain_text_splitters.base import split_text_on_tokens
except ImportError:
    def split_text_on_tokens(*args, **kwargs): pass

try:
    from langchain_text_splitters.character import CharacterTextSplitter
except ImportError:
    class CharacterTextSplitter: pass

try:
    from langchain_text_splitters.html import ElementType, HTMLHeaderTextSplitter
except ImportError:
    class ElementType: pass
    class HTMLHeaderTextSplitter: pass

try:
    from langchain_text_splitters.json import RecursiveJsonSplitter
except ImportError:
    class RecursiveJsonSplitter: pass

try:
    from langchain_text_splitters.konlpy import KonlpyTextSplitter
except ImportError:
    class KonlpyTextSplitter: pass

try:
    from langchain_text_splitters.latex import LatexTextSplitter
except ImportError:
    class LatexTextSplitter: pass

try:
    from langchain_text_splitters.markdown import (
        HeaderType,
        LineType,
        MarkdownHeaderTextSplitter,
        MarkdownTextSplitter,
    )
except ImportError:
    class HeaderType: pass
    class LineType: pass
    class MarkdownHeaderTextSplitter: pass
    class MarkdownTextSplitter: pass

try:
    from langchain_text_splitters.nltk import NLTKTextSplitter
except ImportError:
    class NLTKTextSplitter: pass

try:
    from langchain_text_splitters.python import PythonCodeTextSplitter
except ImportError:
    class PythonCodeTextSplitter: pass

try:
    from langchain_text_splitters.sentence_transformers import (
        SentenceTransformersTokenTextSplitter,
    )
except ImportError:
    class SentenceTransformersTokenTextSplitter: pass

try:
    from langchain_text_splitters.spacy import SpacyTextSplitter
except ImportError:
    class SpacyTextSplitter: pass

__all__ = [
    "CharacterTextSplitter",
    "ElementType",
    "HTMLHeaderTextSplitter",
    "HeaderType",
    "KonlpyTextSplitter",
    "Language",
    "LatexTextSplitter",
    "LineType",
    "MarkdownHeaderTextSplitter",
    "MarkdownTextSplitter",
    "NLTKTextSplitter",
    "PythonCodeTextSplitter",
    "RecursiveCharacterTextSplitter",
    "RecursiveJsonSplitter",
    "SentenceTransformersTokenTextSplitter",
    "SpacyTextSplitter",
    "TextSplitter",
    "TokenTextSplitter",
    "Tokenizer",
    "split_text_on_tokens",
]


class BaseTextSplitter(InvokableEngine):
    """A base class for text splitters."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the base text splitter.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
