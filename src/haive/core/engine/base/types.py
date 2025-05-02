from enum import Enum


class EngineType(str, Enum):
    """Types of engines the system can use."""

    LLM = "llm"
    VECTOR_STORE = "vector_store"
    RETRIEVER = "retriever"
    TOOL = "tool"
    EMBEDDINGS = "embeddings"
    AGENT = "agent"
    DOCUMENT_LOADER = "document_loader"
    DOCUMENT_TRANSFORMER = "document_transformer"
    DOCUMENT_SPLITTER = "document_splitter"
