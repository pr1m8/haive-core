"""RAG (Retrieval-Augmented Generation) state schema for haive agents."""

from typing import Any

from langchain_core.documents import Document
from pydantic import Field

from haive.core.schema.prebuilt.messages_state import MessagesState


class RAGState(MessagesState):
    """State schema for RAG (Retrieval-Augmented Generation) workflows.

    This schema extends MessagesState with fields specific to RAG operations:
    - Document retrieval and storage
    - Query/question tracking
    - Context management
    - Retrieved document scoring and metadata
    """

    # Query/Question fields
    query: str = Field(default="", description="Current query/question being processed")
    original_query: str | None = Field(
        default=None, description="Original query before any transformations"
    )

    # Document fields
    documents: list[Document] = Field(
        default_factory=list, description="Retrieved documents for the current query"
    )

    # Context fields
    context: str = Field(
        default="", description="Formatted context from retrieved documents"
    )

    # Retrieval metadata
    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the retrieval process (scores, sources, etc.)",
    )

    # Answer/Response fields
    answer: str | None = Field(
        default=None, description="Generated answer based on retrieved context"
    )

    # Optional fields for advanced RAG workflows
    hypothetical_document: str | None = Field(
        default=None, description="Hypothetical document for HyDE RAG workflows"
    )

    reranked_documents: list[Document] | None = Field(
        default=None, description="Documents after reranking process"
    )

    # Shared fields for multi-agent RAG
    __shared_fields__ = ["messages", "query", "documents", "context", "answer"]

    def format_documents_as_context(self, separator: str = "\n\n") -> str:
        """Format documents into a context string."""
        if not self.documents:
            return ""

        context_parts = []
        for i, doc in enumerate(self.documents):
            # Include metadata if available
            source = doc.metadata.get("source", f"Document {i + 1}")
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")

        self.context = separator.join(context_parts)
        return self.context

    def add_document(self, document: Document) -> None:
        """Add a document to the retrieved documents list."""
        self.documents.append(document)

    def clear_documents(self) -> None:
        """Clear all retrieved documents."""
        self.documents.clear()
        self.context = ""
        self.retrieval_metadata.clear()

    def get_top_documents(self, k: int = 5) -> list[Document]:
        """Get the top k documents based on retrieval scores."""
        # If we have reranked documents, use those
        docs = self.reranked_documents if self.reranked_documents else self.documents
        return docs[:k]

    def update_retrieval_metadata(self, metadata: dict[str, Any]) -> None:
        """Update retrieval metadata."""
        self.retrieval_metadata.update(metadata)

    @classmethod
    def from_query(cls, query: str) -> "RAGState":
        """Create a RAGState from a query string."""
        return cls(query=query, original_query=query)
