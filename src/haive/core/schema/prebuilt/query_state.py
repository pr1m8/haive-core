"""Query State Schema for Advanced RAG and Document Processing.

This module provides comprehensive query state management for advanced RAG workflows,
document processing, and multi-query scenarios. It builds on top of MessagesState
and DocumentState to provide a unified query processing interface.

The QueryState enables:
- Multi-query processing and refinement
- Query expansion and optimization
- Retrieval strategy management
- Context tracking and memory
- Source citation and provenance
- Time-weighted and filtered queries
- Self-query and adaptive retrieval
- Query result caching and optimization

Examples:
    Basic query processing::

        from haive.core.schema.prebuilt.query_state import QueryState

        state = QueryState(
            original_query="What are the latest trends in AI?",
            query_type="research",
            retrieval_strategy="adaptive"
        )

    Advanced multi-query workflow::

        state = QueryState(
            original_query="Analyze Q4 2024 financial performance",
            refined_queries=[
                "Q4 2024 revenue growth analysis",
                "Fourth quarter 2024 profit margins",
                "2024 Q4 market performance comparison"
            ],
            query_expansion_enabled=True,
            time_weighted_retrieval=True,
            source_filters=["financial_reports", "earnings_calls"]
        )

    Self-query with structured output::

        from haive.core.schema.prebuilt.query_state import QueryType, RetrievalStrategy

        state = QueryState(
            original_query="Find all documents about machine learning published after 2023",
            query_type=QueryType.STRUCTURED,
            retrieval_strategy=RetrievalStrategy.SELF_QUERY,
            structured_query_enabled=True,
            metadata_filters={"year": {"$gt": 2023}, "topic": "machine_learning"}
        )

Author: Claude (Haive AI Agent Framework)
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

# Conditionally import DocumentState to avoid auto-registry initialization
try:
    # Try to import without triggering the full document system
    # This is a stub to check if DocumentState is available
    import sys

    if "haive.core.engine.document" in sys.modules:
        from haive.core.schema.prebuilt.document_state import DocumentState

        _HAS_DOCUMENT_STATE = True
    else:
        DocumentState = None
        _HAS_DOCUMENT_STATE = False
except ImportError:
    DocumentState = None
    _HAS_DOCUMENT_STATE = False

from haive.core.schema.prebuilt.messages_state import MessagesState


class QueryType(str, Enum):
    """Types of queries supported by the query processing system."""

    SIMPLE = "simple"  # Basic question answering
    RESEARCH = "research"  # Research and analysis queries
    STRUCTURED = "structured"  # Structured data extraction
    MULTI_STEP = "multi_step"  # Multi-step reasoning
    COMPARISON = "comparison"  # Comparative analysis
    SUMMARIZATION = "summarization"  # Document summarization
    EXTRACTION = "extraction"  # Information extraction
    CLASSIFICATION = "classification"  # Document classification
    RECOMMENDATION = "recommendation"  # Recommendation queries
    TEMPORAL = "temporal"  # Time-based queries
    SPATIAL = "spatial"  # Location-based queries
    ANALYTICAL = "analytical"  # Data analysis queries


class RetrievalStrategy(str, Enum):
    """Retrieval strategies for query processing."""

    BASIC = "basic"  # Basic vector similarity
    ADAPTIVE = "adaptive"  # Adaptive retrieval based on query
    SELF_QUERY = "self_query"  # Self-querying retrieval
    PARENT_DOCUMENT = "parent_document"  # Parent document retrieval
    MULTI_QUERY = "multi_query"  # Multiple query variations
    ENSEMBLE = "ensemble"  # Ensemble of multiple retrievers
    TIME_WEIGHTED = "time_weighted"  # Time-weighted retrieval
    CONTEXTUAL = "contextual"  # Contextual compression
    HYBRID = "hybrid"  # Hybrid semantic + keyword
    RERANKING = "reranking"  # Retrieval with reranking


class QueryComplexity(str, Enum):
    """Query complexity levels for processing optimization."""

    LOW = "low"  # Simple factual queries
    MEDIUM = "medium"  # Moderate reasoning required
    HIGH = "high"  # Complex multi-step reasoning
    EXPERT = "expert"  # Expert-level analysis required


class QueryIntent(str, Enum):
    """Intent classification for query processing."""

    INFORMATION_SEEKING = "information_seeking"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    COMPARISON = "comparison"
    PLANNING = "planning"
    TROUBLESHOOTING = "troubleshooting"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"


class QueryProcessingConfig(BaseModel):
    """Configuration for query processing behavior."""

    max_query_variations: int = Field(default=5, ge=1, le=20)
    enable_query_expansion: bool = Field(default=True)
    enable_query_refinement: bool = Field(default=True)
    enable_context_compression: bool = Field(default=True)
    enable_result_reranking: bool = Field(default=False)
    enable_citation_tracking: bool = Field(default=True)
    enable_confidence_scoring: bool = Field(default=True)
    max_context_documents: int = Field(default=10, ge=1, le=50)
    context_window_size: int = Field(default=4000, ge=1000, le=16000)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    time_weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, ge=60)


class QueryMetrics(BaseModel):
    """Metrics and analytics for query processing."""

    processing_time: float = Field(default=0.0, ge=0.0)
    retrieval_time: float = Field(default=0.0, ge=0.0)
    generation_time: float = Field(default=0.0, ge=0.0)
    total_documents_searched: int = Field(default=0, ge=0)
    relevant_documents_found: int = Field(default=0, ge=0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    query_complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    context_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class QueryResult(BaseModel):
    """Result container for query processing."""

    query_id: str = Field(description="Unique identifier for the query")
    response: str = Field(description="Generated response to the query")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_documents: list[Document] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    processing_metrics: QueryMetrics = Field(default_factory=QueryMetrics)

    class Config:
        arbitrary_types_allowed = True


# Define QueryState with conditional inheritance based on DocumentState availability
if _HAS_DOCUMENT_STATE and DocumentState is not None:

    class QueryState(MessagesState, DocumentState):
        pass

else:

    class QueryState(MessagesState):
        pass


# Now redefine the actual QueryState class with its full implementation
class QueryState(QueryState):
    """Comprehensive query state for advanced RAG and document processing.

    This state schema combines messages, documents, and query-specific information
    to provide a complete context for query processing workflows. It supports
    multi-query scenarios, retrieval strategies, and advanced RAG features.

    The state includes:
    - Query processing and refinement
    - Document context and retrieval
    - Multi-query coordination
    - Retrieval strategy management
    - Results and metrics tracking
    - Source citation and provenance
    - Time-weighted and filtered queries
    - Adaptive and self-query capabilities

    Examples:
        Basic query state::

            state = QueryState(
                original_query="What is quantum computing?",
                query_type=QueryType.SIMPLE,
                retrieval_strategy=RetrievalStrategy.BASIC
            )

        Advanced research query::

            state = QueryState(
                original_query="Analyze the impact of AI on healthcare",
                query_type=QueryType.RESEARCH,
                retrieval_strategy=RetrievalStrategy.ADAPTIVE,
                query_expansion_enabled=True,
                time_weighted_retrieval=True,
                source_filters=["medical_journals", "clinical_trials"],
                metadata_filters={"publication_year": {"$gte": 2020}}
            )

        Multi-query workflow::

            state = QueryState(
                original_query="Compare Q3 vs Q4 2024 performance",
                refined_queries=[
                    "Q3 2024 financial results analysis",
                    "Q4 2024 earnings report summary",
                    "Q3 Q4 2024 performance comparison"
                ],
                query_type=QueryType.COMPARISON,
                retrieval_strategy=RetrievalStrategy.MULTI_QUERY
            )
    """

    # Core Query Information
    original_query: str = Field(description="The original user query")
    query_id: str = Field(default_factory=lambda: f"query_{datetime.now().timestamp()}")
    query_type: QueryType = Field(default=QueryType.SIMPLE)
    query_intent: QueryIntent = Field(default=QueryIntent.INFORMATION_SEEKING)
    query_complexity: QueryComplexity = Field(default=QueryComplexity.MEDIUM)

    # Query Processing
    refined_queries: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    query_variations: list[str] = Field(default_factory=list)
    processed_query: str = Field(default="")

    # Retrieval Configuration
    retrieval_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.ADAPTIVE)
    retrieval_config: dict[str, Any] = Field(default_factory=dict)

    # Query Enhancement
    query_expansion_enabled: bool = Field(default=True)
    query_refinement_enabled: bool = Field(default=True)
    multi_query_enabled: bool = Field(default=False)
    structured_query_enabled: bool = Field(default=False)
    time_weighted_retrieval: bool = Field(default=False)

    # Filtering and Constraints
    source_filters: list[str] = Field(default_factory=list)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    time_range_filter: dict[str, datetime] | None = Field(default=None)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)

    # Context and Memory
    context_documents: list[Document] = Field(default_factory=list)
    retrieved_documents: list[Document] = Field(default_factory=list)
    relevant_contexts: list[str] = Field(default_factory=list)
    memory_contexts: list[str] = Field(default_factory=list)

    # Results and Tracking
    query_results: list[QueryResult] = Field(default_factory=list)
    current_result: QueryResult | None = Field(default=None)
    intermediate_results: list[dict[str, Any]] = Field(default_factory=list)

    # Citations and Provenance
    citations: list[dict[str, Any]] = Field(default_factory=list)
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    confidence_scores: dict[str, float] = Field(default_factory=dict)

    # Processing Configuration
    processing_config: QueryProcessingConfig = Field(
        default_factory=QueryProcessingConfig
    )

    # Metrics and Analytics
    processing_metrics: QueryMetrics = Field(default_factory=QueryMetrics)
    query_history: list[dict[str, Any]] = Field(default_factory=list)

    # Execution State
    current_stage: str = Field(default="initialized")
    execution_path: list[str] = Field(default_factory=list)
    error_history: list[dict[str, Any]] = Field(default_factory=list)

    # Caching and Optimization
    cache_key: str | None = Field(default=None)
    cached_results: dict[str, Any] = Field(default_factory=dict)
    optimization_hints: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @field_validator("original_query")
    @classmethod
    def validate_original_query(cls, v: str) -> str:
        """Validate that the original query is not empty."""
        if not v or not v.strip():
            raise ValueError("Original query cannot be empty")
        return v.strip()

    @field_validator("refined_queries")
    @classmethod
    def validate_refined_queries(cls, v: list[str]) -> list[str]:
        """Validate refined queries are not empty."""
        return [q.strip() for q in v if q and q.strip()]

    @field_validator("time_range_filter")
    @classmethod
    def validate_time_range(
        cls, v: dict[str, datetime] | None
    ) -> dict[str, datetime] | None:
        """Validate time range filter has valid start and end dates."""
        if v and "start" in v and "end" in v and v["start"] > v["end"]:
            raise ValueError("Start date must be before end date")
        return v

    def add_refined_query(self, query: str) -> None:
        """Add a refined query to the list."""
        if query and query.strip() and query not in self.refined_queries:
            self.refined_queries.append(query.strip())

    def add_expanded_query(self, query: str) -> None:
        """Add an expanded query to the list."""
        if query and query.strip() and query not in self.expanded_queries:
            self.expanded_queries.append(query.strip())

    def add_query_variation(self, query: str) -> None:
        """Add a query variation to the list."""
        if query and query.strip() and query not in self.query_variations:
            self.query_variations.append(query.strip())

    def add_context_document(self, document: Document) -> None:
        """Add a context document to the state."""
        if document not in self.context_documents:
            self.context_documents.append(document)

    def add_retrieved_document(self, document: Document) -> None:
        """Add a retrieved document to the state."""
        if document not in self.retrieved_documents:
            self.retrieved_documents.append(document)

    def add_citation(self, citation: dict[str, Any]) -> None:
        """Add a citation to the state."""
        if citation not in self.citations:
            self.citations.append(citation)

    def set_confidence_score(self, source: str, score: float) -> None:
        """Set confidence score for a source."""
        if 0.0 <= score <= 1.0:
            self.confidence_scores[source] = score

    def get_confidence_score(self, source: str) -> float:
        """Get confidence score for a source."""
        return self.confidence_scores.get(source, 0.0)

    def add_intermediate_result(self, result: dict[str, Any]) -> None:
        """Add an intermediate result to tracking."""
        result["timestamp"] = datetime.now().isoformat()
        self.intermediate_results.append(result)

    def update_stage(self, stage: str) -> None:
        """Update the current processing stage."""
        self.current_stage = stage
        self.execution_path.append(stage)

    def add_error(self, error: str, context: dict[str, Any] | None = None) -> None:
        """Add an error to the history."""
        error_entry = {
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "stage": self.current_stage,
            "context": context or {},
        }
        self.error_history.append(error_entry)

    def get_all_queries(self) -> list[str]:
        """Get all queries including original, refined, and expanded."""
        all_queries = [self.original_query]
        all_queries.extend(self.refined_queries)
        all_queries.extend(self.expanded_queries)
        all_queries.extend(self.query_variations)
        return list(set(all_queries))  # Remove duplicates

    def get_all_documents(self) -> list[Document]:
        """Get all documents including raw, context, and retrieved."""
        all_docs = []
        # Use raw_documents from DocumentState
        all_docs.extend(self.raw_documents)
        all_docs.extend(self.context_documents)
        all_docs.extend(self.retrieved_documents)

        # Remove duplicates based on content
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    def get_processing_summary(self) -> dict[str, Any]:
        """Get a summary of processing statistics."""
        return {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "retrieval_strategy": self.retrieval_strategy.value,
            "total_queries": len(self.get_all_queries()),
            "total_documents": len(self.get_all_documents()),
            "context_documents": len(self.context_documents),
            "retrieved_documents": len(self.retrieved_documents),
            "citations": len(self.citations),
            "current_stage": self.current_stage,
            "execution_path": self.execution_path,
            "processing_time": self.processing_metrics.processing_time,
            "confidence_score": self.processing_metrics.confidence_score,
            "errors": len(self.error_history),
        }

    def is_multi_query_workflow(self) -> bool:
        """Check if this is a multi-query workflow."""
        return (
            self.multi_query_enabled
            or len(self.refined_queries) > 1
            or len(self.expanded_queries) > 1
            or len(self.query_variations) > 1
        )

    def requires_structured_output(self) -> bool:
        """Check if structured output is required."""
        return self.structured_query_enabled or self.query_type in [
            QueryType.STRUCTURED,
            QueryType.EXTRACTION,
            QueryType.CLASSIFICATION,
        ]

    def get_active_filters(self) -> dict[str, Any]:
        """Get all active filters for the query."""
        filters = {}

        if self.source_filters:
            filters["sources"] = self.source_filters

        if self.metadata_filters:
            filters["metadata"] = self.metadata_filters

        if self.time_range_filter:
            filters["time_range"] = self.time_range_filter

        if self.similarity_threshold < 1.0:
            filters["similarity_threshold"] = self.similarity_threshold

        return filters

    def create_cache_key(self) -> str:
        """Create a cache key for the current query state."""
        import hashlib

        # Create a hash based on query and key parameters
        cache_components = [
            self.original_query,
            str(self.query_type.value),
            str(self.retrieval_strategy.value),
            str(sorted(self.source_filters)),
            str(self.metadata_filters),
            str(self.similarity_threshold),
            str(self.max_results),
        ]

        cache_string = "|".join(cache_components)
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()

        self.cache_key = cache_key
        return cache_key


# Alias for backward compatibility
QueryProcessingState = QueryState

# Export all classes
__all__ = [
    "QueryComplexity",
    "QueryIntent",
    "QueryMetrics",
    "QueryProcessingConfig",
    "QueryProcessingState",
    "QueryResult",
    "QueryState",
    "QueryType",
    "RetrievalStrategy",
]
