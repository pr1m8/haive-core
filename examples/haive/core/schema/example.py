"""Example schema module.

This module provides example functionality for the Haive framework.

Classes:
    ConversationState: ConversationState implementation.
    MetricsState: MetricsState implementation.
    ParentState: ParentState implementation.

Functions:
    example_basic_state_creation: Example Basic State Creation functionality.
    example_dynamic_composition: Example Dynamic Composition functionality.
    example_reducer_behavior: Example Reducer Behavior functionality.
"""

#!/usr/bin/env python3
"""From typing import Any
Haive Schema System Examples.

This file demonstrates comprehensive usage of the Haive Schema System,
including state creation, dynamic composition, reducers, and real-world patterns.
"""

import contextlib
import operator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import Field, validator

from haive.core.schema import (
    SchemaComposer,
    StateSchema,
    create_agent_state,
    create_message_state,
    get_schema_info,
    validate_schema,
)
from haive.core.schema.ui import SchemaUI


def example_basic_state_creation() -> None:
    """Example 1: Basic state creation patterns."""

    # Method 1: Declarative definition
    class ConversationState(StateSchema):
        """Simple conversation state."""

        messages: list[BaseMessage] = Field(default_factory=list)
        topic: str = Field(default="general chat")
        turn_count: int = Field(default=0)

        __reducer_fields__ = {
            "messages": "add_messages",  # Use built-in reducer
            "turn_count": operator.add,  # Sum turn counts
        }

    # Create and use state
    state = ConversationState(topic="AI Ethics")
    state.add_message(HumanMessage(content="What are the main concerns?"))
    state.add_messages(
        [
            AIMessage(content="The main concerns include..."),
            HumanMessage(content="What about bias?"),
        ]
    )

    # Method 2: Using factory function
    QuickState = create_message_state(
        {"user_id": (str, Field(default="")), "session_id": (str, Field(default=""))}
    )

    QuickState(user_id="user123")


def example_dynamic_composition() -> None:
    """Example 2: Dynamic schema composition."""
    # Create composer
    composer = SchemaComposer(name="DynamicRAGState")

    # Add fields programmatically
    composer.add_field(
        name="query",
        field_type=str,
        default="",
        description="User query",
        input_for=["retriever", "reranker"],
    )

    composer.add_field(
        name="documents",
        field_type=list[dict[str, Any]],
        default_factory=list,
        description="Retrieved documents",
        output_from=["retriever"],
        input_for=["reranker"],
        reducer=operator.add,  # Accumulate documents
    )

    composer.add_field(
        name="context",
        field_type=list[str],
        default_factory=list,
        description="Reranked context",
        output_from=["reranker"],
        input_for=["generator"],
        shared=True,  # Share with parent graph
    )

    composer.add_field(
        name="response",
        field_type=str,
        default="",
        description="Generated response",
        output_from=["generator"],
    )

    # Configure messages field
    composer.configure_messages_field(
        default_factory=list,
        description="Conversation history",
        shared=True,
        with_reducer=True,
    )

    # Build the schema
    DynamicRAGState = composer.build()

    # Use the dynamic schema
    DynamicRAGState(query="What is quantum computing?")


def example_reducer_behavior() -> None:
    """Example 3: Understanding reducer behavior."""

    class MetricsState(StateSchema):
        """State with various reducer types."""

        # List concatenation
        events: list[str] = Field(default_factory=list)

        # Number addition
        total_count: int = Field(default=0)

        # Replace (last write wins)
        status: str = Field(default="idle")

        # Custom reducer
        tags: list[str] = Field(default_factory=list)

        # Max value reducer
        high_score: float = Field(default=0.0)

        __reducer_fields__ = {
            "events": operator.add,  # Concatenate lists
            "total_count": operator.add,  # Sum numbers
            "status": lambda old, new: new,  # Replace
            "tags": lambda old, new: list(set(old + new)),  # Unique merge
            "high_score": max,  # Keep maximum
        }

    # Initial state
    state = MetricsState()

    # First update
    update1 = {
        "events": ["started", "processing"],
        "total_count": 5,
        "status": "running",
        "tags": ["urgent", "ml"],
        "high_score": 0.85,
    }
    state.apply_reducers(update1)

    # Second update
    update2 = {
        "events": ["completed"],
        "total_count": 3,
        "status": "done",
        "tags": ["ml", "production"],  # ml is duplicate
        "high_score": 0.92,
    }
    state.apply_reducers(update2)


def example_parent_child_sharing() -> None:
    """Example 4: Parent-child graph state sharing."""

    # Parent graph state
    class ParentState(StateSchema):
        """Main workflow state."""

        messages: list[BaseMessage] = Field(default_factory=list)
        user_query: str = Field(default="")
        final_answer: str = Field(default="")
        metadata: dict[str, Any] = Field(default_factory=dict)

        # Only messages and query are shared
        __shared_fields__ = ["messages", "user_query"]

        __reducer_fields__ = {
            "messages": "add_messages",
            "metadata": lambda old, new: {**old, **new},  # Merge dicts
        }

    # Child graph state
    class ResearchState(StateSchema):
        """Research subgraph state."""

        # Shared from parent
        messages: list[BaseMessage] = Field(default_factory=list)
        user_query: str = Field(default="")

        # Local to child
        search_results: list[dict[str, str]] = Field(default_factory=list)
        research_summary: str = Field(default="")

        __shared_fields__ = ["messages", "user_query"]  # Must match parent
        __reducer_fields__ = {
            "messages": "add_messages",
            "search_results": operator.add,
        }

    # Simulate parent state
    parent = ParentState(user_query="Explain quantum entanglement")
    parent.add_message(HumanMessage(content="Explain quantum entanglement"))

    # Child would receive shared fields
    child = ResearchState(
        messages=parent.messages, user_query=parent.user_query  # Shared fields
    )

    # Child does work
    child.search_results = [
        {"title": "Quantum Entanglement Basics", "url": "..."},
        {"title": "EPR Paradox Explained", "url": "..."},
    ]
    child.add_message(AIMessage(content="I found several resources..."))

    # Updates to shared fields would propagate back


def example_engine_io_tracking() -> None:
    """Example 5: Engine I/O tracking for complex workflows."""

    class MultiEngineState(StateSchema):
        """State for multi-engine workflow."""

        # Input fields
        raw_text: str = Field(default="")
        language: str = Field(default="en")

        # Intermediate fields
        entities: list[dict[str, str]] = Field(default_factory=list)
        sentiment: dict[str, float] = Field(default_factory=dict)
        keywords: list[str] = Field(default_factory=list)

        # Output fields
        summary: str = Field(default="")
        insights: dict[str, Any] = Field(default_factory=dict)

        __engine_io_mappings__ = {
            "ner_engine": {"inputs": ["raw_text", "language"], "outputs": ["entities"]},
            "sentiment_engine": {
                "inputs": ["raw_text"],
                "outputs": ["sentiment"],
            },
            "keyword_engine": {
                "inputs": ["raw_text", "language"],
                "outputs": ["keywords"],
            },
            "summary_engine": {
                "inputs": ["raw_text", "entities", "keywords"],
                "outputs": ["summary"],
            },
            "insight_engine": {
                "inputs": ["entities", "sentiment", "keywords"],
                "outputs": ["insights"],
            },
        }

    # Create state
    state = MultiEngineState(
        raw_text="Apple announced record profits...", language="en"
    )

    # Prepare input for specific engine
    state.prepare_for_engine("ner_engine")

    # Simulate engine outputs
    state.entities = [{"text": "Apple", "type": "ORG"}]
    state.keywords = ["profits", "record", "announcement"]

    # Get inputs for summary engine
    state.prepare_for_engine("summary_engine")

    # Display engine dependencies
    for _engine, _io in state.__engine_io_mappings__.items():
        pass


def example_validation_and_custom_methods() -> Any:
    """Example 6: Validation and custom methods."""

    class ValidatedState(StateSchema):
        """State with validation and custom methods."""

        email: str = Field(default="")
        age: int = Field(default=0)
        score: float = Field(default=0.0)
        tags: list[str] = Field(default_factory=list)

        @validator("email")
        def validate_email(self, v) -> Any:
            if v and "@" not in v:
                raise ValueError("Invalid email format")
            return v.lower()  # Normalize

        @validator("age")
        def validate_age(self, v) -> Any:
            if v < 0:
                raise ValueError("Age cannot be negative")
            if v > 150:
                raise ValueError("Age seems unrealistic")
            return v

        @validator("score")
        def validate_score(self, v) -> Any:
            return max(0.0, min(1.0, v))  # Clamp to [0, 1]

        @validator("tags")
        def validate_tags(self, v) -> Any:
            # Remove duplicates and empty strings
            return list({tag.strip() for tag in v if tag.strip()})

        def get_risk_level(self) -> str:
            """Calculate risk level based on score."""
            if self.score < 0.3:
                return "low"
            if self.score < 0.7:
                return "medium"
            return "high"

        def add_tag(self, tag: str) -> None:
            """Add a tag if not already present."""
            tag = tag.strip()
            if tag and tag not in self.tags:
                self.tags.append(tag)

        def to_summary(self) -> dict[str, Any]:
            """Generate summary of state."""
            return {
                "email": self.email,
                "age_group": "adult" if self.age >= 18 else "minor",
                "risk_level": self.get_risk_level(),
                "tag_count": len(self.tags),
            }

    # Test validation
    with contextlib.suppress(ValueError):
        state = ValidatedState(email="invalid-email", age=200)

    # Create valid state
    state = ValidatedState(email="user@example.com", age=25, score=1.5)  # Score clamped

    # Use custom methods
    state.add_tag("important")
    state.add_tag("urgent")
    state.add_tag("important")  # Duplicate, won't be added


def example_schema_merging() -> None:
    """Example 7: Merging schemas from multiple sources."""
    # Create first schema
    composer1 = SchemaComposer(name="SearchSchema")
    composer1.add_field("query", str, default="")
    composer1.add_field("results", list[dict[str, Any]], default_factory=list)

    # Create second schema
    composer2 = SchemaComposer(name="AnalysisSchema")
    composer2.add_field("text", str, default="")
    composer2.add_field("analysis", dict[str, Any], default_factory=dict)

    # Merge schemas
    merged = SchemaComposer.merge(
        composer1, composer2, name="SearchAndAnalyzeState", messages_field=True
    )

    # Build merged schema
    MergedState = merged.build()

    # Check fields
    for _field_name, _field_info in MergedState.model_fields.items():
        pass

    # Use merged state
    MergedState(query="AI safety", text="Recent research shows...")


def example_real_world_agent_state() -> None:
    """Example 8: Real-world agent state pattern."""
    # Use factory for common agent pattern
    AgentState = create_agent_state(
        name="SmartAgentState",
        additional_fields={
            "plan": (list[str], Field(default_factory=list)),
            "current_step": (int, Field(default=0)),
            "tools_used": (list[str], Field(default_factory=list)),
            "reasoning": (str, Field(default="")),
        },
        shared_fields=["messages", "context"],
        reducers={
            "plan": operator.add,
            "tools_used": operator.add,
            "current_step": max,  # Keep highest step
        },
    )

    # Create agent state
    state = AgentState()

    # Simulate agent execution
    state.add_message(HumanMessage(content="Research climate change solutions"))
    state.plan = ["Search for recent papers", "Analyze findings", "Generate summary"]
    state.reasoning = "User wants comprehensive climate solutions research"

    # Execute first step
    state.current_step = 1
    state.tools_used.append("search_tool")
    state.context.append("Found 15 relevant papers on renewable energy")

    # Execute second step
    state.apply_reducers(
        {
            "current_step": 2,
            "tools_used": ["analysis_tool"],
            "context": ["Key finding: Solar efficiency increased 40%"],
        }
    )


def example_serialization_patterns() -> None:
    """Example 9: Serialization and persistence patterns."""

    class PersistentState(StateSchema):
        """State designed for persistence."""

        session_id: str = Field(default="")
        messages: list[BaseMessage] = Field(default_factory=list)
        metadata: dict[str, Any] = Field(default_factory=dict)
        created_at: str = Field(default="")

        __reducer_fields__ = {
            "messages": "add_messages",
            "metadata": lambda old, new: {**old, **new},
        }

        class Config:
            # Custom JSON encoders for special types
            json_encoders = {
                BaseMessage: lambda v: {
                    "type": v.__class__.__name__,
                    "content": v.content,
                    "additional_kwargs": v.additional_kwargs,
                }
            }

    # Create state
    import datetime

    state = PersistentState(
        session_id="sess_123",
        created_at=datetime.datetime.now().isoformat(),
        metadata={"user": "alice", "topic": "chatbot"},
    )

    state.add_messages(
        [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there! How can I help?"),
        ]
    )

    # Serialize to JSON
    json_data = state.to_json()

    # Serialize to dict
    state.to_dict()

    # Restore from JSON
    restored = PersistentState.from_json(json_data)

    # Pretty print for debugging
    restored.pretty_print()


def example_schema_visualization() -> None:
    """Example 10: Schema visualization and introspection."""

    class ComplexState(StateSchema):
        """Complex state for visualization demo."""

        # Message handling
        messages: list[BaseMessage] = Field(
            default_factory=list, description="Conversation messages"
        )

        # Query processing
        query: str = Field(default="", description="User query")
        query_embedding: list[float] | None = Field(
            default=None, description="Query vector"
        )

        # Results
        results: list[dict[str, Any]] = Field(
            default_factory=list, description="Search results"
        )
        score: float = Field(default=0.0, description="Relevance score")

        # Metadata
        metadata: dict[str, Any] = Field(
            default_factory=dict, description="Additional metadata"
        )

        __shared_fields__ = ["messages", "query"]
        __reducer_fields__ = {
            "messages": "add_messages",
            "results": operator.add,
            "score": max,
            "metadata": lambda old, new: {**old, **new},
        }
        __engine_io_mappings__ = {
            "embeddef": {"inputs": ["query"], "outputs": ["query_embedding"]},
            "searchef": {
                "inputs": ["query", "query_embedding"],
                "outputs": ["results", "score"],
            },
        }

    # Display schema information
    SchemaUI.display_schema(ComplexState)

    # Get schema info programmatically
    get_schema_info(ComplexState)

    # Generate code representation
    ComplexState.to_python_code()

    # Validate schema structure
    validate_schema(ComplexState)


def main() -> None:
    """Run all examples."""
    examples = [
        example_basic_state_creation,
        example_dynamic_composition,
        example_reducer_behavior,
        example_parent_child_sharing,
        example_engine_io_tracking,
        example_validation_and_custom_methods,
        example_schema_merging,
        example_real_world_agent_state,
        example_serialization_patterns,
        example_schema_visualization,
    ]

    for example in examples:
        example()
        input("\nPress Enter to continue to next example...")


if __name__ == "__main__":
    # Run specific example or all
    import sys

    if len(sys.argv) > 1:
        example_num = int(sys.argv[1]) - 1
        examples = [
            example_basic_state_creation,
            example_dynamic_composition,
            example_reducer_behavior,
            example_parent_child_sharing,
            example_engine_io_tracking,
            example_validation_and_custom_methods,
            example_schema_merging,
            example_real_world_agent_state,
            example_serialization_patterns,
            example_schema_visualization,
        ]
        if 0 <= example_num < len(examples):
            examples[example_num]()
        else:
            pass
    else:
        main()
