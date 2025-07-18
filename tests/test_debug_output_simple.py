"""Test to reproduce the messy debug output issue without persistence."""

import logging
from typing import List

from haive.agents.simple.agent_v2 import SimpleAgentV2
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig

# Disable most logging to focus on the issue
logging.disable(logging.CRITICAL)


class QueryRefinementSuggestion(BaseModel):
    """A suggestion for refining the query."""

    refined_query: str = Field(description="The refined version of the query")
    improvement_type: str = Field(description="Type of improvement")
    rationale: str = Field(
        description="Explanation of why this refinement is beneficial"
    )
    expected_benefit: str = Field(description="Expected benefit of this refinement")


class QueryRefinementResponse(BaseModel):
    """Response for query refinement analysis."""

    original_query: str = Field(description="The original user query")
    query_analysis: str = Field(
        description="Analysis of the query's strengths and weaknesses"
    )
    query_type: str = Field(description="Type of query")
    complexity_level: str = Field(description="Complexity level")
    refinement_suggestions: List[QueryRefinementSuggestion] = Field(
        description="List of refinement suggestions"
    )
    best_refined_query: str = Field(
        description="The best refined query from the suggestions"
    )
    search_strategy_recommendations: List[str] = Field(
        description="Recommended search strategies"
    )


def test_debug_output_simple():
    """Test the debug output issue with structured output."""
    print("=== TESTING DEBUG OUTPUT ISSUE ===\n")

    # Create agent with structured output but no persistence
    engine = AugLLMConfig(
        temperature=0.3, structured_output_model=QueryRefinementResponse
    )

    agent = SimpleAgentV2(
        name="query_refiner",
        engine=engine,
        structured_output_model=QueryRefinementResponse,
    )

    # Disable persistence to avoid database issues
    agent.checkpointer = None
    agent.store = None

    print("Running agent with debug=True (no persistence)...")
    try:
        result = agent.run(
            "what is the tallest tower in north america",
            debug=True,
            thread_id=None,  # No persistence
        )

        print(f"\n=== RESULT TYPE: {type(result)} ===")
        print(f"Result: {result}")

        # Check what we're getting
        if hasattr(result, "query_refinement_response"):
            print(f"\n=== STRUCTURED OUTPUT FIELD ===")
            print(f"Field value: {result.query_refinement_response}")
            print(f"Field type: {type(result.query_refinement_response)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_debug_output_simple()
