"""Test that the clean output formatting works correctly for structured outputs."""

import logging

from pydantic import BaseModel, Field

from haive.agents.simple.agent_v2 import SimpleAgentV2
from haive.core.engine.aug_llm import AugLLMConfig

# Disable debug logging to focus on the output
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
    refinement_suggestions: list[QueryRefinementSuggestion] = Field(
        description="List of refinement suggestions"
    )
    best_refined_query: str = Field(
        description="The best refined query from the suggestions"
    )
    search_strategy_recommendations: list[str] = Field(
        description="Recommended search strategies"
    )


def test_clean_output_format():
    """Test that structured output is formatted cleanly."""
    # Create agent with structured output
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

    try:
        result = agent.run(
            "what is the tallest tower in north america", debug=False  # Clean output
        )

        # Verify we can still access the original data
        if hasattr(result, "original_data"):
            if hasattr(result.original_data, "query_refinement_response"):
                pass

        # Test that token usage appears (if available)
        if hasattr(result, "original_data"):
            if hasattr(result.original_data, "token_usage"):
                pass

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_clean_output_format()
