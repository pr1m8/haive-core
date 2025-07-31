"""Compare the old messy output vs the new clean output."""

import logging

from pydantic import BaseModel, Field

from haive.agents.simple.agent_v2 import SimpleAgentV2
from haive.core.engine.aug_llm import AugLLMConfig

# Only show WARNING and above to reduce noise
logging.getLogger().setLevel(logging.WARNING)


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


def test_output_comparison():
    """Compare old messy output vs new clean output."""
    # Create agent with structured output
    engine = AugLLMConfig(
        temperature=0.1,  # Low temperature for more consistent output
        structured_output_model=QueryRefinementResponse,
    )

    agent = SimpleAgentV2(
        name="query_refiner",
        engine=engine,
        structured_output_model=QueryRefinementResponse,
    )

    # Disable persistence to avoid database issues
    agent.checkpointer = None
    agent.store = None

    # Test with the user's original query
    query = "what is the tallest tower in north america"

    result = agent.run(query, debug=False)

    if hasattr(result, "original_data"):
        pass


if __name__ == "__main__":
    test_output_comparison()
