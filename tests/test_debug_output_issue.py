"""Test to reproduce the messy debug output issue."""

import logging

from pydantic import BaseModel, Field

from haive.agents.simple.agent_v2 import SimpleAgentV2
from haive.core.engine.aug_llm import AugLLMConfig

# Set up logging to see debug output
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class QueryRefinementSuggestion(BaseModel):
    """A suggestion for refining the query."""

    refined_query: str = Field(description="The refined version of the query")
    improvement_type: str = Field(
        description="Type of improvement (e.g., 'Add Specificity', 'Clarify Intent')"
    )
    rationale: str = Field(description="Explanation of why this refinement is beneficial")
    expected_benefit: str = Field(description="Expected benefit of this refinement")


class QueryRefinementResponse(BaseModel):
    """Response for query refinement analysis."""

    original_query: str = Field(description="The original user query")
    query_analysis: str = Field(description="Analysis of the query's strengths and weaknesses")
    query_type: str = Field(
        description="Type of query (e.g., 'Factual', 'Comparative', 'Exploratory')"
    )
    complexity_level: str = Field(description="Complexity level (simple, moderate, complex)")
    refinement_suggestions: list[QueryRefinementSuggestion] = Field(
        description="List of refinement suggestions"
    )
    best_refined_query: str = Field(description="The best refined query from the suggestions")
    search_strategy_recommendations: list[str] = Field(description="Recommended search strategies")


def test_debug_output_issue():
    """Test the debug output issue with structured output."""
    # Create agent with structured output
    engine = AugLLMConfig(temperature=0.3, structured_output_model=QueryRefinementResponse)

    agent = SimpleAgentV2(
        name="query_refiner",
        engine=engine,
        structured_output_model=QueryRefinementResponse,
    )

    # Test with debug=True to see the messy output
    result = agent.run("what is the tallest tower in north america", debug=True)

    # Check what we're getting
    if hasattr(result, "query_refinement_response"):
        pass


if __name__ == "__main__":
    test_debug_output_issue()
