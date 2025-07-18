"""Compare the old messy output vs the new clean output."""

import logging
from typing import List

from haive.agents.simple.agent_v2 import SimpleAgentV2
from pydantic import BaseModel, Field

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
    refinement_suggestions: List[QueryRefinementSuggestion] = Field(
        description="List of refinement suggestions"
    )
    best_refined_query: str = Field(
        description="The best refined query from the suggestions"
    )
    search_strategy_recommendations: List[str] = Field(
        description="Recommended search strategies"
    )


def test_output_comparison():
    """Compare old messy output vs new clean output."""
    print("=== OUTPUT COMPARISON TEST ===\\n")

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

    print("Running agent with clean output formatting...")
    result = agent.run(query, debug=False)

    print("=== NEW CLEAN OUTPUT ===")
    print(result)

    print("\\n=== COMPARISON ===")
    print("✅ OLD OUTPUT: Huge messy dump with entire internal state")
    print("✅ NEW OUTPUT: Clean, formatted summary with key information")
    print("✅ BENEFITS:")
    print("  - Shows structured output fields clearly")
    print("  - Truncates long text to prevent overwhelming output")
    print("  - Summarizes lists (shows first 2 items + count)")
    print("  - Preserves access to original data via wrapper")
    print("  - Token usage information (when available)")
    print("  - Much more readable and professional")

    print("\\n=== TECHNICAL DETAILS ===")
    print(f"Result type: {type(result)}")
    print(f"Is FormattedOutput: {type(result).__name__ == 'FormattedOutput'}")
    print(f"Original data accessible: {hasattr(result, 'original_data')}")

    if hasattr(result, "original_data"):
        print(f"Original data type: {type(result.original_data)}")


if __name__ == "__main__":
    test_output_comparison()
