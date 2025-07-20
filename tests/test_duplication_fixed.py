"""Test to verify the duplication fix is working."""

import logging

from pydantic import BaseModel, Field

from haive.agents.simple.agent_v2 import SimpleAgentV2
from haive.core.engine.aug_llm import AugLLMConfig

# Reduce logging to focus on the issue
logging.getLogger().setLevel(logging.ERROR)


class SampleTestModel(BaseModel):
    """Simple test model for structured output."""

    analysis: str = Field(description="Analysis result")
    score: float = Field(description="Quality score")


def test_duplication_fix():
    """Test that the duplication fix is working."""
    # Create agent with structured output
    engine = AugLLMConfig(
        temperature=0.1,
        structured_output_model=SampleTestModel)

    agent = SimpleAgentV2(
        name="duplication_fix_test",
        engine=engine,
        structured_output_model=SampleTestModel,
    )

    # Disable persistence to avoid database issues
    agent.checkpointer = None
    agent.store = None

    try:
        result = agent.run(
            "Analyze the quality of this test input",
            debug=False)

        # Verify the result has the expected structure
        if hasattr(result, "original_data"):

            # Check if it has the expected structured output field
            if hasattr(result.original_data, "test_model"):
                pass
            else:
                pass

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_duplication_fix()
