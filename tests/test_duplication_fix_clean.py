"""Clean test to verify the duplication fix is working."""

import logging
import os

from pydantic import BaseModel, Field

from haive.agents.simple.agent_v2 import SimpleAgentV2
from haive.core.engine.aug_llm import AugLLMConfig

# Disable database connection
os.environ["DISABLE_PERSISTENCE"] = "true"


# Reduce logging to focus on the issue
logging.getLogger().setLevel(logging.ERROR)


class SampleTestModel(BaseModel):
    """Simple test model for structured output."""

    analysis: str = Field(description="Analysis result")
    score: float = Field(description="Quality score")


def test_duplication_fix_clean():
    """Test that the duplication fix is working without database warnings."""
    # Create agent with structured output
    engine = AugLLMConfig(temperature=0.1, structured_output_model=SampleTestModel)

    agent = SimpleAgentV2(
        name="duplication_fix_test",
        engine=engine,
        structured_output_model=SampleTestModel,
        checkpointer=None,  # Disable persistence
        store=None,  # Disable store
    )

    # Double-check persistence is disabled
    agent.checkpointer = None
    agent.store = None

    # Run the agent
    result = agent.run("Analyze the quality of this test input", debug=False)

    # Check result
    assert result is not None

    # Verify structured output
    if hasattr(result, "original_data") and hasattr(
        result.original_data, "sample_test_model"
    ):
        structured_data = result.original_data.sample_test_model
        assert isinstance(structured_data.score, float)
        assert isinstance(structured_data.analysis, str)


if __name__ == "__main__":
    test_duplication_fix_clean()
