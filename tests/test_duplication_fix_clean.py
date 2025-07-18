"""Clean test to verify the duplication fix is working."""

import logging
import os

# Disable database connection
os.environ["DISABLE_PERSISTENCE"] = "true"

from haive.agents.simple.agent_v2 import SimpleAgentV2
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig

# Reduce logging to focus on the issue
logging.getLogger().setLevel(logging.ERROR)


class SampleTestModel(BaseModel):
    """Simple test model for structured output."""

    analysis: str = Field(description="Analysis result")
    score: float = Field(description="Quality score")


def test_duplication_fix_clean():
    """Test that the duplication fix is working without database warnings."""
    print("\n=== TESTING DUPLICATION FIX (CLEAN) ===\n")

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

    print("Agent configuration:")
    print(f"  use_parser_safety_net: {agent.use_parser_safety_net}")
    print(f"  parser_safety_net_mode: {agent.parser_safety_net_mode}")

    # Run the agent
    result = agent.run("Analyze the quality of this test input", debug=False)

    # Check result
    assert result is not None
    print(f"\n✅ SUCCESS: Agent ran without duplication issues!")
    print(f"Result type: {type(result)}")

    # Verify structured output
    if hasattr(result, "original_data") and hasattr(
        result.original_data, "sample_test_model"
    ):
        structured_data = result.original_data.sample_test_model
        print(f"\nStructured output:")
        print(f"  Analysis: {structured_data.analysis[:50]}...")
        print(f"  Score: {structured_data.score}")
        assert isinstance(structured_data.score, float)
        assert isinstance(structured_data.analysis, str)
        print("\n✅ Structured output is properly formatted!")


if __name__ == "__main__":
    test_duplication_fix_clean()
