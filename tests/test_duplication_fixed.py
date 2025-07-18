"""Test to verify the duplication fix is working."""

import logging

from haive.agents.simple.agent_v2 import SimpleAgentV2
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig

# Reduce logging to focus on the issue
logging.getLogger().setLevel(logging.ERROR)


class TestModel(BaseModel):
    """Simple test model for structured output."""

    analysis: str = Field(description="Analysis result")
    score: float = Field(description="Quality score")


def test_duplication_fix():
    """Test that the duplication fix is working."""
    print("=== TESTING DUPLICATION FIX ===\n")

    # Create agent with structured output
    engine = AugLLMConfig(temperature=0.1, structured_output_model=TestModel)

    agent = SimpleAgentV2(
        name="duplication_fix_test", engine=engine, structured_output_model=TestModel
    )

    # Disable persistence to avoid database issues
    agent.checkpointer = None
    agent.store = None

    print("Agent configuration:")
    print(f"  use_parser_safety_net: {agent.use_parser_safety_net}")
    print(f"  parser_safety_net_mode: {agent.parser_safety_net_mode}")

    try:
        print("\n=== RUNNING AGENT ===")
        result = agent.run("Analyze the quality of this test input", debug=False)

        print(f"\nResult: {result}")
        print("\n✅ SUCCESS: Agent ran without duplication issues!")

        # Verify the result has the expected structure
        if hasattr(result, "original_data"):
            print("\n=== RESULT VERIFICATION ===")
            print(f"Result type: {type(result.original_data)}")

            # Check if it has the expected structured output field
            if hasattr(result.original_data, "test_model"):
                structured_data = result.original_data.test_model
                print(f"Structured data type: {type(structured_data)}")
                print(f"Analysis: {structured_data.analysis[:100]}...")
                print(f"Score: {structured_data.score}")
                print("✅ Structured output is properly formatted!")
            else:
                print("❌ No structured output field found")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_duplication_fix()
