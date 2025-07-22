"""Test to understand and fix the duplication issue in SimpleAgentV2."""

import logging

from haive.agents.simple.agent_v2 import SimpleAgentV2
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig

# Reduce logging to focus on the issue
logging.getLogger().setLevel(logging.WARNING)


class TestModel(BaseModel):
    """Simple test model for structured output."""

    result: str = Field(description="Test result")
    confidence: float = Field(description="Confidence score")


def test_with_safety_net_disabled():
    """Test agent with safety net disabled to see if it fixes duplication."""
    # Create agent with safety net DISABLED
    engine = AugLLMConfig(temperature=0.1, structured_output_model=TestModel)

    agent = SimpleAgentV2(
        name="no_safety_net",
        engine=engine,
        structured_output_model=TestModel,
        use_parser_safety_net=False,  # DISABLE safety net
        parser_safety_net_mode="ignore",  # Ignore mode
    )

    # Disable persistence to avoid database issues
    agent.checkpointer = None
    agent.store = None

    try:
        agent.run("Analyze the quality of this test", debug=False)

    except Exception:
        import traceback

        traceback.print_exc()


def test_with_v1_parser():
    """Test agent with V1 parser instead of V2."""
    # Create agent using V1 parser
    engine = AugLLMConfig(temperature=0.1, structured_output_model=TestModel)

    agent = SimpleAgentV2(
        name="v1_parser",
        engine=engine,
        structured_output_model=TestModel,
        use_parser_safety_net=False,  # Use V1 parser
    )

    # Disable persistence to avoid database issues
    agent.checkpointer = None
    agent.store = None

    try:
        agent.run("Analyze the quality of this test", debug=False)

    except Exception:
        import traceback

        traceback.print_exc()


def test_minimal_setup():
    """Test with minimal setup to isolate the issue."""
    # Create agent with absolute minimal configuration
    engine = AugLLMConfig(temperature=0.1, structured_output_model=TestModel)

    agent = SimpleAgentV2(
        name="minimal",
        engine=engine,
        structured_output_model=TestModel,
        use_parser_safety_net=False,
        parser_safety_net_mode="ignore",
    )

    # Disable ALL persistence
    agent.checkpointer = None
    agent.store = None
    agent._disable_checkpointing = True

    try:
        agent.run("Test minimal setup", debug=False)

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_with_safety_net_disabled()
    test_with_v1_parser()
    test_minimal_setup()
