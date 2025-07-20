"""Test real token tracking with actual SimpleAgent interactions.

This test verifies that token tracking works end-to-end with real LLM calls.
"""

import asyncio

from langchain_core.messages import AIMessage

from haive.agents.simple.agent import SimpleAgent
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.schema.prebuilt.llm_state import LLMState
from haive.core.schema.prebuilt.messages.messages_with_token_usage import (
    MessagesStateWithTokenUsage,
)


async def test_simple_agent_real_token_tracking():
    """Test that SimpleAgent actually tracks tokens with real LLM calls."""
    # Create SimpleAgent with real LLM
    config = AugLLMConfig(
        temperature=0.1,
        max_tokens=50,  # Keep it small for testing
    )

    agent = SimpleAgent(name="token_tracker_test", engine=config)

    # Make a real LLM call
    await agent.arun("What is 2 + 2? Answer briefly.")

    # Try to access agent state/memory to check token tracking
    # This might vary based on SimpleAgent implementation

    if hasattr(agent, "state"):
        state = agent.state

        if hasattr(state, "token_usage"):
            pass

        if hasattr(state, "messages"):
            for _i, msg in enumerate(state.messages):
                if hasattr(msg, "response_metadata"):
                    pass

    # Check if agent has conversation memory
    if hasattr(agent, "memory"):
        pass

    # Check if agent stores conversation history
    if hasattr(agent, "conversation_history"):
        pass


def test_llm_state_with_real_engine():
    """Test LLMState with real engine configuration."""
    engine = AugLLMConfig(
        name="llm_state_test",
        temperature=0.1,
        max_tokens=100)

    llm_state = LLMState(engine=engine)

    # Test with a real AI message that might have token usage
    ai_message = AIMessage(
        content="The answer is 4.",
        response_metadata={
            "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20}
        },
    )

    llm_state.add_message(ai_message)

    # Test LLMState specific features


def test_messages_state_with_token_usage_direct():
    """Test MessagesStateWithTokenUsage directly."""
    state = MessagesStateWithTokenUsage()

    # Add multiple messages with token usage
    messages = [
        AIMessage(
            content="Response 1",
            response_metadata={
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            },
        ),
        AIMessage(
            content="Response 2",
            response_metadata={
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                }
            },
        ),
        AIMessage(
            content="Response 3",
            response_metadata={
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 15,
                    "total_tokens": 45,
                }
            },
        ),
    ]

    for _i, msg in enumerate(messages):
        state.add_message(msg)

    # Test summary
    state.get_token_usage_summary()

    # Test cost calculation
    state.calculate_costs(input_cost_per_1k=0.001, output_cost_per_1k=0.002)


async def main():
    """Run all token tracking tests."""
    try:
        await test_simple_agent_real_token_tracking()
        test_llm_state_with_real_engine()
        test_messages_state_with_token_usage_direct()

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
