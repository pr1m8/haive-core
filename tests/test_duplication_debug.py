"""Debug test to investigate tool call and AI message duplication in SimpleAgentV2."""

import logging

from pydantic import BaseModel, Field

from haive.agents.simple.agent_v2 import SimpleAgentV2
from haive.core.engine.aug_llm import AugLLMConfig

# Enable debug logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestModel(BaseModel):
    """Simple test model for structured output."""

    result: str = Field(description="Test result")
    value: int = Field(description="Test value")


def test_duplication_debug():
    """Test to debug duplication issues."""
    # Create agent with structured output
    engine = AugLLMConfig(temperature=0.1, structured_output_model=TestModel)

    agent = SimpleAgentV2(
        name="duplication_test",
        engine=engine,
        structured_output_model=TestModel,
        use_parser_safety_net=True,  # This might be the culprit
        parser_safety_net_mode="create",  # Default mode
    )

    # Disable persistence to avoid database issues
    agent.checkpointer = None
    agent.store = None

    # Check the graph structure
    if hasattr(agent, "graph"):
        pass

    try:
        agent.run(
            "Test input for duplication",
            debug=True,  # Enable debug to see execution flow
        )

        # Check if there are any duplicates in the state
        if hasattr(agent, "_app") and agent._app:
            try:
                # Get the final state
                final_state = agent._app.get_state(
                    {"configurable": {"thread_id": "test"}}
                )
                if final_state and hasattr(final_state, "values"):
                    messages = final_state.values.get("messages", [])

                    from langchain_core.messages import AIMessage, ToolMessage

                    ai_messages = [
                        m for m in messages if isinstance(
                            m, AIMessage)]
                    tool_messages = [
                        m for m in messages if isinstance(
                            m, ToolMessage)]

                    # Check for duplicates
                    tool_call_ids = []
                    for ai_msg in ai_messages:
                        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                            for tc in ai_msg.tool_calls:
                                tool_call_ids.append(tc.get("id"))

                    tool_message_ids = []
                    for tool_msg in tool_messages:
                        if hasattr(tool_msg, "tool_call_id"):
                            tool_message_ids.append(tool_msg.tool_call_id)

                    # Check for duplicates
                    duplicate_tool_calls = len(
                        tool_call_ids) != len(set(tool_call_ids))
                    duplicate_tool_messages = len(tool_message_ids) != len(
                        set(tool_message_ids)
                    )

                    if duplicate_tool_calls or duplicate_tool_messages:
                        pass
                    else:
                        pass

            except Exception:
                pass

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_duplication_debug()
