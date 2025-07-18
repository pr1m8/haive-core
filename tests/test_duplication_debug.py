"""Debug test to investigate tool call and AI message duplication in SimpleAgentV2."""

import logging
from typing import List

from haive.agents.simple.agent_v2 import SimpleAgentV2
from pydantic import BaseModel, Field

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
    print("=== DEBUGGING DUPLICATION ISSUES ===\n")

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

    print("Agent configuration:")
    print(f"  use_parser_safety_net: {agent.use_parser_safety_net}")
    print(f"  parser_safety_net_mode: {agent.parser_safety_net_mode}")
    print(f"  structured_output_model: {agent.structured_output_model}")

    # Check the graph structure
    if hasattr(agent, "graph"):
        print(f"\nGraph nodes: {agent.graph.metadata.get('available_nodes', [])}")
        print(f"Tool routes: {agent.get_tool_routes()}")

    try:
        print("\n=== RUNNING AGENT ===")
        result = agent.run(
            "Test input for duplication",
            debug=True,  # Enable debug to see execution flow
        )

        print(f"\n=== RESULT ===")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")

        # Check if there are any duplicates in the state
        if hasattr(agent, "_app") and agent._app:
            try:
                # Get the final state
                final_state = agent._app.get_state(
                    {"configurable": {"thread_id": "test"}}
                )
                if final_state and hasattr(final_state, "values"):
                    messages = final_state.values.get("messages", [])
                    print(f"\n=== FINAL STATE MESSAGES ===")
                    print(f"Total messages: {len(messages)}")

                    from langchain_core.messages import AIMessage, ToolMessage

                    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
                    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

                    print(f"AI Messages: {len(ai_messages)}")
                    print(f"Tool Messages: {len(tool_messages)}")

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

                    print(f"Tool call IDs: {tool_call_ids}")
                    print(f"Tool message IDs: {tool_message_ids}")

                    # Check for duplicates
                    duplicate_tool_calls = len(tool_call_ids) != len(set(tool_call_ids))
                    duplicate_tool_messages = len(tool_message_ids) != len(
                        set(tool_message_ids)
                    )

                    print(f"Duplicate tool calls: {duplicate_tool_calls}")
                    print(f"Duplicate tool messages: {duplicate_tool_messages}")

                    if duplicate_tool_calls or duplicate_tool_messages:
                        print("\n❌ DUPLICATION DETECTED!")
                        print("This is likely caused by:")
                        print("  1. Safety net creating duplicate ToolMessages")
                        print(
                            "  2. Multiple parser nodes processing the same tool call"
                        )
                        print("  3. Tool call ID mismatches")
                    else:
                        print("\n✅ No duplicates found")

            except Exception as e:
                print(f"Error checking state: {e}")

    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_duplication_debug()
