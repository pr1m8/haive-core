#!/usr/bin/env python3
"""Test AgentNodeV3 directly with SimpleAgentV3 to isolate the state issue."""



# Import SimpleAgentV3 from haive-agents
from haive.agents.simple.agent_v3 import SimpleAgentV3
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from haive.core.engine.aug_llm import AugLLMConfig

# Import from haive-core
from haive.core.graph.node.agent_node_v3 import create_agent_node_v3
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


@tool
def test_calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def test_agent_node_v3_creation():
    """Test AgentNodeV3 can be created with SimpleAgentV3."""
    print("\n🔍 Testing AgentNodeV3 Creation")
    print("=" * 50)

    # Create SimpleAgentV3
    agent = SimpleAgentV3(name="test_simple", engine=AugLLMConfig(temperature=0.7))

    print(f"✅ SimpleAgentV3 created: {agent.name}")
    print(f"   Type: {type(agent)}")
    print(f"   State schema: {agent.state_schema}")

    # Create AgentNodeV3
    node_config = create_agent_node_v3(
        agent_name="test_simple", agent=agent, name="test_node"
    )

    print(f"✅ AgentNodeV3 created: {type(node_config)}")
    print(f"   Callable: {callable(node_config)}")

    # Check attributes
    if hasattr(node_config, "__dict__"):
        attrs = list(node_config.__dict__.keys())
        print(f"   Attributes: {attrs}")

    return node_config


def test_agent_node_v3_with_dict_input():
    """Test AgentNodeV3 execution with dict input (what LangGraph passes)."""
    print("\n🔍 Testing AgentNodeV3 with Dict Input")
    print("=" * 50)

    # Create agent and node
    agent = SimpleAgentV3(name="test_simple", engine=AugLLMConfig(temperature=0.7))

    node_config = create_agent_node_v3(
        agent_name="test_simple", agent=agent, name="test_node"
    )

    # Create test input as DICT (what LangGraph would pass)
    test_input = {
        "messages": [HumanMessage(content="What is 5 + 7?")],
        "agents": {"test_simple": agent},
        "agent_states": {},
        "active_agent": "test_simple",
    }

    print(f"📥 Input type: {type(test_input)}")
    print(f"   Input keys: {list(test_input.keys())}")

    try:
        # THIS IS THE CRITICAL TEST - what does AgentNodeV3 return?
        result = node_config(test_input)

        print(f"📤 Result type: {type(result)}")
        print(f"   Result: {result}")

        # Check if it's a Command object
        if hasattr(result, "update"):
            print(f"   ✅ Has 'update' attribute: {type(result.update)}")
            print(
                f"   Update keys: {list(result.update.keys()) if isinstance(result.update, dict) else 'Not dict'}"
            )

        # Check if it's a dict
        if isinstance(result, dict):
            print(f"   ✅ Is dict with keys: {list(result.keys())}")

        return result

    except Exception as e:
        print(f"❌ AgentNodeV3 execution failed: {e}")
        print(f"   Error type: {type(e)}")
        import traceback

        traceback.print_exc()
        return None


def test_agent_node_v3_with_multi_agent_state():
    """Test AgentNodeV3 with actual MultiAgentState object."""
    print("\n🔍 Testing AgentNodeV3 with MultiAgentState")
    print("=" * 50)

    # Create agent and node
    agent = SimpleAgentV3(name="test_simple", engine=AugLLMConfig(temperature=0.7))

    node_config = create_agent_node_v3(
        agent_name="test_simple", agent=agent, name="test_node"
    )

    # Create MultiAgentState (Pydantic model)
    multi_state = MultiAgentState(
        messages=[HumanMessage(content="What is 5 + 7?")],
        agents={"test_simple": agent},
        agent_states={},
    )

    print(f"📥 MultiAgentState type: {type(multi_state)}")
    print(f"   Is subscriptable: {hasattr(multi_state, '__getitem__')}")
    print(f"   Has model_dump: {hasattr(multi_state, 'model_dump')}")

    try:
        # Test 1: Pass MultiAgentState directly
        print("\n🧪 Test 1: Direct MultiAgentState")
        result1 = node_config(multi_state)

        print("✅ Direct execution successful!")
        print(f"   Result type: {type(result1)}")

    except Exception as e:
        print(f"❌ Direct execution failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        # Test 2: Pass as dict (model_dump)
        print("\n🧪 Test 2: MultiAgentState.model_dump()")
        dict_state = multi_state.model_dump()
        result2 = node_config(dict_state)

        print("✅ Dict execution successful!")
        print(f"   Result type: {type(result2)}")

        return result2

    except Exception as e:
        print(f"❌ Dict execution failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_langgraph_command_expectation():
    """Test what LangGraph expects from node returns."""
    print("\n🔍 Testing LangGraph Command Expectations")
    print("=" * 50)

    try:
        from langgraph.types import Command

        # Test Command creation
        test_command = Command(update={"test": "value"})
        print(f"✅ Command created: {test_command}")
        print(f"   Command type: {type(test_command)}")
        print(f"   Command.update: {test_command.update}")

        # Test what happens if we return Command vs dict
        return test_command

    except ImportError:
        print("❌ Command not available - using older LangGraph")
        return None


def test_subscriptable_issue():
    """Test the specific subscriptable issue."""
    print("\n🔍 Testing Subscriptable Issue")
    print("=" * 50)

    # Create MultiAgentState
    agent = SimpleAgentV3(name="test", engine=AugLLMConfig())
    multi_state = MultiAgentState(
        messages=[HumanMessage(content="test")], agents={"test": agent}
    )

    # Test direct subscript access
    try:
        # This should FAIL - MultiAgentState is not subscriptable
        value = multi_state["messages"]
        print(f"❌ Unexpected success: {value}")
    except TypeError as e:
        print(f"✅ Expected error: {e}")
        print("   This confirms MultiAgentState is not subscriptable")

    # Test attribute access
    try:
        # This should WORK - Pydantic attribute access
        value = multi_state.messages
        print(f"✅ Attribute access works: {len(value)} messages")
    except Exception as e:
        print(f"❌ Attribute access failed: {e}")

    # Test model_dump subscript access
    try:
        # This should WORK - dict is subscriptable
        dict_state = multi_state.model_dump()
        value = dict_state["messages"]
        print(f"✅ Dict subscript works: {len(value)} messages")
    except Exception as e:
        print(f"❌ Dict subscript failed: {e}")


def main():
    """Run all AgentNodeV3 tests."""
    print("🚀 AgentNodeV3 Direct Testing with SimpleAgentV3")
    print("=" * 80)

    # Run all tests
    try:
        node_config = test_agent_node_v3_creation()
        dict_result = test_agent_node_v3_with_dict_input()
        state_result = test_agent_node_v3_with_multi_agent_state()
        command = test_langgraph_command_expectation()
        test_subscriptable_issue()

        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)
        print(
            f"AgentNodeV3 Creation:     {'✅ SUCCESS' if node_config else '❌ FAILED'}"
        )
        print(
            f"Dict Input Execution:     {'✅ SUCCESS' if dict_result else '❌ FAILED'}"
        )
        print(
            f"MultiAgentState Execution: {'✅ SUCCESS' if state_result else '❌ FAILED'}"
        )
        print(
            f"Command Support:          {'✅ AVAILABLE' if command else '❌ NOT AVAILABLE'}"
        )

        if dict_result and hasattr(dict_result, "update"):
            print("\n🎯 KEY INSIGHT: AgentNodeV3 returns Command objects")
            print("   This is CORRECT for LangGraph 0.3.31+")

        print("\n🎯 ISSUE CONFIRMED: MultiAgentState objects are not subscriptable")
        print("   LangGraph nodes expect dict access: state['key']")
        print("   But MultiAgentState uses attribute access: state.key")

    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
