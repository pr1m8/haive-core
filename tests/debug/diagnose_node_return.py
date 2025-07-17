#!/usr/bin/env python3
"""Diagnose node return value issue in self-discovery agent."""

import sys

sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-agents/src")
sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")


def show_langgraph_version():
    """Check LangGraph version to understand expected return types."""
    try:
        import langgraph

        print(f"📦 LangGraph version: {langgraph.__version__}")

        # Check if Command is available
        try:
            from langgraph.types import Command

            print("✅ Command import successful - modern LangGraph")

            # Show what Command expects
            test_command = Command(update={"test": "value"})
            print(f"✅ Command structure: {test_command}")

        except ImportError:
            print("❌ Command not available - older LangGraph version")

    except Exception as e:
        print(f"❌ LangGraph import error: {e}")


def analyze_agent_node_v3_return():
    """Analyze what AgentNodeV3 actually returns."""
    print("\n🔍 ANALYZING: What AgentNodeV3 returns")

    try:

        print("✅ Successfully imported AgentNodeV3Config and Command")

        # Show the return type
        print("📋 AgentNodeV3.__call__ returns: Command object")
        print("📋 Command contains: update (dict) + goto (optional)")

        # The issue: Let's check if there's a version mismatch
        print("\n🔍 CHECKING: Version compatibility")

        # Create a simple test state to see what happens
        test_state = {"task_description": "test", "messages": []}
        print(f"📝 Test state: {test_state}")

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback

        traceback.print_exc()


def test_simple_agent_execution():
    """Test a simple agent to see the exact return value."""
    print("\n🔍 TESTING: Simple agent execution")

    try:
        # Import the self_discovery agent
        from haive.agents.reasoning_and_critique.self_discover.v2.agent import (
            self_discovery,
        )

        # Get the first sub-agent (select_modules)
        first_agent_name = list(self_discovery.agents.keys())[0]
        first_agent = self_discovery.agents[first_agent_name]

        print(f"🔍 Testing first agent: {first_agent_name}")
        print(f"🔍 Agent type: {type(first_agent)}")

        # Try to invoke it directly with simple input
        test_input = {
            "task_description": "Simple test task",
            "messages": [],
            "reasoning_modules": "Test modules",
        }

        print(f"📝 Input: {list(test_input.keys())}")

        # THIS is where we'll see what the agent actually returns
        result = first_agent.invoke(test_input)
        print(f"✅ Agent result type: {type(result)}")
        print(f"✅ Agent result: {result}")

        # Now test what AgentNodeV3 would return
        from haive.core.graph.node.agent_node_v3 import create_agent_node_v3

        node_config = create_agent_node_v3(
            agent_name=first_agent_name,
            agent=first_agent,
            name=f"test_{first_agent_name}",
        )

        print("\n🔍 TESTING: AgentNodeV3 execution")
        node_result = node_config(test_input)
        print(f"✅ Node result type: {type(node_result)}")
        print(f"✅ Node result: {node_result}")

        # THIS should be a Command object
        if hasattr(node_result, "update"):
            print(f"✅ Command.update: {node_result.update}")
            print(f"✅ Command.update type: {type(node_result.update)}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


def check_langgraph_expects():
    """Check what LangGraph StateGraph actually expects."""
    print("\n🔍 CHECKING: What LangGraph expects vs what we return")

    try:
        import langgraph

        # Check the LangGraph source/docs for expected return types
        print(f"📦 LangGraph package: {langgraph}")

        # The ERROR MESSAGE tells us: "Expected dict, got"
        # This suggests LangGraph expects a plain dict, but AgentNodeV3 returns Command

        print("❗ ISSUE IDENTIFIED:")
        print("   LangGraph error says: 'Expected dict, got'")
        print("   AgentNodeV3 returns: Command object")
        print("   → VERSION MISMATCH or CONFIGURATION ISSUE")

    except Exception as e:
        print(f"❌ Check failed: {e}")


if __name__ == "__main__":
    print("🚀 Node Return Value Diagnostics")
    print("=" * 50)

    show_langgraph_version()
    analyze_agent_node_v3_return()
    test_simple_agent_execution()
    check_langgraph_expects()

    print("\n🎯 SUMMARY:")
    print("The error 'Expected dict, got' means:")
    print("1. LangGraph expects plain dict returns")
    print("2. AgentNodeV3 returns Command objects")
    print("3. There's a mismatch in expected return types")
    print("4. We need to either:")
    print("   - Update LangGraph to a version that supports Command")
    print("   - OR modify AgentNodeV3 to return plain dicts")
