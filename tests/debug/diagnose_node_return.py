#!/usr/bin/env python3
"""Diagnose node return value issue in self-discovery agent."""

import sys

sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-agents/src")
sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")


def show_langgraph_version():
    """Check LangGraph version to understand expected return types."""
    try:
        # Check if Command is available
        try:
            from langgraph.types import Command

            # Show what Command expects
            Command(update={"test": "value"})

        except ImportError:
            pass

    except Exception:
        pass


def analyze_agent_node_v3_return():
    """Analyze what AgentNodeV3 actually returns."""
    try:
        # Show the return type

        # The issue: Let's check if there's a version mismatch

        # Create a simple test state to see what happens
        pass

    except Exception:
        import traceback

        traceback.print_exc()


def test_simple_agent_execution():
    """Test a simple agent to see the exact return value."""
    try:
        # Import the self_discovery agent
        from haive.agents.reasoning_and_critique.self_discover.v2.agent import (
            self_discovery,
        )

        # Get the first sub-agent (select_modules)
        first_agent_name = next(iter(self_discovery.agents.keys()))
        first_agent = self_discovery.agents[first_agent_name]

        # Try to invoke it directly with simple input
        test_input = {
            "task_description": "Simple test task",
            "messages": [],
            "reasoning_modules": "Test modules",
        }

        # THIS is where we'll see what the agent actually returns
        first_agent.invoke(test_input)

        # Now test what AgentNodeV3 would return
        from haive.core.graph.node.agent_node_v3 import create_agent_node_v3

        node_config = create_agent_node_v3(
            agent_name=first_agent_name,
            agent=first_agent,
            name=f"test_{first_agent_name}",
        )

        node_result = node_config(test_input)

        # THIS should be a Command object
        if hasattr(node_result, "update"):
            pass

    except Exception:
        import traceback

        traceback.print_exc()


def check_langgraph_expects():
    """Check what LangGraph StateGraph actually expects."""
    try:
        # Check the LangGraph source/docs for expected return types

        # The ERROR MESSAGE tells us: "Expected dict, got"
        # This suggests LangGraph expects a plain dict, but AgentNodeV3 returns
        # Command

        pass

    except Exception:
        pass


if __name__ == "__main__":
    show_langgraph_version()
    analyze_agent_node_v3_return()
    test_simple_agent_execution()
    check_langgraph_expects()
