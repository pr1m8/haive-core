#!/usr/bin/env python3
"""Show the actual output of AgentNodeV3 without any other dependencies."""

import sys

# Add the paths
sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")


def show_agentnodev3_output():
    """Show exactly what AgentNodeV3 outputs."""
    try:
        # Import just the essential parts

        # Import AgentNodeV3Config directly
        from haive.core.graph.node.agent_node_v3 import AgentNodeV3Config

        # Create a minimal mock agent
        class SimpleAgent:
            def __init__(self):
                self.name = "test_agent"

            def invoke(self, input_data, config=None):
                # Return a simple result
                return {"messages": [], "result": "Agent processed the input"}

        # Create simple test state
        test_state = {"messages": [], "task": "test task"}

        # Create AgentNodeV3Config
        node = AgentNodeV3Config(
            agent_name="test_agent", agent=SimpleAgent(), name="test_node"
        )

        # THIS IS THE KEY PART - Call the node and see what it returns
        result = node(test_state)

        # Check if it's a Command
        if hasattr(result, "update") or isinstance(result, dict):
            pass

        else:
            pass

        return result

    except Exception:
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":

    actual_output = show_agentnodev3_output()

    if actual_output is not None:
        if hasattr(actual_output, "update") or isinstance(actual_output, dict):
            pass
    else:
        pass
