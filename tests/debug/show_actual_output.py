#!/usr/bin/env python3
"""Show the actual output of AgentNodeV3 without any other dependencies."""

import sys

# Add the paths
sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")


def show_agentnodev3_output():
    """Show exactly what AgentNodeV3 outputs."""
    print("🔍 SHOWING: Exact AgentNodeV3 Output")

    try:
        # Import just the essential parts

        # Import AgentNodeV3Config directly
        from haive.core.graph.node.agent_node_v3 import AgentNodeV3Config

        print("✅ Imports successful")

        # Create a minimal mock agent
        class SimpleAgent:
            def __init__(self):
                self.name = "test_agent"

            def invoke(self, input_data, config=None):
                print(f"  📨 Agent received: {list(input_data.keys())}")
                # Return a simple result
                return {"messages": [], "result": "Agent processed the input"}

        # Create simple test state
        test_state = {"messages": [], "task": "test task"}

        print(f"📝 Test state: {test_state}")

        # Create AgentNodeV3Config
        node = AgentNodeV3Config(
            agent_name="test_agent", agent=SimpleAgent(), name="test_node"
        )

        print(f"✅ Created node: {type(node)}")

        # THIS IS THE KEY PART - Call the node and see what it returns
        print("\n🚀 CALLING NODE...")
        result = node(test_state)

        print("\n📋 EXACT RESULT:")
        print(f"🔍 Type: {type(result)}")
        print(f"🔍 Repr: {result!r}")
        print(f"🔍 Str: {result!s}")

        # Check if it's a Command
        if hasattr(result, "update"):
            print("\n🔍 IT'S A COMMAND:")
            print(f"  Update type: {type(result.update)}")
            print(f"  Update content: {result.update}")
            print(f"  Goto: {getattr(result, 'goto', 'No goto')}")

        # Check if it's a dict
        elif isinstance(result, dict):
            print("\n🔍 IT'S A DICT:")
            print(f"  Keys: {list(result.keys())}")
            print(f"  Content: {result}")

        else:
            print(f"\n❓ UNKNOWN TYPE: {type(result)}")
            print(f"  Content: {result}")

        return result

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🚀 AgentNodeV3 Output Inspector")
    print("=" * 50)

    actual_output = show_agentnodev3_output()

    if actual_output is not None:
        print("\n🎯 SUMMARY:")
        print(f"AgentNodeV3 returns: {type(actual_output)}")
        if hasattr(actual_output, "update"):
            print(f"Command.update = {actual_output.update}")
        elif isinstance(actual_output, dict):
            print(f"Dict keys = {list(actual_output.keys())}")
    else:
        print("\n💥 Could not determine output")
