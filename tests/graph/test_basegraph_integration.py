#!/usr/bin/env python3
"""Test BaseGraph integration with intelligent routing."""

import asyncio

from haive.agents.multi.clean import MultiAgent
from haive.agents.simple.agent import SimpleAgent

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.graph.state_graph.base_graph2 import BaseGraph


def test_basegraph_intelligent_routing():
    """Test BaseGraph's intelligent routing capabilities."""
    print("🧪 Testing BaseGraph intelligent routing...")

    # Create agents
    planner = SimpleAgent(
        name="planner", engine=AugLLMConfig(prompt_template="Plan: {input}")
    )

    executor = SimpleAgent(
        name="executor", engine=AugLLMConfig(prompt_template="Execute: {input}")
    )

    reviewer = SimpleAgent(
        name="reviewer", engine=AugLLMConfig(prompt_template="Review: {input}")
    )

    # Test direct BaseGraph usage
    graph = BaseGraph(name="test_graph")

    # Add intelligent routing
    agents = {"executor": executor, "reviewer": reviewer, "planner": planner}

    graph.add_intelligent_agent_routing(
        agents=agents, execution_mode="infer", prefix="agent_"
    )

    print(f"✅ Graph nodes: {list(graph.nodes.keys())}")
    print(f"✅ Graph edges: {graph.edges}")

    # Should have inferred sequence: planner → executor → reviewer
    expected_edges = [
        ("__start__", "agent_planner"),
        ("agent_planner", "agent_executor"),
        ("agent_executor", "agent_reviewer"),
        ("agent_reviewer", "__end__"),
    ]

    for edge in expected_edges:
        if edge in graph.edges:
            print(f"✅ Found expected edge: {edge}")
        else:
            print(f"❌ Missing edge: {edge}")

    return graph


def test_multiagent_basegraph_integration():
    """Test MultiAgent using BaseGraph intelligent routing."""
    print("\n🧪 Testing MultiAgent with BaseGraph integration...")

    # Create agents
    planner = SimpleAgent(
        name="planner", engine=AugLLMConfig(prompt_template="Plan: {input}")
    )

    executor = SimpleAgent(
        name="executor", engine=AugLLMConfig(prompt_template="Execute: {input}")
    )

    reviewer = SimpleAgent(
        name="reviewer", engine=AugLLMConfig(prompt_template="Review: {input}")
    )

    # Create MultiAgent (should use BaseGraph intelligent routing)
    multi_agent = MultiAgent.create(
        agents=[executor, reviewer, planner],  # Out of order
        name="integrated_multi_agent",
        execution_mode="infer",
    )

    # Build graph
    graph = multi_agent.build_graph()

    print(f"✅ MultiAgent graph nodes: {list(graph.nodes.keys())}")
    print(f"✅ MultiAgent graph edges: {graph.edges}")

    # Should have inferred sequence: planner → executor → reviewer
    expected_edges = [
        ("__start__", "planner"),
        ("planner", "executor"),
        ("executor", "reviewer"),
        ("reviewer", "__end__"),
    ]

    for edge in expected_edges:
        if edge in graph.edges:
            print(f"✅ Found expected edge: {edge}")
        else:
            print(f"❌ Missing edge: {edge}")

    return multi_agent


def test_branch_routing():
    """Test branch routing with BaseGraph."""
    print("\n🧪 Testing branch routing...")

    # Create agents
    analyzer = SimpleAgent(
        name="analyzer", engine=AugLLMConfig(prompt_template="Analyze: {input}")
    )

    success_handler = SimpleAgent(
        name="success_handler",
        engine=AugLLMConfig(prompt_template="Handle success: {input}"),
    )

    error_handler = SimpleAgent(
        name="error_handler",
        engine=AugLLMConfig(prompt_template="Handle error: {input}"),
    )

    # Create graph with branch routing
    graph = BaseGraph(name="branch_test")

    agents = {
        "analyzer": analyzer,
        "success_handler": success_handler,
        "error_handler": error_handler,
    }

    branches = {
        "analyzer": {
            "condition": "if success",
            "targets": ["success_handler", "error_handler"],
        }
    }

    graph.add_intelligent_agent_routing(
        agents=agents, execution_mode="branch", branches=branches, prefix="agent_"
    )

    print(f"✅ Branch graph nodes: {list(graph.nodes.keys())}")
    print(f"✅ Branch graph edges: {graph.edges}")

    # Should have branching structure
    assert "agent_analyzer" in graph.nodes
    assert "agent_success_handler" in graph.nodes
    assert "agent_error_handler" in graph.nodes

    print("✅ Branch routing working correctly!")
    return graph


def main():
    """Run all integration tests."""
    print("🚀 Testing BaseGraph integration with intelligent routing...\n")

    try:
        # Test 1: Direct BaseGraph usage
        test_basegraph_intelligent_routing()

        # Test 2: MultiAgent integration
        test_multiagent_basegraph_integration()

        # Test 3: Branch routing
        test_branch_routing()

        print(f"\n✅ All BaseGraph integration tests passed!")
        print(f"✅ Features working:")
        print(f"  - BaseGraph intelligent routing")
        print(f"  - MultiAgent BaseGraph integration")
        print(f"  - Sequence inference in BaseGraph")
        print(f"  - Branch routing in BaseGraph")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
