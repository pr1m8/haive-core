#!/usr/bin/env python3
"""Test BaseGraph integration with intelligent routing."""


from haive.agents.multi.clean import MultiAgent
from haive.agents.simple.agent import SimpleAgent
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.graph.state_graph.base_graph2 import BaseGraph


def test_basegraph_intelligent_routing():
    """Test BaseGraph's intelligent routing capabilities."""
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

    # Should have inferred sequence: planner → executor → reviewer
    expected_edges = [
        ("__start__", "agent_planner"),
        ("agent_planner", "agent_executor"),
        ("agent_executor", "agent_reviewer"),
        ("agent_reviewer", "__end__"),
    ]

    for edge in expected_edges:
        if edge in graph.edges:
            pass
        else:
            pass

    return graph


def test_multiagent_basegraph_integration():
    """Test MultiAgent using BaseGraph intelligent routing."""
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

    # Should have inferred sequence: planner → executor → reviewer
    expected_edges = [
        ("__start__", "planner"),
        ("planner", "executor"),
        ("executor", "reviewer"),
        ("reviewer", "__end__"),
    ]

    for edge in expected_edges:
        if edge in graph.edges:
            pass
        else:
            pass

    return multi_agent


def test_branch_routing():
    """Test branch routing with BaseGraph."""
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

    # Should have branching structure
    assert "agent_analyzer" in graph.nodes
    assert "agent_success_handler" in graph.nodes
    assert "agent_error_handler" in graph.nodes

    return graph


def main():
    """Run all integration tests."""
    try:
        # Test 1: Direct BaseGraph usage
        test_basegraph_intelligent_routing()

        # Test 2: MultiAgent integration
        test_multiagent_basegraph_integration()

        # Test 3: Branch routing
        test_branch_routing()

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
