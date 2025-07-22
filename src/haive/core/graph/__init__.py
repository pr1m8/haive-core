"""Graph system for building AI workflows in the Haive framework.

This module provides a powerful, flexible system for creating graph-based workflows
that orchestrate AI agents, tools, and data processing pipelines. The graph system
is built on top of LangGraph and extends it with Haive-specific features for
agent coordination, dynamic state management, and advanced workflow patterns.

The graph system enables complex AI workflows through composable nodes, conditional
routing, parallel processing, and state persistence. It's designed to handle
everything from simple linear workflows to complex multi-agent orchestration.

Key Components:
    BaseGraph: Foundation class for all graph implementations
        - State schema management and validation
        - Node registration and execution
        - Edge definition and routing logic
        - Built-in persistence and checkpointing
        - Visual graph representation and debugging

    Graph Builder Components:
        - Dynamic graph construction from configuration
        - Pattern-based graph templates
        - Node factory system for component creation
        - Advanced routing and branching logic

    State Management:
        - Schema composition and validation
        - State sharing between parent and child graphs
        - Reducer functions for intelligent state merging
        - Field-level access control and visibility

Features:
    - Dynamic graph construction and modification
    - Schema-aware state management
    - Parallel and conditional execution
    - Built-in persistence and checkpointing
    - Visual graph debugging and analysis
    - Pattern-based workflow templates
    - Tool and agent integration
    - Error handling and recovery
    - Performance monitoring and optimization

Examples:
    Basic graph creation::

        from haive.core.graph import BaseGraph
        from haive.core.schema import StateSchema

        class MyWorkflowState(StateSchema):
            query: str = ""
            results: List[str] = []

        graph = BaseGraph(state_schema=MyWorkflowState)

        # Add nodes
        graph.add_node("process", processing_function)
        graph.add_node("validate", validation_function)

        # Define flow
        graph.set_entry_point("process")
        graph.add_edge("process", "validate")
        graph.set_finish_point("validate")

        # Compile and run
        compiled_graph = graph.compile()
        result = compiled_graph.invoke({"query": "What is AI?"})

    Agent integration::

        from haive.core.graph import BaseGraph
        from haive.agents.simple import SimpleAgent

        # Create agents
        research_agent = SimpleAgent(name="researcher")
        writer_agent = SimpleAgent(name="writer")

        # Build workflow
        graph = BaseGraph()
        graph.add_agent_node("research", research_agent)
        graph.add_agent_node("write", writer_agent)

        # Sequential workflow
        graph.set_entry_point("research")
        graph.add_edge("research", "write")
        graph.set_finish_point("write")

    Conditional routing::

        def route_logic(state):
            if state["requires_verification"]:
                return "verify"
            return "finalize"

        graph.add_conditional_edges(
            source="process",
            path=route_logic,
            path_map={"verify": "verification", "finalize": "finalization"}
        )

See Also:
    - Node system: haive.core.graph.node
    - State management: haive.core.schema
    - Agent integration: haive.agents
    - Workflow patterns: haive.core.graph.patterns
"""

# Import current graph implementation
from haive.core.graph.state_graph.base_graph2 import BaseGraph

__all__ = [
    "BaseGraph",
]
