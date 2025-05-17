# haive/core/graph/graph.py

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import ConfigDict, Field
from rich.console import Console

from haive.core.graph.common.types import ConfigLike, NodeLike, StateLike
from haive.core.graph.state_graph.base_graph2 import BaseGraph
from haive.core.graph.state_graph.schema_mixin import GraphSchemaMixin

logger = logging.getLogger(__name__)
console = Console()


class SchemaGraph(BaseGraph, GraphSchemaMixin[StateLike, Optional[ConfigLike]]):
    """
    Graph implementation with schema management capabilities.

    SchemaGraph extends BaseGraph with state schema management, enabling
    seamless integration with LangGraph and providing type validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_langgraph(self) -> StateGraph:
        """
        Convert to a LangGraph StateGraph with schema information.

        Returns:
            LangGraph StateGraph instance
        """
        # Create the StateGraph with schema information
        graph = StateGraph(
            self.state_schema,
            input=self.input_schema,
            output=self.output_schema,
            config=self.config_schema,
        )

        # Add all nodes
        for name, node in self.nodes.items():
            # Handle different node types
            if hasattr(node, "__call__"):
                # Callable nodes can be added directly
                graph.add_node(name, node)

                # Add retry policy if present
                if name in self.node_retry_policies:
                    graph.set_retry_policy(name, self.node_retry_policies[name])
            elif hasattr(node, "create_node_function"):
                # Use create_node_function if available
                node_func = node.create_node_function()
                graph.add_node(name, node_func)

                # Add retry policy if present
                if name in self.node_retry_policies:
                    graph.set_retry_policy(name, self.node_retry_policies[name])
            elif isinstance(node, BaseGraph) and hasattr(node, "to_langgraph"):
                # Convert subgraph to LangGraph
                subgraph = node.to_langgraph()
                graph.add_node(name, subgraph)

                # Retry policies don't apply to subgraphs directly
            else:
                # Unsupported node type - try to extract callable
                node_func = None

                # Try common attributes for callable
                for attr in ["__call__", "invoke", "run"]:
                    if hasattr(node, attr) and callable(getattr(node, attr)):
                        node_func = getattr(node, attr)
                        break

                if node_func:
                    graph.add_node(name, node_func)

                    # Add retry policy if present
                    if name in self.node_retry_policies:
                        graph.set_retry_policy(name, self.node_retry_policies[name])
                else:
                    logger.warning(f"Unsupported node type for {name}: {type(node)}")

        # Add all edges
        for source, target in self.edges:
            graph.add_edge(source, target)

        # Add all conditional edges
        for edge_id, edge in self.conditional_edges.items():
            source = edge["source"]

            # Use the branch object directly if available
            if edge.get("branch") and hasattr(
                edge["branch"], "create_langgraph_branch"
            ):
                branch = edge["branch"].create_langgraph_branch()
                graph.add_conditional_edges(
                    source, branch, edge["destinations"], default=edge.get("default")
                )
            else:
                # Use the condition function
                condition = edge["condition"]
                graph.add_conditional_edges(
                    source, condition, edge["destinations"], default=edge.get("default")
                )

        # Compile if needed
        if self.compiled:
            return graph.compile()

        return graph

    def compile(self) -> StateGraph:
        """
        Validate and compile the graph to a runnable StateGraph.

        Returns:
            Compiled LangGraph StateGraph
        """
        # Mark as compiled in BaseGraph
        super().compile()

        # Convert to LangGraph and compile
        return self.to_langgraph().compile()

    def display(self):
        """
        Display a visual representation of the graph structure.

        Outputs information about nodes, edges, and branches,
        and generates a Mermaid diagram for visualization.
        """
        print("\n" + "=" * 50)
        print(f"GRAPH: {self.name}")
        print("=" * 50)

        # Display basic graph info
        print(f"\nNodes ({len(self.nodes)}):")
        for name, node in self.nodes.items():
            node_type = self.node_types.get(name, "unknown")
            print(f"  - {name} ({node_type})")

        print(f"\nEdges ({len(self.edges)}):")
        for src, dst in self.edges:
            print(f"  - {src} → {dst}")

        print(f"\nBranches ({len(self.branches)}):")
        for branch_id, branch in self.branches.items():
            print(f"  - {branch.name} (from {branch.source_node}):")
            for cond, dest in branch.destinations.items():
                print(f"    - {cond} → {dest}")
            if branch.default:
                print(f"    - default → {branch.default}")

        # Generate and print Mermaid diagram
        print("\nMermaid Diagram:")
        print("```mermaid")
        print(self.to_mermaid())
        print("```")

        # Display validation issues if any
        issues = self.check_graph_validity()
        if issues:
            print("\nWARNING: Graph Validation Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nGraph validation: OK")

        print("\n" + "=" * 50 + "\n")

        return self
