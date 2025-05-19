"""
Graph visualization utilities for Haive graphs.

This module provides enhanced visualization capabilities for Haive graphs,
including Mermaid diagram generation with custom styling.
"""

import os
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from haive.core.graph.common.types import NodeType
from haive.core.utils.mermaid_utils import (
    Environment,
    detect_environment,
    display_mermaid,
    mermaid_to_png,
)


class GraphVisualizer:
    """
    Visualization utilities for Haive graph structures.

    This class provides methods to generate and display visual representations
    of graph structures using Mermaid diagrams.
    """

    # Color scheme for different node types
    NODE_COLORS = {
        NodeType.ENGINE: "#90EE90",  # Light green
        NodeType.TOOL: "#FFD700",  # Gold
        NodeType.VALIDATION: "#B0E0E6",  # Light blue
        NodeType.SUBGRAPH: "#FFA07A",  # Light salmon
        NodeType.CALLABLE: "#F5F5DC",  # Beige
        NodeType.CUSTOM: "#DDA0DD",  # Plum
        # Add more colors if more node types are added to the NodeType enum
    }

    # Special node colors
    START_COLOR = "#5D8AA8"  # Blue
    END_COLOR = "#FF6347"  # Tomato red

    # Edge styles
    DIRECT_EDGE_STYLE = "stroke:#333,stroke-width:2px;"
    BRANCH_EDGE_STYLE = "stroke:#333,stroke-width:1.5px,stroke-dasharray:5 5;"

    @classmethod
    def generate_mermaid(
        cls,
        graph: Any,
        include_subgraphs: bool = True,
        highlight_nodes: Optional[List[str]] = None,
        highlight_color: str = "#FF69B4",
        theme: str = "default",
    ) -> str:
        """
        Generate a Mermaid diagram for a graph.

        Args:
            graph: Graph object (BaseGraph instance)
            include_subgraphs: Whether to visualize subgraphs as clusters
            highlight_nodes: List of node names to highlight
            highlight_color: Color to use for highlighted nodes
            theme: Mermaid theme name (default, forest, dark, neutral)

        Returns:
            Mermaid diagram code as string
        """
        from langgraph.graph import END, START

        # Initialize Mermaid code with directives and styling
        lines = [
            f"%%{{ init: {{ 'theme': '{theme}', 'flowchart': {{ 'curve': 'basis' }} }} }}%%",
            "flowchart TD;",
            "    %% Node styling classes",
        ]

        # Define node type classes
        node_type_classes = {}
        for node_type, color in cls.NODE_COLORS.items():
            class_name = f"nodeType{node_type.value.capitalize()}"
            node_type_classes[node_type] = class_name
            lines.append(
                f"    classDef {class_name} fill:{color},stroke:#333,stroke-width:1px;"
            )

        # Add special node classes
        lines.append(
            f"    classDef startNode fill:{cls.START_COLOR},color:white,font-weight:bold;"
        )
        lines.append(
            f"    classDef endNode fill:{cls.END_COLOR},color:white,font-weight:bold;"
        )
        lines.append(
            f"    classDef highlightNode fill:{highlight_color},stroke:#fff,stroke-width:2px,color:white;"
        )

        # Set for tracking processed node IDs to avoid duplicates
        processed_nodes = set()

        # Helper function to get safe node ID for Mermaid
        def get_safe_node_id(node_name: str) -> str:
            # Replace characters that might cause issues in Mermaid
            safe_id = node_name.replace(" ", "_").replace("-", "_")

            # Check if it starts with a number, prefix with n_ if it does
            if safe_id and safe_id[0].isdigit():
                safe_id = f"n_{safe_id}"

            return safe_id

        # Add subgraphs first if enabled
        if include_subgraphs and hasattr(graph, "subgraphs"):
            subgraph_nodes = set()
            for sg_name, subgraph in graph.subgraphs.items():
                # Create a subgraph cluster with its own ID
                subgraph_id = f"cluster_{get_safe_node_id(sg_name)}"

                lines.append(f"    subgraph {subgraph_id}[{sg_name}]")

                # Add nodes in the subgraph
                for sub_node_name, sub_node in subgraph.nodes.items():
                    if sub_node is None:
                        continue

                    # Skip special nodes
                    if sub_node_name in (START, END):
                        continue

                    safe_id = f"{get_safe_node_id(sub_node_name)}"
                    subgraph_nodes.add(safe_id)

                    # Get node type and class
                    if (
                        hasattr(subgraph, "node_types")
                        and sub_node_name in subgraph.node_types
                    ):
                        node_type = subgraph.node_types[sub_node_name]
                        class_name = node_type_classes.get(node_type, "defaultNode")
                    else:
                        node_type = getattr(sub_node, "node_type", NodeType.CALLABLE)
                        class_name = node_type_classes.get(node_type, "defaultNode")

                    # Add node with class
                    lines.append(
                        f'        {safe_id}["{sub_node_name} ({node_type.value})"]:::{class_name};'
                    )
                    processed_nodes.add(safe_id)

                lines.append("    end")

                # Add subgraph style
                lines.append(
                    f"    style {subgraph_id} fill:#f0f0f0,stroke:#666,stroke-width:2px,stroke-dasharray:5 5;"
                )

        # Add nodes
        lines.append("    %% Nodes")
        for name, node in graph.nodes.items():
            # Skip None nodes and already processed nodes
            if node is None or get_safe_node_id(name) in processed_nodes:
                continue

            # Skip special nodes
            if name in (START, END):
                continue

            safe_id = get_safe_node_id(name)

            # Get node type and class
            if hasattr(graph, "node_types") and name in graph.node_types:
                node_type = graph.node_types[name]
                class_name = node_type_classes.get(node_type, "defaultNode")
            else:
                node_type = getattr(node, "node_type", NodeType.CALLABLE)
                class_name = node_type_classes.get(node_type, "defaultNode")

            # Add node with class
            lines.append(f'    {safe_id}["{name} ({node_type.value})"]:::{class_name};')
            processed_nodes.add(safe_id)

        # Add special nodes with safe class names
        start_id = get_safe_node_id(START)
        end_id = get_safe_node_id(END)

        # Add START and END nodes with the correct style classes
        lines.append(f'    {start_id}["{START}"]:::startNode;')
        lines.append(f'    {end_id}["{END}"]:::endNode;')

        # Track processed edges to avoid duplicates
        processed_edges = set()

        # Add direct edges
        lines.append("    %% Direct edges")
        for source, target in graph.edges:
            source_id = get_safe_node_id(source)
            target_id = get_safe_node_id(target)

            edge_key = f"{source_id}->{target_id}"
            if edge_key in processed_edges:
                continue

            lines.append(f"    {source_id} --> {target_id};")
            processed_edges.add(edge_key)

        # Add branch edges
        if hasattr(graph, "branches") and graph.branches:
            lines.append("    %% Branch connections")

            for branch_id, branch in graph.branches.items():
                source = branch.source_node
                source_id = get_safe_node_id(source)

                # Add branch mode comment
                if hasattr(branch, "mode"):
                    lines.append(f"    %% {branch.mode} Branch: {branch.name}")
                else:
                    lines.append(f"    %% Branch: {branch.name}")

                # Draw connections for destinations with dotted lines
                for condition, target in branch.destinations.items():
                    target_id = get_safe_node_id(target)

                    edge_key = f"{source_id}-->{target_id}"
                    if edge_key in processed_edges:
                        continue

                    # Format condition string
                    condition_str = str(condition)
                    if len(condition_str) > 15:
                        condition_str = condition_str[:12] + "..."

                    # Add branch edge with dash style
                    lines.append(
                        f'    {source_id} -.->|"{condition_str}"| {target_id};'
                    )
                    processed_edges.add(edge_key)

                # Draw default connection if different from destinations
                if hasattr(branch, "default") and branch.default:
                    target_id = get_safe_node_id(branch.default)

                    # Skip if this is already connected by a specific condition
                    is_duplicate = False
                    for dest in branch.destinations.values():
                        if dest == branch.default:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        edge_key = f"{source_id}-->{target_id}"
                        if edge_key not in processed_edges:
                            lines.append(
                                f'    {source_id} -.->|"default"| {target_id};'
                            )
                            processed_edges.add(edge_key)

        # Apply highlights if any
        if highlight_nodes:
            highlight_list = []
            for node in highlight_nodes:
                safe_id = get_safe_node_id(node)
                if safe_id in processed_nodes or safe_id in [start_id, end_id]:
                    highlight_list.append(safe_id)

            if highlight_list:
                lines.append("    %% Highlight specified nodes")
                lines.append(f"    class {','.join(highlight_list)} highlightNode;")

        return "\n".join(lines)

    @classmethod
    def display_graph(
        cls,
        graph: Any,
        output_path: Optional[str] = None,
        include_subgraphs: bool = True,
        highlight_nodes: Optional[List[str]] = None,
        highlight_paths: Optional[List[List[str]]] = None,
        save_png: bool = False,
        width: str = "100%",
        theme: str = "default",
    ) -> None:
        """
        Generate and display a Mermaid diagram for a graph.

        Args:
            graph: Graph object (BaseGraph instance)
            output_path: Optional path to save the diagram
            include_subgraphs: Whether to visualize subgraphs as clusters
            highlight_nodes: List of node names to highlight
            highlight_paths: List of paths to highlight (each path is a list of node names)
            save_png: Whether to save the diagram as PNG
            width: Width of the displayed diagram
            theme: Mermaid theme to use
        """
        # Combine nodes from highlight_paths with highlight_nodes
        all_highlight_nodes = highlight_nodes or []

        if highlight_paths:
            for path in highlight_paths:
                all_highlight_nodes.extend(path)

        # Generate the Mermaid code
        mermaid_code = cls.generate_mermaid(
            graph,
            include_subgraphs=include_subgraphs,
            highlight_nodes=all_highlight_nodes if all_highlight_nodes else None,
            theme=theme,
        )

        # Generate a default output path if saving but no path provided
        if save_png and not output_path:
            graph_name = getattr(graph, "name", f"graph_{uuid.uuid4().hex[:8]}")
            output_dir = os.path.join(os.getcwd(), "graph_images")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{graph_name}.png")

        # Display the diagram
        display_mermaid(
            mermaid_code, output_path=output_path, save_png=save_png, width=width
        )

        return mermaid_code
