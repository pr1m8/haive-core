"""
Graph visualization utilities for Haive graphs.

This module provides enhanced visualization capabilities for Haive graphs,
including Mermaid diagram generation with custom styling.
"""

import os
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from langgraph.graph import StateGraph

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
        subgraph_mode: str = "cluster",
        show_default_branches: bool = False,
    ) -> str:
        """
        Generate a Mermaid diagram for a graph.

        Args:
            graph: Graph object (BaseGraph instance)
            include_subgraphs: Whether to visualize subgraphs as clusters
            highlight_nodes: List of node names to highlight
            highlight_color: Color to use for highlighted nodes
            theme: Mermaid theme name (default, forest, dark, neutral)
            subgraph_mode: How to render subgraphs ("cluster", "inline", or "separate")
            show_default_branches: Whether to show default branches

        Returns:
            Mermaid diagram code as string
        """
        from langgraph.graph import END, START

        # Initialize Mermaid code with directives and styling
        lines = [
            f'%%{{ init: {{ "theme": "{theme}", "flowchart": {{ "curve": "basis" }} }} }}%%',
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
        processed_edges = set()
        subgraph_node_names = set()  # Track which nodes belong to subgraphs

        # Helper function to get safe node ID for Mermaid
        def get_safe_node_id(node_name: str, prefix: str = "") -> str:
            # Replace characters that might cause issues in Mermaid
            safe_id = node_name.replace(" ", "_").replace("-", "_")

            # Check if it starts with a number, prefix with n_ if it does
            if safe_id and safe_id[0].isdigit():
                safe_id = f"n_{safe_id}"

            # Handle reserved Mermaid keywords
            reserved_keywords = [
                "subgraph",
                "end",
                "classDef",
                "class",
                "click",
                "style",
            ]
            if safe_id.lower() in reserved_keywords:
                safe_id = f"node_{safe_id}"

            # Add prefix if provided
            if prefix:
                safe_id = f"{prefix}_{safe_id}"

            return safe_id

        def add_subgraph_cluster(sg_name: str, subgraph: Any) -> None:
            """Add a subgraph as a Mermaid cluster with proper START/END handling."""
            subgraph_id = f"cluster_{get_safe_node_id(sg_name)}"

            lines.append(f'    subgraph {subgraph_id}["{sg_name}"]')
            lines.append(f"        direction TB")

            # Add subgraph START and END nodes first
            sg_start_id = get_safe_node_id(START, f"sg_{sg_name}")
            sg_end_id = get_safe_node_id(END, f"sg_{sg_name}")

            lines.append(f'        {sg_start_id}["START"]:::startNode;')
            lines.append(f'        {sg_end_id}["END"]:::endNode;')
            processed_nodes.add(sg_start_id)
            processed_nodes.add(sg_end_id)

            # Add subgraph nodes (excluding START/END)
            for sub_node_name, sub_node in subgraph.nodes.items():
                if sub_node is None or sub_node_name in (START, END):
                    continue

                # Create unique ID for subgraph nodes
                safe_id = get_safe_node_id(sub_node_name, f"sg_{sg_name}")

                # Get node type and class
                if (
                    hasattr(subgraph, "node_types")
                    and sub_node_name in subgraph.node_types
                ):
                    node_type = subgraph.node_types[sub_node_name]
                    class_name = node_type_classes.get(node_type, "nodeTypeCallable")
                else:
                    node_type = getattr(sub_node, "node_type", NodeType.CALLABLE)
                    class_name = node_type_classes.get(node_type, "nodeTypeCallable")

                # Add node with class
                lines.append(f'        {safe_id}["{sub_node_name}"]:::{class_name};')
                processed_nodes.add(safe_id)

            # Add subgraph edges
            for source, target in subgraph.edges:
                source_id = get_safe_node_id(source, f"sg_{sg_name}")
                target_id = get_safe_node_id(target, f"sg_{sg_name}")

                edge_key = f"{source_id}->{target_id}"
                if edge_key not in processed_edges:
                    lines.append(f"        {source_id} --> {target_id};")
                    processed_edges.add(edge_key)

            # Add subgraph branches
            if hasattr(subgraph, "branches") and subgraph.branches:
                for branch_id, branch in subgraph.branches.items():
                    source = branch.source_node
                    source_id = get_safe_node_id(source, f"sg_{sg_name}")

                    # Draw connections for destinations with dotted lines
                    for condition, target in branch.destinations.items():
                        target_id = get_safe_node_id(target, f"sg_{sg_name}")

                        edge_key = f"{source_id}-->{target_id}"
                        if edge_key not in processed_edges:
                            # Format condition string
                            condition_str = str(condition)
                            if len(condition_str) > 15:
                                condition_str = condition_str[:12] + "..."

                            # Add branch edge with dash style
                            lines.append(
                                f'        {source_id} -.->|"{condition_str}"| {target_id};'
                            )
                            processed_edges.add(edge_key)

                    # Draw default connection if different from destinations and if enabled
                    if (
                        show_default_branches
                        and hasattr(branch, "default")
                        and branch.default
                    ):
                        target_id = get_safe_node_id(branch.default, f"sg_{sg_name}")

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
                                    f'        {source_id} -.->|"default"| {target_id};'
                                )
                                processed_edges.add(edge_key)

            lines.append("    end")

            # Add subgraph style
            lines.append(
                f"    style {subgraph_id} fill:#f0f0f0,stroke:#666,stroke-width:2px,stroke-dasharray:5 5;"
            )

        def add_subgraph_inline(sg_name: str, subgraph: Any) -> None:
            """Add subgraph nodes inline with main graph (no clustering)."""
            # Add subgraph nodes with prefixed IDs
            for sub_node_name, sub_node in subgraph.nodes.items():
                if sub_node is None or sub_node_name in (START, END):
                    continue

                # Create unique ID for subgraph nodes
                safe_id = get_safe_node_id(sub_node_name, f"sg_{sg_name}")

                # Get node type and class
                if (
                    hasattr(subgraph, "node_types")
                    and sub_node_name in subgraph.node_types
                ):
                    node_type = subgraph.node_types[sub_node_name]
                    class_name = node_type_classes.get(node_type, "nodeTypeCallable")
                else:
                    node_type = getattr(sub_node, "node_type", NodeType.CALLABLE)
                    class_name = node_type_classes.get(node_type, "nodeTypeCallable")

                # Add node with class and subgraph indicator
                lines.append(
                    f'    {safe_id}["{sub_node_name} ({sg_name})"]:::{class_name};'
                )
                processed_nodes.add(safe_id)

        # Process subgraphs first if enabled
        if include_subgraphs and hasattr(graph, "subgraphs") and graph.subgraphs:
            lines.append("    %% Subgraphs")

            for sg_name, subgraph in graph.subgraphs.items():
                if subgraph_mode == "cluster":
                    add_subgraph_cluster(sg_name, subgraph)
                elif subgraph_mode == "inline":
                    add_subgraph_inline(sg_name, subgraph)

                # Track subgraph node names for connection handling
                for sub_node_name in subgraph.nodes.keys():
                    if sub_node_name not in (START, END):
                        subgraph_node_names.add(f"{sg_name}.{sub_node_name}")

        # Add main graph nodes (excluding subgraph containers and nodes that exist in subgraphs)
        lines.append("    %% Main Graph Nodes")

        # Collect all node names that exist in subgraphs to avoid duplicates
        subgraph_node_names = set()
        if include_subgraphs and hasattr(graph, "subgraphs"):
            for sg_name, subgraph in graph.subgraphs.items():
                if hasattr(subgraph, "nodes"):
                    for sub_node_name in subgraph.nodes.keys():
                        if sub_node_name not in (START, END):
                            subgraph_node_names.add(sub_node_name)

        for name, node in graph.nodes.items():
            # Skip None nodes
            if node is None:
                continue

            # Skip special nodes (will be added separately)
            if name in (START, END):
                continue

            # Handle subgraph container nodes
            if (
                include_subgraphs
                and hasattr(graph, "subgraphs")
                and name in graph.subgraphs
            ):
                # This is a subgraph container node, add it as a subgraph type
                safe_id = get_safe_node_id(name)
                lines.append(f'    {safe_id}["{name}"]:::nodeTypeSubgraph;')
                processed_nodes.add(safe_id)
                continue

            # Skip nodes that exist in subgraphs to avoid duplicates
            # These nodes will be shown inside the subgraph clusters
            if include_subgraphs and name in subgraph_node_names:
                continue

            # For regular nodes, add them to the main graph
            safe_id = get_safe_node_id(name)

            # Skip if already processed (e.g., as part of a subgraph)
            if safe_id in processed_nodes:
                continue

            # Get node type and class
            if hasattr(graph, "node_types") and name in graph.node_types:
                node_type = graph.node_types[name]
                class_name = node_type_classes.get(node_type, "nodeTypeCallable")
            else:
                node_type = getattr(node, "node_type", NodeType.CALLABLE)
                class_name = node_type_classes.get(node_type, "nodeTypeCallable")

            # Add node with class
            lines.append(f'    {safe_id}["{name}"]:::{class_name};')
            processed_nodes.add(safe_id)

        # Add main graph START and END nodes
        start_id = get_safe_node_id(START)
        end_id = get_safe_node_id(END)

        lines.append(f'    {start_id}["START"]:::startNode;')
        lines.append(f'    {end_id}["END"]:::endNode;')

        # Add main graph direct edges
        lines.append("    %% Main Graph Direct Edges")
        for source, target in graph.edges:
            # Skip edges where source or target are in subgraphs (they'll be handled by bridge connections)
            if include_subgraphs and (
                source in subgraph_node_names or target in subgraph_node_names
            ):
                continue

            source_id = get_safe_node_id(source)
            target_id = get_safe_node_id(target)

            edge_key = f"{source_id}->{target_id}"
            if edge_key not in processed_edges:
                lines.append(f"    {source_id} --> {target_id};")
                processed_edges.add(edge_key)

        # Add main graph branch edges
        if hasattr(graph, "branches") and graph.branches:
            lines.append("    %% Main Graph Branch Connections")

            for branch_id, branch in graph.branches.items():
                source = branch.source_node

                # Skip branches where the source is in a subgraph (they'll be handled within the subgraph)
                if include_subgraphs and source in subgraph_node_names:
                    continue

                source_id = get_safe_node_id(source)

                # Add branch mode comment
                if hasattr(branch, "mode"):
                    lines.append(f"    %% {branch.mode} Branch: {branch.name}")
                else:
                    lines.append(f"    %% Branch: {branch.name}")

                # Draw connections for destinations with dotted lines
                for condition, target in branch.destinations.items():
                    # Handle subgraph destinations - these will be handled by bridge connections
                    if (
                        include_subgraphs
                        and hasattr(graph, "subgraphs")
                        and target in graph.subgraphs
                    ):
                        continue

                    # Skip destinations that exist in subgraphs (they're shown in subgraph clusters)
                    if include_subgraphs and target in subgraph_node_names:
                        continue

                    target_id = get_safe_node_id(target)

                    edge_key = f"{source_id}-->{target_id}"
                    if edge_key not in processed_edges:
                        # Format condition string
                        condition_str = str(condition)
                        if len(condition_str) > 15:
                            condition_str = condition_str[:12] + "..."

                        # Add branch edge with dash style
                        lines.append(
                            f'    {source_id} -.->|"{condition_str}"| {target_id};'
                        )
                        processed_edges.add(edge_key)

                # Draw default connection if different from destinations and if enabled
                if (
                    show_default_branches
                    and hasattr(branch, "default")
                    and branch.default
                ):
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

        # Add bridge connections between main graph and subgraphs
        if include_subgraphs and hasattr(graph, "subgraphs") and graph.subgraphs:
            lines.append("    %% Main Graph to Subgraph Bridge Connections")

            for sg_name, subgraph in graph.subgraphs.items():
                if subgraph_mode == "cluster":
                    # Get subgraph START and END node IDs
                    sg_start_id = get_safe_node_id(START, f"sg_{sg_name}")
                    sg_end_id = get_safe_node_id(END, f"sg_{sg_name}")
                    subgraph_container_id = get_safe_node_id(sg_name)

                    # Bridge: Connect subgraph container to subgraph START
                    # This shows how main graph flow enters the subgraph
                    bridge_entry_key = f"{subgraph_container_id}->{sg_start_id}"
                    if bridge_entry_key not in processed_edges:
                        lines.append(f"    {subgraph_container_id} --> {sg_start_id};")
                        processed_edges.add(bridge_entry_key)

                    # Bridge: Connect subgraph END back to main graph
                    # Find what the subgraph container connects to in main graph
                    for source, target in graph.edges:
                        if source == sg_name:
                            target_id = get_safe_node_id(target)
                            bridge_exit_key = f"{sg_end_id}->{target_id}"
                            if bridge_exit_key not in processed_edges:
                                lines.append(f"    {sg_end_id} --> {target_id};")
                                processed_edges.add(bridge_exit_key)

                    # Handle direct edges TO subgraph nodes: redirect them to subgraph container
                    for source, target in graph.edges:
                        if (
                            target in subgraph_node_names
                            and source not in subgraph_node_names
                        ):
                            source_id = get_safe_node_id(source)
                            bridge_to_subgraph_key = (
                                f"{source_id}->{subgraph_container_id}"
                            )
                            if bridge_to_subgraph_key not in processed_edges:
                                lines.append(
                                    f"    {source_id} --> {subgraph_container_id};"
                                )
                                processed_edges.add(bridge_to_subgraph_key)

                    # Handle direct edges FROM subgraph nodes: redirect them from subgraph END
                    for source, target in graph.edges:
                        if (
                            source in subgraph_node_names
                            and target not in subgraph_node_names
                        ):
                            target_id = get_safe_node_id(target)
                            bridge_from_subgraph_key = f"{sg_end_id}->{target_id}"
                            if bridge_from_subgraph_key not in processed_edges:
                                lines.append(f"    {sg_end_id} --> {target_id};")
                                processed_edges.add(bridge_from_subgraph_key)

                    # Handle branch connections: if main graph branches to subgraph,
                    # connect those branches to subgraph container
                    for branch_id, branch in graph.branches.items():
                        if (
                            branch.source_node not in subgraph_node_names
                        ):  # Only main graph branches
                            for condition, target in branch.destinations.items():
                                if target == sg_name:
                                    source_id = get_safe_node_id(branch.source_node)
                                    bridge_branch_key = f"{source_id}-->{sg_start_id}"
                                    if bridge_branch_key not in processed_edges:
                                        condition_str = str(condition)
                                        if len(condition_str) > 15:
                                            condition_str = condition_str[:12] + "..."
                                        lines.append(
                                            f'    {source_id} -.->|"{condition_str}"| {sg_start_id};'
                                        )
                                        processed_edges.add(bridge_branch_key)
                                elif target in subgraph_node_names:
                                    # Branch to a node that's in a subgraph - redirect to subgraph container
                                    source_id = get_safe_node_id(branch.source_node)
                                    bridge_branch_key = (
                                        f"{source_id}-->{subgraph_container_id}"
                                    )
                                    if bridge_branch_key not in processed_edges:
                                        condition_str = str(condition)
                                        if len(condition_str) > 15:
                                            condition_str = condition_str[:12] + "..."
                                        lines.append(
                                            f'    {source_id} -.->|"{condition_str}"| {subgraph_container_id};'
                                        )
                                        processed_edges.add(bridge_branch_key)
                        else:
                            # Handle branches FROM subgraph nodes TO main graph nodes
                            # These need to be bridge connections from subgraph END
                            for condition, target in branch.destinations.items():
                                # Only handle targets that are NOT in subgraphs (main graph nodes)
                                if (
                                    target not in subgraph_node_names
                                    and target not in (START, END)
                                ):
                                    target_id = get_safe_node_id(target)
                                    bridge_branch_key = f"{sg_end_id}-->{target_id}"
                                    if bridge_branch_key not in processed_edges:
                                        condition_str = str(condition)
                                        if len(condition_str) > 15:
                                            condition_str = condition_str[:12] + "..."
                                        lines.append(
                                            f'    {sg_end_id} -.->|"{condition_str}"| {target_id};'
                                        )
                                        processed_edges.add(bridge_branch_key)
                                elif target == END:
                                    # Special case: branch from subgraph to main graph END
                                    end_id = get_safe_node_id(END)
                                    bridge_branch_key = f"{sg_end_id}-->{end_id}"
                                    if bridge_branch_key not in processed_edges:
                                        condition_str = str(condition)
                                        if len(condition_str) > 15:
                                            condition_str = condition_str[:12] + "..."
                                        lines.append(
                                            f'    {sg_end_id} -.->|"{condition_str}"| {end_id};'
                                        )
                                        processed_edges.add(bridge_branch_key)

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
        subgraph_mode: str = "cluster",
        show_default_branches: bool = False,
    ) -> str:
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
            subgraph_mode: How to render subgraphs ("cluster", "inline", or "separate")
            show_default_branches: Whether to show default branches
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
            subgraph_mode=subgraph_mode,
            show_default_branches=show_default_branches,
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
