"""
Enhanced graph visualization utilities for Haive graphs with proper subgraph support.

This module provides improved visualization with embedded subgraphs (x-ray view),
safer professional colors, and better branch handling.
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
    Enhanced visualization utilities with proper subgraph embedding support.

    Provides elegant, accessible graph visualizations with embedded subgraphs
    that show internal structure (x-ray view) and safe, professional colors.
    """

    # Safe, accessible color scheme (WCAG AA compliant)
    NODE_COLORS = {
        NodeType.ENGINE: "#2563EB",  # Safe blue
        NodeType.TOOL: "#DC2626",  # Safe red
        NodeType.VALIDATION: "#059669",  # Safe green
        NodeType.SUBGRAPH: "#7C3AED",  # Safe purple
        NodeType.CALLABLE: "#0891B2",  # Safe cyan
        NodeType.CUSTOM: "#BE185D",  # Safe magenta
    }

    # Safe special node colors
    START_COLOR = "#065F46"  # Dark green
    END_COLOR = "#991B1B"  # Dark red
    HIGHLIGHT_COLOR = "#B45309"  # Safe orange

    # Professional styling
    BACKGROUND_COLOR = "#F9FAFB"
    EDGE_COLOR = "#374151"
    BORDER_COLOR = "#D1D5DB"
    SUBGRAPH_BG = "#F3F4F6"
    SUBGRAPH_BORDER = "#9CA3AF"

    # Edge styles
    DIRECT_EDGE_STYLE = f"stroke:{EDGE_COLOR},stroke-width:2px;"
    BRANCH_EDGE_STYLE = f"stroke:#7C3AED,stroke-width:2px,stroke-dasharray:8 4;"
    SUBGRAPH_STYLE = f"fill:{SUBGRAPH_BG},stroke:{SUBGRAPH_BORDER},stroke-width:2px,stroke-dasharray:3 3;"

    @classmethod
    def _sanitize_node_name(cls, name: str) -> str:
        """Create a clean, readable node name for display."""
        if name in ["__start__", "START"]:
            return "START"
        if name in ["__end__", "END"]:
            return "END"

        # Clean up common patterns
        clean_name = name.replace("_", " ").strip()
        clean_name = clean_name.title()

        # Handle common agent patterns
        replacements = {
            "Agent Node": "Agent",
            "Tool Node": "Tools",
            "Parse Output": "Parser",
            "Validation Node": "Validator",
            "Engine Node": "Engine",
        }

        for old, new in replacements.items():
            clean_name = clean_name.replace(old, new)

        return clean_name

    @classmethod
    def _get_safe_node_id(cls, node_name: str, prefix: str = "") -> str:
        """Create a safe, unique node ID for Mermaid."""
        safe_id = str(node_name)

        # Replace problematic characters
        replacements = {" ": "_", "-": "_", ".": "_", "/": "_", "\\": "_", ":": "_"}
        for old, new in replacements.items():
            safe_id = safe_id.replace(old, new)

        # Handle special cases
        if safe_id in ["__start__", "START"]:
            safe_id = "START"
        elif safe_id in ["__end__", "END"]:
            safe_id = "END"

        # Ensure valid identifier
        if safe_id and safe_id[0].isdigit():
            safe_id = f"n_{safe_id}"

        # Handle reserved keywords
        reserved = [
            "subgraph",
            "end",
            "classDef",
            "class",
            "click",
            "style",
            "graph",
            "flowchart",
        ]
        if safe_id.lower() in reserved:
            safe_id = f"node_{safe_id}"

        # Add prefix for uniqueness
        if prefix:
            safe_id = f"{prefix}_{safe_id}"

        return safe_id

    @classmethod
    def _format_condition_label(cls, condition: str) -> str:
        """Format condition labels for better readability."""
        condition_str = str(condition)

        # Handle boolean values
        if condition_str.lower() == "true":
            return "✓ Yes"
        elif condition_str.lower() == "false":
            return "✗ No"

        # Handle common patterns
        replacements = {
            "has_errors": "❌ Has Errors",
            "no_errors": "✅ No Errors",
            "tool_node": "🔧 Use Tools",
            "parse_output": "📝 Parse Output",
            "no_tools": "➡️ Continue",
            "validation_passed": "✅ Valid",
            "validation_failed": "❌ Invalid",
            "default": "📍 Default",
        }

        if condition_str in replacements:
            return replacements[condition_str]

        # Truncate long conditions
        if len(condition_str) > 15:
            return condition_str[:12] + "..."

        return condition_str.replace("_", " ").title()

    @classmethod
    def generate_mermaid(
        cls,
        graph: Any,
        include_subgraphs: bool = True,
        highlight_nodes: Optional[List[str]] = None,
        highlight_color: str = None,
        theme: str = "base",
        subgraph_mode: str = "cluster",
        show_default_branches: bool = True,
        direction: str = "TD",
        compact_mode: bool = False,
    ) -> str:
        """
        Generate an elegant Mermaid diagram with proper subgraph embedding.

        Args:
            graph: Graph object (BaseGraph instance)
            include_subgraphs: Whether to visualize subgraphs as embedded clusters
            highlight_nodes: List of node names to highlight
            highlight_color: Color to use for highlighted nodes
            theme: Mermaid theme name
            subgraph_mode: How to render subgraphs ("cluster" shows embedded view)
            show_default_branches: Whether to show default branches
            direction: Graph direction (TD, TB, LR, RL)
            compact_mode: Whether to use compact styling

        Returns:
            Mermaid diagram code with embedded subgraphs
        """
        from langgraph.graph import END, START

        highlight_color = highlight_color or cls.HIGHLIGHT_COLOR

        # Initialize with clean styling
        lines = [
            f'%%{{ init: {{ "theme": "{theme}", "flowchart": {{ "curve": "basis", "padding": 15, "nodeSpacing": 50, "rankSpacing": 80 }} }} }}%%',
            f"flowchart {direction};",
            "    %% === SAFE PROFESSIONAL STYLING ===",
        ]

        # Add accessible node type classes
        node_type_classes = {}
        for node_type, color in cls.NODE_COLORS.items():
            class_name = f"nodeType{node_type.value.capitalize()}"
            node_type_classes[node_type] = class_name
            lines.append(
                f"    classDef {class_name} fill:{color},stroke:white,stroke-width:2px,color:white,font-weight:500,font-size:13px,font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            )

        # Add special node classes
        lines.append(
            f"    classDef startNode fill:{cls.START_COLOR},stroke:white,stroke-width:2px,color:white,font-weight:bold,font-size:14px;"
        )
        lines.append(
            f"    classDef endNode fill:{cls.END_COLOR},stroke:white,stroke-width:2px,color:white,font-weight:bold,font-size:14px;"
        )
        lines.append(
            f"    classDef highlightNode fill:{highlight_color},stroke:white,stroke-width:3px,color:white,font-weight:bold;"
        )
        lines.append(
            f"    classDef subgraphNode fill:{cls.SUBGRAPH_BG},stroke:{cls.SUBGRAPH_BORDER},stroke-width:2px,color:{cls.EDGE_COLOR};"
        )

        # Track processed items
        processed_nodes = set()
        processed_edges = set()
        subgraph_containers = set()

        # First, handle embedded subgraphs if enabled
        if include_subgraphs and hasattr(graph, "subgraphs") and graph.subgraphs:
            lines.append("    %% === EMBEDDED SUBGRAPHS (X-RAY VIEW) ===")

            for sg_name, subgraph in graph.subgraphs.items():
                if subgraph_mode == "cluster":
                    cls._add_embedded_subgraph(
                        lines,
                        sg_name,
                        subgraph,
                        node_type_classes,
                        processed_nodes,
                        processed_edges,
                    )
                    subgraph_containers.add(sg_name)

        lines.append("    %% === MAIN GRAPH NODES ===")

        # Add START and END nodes
        start_id = cls._get_safe_node_id(START)
        end_id = cls._get_safe_node_id(END)

        lines.append(
            f'    {start_id}["{cls._sanitize_node_name("START")}"]:::startNode;'
        )
        lines.append(f'    {end_id}["{cls._sanitize_node_name("END")}"]:::endNode;')
        processed_nodes.update([start_id, end_id])

        # Add main graph nodes (excluding subgraph containers)
        for name, node in graph.nodes.items():
            if node is None or name in (START, END):
                continue

            # Skip subgraph container nodes - they're handled as embedded subgraphs
            if name in subgraph_containers:
                continue

            safe_id = cls._get_safe_node_id(name)
            if safe_id in processed_nodes:
                continue

            display_name = cls._sanitize_node_name(name)

            # Get node type and styling
            if hasattr(graph, "node_types") and name in graph.node_types:
                node_type = graph.node_types[name]
                class_name = node_type_classes.get(node_type, "nodeTypeEngine")
            else:
                class_name = node_type_classes.get(NodeType.ENGINE, "nodeTypeEngine")

            lines.append(f'    {safe_id}["{display_name}"]:::{class_name};')
            processed_nodes.add(safe_id)

        lines.append("    %% === MAIN GRAPH EDGES ===")

        # Add direct edges (excluding subgraph internal edges)
        for source, target in graph.edges:
            # Skip edges that connect to subgraph containers - handle separately
            if source in subgraph_containers or target in subgraph_containers:
                continue

            source_id = cls._get_safe_node_id(source)
            target_id = cls._get_safe_node_id(target)

            edge_key = f"{source_id}->{target_id}"
            if edge_key not in processed_edges:
                lines.append(f"    {source_id} --> {target_id};")
                processed_edges.add(edge_key)

        # Handle subgraph bridge connections
        if include_subgraphs and subgraph_containers:
            lines.append("    %% === SUBGRAPH BRIDGE CONNECTIONS ===")
            cls._add_subgraph_bridges(
                lines, graph, subgraph_containers, processed_edges
            )

        # Add branch edges with better styling
        if hasattr(graph, "branches") and graph.branches:
            lines.append("    %% === CONDITIONAL BRANCHES ===")
            cls._add_branch_edges(lines, graph, subgraph_containers, processed_edges)

        # Apply highlights
        if highlight_nodes:
            lines.append("    %% === HIGHLIGHTS ===")
            highlight_list = []
            for node in highlight_nodes:
                safe_id = cls._get_safe_node_id(node)
                if safe_id in processed_nodes:
                    highlight_list.append(safe_id)

            if highlight_list:
                lines.append(f"    class {','.join(highlight_list)} highlightNode;")

        return "\n".join(lines)

    @classmethod
    def _add_embedded_subgraph(
        cls,
        lines: List[str],
        sg_name: str,
        subgraph: Any,
        node_type_classes: Dict,
        processed_nodes: Set,
        processed_edges: Set,
    ):
        """Add an embedded subgraph cluster with x-ray view of internal structure."""
        from langgraph.graph import END, START

        subgraph_id = f"cluster_{cls._get_safe_node_id(sg_name)}"

        lines.append(
            f'    subgraph {subgraph_id}[" 📦 {cls._sanitize_node_name(sg_name)} "]'
        )
        lines.append(f"        direction TB")

        # Add subgraph internal START and END
        sg_start_id = cls._get_safe_node_id(START, f"sg_{sg_name}")
        sg_end_id = cls._get_safe_node_id(END, f"sg_{sg_name}")

        lines.append(f'        {sg_start_id}["⭐ START"]:::startNode;')
        lines.append(f'        {sg_end_id}["🏁 END"]:::endNode;')
        processed_nodes.update([sg_start_id, sg_end_id])

        # Add subgraph internal nodes
        for sub_node_name, sub_node in subgraph.nodes.items():
            if sub_node is None or sub_node_name in (START, END):
                continue

            safe_id = cls._get_safe_node_id(sub_node_name, f"sg_{sg_name}")
            display_name = cls._sanitize_node_name(sub_node_name)

            # Get node type
            if hasattr(subgraph, "node_types") and sub_node_name in subgraph.node_types:
                node_type = subgraph.node_types[sub_node_name]
                class_name = node_type_classes.get(node_type, "nodeTypeEngine")
            else:
                class_name = node_type_classes.get(NodeType.ENGINE, "nodeTypeEngine")

            lines.append(f'        {safe_id}["{display_name}"]:::{class_name};')
            processed_nodes.add(safe_id)

        # Add subgraph internal edges
        for source, target in subgraph.edges:
            source_id = cls._get_safe_node_id(source, f"sg_{sg_name}")
            target_id = cls._get_safe_node_id(target, f"sg_{sg_name}")

            edge_key = f"{source_id}->{target_id}"
            if edge_key not in processed_edges:
                lines.append(f"        {source_id} --> {target_id};")
                processed_edges.add(edge_key)

        # Add subgraph internal branches
        if hasattr(subgraph, "branches") and subgraph.branches:
            for branch_id, branch in subgraph.branches.items():
                source = branch.source_node
                source_id = cls._get_safe_node_id(source, f"sg_{sg_name}")

                destinations = getattr(branch, "destinations", {})
                for condition, target in destinations.items():
                    target_id = cls._get_safe_node_id(target, f"sg_{sg_name}")
                    condition_label = cls._format_condition_label(condition)

                    edge_key = f"{source_id}-->{target_id}_{condition}"
                    if edge_key not in processed_edges:
                        lines.append(
                            f'        {source_id} -.->|"{condition_label}"| {target_id};'
                        )
                        processed_edges.add(edge_key)

        lines.append("    end")

        # Style the subgraph
        lines.append(f"    style {subgraph_id} {cls.SUBGRAPH_STYLE}")

    @classmethod
    def _add_subgraph_bridges(
        cls,
        lines: List[str],
        graph: Any,
        subgraph_containers: Set,
        processed_edges: Set,
    ):
        """Add bridge connections between main graph and embedded subgraphs."""
        from langgraph.graph import END, START

        for sg_name in subgraph_containers:
            sg_start_id = cls._get_safe_node_id(START, f"sg_{sg_name}")
            sg_end_id = cls._get_safe_node_id(END, f"sg_{sg_name}")

            # Find edges TO this subgraph
            for source, target in graph.edges:
                if target == sg_name:
                    source_id = cls._get_safe_node_id(source)
                    edge_key = f"{source_id}->{sg_start_id}"
                    if edge_key not in processed_edges:
                        lines.append(f"    {source_id} --> {sg_start_id};")
                        processed_edges.add(edge_key)

            # Find edges FROM this subgraph
            for source, target in graph.edges:
                if source == sg_name:
                    target_id = cls._get_safe_node_id(target)
                    edge_key = f"{sg_end_id}->{target_id}"
                    if edge_key not in processed_edges:
                        lines.append(f"    {sg_end_id} --> {target_id};")
                        processed_edges.add(edge_key)

    @classmethod
    def _add_branch_edges(
        cls,
        lines: List[str],
        graph: Any,
        subgraph_containers: Set,
        processed_edges: Set,
    ):
        """Add conditional branch edges with proper styling."""
        for branch_id, branch in graph.branches.items():
            source = branch.source_node
            source_id = cls._get_safe_node_id(source)

            # Add branch comment
            branch_name = getattr(branch, "name", branch_id)
            lines.append(f"    %% Branch: {branch_name}")

            # Handle destinations
            destinations = getattr(branch, "destinations", {})
            if hasattr(branch, "condition_map"):
                destinations = branch.condition_map
            elif hasattr(branch, "routes"):
                destinations = branch.routes

            for condition, target in destinations.items():
                # Handle subgraph targets
                if target in subgraph_containers:
                    target_id = cls._get_safe_node_id("START", f"sg_{target}")
                else:
                    target_id = cls._get_safe_node_id(target)

                condition_label = cls._format_condition_label(condition)

                edge_key = f"{source_id}-->{target_id}_{condition}"
                if edge_key not in processed_edges:
                    lines.append(
                        f'    {source_id} -.->|"{condition_label}"| {target_id};'
                    )
                    processed_edges.add(edge_key)

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
        theme: str = "base",
        subgraph_mode: str = "cluster",
        show_default_branches: bool = True,
        direction: str = "TD",
        compact_mode: bool = False,
        title: Optional[str] = None,
    ) -> str:
        """
        Generate and display an elegant Mermaid diagram with embedded subgraphs.

        Args:
            graph: Graph object (BaseGraph instance)
            output_path: Optional path to save the diagram
            include_subgraphs: Whether to visualize subgraphs as embedded clusters
            highlight_nodes: List of node names to highlight
            highlight_paths: List of paths to highlight
            save_png: Whether to save the diagram as PNG
            width: Width of the displayed diagram
            theme: Mermaid theme to use
            subgraph_mode: How to render subgraphs ("cluster" for embedded view)
            show_default_branches: Whether to show default branches
            direction: Graph direction (TD, TB, LR, RL)
            compact_mode: Whether to use compact styling
            title: Optional title for the graph
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
            direction=direction,
            compact_mode=compact_mode,
        )

        # Add title if provided
        if title:
            title_line = f"    %% {title}"
            mermaid_lines = mermaid_code.split("\n")
            mermaid_lines.insert(2, title_line)
            mermaid_code = "\n".join(mermaid_lines)

        # Generate output path
        if save_png and not output_path:
            graph_name = getattr(graph, "name", f"graph_{uuid.uuid4().hex[:8]}")
            clean_name = graph_name.replace(" ", "_").replace("/", "_")
            output_dir = os.path.join(os.getcwd(), "graph_images")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{clean_name}.png")

        # Display the diagram
        display_mermaid(
            mermaid_code, output_path=output_path, save_png=save_png, width=width
        )

        return mermaid_code

    @classmethod
    def debug_graph_structure(cls, graph: Any) -> Dict[str, Any]:
        """Debug helper to understand graph structure including subgraphs."""
        info = {
            "nodes": {},
            "edges": [],
            "branches": {},
            "node_types": {},
            "subgraphs": {},
        }

        # Analyze main graph
        if hasattr(graph, "nodes"):
            for name, node in graph.nodes.items():
                info["nodes"][name] = {
                    "type": type(node).__name__ if node else "None",
                    "has_node_type": hasattr(node, "node_type") if node else False,
                }

        if hasattr(graph, "edges"):
            info["edges"] = list(graph.edges)

        if hasattr(graph, "branches"):
            for branch_id, branch in graph.branches.items():
                info["branches"][branch_id] = {
                    "source": getattr(branch, "source_node", "unknown"),
                    "destinations": getattr(branch, "destinations", {}),
                    "default": getattr(branch, "default", None),
                    "type": type(branch).__name__,
                }

        if hasattr(graph, "node_types"):
            info["node_types"] = dict(graph.node_types)

        # Analyze subgraphs
        if hasattr(graph, "subgraphs"):
            for sg_name, subgraph in graph.subgraphs.items():
                sg_info = {
                    "nodes": (
                        list(subgraph.nodes.keys())
                        if hasattr(subgraph, "nodes")
                        else []
                    ),
                    "edges": list(subgraph.edges) if hasattr(subgraph, "edges") else [],
                    "branches": (
                        list(subgraph.branches.keys())
                        if hasattr(subgraph, "branches")
                        else []
                    ),
                }
                info["subgraphs"][sg_name] = sg_info

        return info
