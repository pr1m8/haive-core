"""
Enhanced graph visualization utilities for Haive graphs with automatic Agent detection and expansion.

This module provides comprehensive visualization that automatically detects nodes with Agent engines
or nodes that are Agents themselves, expanding them to show their complete internal graph structure.
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from langgraph.graph import END, START

from haive.core.graph.common.types import NodeType
from haive.core.utils.mermaid_utils import (
    Environment,
    detect_environment,
    display_mermaid,
    mermaid_to_png,
)

# Set up module logger
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """
    Enhanced visualization utilities with automatic Agent detection and expansion.

    Detects:
    1. EngineNodeConfig where engine.engine_type == 'agent'
    2. Nodes that are Agent instances themselves
    3. Any node with a graph attribute
    """

    # Professional color scheme
    NODE_COLORS = {
        NodeType.ENGINE: "#2563EB",  # Blue
        NodeType.TOOL: "#DC2626",  # Red
        NodeType.VALIDATION: "#059669",  # Green
        NodeType.SUBGRAPH: "#7C3AED",  # Purple (for agents)
        NodeType.CALLABLE: "#0891B2",  # Cyan
        NodeType.CUSTOM: "#BE185D",  # Pink
    }

    # Special node colors
    START_COLOR = "#065F46"  # Dark green
    END_COLOR = "#991B1B"  # Dark red
    HIGHLIGHT_COLOR = "#B45309"  # Orange

    # Subgraph styling
    SUBGRAPH_BG = "#F3F4F6"
    SUBGRAPH_BORDER = "#9CA3AF"
    AGENT_BG = "#EDE9FE"  # Light purple for agent subgraphs
    AGENT_BORDER = "#7C3AED"  # Purple border for agents

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
        debug: bool = False,
    ) -> str:
        """
        Generate a comprehensive Mermaid diagram with automatic Agent expansion.

        Args:
            graph: The graph to visualize
            include_subgraphs: Whether to expand Agent nodes
            highlight_nodes: Nodes to highlight
            highlight_color: Color for highlights
            theme: Mermaid theme
            subgraph_mode: How to display subgraphs
            show_default_branches: Show default branch labels
            direction: Graph direction
            compact_mode: Use compact styling
            debug: Enable debug output

        Returns:
            Complete Mermaid diagram string
        """
        logger.info(
            f"Starting Mermaid generation for graph: {getattr(graph, 'name', 'Unnamed')}"
        )

        if debug:
            print("\n" + "=" * 80)
            print(f"🎨 GRAPH VISUALIZATION DEBUG")
            print("=" * 80)
            print(f"Graph: {getattr(graph, 'name', 'Unnamed Graph')}")
            print(f"Type: {type(graph).__name__}")
            print(f"Include subgraphs: {include_subgraphs}")
            print(f"Debug mode: {debug}")
            print("=" * 80 + "\n")

        highlight_color = highlight_color or cls.HIGHLIGHT_COLOR

        # Initialize diagram
        lines = [
            f'%%{{ init: {{ "theme": "{theme}", "flowchart": {{ "curve": "basis", "padding": 15, "nodeSpacing": 50, "rankSpacing": 80 }} }} }}%%',
            f"flowchart {direction};",
            "    %% === SAFE PROFESSIONAL STYLING ===",
        ]

        # Add style definitions
        cls._add_style_definitions(lines)

        # Track state
        processed_nodes = set()
        processed_edges = set()
        agent_subgraphs = {}  # Map node_name -> (node, agent_graph)

        # CRITICAL: Detect Agent nodes
        if include_subgraphs:
            logger.info("Detecting agent nodes...")
            agent_subgraphs = cls._detect_agent_nodes(graph, debug)

            if debug:
                print(f"\n🤖 AGENT DETECTION RESULTS:")
                print(f"   Found {len(agent_subgraphs)} agent nodes")
                for name, (node, agent_graph) in agent_subgraphs.items():
                    print(f"   - {name}: {type(node).__name__}")
                    if agent_graph and hasattr(agent_graph, "nodes"):
                        print(
                            f"     └─ Internal nodes: {list(agent_graph.nodes.keys())}"
                        )
                print()

        # Add START and END nodes
        lines.append("    %% === MAIN GRAPH NODES ===")
        lines.append('    START["START"]:::startNode;')
        lines.append('    END["END"]:::endNode;')
        processed_nodes.update(["START", "END"])

        # Process main graph nodes
        if hasattr(graph, "nodes"):
            logger.info(f"Processing {len(graph.nodes)} main graph nodes")

            for node_name, node in (graph.nodes or {}).items():
                if node_name in (START, END, "__start__", "__end__"):
                    continue

                if debug:
                    print(f"\n📍 Processing node: {node_name}")
                    print(f"   Type: {type(node).__name__ if node else 'None'}")

                # Check if this is an Agent node that should be expanded
                if node_name in agent_subgraphs and include_subgraphs:
                    node_obj, agent_graph = agent_subgraphs[node_name]

                    if debug:
                        print(f"   ✓ Expanding as agent subgraph")

                    # Add as expanded subgraph
                    cls._add_agent_subgraph(
                        lines,
                        node_name,
                        node_obj,
                        agent_graph,
                        processed_nodes,
                        processed_edges,
                        debug,
                    )
                else:
                    # Regular node
                    safe_id = cls._get_safe_node_id(node_name)
                    label = cls._sanitize_node_name(node_name)
                    style = cls._get_node_style(node, graph, node_name)

                    if debug:
                        print(f"   → Regular node: {safe_id} [{label}] style={style}")

                    lines.append(f'    {safe_id}["{label}"]:::{style};')
                    processed_nodes.add(safe_id)

        # Add edges
        lines.append("")
        lines.append("    %% === MAIN GRAPH EDGES ===")

        if hasattr(graph, "edges"):
            logger.info(f"Processing {len(graph.edges)} edges")

            for source, target in graph.edges or []:
                source_id = cls._get_edge_node_id(source, agent_subgraphs, "exit")
                target_id = cls._get_edge_node_id(target, agent_subgraphs, "entry")

                if debug:
                    print(f"Edge: {source} ({source_id}) → {target} ({target_id})")

                edge_key = f"{source_id}->{target_id}"
                if edge_key not in processed_edges:
                    lines.append(f"    {source_id} --> {target_id};")
                    processed_edges.add(edge_key)

        # Add branches
        if hasattr(graph, "branches") and graph.branches:
            lines.append("")
            lines.append("    %% === CONDITIONAL BRANCHES ===")
            logger.info(f"Processing {len(graph.branches)} branches")

            for branch_id, branch in graph.branches.items():
                cls._add_branch_edges(
                    lines, branch, agent_subgraphs, processed_edges, branch_id, debug
                )

        # Add highlights
        if highlight_nodes:
            lines.append("")
            lines.append("    %% === HIGHLIGHTS ===")
            highlight_ids = []
            for node in highlight_nodes:
                if node in processed_nodes:
                    highlight_ids.append(cls._get_safe_node_id(node))

            if highlight_ids:
                lines.append(f"    class {','.join(highlight_ids)} highlightNode;")

        mermaid_code = "\n".join(lines)
        logger.info(f"Generated Mermaid code: {len(mermaid_code)} characters")

        if debug:
            print("\n" + "=" * 80)
            print("✅ VISUALIZATION COMPLETE")
            print(f"   Total lines: {len(lines)}")
            print(f"   Processed nodes: {len(processed_nodes)}")
            print(f"   Processed edges: {len(processed_edges)}")
            print(f"   Agent subgraphs: {len(agent_subgraphs)}")
            print("=" * 80 + "\n")

        return mermaid_code

    @classmethod
    def _detect_agent_nodes(
        cls, graph: Any, debug: bool = False
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        Detect nodes that are Agents or have Agent engines.

        Returns:
            Dict mapping node_name -> (node_object, agent_graph)
        """
        agent_subgraphs = {}

        if not hasattr(graph, "nodes"):
            logger.warning("Graph has no nodes attribute")
            return agent_subgraphs

        logger.info(f"Checking {len(graph.nodes)} nodes for agents...")

        for node_name, node in (graph.nodes or {}).items():
            if node is None:
                continue

            if debug:
                print(f"\n🔍 Checking node: {node_name}")
                print(f"   Node type: {type(node).__name__}")

            agent_graph = None
            is_agent = False

            # Method 1: Check if node is EngineNodeConfig with Agent engine
            if (
                hasattr(node, "__class__")
                and "EngineNodeConfig" in node.__class__.__name__
            ):
                if debug:
                    print(f"   → Node is EngineNodeConfig")

                if hasattr(node, "engine") and node.engine:
                    engine = node.engine

                    if debug:
                        print(f"   → Has engine: {type(engine).__name__}")

                    # Check engine_type
                    if hasattr(engine, "engine_type"):
                        engine_type_value = None

                        # Handle enum or string
                        if hasattr(engine.engine_type, "value"):
                            engine_type_value = engine.engine_type.value
                        else:
                            engine_type_value = str(engine.engine_type)

                        if debug:
                            print(f"   → Engine type: {engine_type_value}")

                        if engine_type_value == "agent":
                            is_agent = True
                            logger.info(f"   ✓ Found agent engine in {node_name}")

                            # Get the graph from the engine
                            if hasattr(engine, "graph") and engine.graph:
                                agent_graph = engine.graph
                                if debug:
                                    print(f"   → Got graph from engine.graph")
                            elif hasattr(engine, "build_graph"):
                                try:
                                    if debug:
                                        print(f"   → Attempting to build graph...")
                                    agent_graph = engine.build_graph()
                                    if debug:
                                        print(f"   → Successfully built graph")
                                except Exception as e:
                                    logger.error(
                                        f"Failed to build graph for {node_name}: {e}"
                                    )
                                    if debug:
                                        print(f"   ⚠️ Failed to build graph: {e}")

            # Method 2: Check if node itself is an Agent
            if not is_agent:
                # Check class name
                class_name = type(node).__name__
                if "Agent" in class_name:
                    is_agent = True
                    logger.info(f"   ✓ Node {node_name} is an Agent ({class_name})")

                    if debug:
                        print(f"   → Node class contains 'Agent': {class_name}")

                # Check MRO
                if not is_agent:
                    for base in type(node).__mro__:
                        if base.__name__ == "Agent":
                            is_agent = True
                            logger.info(f"   ✓ Node {node_name} inherits from Agent")

                            if debug:
                                print(f"   → Found Agent in MRO")
                            break

                # If it's an agent, get its graph
                if is_agent:
                    if hasattr(node, "graph") and node.graph:
                        agent_graph = node.graph
                        if debug:
                            print(f"   → Got graph from node.graph")
                    elif hasattr(node, "build_graph"):
                        try:
                            if debug:
                                print(f"   → Attempting to build graph...")
                            agent_graph = node.build_graph()
                            if debug:
                                print(f"   → Successfully built graph")
                        except Exception as e:
                            logger.error(f"Failed to build graph for {node_name}: {e}")
                            if debug:
                                print(f"   ⚠️ Failed to build graph: {e}")

            # Method 3: Check for graph attribute directly
            if not is_agent and not agent_graph:
                if hasattr(node, "graph") and node.graph:
                    is_agent = True
                    agent_graph = node.graph
                    logger.info(f"   ✓ Node {node_name} has graph attribute")

                    if debug:
                        print(f"   → Node has graph attribute")

            # Store if we found an agent
            if is_agent and agent_graph:
                agent_subgraphs[node_name] = (node, agent_graph)

                if debug:
                    node_count = len(getattr(agent_graph, "nodes", {}))
                    edge_count = len(getattr(agent_graph, "edges", []))
                    print(
                        f"   ✅ AGENT CONFIRMED: {node_count} nodes, {edge_count} edges"
                    )
            elif is_agent:
                logger.warning(f"   ⚠️ Node {node_name} is agent but has no graph")
                if debug:
                    print(f"   ⚠️ Agent detected but no graph found")

        logger.info(f"Agent detection complete: found {len(agent_subgraphs)} agents")
        return agent_subgraphs

    @classmethod
    def _add_agent_subgraph(
        cls,
        lines: List[str],
        node_name: str,
        node: Any,
        agent_graph: Any,
        processed_nodes: Set[str],
        processed_edges: Set[str],
        debug: bool = False,
    ):
        """Add a complete Agent subgraph expansion."""
        logger.info(f"Adding agent subgraph for: {node_name}")

        subgraph_id = f"cluster_{cls._get_safe_node_id(node_name)}"
        clean_name = cls._sanitize_node_name(node_name)

        lines.append("")
        lines.append(f"    %% === AGENT SUBGRAPH: {node_name} ===")
        lines.append(f'    subgraph {subgraph_id}[" 🤖 {clean_name} Agent "]')
        lines.append("        direction TB")

        # Add entry/exit points
        entry_id = f"{node_name}_entry"
        exit_id = f"{node_name}_exit"
        safe_entry = cls._get_safe_node_id(entry_id)
        safe_exit = cls._get_safe_node_id(exit_id)

        lines.append(f'        {safe_entry}["◉ Entry"]:::startNode;')
        lines.append(f'        {safe_exit}["◉ Exit"]:::endNode;')
        processed_nodes.update([safe_entry, safe_exit])

        if debug:
            print(f"\n📦 Building subgraph for: {node_name}")
            print(f"   Entry: {safe_entry}")
            print(f"   Exit: {safe_exit}")

        # Map internal nodes
        internal_node_map = {}

        # Add internal nodes
        if hasattr(agent_graph, "nodes"):
            internal_nodes = agent_graph.nodes or {}
            logger.info(f"   Adding {len(internal_nodes)} internal nodes")

            for int_name, int_node in internal_nodes.items():
                if int_name in (START, END, "__start__", "__end__"):
                    continue

                int_safe_id = f"{node_name}_{cls._get_safe_node_id(int_name)}"
                int_label = cls._sanitize_node_name(int_name)
                int_style = cls._get_node_style(int_node, agent_graph, int_name)

                if debug:
                    print(f"   + Internal node: {int_name} → {int_safe_id}")

                lines.append(f'        {int_safe_id}["{int_label}"]:::{int_style};')
                processed_nodes.add(int_safe_id)
                internal_node_map[int_name] = int_safe_id

        # Add internal edges
        if hasattr(agent_graph, "edges"):
            internal_edges = agent_graph.edges or []
            logger.info(f"   Adding {len(internal_edges)} internal edges")

            for source, target in internal_edges:
                # Map START/END to entry/exit
                if source in (START, "__start__"):
                    source_id = safe_entry
                elif source in internal_node_map:
                    source_id = internal_node_map[source]
                else:
                    logger.warning(f"   Unknown source node: {source}")
                    continue

                if target in (END, "__end__"):
                    target_id = safe_exit
                elif target in internal_node_map:
                    target_id = internal_node_map[target]
                else:
                    logger.warning(f"   Unknown target node: {target}")
                    continue

                edge_key = f"{source_id}->{target_id}"
                if edge_key not in processed_edges:
                    lines.append(f"        {source_id} --> {target_id};")
                    processed_edges.add(edge_key)

                    if debug:
                        print(f"   + Internal edge: {source} → {target}")

        # Add internal branches
        if hasattr(agent_graph, "branches") and agent_graph.branches:
            logger.info(f"   Adding {len(agent_graph.branches)} internal branches")

            for branch_id, branch in agent_graph.branches.items():
                source_node = getattr(branch, "source_node", None)
                if source_node and source_node in internal_node_map:
                    source_id = internal_node_map[source_node]

                    destinations = cls._get_branch_destinations(branch)
                    for condition, target in destinations.items():
                        if target in (END, "__end__"):
                            target_id = safe_exit
                        elif target in internal_node_map:
                            target_id = internal_node_map[target]
                        else:
                            continue

                        condition_label = cls._format_condition_label(condition)
                        edge_key = f"{source_id}-->{target_id}_{condition}"

                        if edge_key not in processed_edges:
                            lines.append(
                                f'        {source_id} -.->|"{condition_label}"| {target_id};'
                            )
                            processed_edges.add(edge_key)

                            if debug:
                                print(
                                    f"   + Internal branch: {source_node} --{condition}--> {target}"
                                )

        lines.append("    end")

        # Style the subgraph
        lines.append(
            f"    style {subgraph_id} fill:{cls.AGENT_BG},stroke:{cls.AGENT_BORDER},stroke-width:3px;"
        )

        # Mark this node as processed
        processed_nodes.add(cls._get_safe_node_id(node_name))

        logger.info(f"   Agent subgraph complete for: {node_name}")

    @classmethod
    def _add_style_definitions(cls, lines: List[str]):
        """Add all style class definitions."""
        # Node type styles
        for node_type, color in cls.NODE_COLORS.items():
            class_name = f"nodeType{node_type.value.capitalize()}"
            lines.append(
                f"    classDef {class_name} fill:{color},stroke:white,stroke-width:2px,color:white,font-weight:500,font-size:13px,font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            )

        # Special styles
        lines.append(
            f"    classDef startNode fill:{cls.START_COLOR},stroke:white,stroke-width:2px,color:white,font-weight:bold,font-size:14px;"
        )
        lines.append(
            f"    classDef endNode fill:{cls.END_COLOR},stroke:white,stroke-width:2px,color:white,font-weight:bold,font-size:14px;"
        )
        lines.append(
            f"    classDef highlightNode fill:{cls.HIGHLIGHT_COLOR},stroke:white,stroke-width:3px,color:white,font-weight:bold;"
        )
        lines.append(
            f"    classDef subgraphNode fill:{cls.SUBGRAPH_BG},stroke:{cls.SUBGRAPH_BORDER},stroke-width:2px,color:#374151;"
        )

    @classmethod
    def _get_safe_node_id(cls, node_name: str) -> str:
        """Convert node name to safe Mermaid ID."""
        if node_name in ("START", "__start__"):
            return "START"
        if node_name in ("END", "__end__"):
            return "node_END"  # Avoid keyword collision

        # Replace problematic characters
        safe_id = str(node_name)
        for char in " -./\\:()[]{}@#$%^&*+=|<>?~`\"',;!":
            safe_id = safe_id.replace(char, "_")

        # Ensure valid identifier
        if safe_id and safe_id[0].isdigit():
            safe_id = f"node_{safe_id}"

        # Handle reserved words
        reserved = {
            "end",
            "subgraph",
            "class",
            "classDef",
            "click",
            "style",
            "graph",
            "flowchart",
        }
        if safe_id.lower() in reserved:
            safe_id = f"node_{safe_id}"

        return safe_id

    @classmethod
    def _get_edge_node_id(
        cls, node_name: str, agent_subgraphs: Dict[str, Tuple[Any, Any]], suffix: str
    ) -> str:
        """Get node ID for edge connections, handling agent subgraphs."""
        if node_name in (START, "__start__"):
            return "START"
        if node_name in (END, "__end__"):
            return "node_END"

        # If it's an agent node, connect to entry/exit
        if node_name in agent_subgraphs:
            return cls._get_safe_node_id(f"{node_name}_{suffix}")

        return cls._get_safe_node_id(node_name)

    @classmethod
    def _sanitize_node_name(cls, name: str) -> str:
        """Create clean, readable node label."""
        if name in ("__start__", "START"):
            return "START"
        if name in ("__end__", "END"):
            return "END"

        # Clean up
        clean_name = name.replace("_", " ").strip()
        clean_name = clean_name.title()

        # Common replacements
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
    def _get_node_style(cls, node: Any, graph: Any, node_name: str) -> str:
        """Determine the style class for a node."""
        # Check node_types mapping
        if hasattr(graph, "node_types") and node_name in graph.node_types:
            node_type = graph.node_types[node_name]
            return f"nodeType{node_type.value.capitalize()}"

        # Check node.node_type
        if hasattr(node, "node_type"):
            return f"nodeType{node.node_type.value.capitalize()}"

        # Pattern matching
        if node is not None:
            class_name = type(node).__name__.lower()
            if "tool" in class_name:
                return "nodeTypeTool"
            elif "validation" in class_name:
                return "nodeTypeValidation"
            elif "agent" in class_name:
                return "nodeTypeSubgraph"

        # Default
        return "nodeTypeEngine"

    @classmethod
    def _add_branch_edges(
        cls,
        lines: List[str],
        branch: Any,
        agent_subgraphs: Dict[str, Tuple[Any, Any]],
        processed_edges: Set[str],
        branch_id: str,
        debug: bool = False,
    ):
        """Add conditional branch edges."""
        source = getattr(branch, "source_node", None)
        if not source:
            return

        source_id = cls._get_edge_node_id(source, agent_subgraphs, "exit")
        destinations = cls._get_branch_destinations(branch)

        # Add comment
        lines.append(f"    %% Branch: {branch_id}")

        if debug:
            print(f"\n🔀 Branch: {branch_id}")
            print(f"   Source: {source} ({source_id})")

        for condition, target in destinations.items():
            target_id = cls._get_edge_node_id(target, agent_subgraphs, "entry")
            condition_label = cls._format_condition_label(condition)

            if debug:
                print(f"   {condition} → {target} ({target_id})")

            edge_key = f"{source_id}-->{target_id}_{condition}"
            if edge_key not in processed_edges:
                lines.append(f'    {source_id} -.->|"{condition_label}"| {target_id};')
                processed_edges.add(edge_key)

    @classmethod
    def _get_branch_destinations(cls, branch: Any) -> Dict[str, str]:
        """Extract destinations from branch object."""
        # Try different attributes
        for attr in ["destinations", "condition_map", "routes"]:
            if hasattr(branch, attr):
                dest = getattr(branch, attr)
                if isinstance(dest, dict):
                    return dest
        return {}

    @classmethod
    def _format_condition_label(cls, condition: str) -> str:
        """Format condition label for display."""
        condition_str = str(condition)

        # Boolean values
        if condition_str.lower() == "true":
            return "✓ Yes"
        elif condition_str.lower() == "false":
            return "✗ No"

        # Common patterns
        replacements = {
            "has_errors": "❌ Has Errors",
            "no_errors": "✅ No Errors",
            "tool_node": "🔧 Use Tools",
            "parse_output": "📝 Parse Output",
            "validation_passed": "✅ Valid",
            "validation_failed": "❌ Invalid",
            "default": "📍 Default",
        }

        return replacements.get(condition_str, condition_str.replace("_", " ").title())

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
        debug: bool = False,
    ) -> str:
        """
        Generate and display graph with Agent expansion.

        Returns:
            Generated Mermaid code
        """
        logger.info(f"Displaying graph: {getattr(graph, 'name', 'Unnamed')}")

        # Combine highlight nodes
        all_highlight_nodes = highlight_nodes or []
        if highlight_paths:
            for path in highlight_paths:
                all_highlight_nodes.extend(path)

        # Generate Mermaid
        mermaid_code = cls.generate_mermaid(
            graph,
            include_subgraphs=include_subgraphs,
            highlight_nodes=all_highlight_nodes if all_highlight_nodes else None,
            theme=theme,
            subgraph_mode=subgraph_mode,
            show_default_branches=show_default_branches,
            direction=direction,
            compact_mode=compact_mode,
            debug=debug,
        )

        # Add title if provided
        if title:
            title_line = f"    %% {title}"
            mermaid_lines = mermaid_code.split("\n")
            mermaid_lines.insert(2, title_line)
            mermaid_code = "\n".join(mermaid_lines)

        # Generate output path if needed
        if save_png and not output_path:
            graph_name = getattr(graph, "name", f"graph_{uuid.uuid4().hex[:8]}")
            clean_name = graph_name.replace(" ", "_").replace("/", "_")
            output_dir = os.path.join(os.getcwd(), "graph_images")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{clean_name}.png")

        # Display
        display_mermaid(
            mermaid_code, output_path=output_path, save_png=save_png, width=width
        )

        return mermaid_code

    @classmethod
    def debug_graph_structure(cls, graph: Any) -> Dict[str, Any]:
        """Debug helper to analyze graph structure."""
        info = {
            "graph_name": getattr(graph, "name", "Unnamed"),
            "graph_type": type(graph).__name__,
            "nodes": {},
            "edges": [],
            "branches": {},
            "node_types": {},
            "agent_nodes": {},
        }

        # Analyze nodes
        if hasattr(graph, "nodes"):
            for name, node in (graph.nodes or {}).items():
                node_info = {
                    "type": type(node).__name__ if node else "None",
                    "has_graph": hasattr(node, "graph") and node.graph is not None,
                }

                # Check for EngineNodeConfig
                if (
                    hasattr(node, "__class__")
                    and "EngineNodeConfig" in node.__class__.__name__
                ):
                    node_info["is_engine_node_config"] = True

                    # Check engine
                    if hasattr(node, "engine"):
                        engine = node.engine
                        node_info["engine_type"] = (
                            type(engine).__name__ if engine else "None"
                        )

                        if hasattr(engine, "engine_type"):
                            if hasattr(engine.engine_type, "value"):
                                node_info["engine_type_value"] = (
                                    engine.engine_type.value
                                )
                            else:
                                node_info["engine_type_value"] = str(engine.engine_type)

                        node_info["engine_has_graph"] = (
                            hasattr(engine, "graph") and engine.graph is not None
                        )

                        # Check if engine is Agent
                        engine_class_name = type(engine).__name__ if engine else ""
                        node_info["engine_is_agent"] = "Agent" in engine_class_name

                        # Check MRO
                        if engine:
                            for base in type(engine).__mro__:
                                if base.__name__ == "Agent":
                                    node_info["engine_inherits_agent"] = True
                                    break

                # Check if node itself is Agent
                node_class_name = type(node).__name__ if node else ""
                node_info["is_agent"] = "Agent" in node_class_name

                # Check MRO
                if node:
                    for base in type(node).__mro__:
                        if base.__name__ == "Agent":
                            node_info["inherits_agent"] = True
                            break

                info["nodes"][name] = node_info

                # Track agent nodes
                is_agent = (
                    node_info.get("is_agent")
                    or node_info.get("inherits_agent")
                    or node_info.get("engine_is_agent")
                    or node_info.get("engine_inherits_agent")
                    or (node_info.get("engine_type_value") == "agent")
                )

                if is_agent:
                    info["agent_nodes"][name] = {
                        "reason": [],
                        "has_graph": node_info.get("has_graph")
                        or node_info.get("engine_has_graph", False),
                    }

                    if node_info.get("is_agent"):
                        info["agent_nodes"][name]["reason"].append("node_is_agent")
                    if node_info.get("inherits_agent"):
                        info["agent_nodes"][name]["reason"].append(
                            "node_inherits_agent"
                        )
                    if node_info.get("engine_is_agent"):
                        info["agent_nodes"][name]["reason"].append("engine_is_agent")
                    if node_info.get("engine_inherits_agent"):
                        info["agent_nodes"][name]["reason"].append(
                            "engine_inherits_agent"
                        )
                    if node_info.get("engine_type_value") == "agent":
                        info["agent_nodes"][name]["reason"].append(
                            "engine_type_is_agent"
                        )

        # Edges
        if hasattr(graph, "edges"):
            info["edges"] = list(graph.edges)

        # Branches
        if hasattr(graph, "branches"):
            for branch_id, branch in graph.branches.items():
                info["branches"][branch_id] = {
                    "source": getattr(branch, "source_node", "unknown"),
                    "destinations": cls._get_branch_destinations(branch),
                    "type": type(branch).__name__,
                }

        # Node types
        if hasattr(graph, "node_types"):
            info["node_types"] = {k: str(v) for k, v in graph.node_types.items()}

        return info
