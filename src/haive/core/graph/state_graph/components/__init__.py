"""Components module for the Haive state graph system.

This module provides the core components used in the state graph system for building
computational graphs with rich state management and control flow capabilities.

The components module includes:
    - Node: Base processing unit in a graph that handles state transformation
    - Branch: Conditional routing component for decision points in the graph

These components are designed to be used both within the graph system and independently,
allowing for flexible composition and reuse across different graph configurations.

Example:
    Creating and using components directly:
    ```python
    from haive.core.graph.state_graph.components import Node, Branch
    from haive.core.graph.common.types import NodeType
    from haive.core.graph.branches.types import BranchMode

    # Create a processing node
    node = Node(
        name="process_data",
        node_type=NodeType.CALLABLE,
        metadata={"callable": lambda state: {"processed": state["input"].upper()}}
    )

    # Create a decision branch
    branch = Branch(
        name="route_data",
        source_node="check_data",
        mode=BranchMode.FUNCTION,
        function=lambda state: "valid" if len(state.get("data", "")) > 0 else "invalid",
        destinations={"valid": "process_valid", "invalid": "handle_error"}
    )
    ```
"""

from haive.core.graph.state_graph.components.branch import Branch
from haive.core.graph.state_graph.components.node import Node

__all__ = ["Node", "Branch"]
