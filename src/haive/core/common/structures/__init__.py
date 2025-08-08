"""Common data structures for the Haive framework.

This module provides various data structures used throughout the framework:
- tree_leaf: Enhanced tree/leaf structure with advanced generics
- tree: AutoTree for automatic tree generation from BaseModels
- generic_tree: Generic tree implementation with status tracking
- named_dict: Named dictionary utilities
"""

# Export tree_leaf module
from haive.core.common.structures.tree_leaf import (  # Base classes; Type variables; Auto tree
    AutoTree,
    ChildT,
    ContentT,
    DefaultContent,
    DefaultResult,
    Leaf,
    ResultT,
    Tree,
    TreeNode,
    auto_tree,
)

# Also export convenience names
TreeLeaf = Tree  # Alias for backward compatibility

__all__ = [
    # From tree_leaf
    "TreeNode",
    "Leaf",
    "Tree",
    "TreeLeaf",  # Alias
    "ContentT",
    "ChildT",
    "ResultT",
    "DefaultContent",
    "DefaultResult",
    "AutoTree",
    "auto_tree",
]
