"""Enhanced Tree/Leaf Structure with Advanced Generics.

This mini-module provides an improved tree structure with:
- Better generic type support using TypeVar with bounds and defaults
- Overloaded constructors for different use cases
- Auto-inference of tree structure from BaseModel fields
- Support for mixed content types
- Built-in indexing and path tracking

Key improvements over AutoTree:
1. Multiple TypeVars for content, children, and result types
2. Default TypeVars for common use cases
3. Overloaded methods for type-safe operations
4. Better Union type handling
5. Computed properties with proper type inference
"""

from haive.core.common.structures.tree_leaf.auto import AutoTree, auto_tree
from haive.core.common.structures.tree_leaf.base import Leaf, Tree, TreeNode
from haive.core.common.structures.tree_leaf.generics import (
    ChildT,
    ContentT,
    DefaultContent,
    DefaultResult,
    ResultT,
)

__all__ = [
    # Base classes
    "TreeNode",
    "Leaf",
    "Tree",
    # Type variables
    "ContentT",
    "ChildT",
    "ResultT",
    "DefaultContent",
    "DefaultResult",
    # Auto tree
    "AutoTree",
    "auto_tree",
]
