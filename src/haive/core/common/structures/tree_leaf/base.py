"""Base tree node classes with advanced generic support."""

from abc import ABC, abstractmethod
from typing import (
    Generic,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from .generics import ChildT, ContentT, DefaultContent, DefaultResult, ResultT


class TreeNode(BaseModel, Generic[ContentT, ResultT], ABC):
    """Abstract base class for all tree nodes.

    Uses bounded TypeVars for better type safety and inference.
    """

    content: ContentT
    result: Optional[ResultT] = None

    # Auto-indexing (hidden)
    _index: int = PrivateAttr(default=0)
    _parent: Optional["TreeNode"] = PrivateAttr(default=None)
    _depth: int = PrivateAttr(default=0)
    _path: tuple[int, ...] = PrivateAttr(default=())

    @abstractmethod
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        ...

    @computed_field
    @property
    def node_id(self) -> str:
        """Unique identifier based on path."""
        if not self._path:
            return "root"
        return ".".join(str(i) for i in self._path)

    @computed_field
    @property
    def level(self) -> int:
        """Tree level (alias for depth)."""
        return self._depth


class Leaf(TreeNode[ContentT, ResultT], Generic[ContentT, ResultT]):
    """Leaf node - has content but no children.

    Example:
        # With explicit types
        leaf: Leaf[TaskContent, TaskResult] = Leaf(
            content=TaskContent(name="Calculate", action="add", params={"a": 1, "b": 2})
        )

        # With default types
        simple_leaf = Leaf(content=DefaultContent(name="Task1"))
    """

    def is_leaf(self) -> bool:
        return True


class Tree(TreeNode[ContentT, ResultT], Generic[ContentT, ChildT, ResultT]):
    """Tree node - has content and children.

    The ChildT parameter allows for heterogeneous trees where children
    can be of different types (but all extending the bound).

    Example:
        # Homogeneous tree (all children same type)
        tree: Tree[PlanContent, PlanNode, PlanResult] = Tree(
            content=PlanContent(objective="Main Plan")
        )

        # Heterogeneous tree (mixed children)
        mixed: Tree[DefaultContent, TreeNode, DefaultResult] = Tree(
            content=DefaultContent(name="Root")
        )
    """

    children: List[ChildT] = Field(default_factory=list)

    # Private counter for auto-indexing
    _child_counter: int = PrivateAttr(default=0)

    def is_leaf(self) -> bool:
        return False

    @overload
    def add_child(self, child: ChildT) -> ChildT:
        """Add a single child."""
        ...

    @overload
    def add_child(self, *children: ChildT) -> List[ChildT]:
        """Add multiple children."""
        ...

    def add_child(self, *children: ChildT) -> Union[ChildT, List[ChildT]]:
        """Add one or more children with auto-indexing."""
        if len(children) == 1:
            child = children[0]
            self._index_child(child)
            self.children.append(child)
            return child
        else:
            indexed_children = []
            for child in children:
                self._index_child(child)
                self.children.append(child)
                indexed_children.append(child)
            return indexed_children

    def _index_child(self, child: ChildT) -> None:
        """Set up indexing for a child node."""
        if hasattr(child, "_index"):
            child._index = self._child_counter
        if hasattr(child, "_parent"):
            child._parent = self
        if hasattr(child, "_depth"):
            child._depth = self._depth + 1
        if hasattr(child, "_path"):
            child._path = self._path + (self._child_counter,)

        self._child_counter += 1

    @computed_field
    @property
    def child_count(self) -> int:
        """Number of direct children."""
        return len(self.children)

    @computed_field
    @property
    def descendant_count(self) -> int:
        """Total number of descendants."""
        count = 0
        for child in self.children:
            count += 1
            if hasattr(child, "descendant_count"):
                count += child.descendant_count
        return count

    @computed_field
    @property
    def height(self) -> int:
        """Height of the subtree rooted at this node."""
        if not self.children:
            return 0

        max_child_height = 0
        for child in self.children:
            child_height = 0
            if hasattr(child, "height"):
                child_height = child.height
            max_child_height = max(max_child_height, child_height)

        return max_child_height + 1

    def find_by_path(self, *indices: int) -> Optional[ChildT]:
        """Find a descendant by path indices."""
        if not indices:
            return self

        if indices[0] >= len(self.children):
            return None

        child = self.children[indices[0]]
        if len(indices) == 1:
            return child

        if hasattr(child, "find_by_path"):
            return child.find_by_path(*indices[1:])

        return None


# Convenience type aliases for common patterns
SimpleTree = Tree[
    DefaultContent, TreeNode[DefaultContent, DefaultResult], DefaultResult
]
SimpleLeaf = Leaf[DefaultContent, DefaultResult]
SimpleBranch = Tree[
    DefaultContent, TreeNode[DefaultContent, DefaultResult], DefaultResult
]
