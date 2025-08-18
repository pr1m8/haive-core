"""Generic Tree/Leaf model with auto-indexing and type safety.

This module provides a powerful generic tree structure that can be used for
any hierarchical data, with automatic indexing, computed properties, and
full type safety through generics.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar, Union

from pydantic import BaseModel, Field, PrivateAttr, computed_field

# Type variables for content and result types
ContentType = TypeVar("ContentType")
ResultType = TypeVar("ResultType")


class NodeStatus(str, Enum):
    """Status for tree nodes."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TreeNode(BaseModel, Generic[ContentType, ResultType], ABC):
    """Abstract base class for tree nodes.

    Provides common functionality for both leaf and branch nodes.
    """

    # User-visible fields
    content: ContentType = Field(..., description="The node's content/objective")
    result: ResultType | None = Field(None, description="The execution result")
    status: NodeStatus = Field(NodeStatus.PENDING, description="Current node status")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Private auto-indexing fields
    _index: int = PrivateAttr(default=0)
    _parent_index: int | None = PrivateAttr(default=None)
    _depth: int = PrivateAttr(default=0)
    _path: list[int] = PrivateAttr(default_factory=list)

    @abstractmethod
    def is_leaf(self) -> bool:
        """Whether this is a leaf node."""

    def set_result(
        self, result: ResultType, status: NodeStatus = NodeStatus.COMPLETED
    ) -> None:
        """Set the result and update status."""
        self.result = result
        self.status = status

    def mark_failed(self, error: str) -> None:
        """Mark node as failed with error message."""
        self.status = NodeStatus.FAILED
        self.metadata["error"] = error


class Leaf(TreeNode[ContentType, ResultType]):
    """Leaf node - contains content but no children."""

    def is_leaf(self) -> bool:
        """Is Leaf.

        Returns:
            [TODO: Add return description]
        """
        return True

    @computed_field
    @property
    def is_complete(self) -> bool:
        """Whether this leaf is complete."""
        return self.status == NodeStatus.COMPLETED


class Branch(TreeNode[ContentType, ResultType], Generic[ContentType, ResultType]):
    """Branch node - contains content and children."""

    children: list[
        Union[Leaf[ContentType, ResultType], "Branch[ContentType, ResultType]"]
    ] = Field(default_factory=list, description="Child nodes (leaves or branches)")

    # Private indexing for children
    _next_child_index: int = PrivateAttr(default=0)

    def is_leaf(self) -> bool:
        """Is Leaf.

        Returns:
            [TODO: Add return description]
        """
        return False

    def add_child(
        self,
        child: Union[Leaf[ContentType, ResultType], "Branch[ContentType, ResultType]"],
        auto_index: bool = True,
    ) -> Union[Leaf[ContentType, ResultType], "Branch[ContentType, ResultType]"]:
        """Add a child node with auto-indexing."""
        if auto_index:
            child._index = self._next_child_index
            child._parent_index = self._index
            child._depth = self._depth + 1
            child._path = self._path + [self._next_child_index]
            self._next_child_index += 1

        self.children.append(child)
        return child

    def add_leaf(self, content: ContentType) -> Leaf[ContentType, ResultType]:
        """Convenience method to add a leaf child."""
        leaf = Leaf[ContentType, ResultType](content=content)
        return self.add_child(leaf)

    def add_branch(self, content: ContentType) -> "Branch[ContentType, ResultType]":
        """Convenience method to add a branch child."""
        branch = Branch[ContentType, ResultType](content=content)
        return self.add_child(branch)

    def add_parallel_children(
        self,
        children: list[
            Union[Leaf[ContentType, ResultType], "Branch[ContentType, ResultType]"]
        ],
    ) -> list[Union[Leaf[ContentType, ResultType], "Branch[ContentType, ResultType]"]]:
        """Add multiple children that represent parallel execution."""
        base_index = self._next_child_index

        for i, child in enumerate(children):
            child._index = base_index  # Same base index for parallel
            child._parent_index = self._index
            child._depth = self._depth + 1
            child._path = self._path + [base_index, i]  # Extra index for parallel
            child.metadata["parallel_group"] = base_index
            child.metadata["parallel_index"] = i
            self.children.append(child)

        self._next_child_index = base_index + 1
        return children

    # Computed properties for tree analysis

    @computed_field
    @property
    def total_nodes(self) -> int:
        """Total number of nodes in this subtree."""
        count = 1  # This node
        for child in self.children:
            if isinstance(child, Branch):
                count += child.total_nodes
            else:
                count += 1
        return count

    @computed_field
    @property
    def leaf_count(self) -> int:
        """Number of leaf nodes in this subtree."""
        count = 0
        for child in self.children:
            if isinstance(child, Branch):
                count += child.leaf_count
            else:
                count += 1
        return count

    @computed_field
    @property
    def completed_count(self) -> int:
        """Number of completed nodes."""
        count = 0
        if self.status == NodeStatus.COMPLETED:
            count += 1

        for child in self.children:
            if isinstance(child, Branch):
                count += child.completed_count
            elif child.status == NodeStatus.COMPLETED:
                count += 1
        return count

    @computed_field
    @property
    def failed_count(self) -> int:
        """Number of failed nodes."""
        count = 0
        if self.status == NodeStatus.FAILED:
            count += 1

        for child in self.children:
            if isinstance(child, Branch):
                count += child.failed_count
            elif child.status == NodeStatus.FAILED:
                count += 1
        return count

    @computed_field
    @property
    def progress_percentage(self) -> float:
        """Percentage of completion (0-100)."""
        total = self.total_nodes
        if total == 0:
            return 0.0
        return (self.completed_count / total) * 100

    @computed_field
    @property
    def current_active_nodes(
        self,
    ) -> list[Union[Leaf[ContentType, ResultType], "Branch[ContentType, ResultType]"]]:
        """Get all nodes currently in progress."""
        active = []

        if self.status == NodeStatus.IN_PROGRESS:
            active.append(self)

        for child in self.children:
            if isinstance(child, Branch):
                active.extend(child.current_active_nodes)
            elif child.status == NodeStatus.IN_PROGRESS:
                active.append(child)

        return active

    @computed_field
    @property
    def next_pending_leaf(self) -> Leaf[ContentType, ResultType] | None:
        """Get the next pending leaf node (depth-first)."""
        for child in self.children:
            if isinstance(child, Leaf) and child.status == NodeStatus.PENDING:
                return child
            elif isinstance(child, Branch):
                next_leaf = child.next_pending_leaf
                if next_leaf:
                    return next_leaf
        return None

    @computed_field
    @property
    def is_complete(self) -> bool:
        """Whether all nodes in this subtree are complete."""
        if self.status != NodeStatus.COMPLETED:
            return False

        for child in self.children:
            if not child.is_complete:
                return False

        return True

    @computed_field
    @property
    def has_failures(self) -> bool:
        """Whether any node in this subtree has failed."""
        return self.failed_count > 0

    def get_all_leaves(self) -> list[Leaf[ContentType, ResultType]]:
        """Get all leaf nodes in this subtree."""
        leaves = []
        for child in self.children:
            if isinstance(child, Branch):
                leaves.extend(child.get_all_leaves())
            else:
                leaves.append(child)
        return leaves

    def find_node_by_path(
        self, path: list[int]
    ) -> Union[Leaf[ContentType, ResultType], "Branch[ContentType, ResultType]"] | None:
        """Find a node by its path indices."""
        if not path:
            return self

        if path[0] >= len(self.children):
            return None

        child = self.children[path[0]]
        if len(path) == 1:
            return child
        elif isinstance(child, Branch):
            return child.find_node_by_path(path[1:])
        else:
            return None


# Enable forward references
Branch.model_rebuild()
