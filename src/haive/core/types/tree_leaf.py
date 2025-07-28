"""Tree_Leaf core module.

This module provides tree leaf functionality for the Haive framework.

Classes:
    Leaf: Leaf implementation.
    Branch: Branch implementation.
    NodeMixin: NodeMixin implementation.

Functions:
    is_leaf: Is Leaf functionality.
    add: Add functionality.
    size: Size functionality.
"""

# tree_model.py
from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, computed_field

T = TypeVar("T")  # payload type


class Leaf(BaseModel, Generic[T]):
    kind: Literal["leaf"] = "leaf"  # discriminator
    value: T

    model_config = {"extra": "forbid"}


class Branch(BaseModel, Generic[T]):
    kind: Literal["branch"] = "branch"
    children: list[Node[T]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


# Self‑recursive union
Node = Leaf[T] | Branch[T]


# ───────────────────────── helper mixin (optional) ─────────────────────── #
class NodeMixin(BaseModel, Generic[T]):
    """Adds utility methods while preserving the union above."""

    @computed_field
    def is_leaf(self) -> bool:  # appears in .model_dump()
        return isinstance(self, Leaf)

    def add(self, *kids: Node[T]) -> None:
        if isinstance(self, Branch):
            self.children.extend(kids)
        else:
            raise TypeError("Can't add children to a leaf")

    def size(self) -> int:  # total nodes in subtree
        if isinstance(self, Leaf):
            return 1
        # type: ignore[attr-defined]
        return 1 + sum(child.size() for child in self.children)


# Re‑define aliases so NodeMixin is recognised
Leaf.__bases__ += (NodeMixin[T],)
Branch.__bases__ += (NodeMixin[T],)
