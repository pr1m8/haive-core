"""Generic type definitions for tree structures.

Provides TypeVars with proper bounds and defaults for maximum flexibility.
"""

from typing import Any, TypeVar

from pydantic import BaseModel

# Basic content type - what each node contains
ContentT = TypeVar("ContentT", bound=BaseModel)

# Child type - what type of children a branch can have
# This allows mixed trees (e.g., branches containing both leaves and branches)
ChildT = TypeVar("ChildT", bound=BaseModel)

# Result type - what executing a node produces
ResultT = TypeVar("ResultT")


# Default types for common use cases
class DefaultContent(BaseModel):
    """Default content type with just a name/value."""

    name: str
    value: Any = None


class DefaultResult(BaseModel):
    """Default result type with status and data."""

    success: bool
    data: Any = None
    error: str | None = None


# Bounded TypeVars for specific use cases
PlanContentT = TypeVar("PlanContentT", bound="PlanContent")
TaskContentT = TypeVar("TaskContentT", bound="TaskContent")


# Forward references for planning use case
class PlanContent(BaseModel):
    """Content type for planning trees."""

    objective: str
    description: str = ""


class TaskContent(BaseModel):
    """Content type for task trees."""

    name: str
    action: str
    params: dict[str, Any] = {}
