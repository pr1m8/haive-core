"""
Type definitions for the Branch system.
"""

from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from pydantic import BaseModel, ConfigDict, Field

# Type variables for improved type checking
StateLike = TypeVar("StateLike")
ConfigLike = TypeVar("ConfigLike")
BranchResult = TypeVar("BranchResult", str, Send, List[Send], Command)


class ComparisonType(str, Enum):
    """Comparison operations supported by Branch."""

    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUALS = ">="
    LESS_EQUALS = "<="
    IN = "in"
    CONTAINS = "contains"
    IS = "is"
    IS_NOT = "is not"
    EXISTS = "exists"
    NOT_EXISTS = "not exists"
    MATCHES = "matches"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    HAS_LENGTH = "has_length"
    MESSAGE_CONTAINS = "message_contains"


class BranchMode(str, Enum):
    """Branch evaluation modes."""

    DIRECT = "direct"  # Direct key/value comparison
    FUNCTION = "function"  # Uses a function to evaluate
    CHAIN = "chain"  # Chain of branches
    CONDITION = "condition"  # Conditional branch
    SEND_MAPPER = "send_mapper"  # Maps to Send objects
    DYNAMIC = "dynamic"  # Dynamic output based on state


class BranchCallable(Protocol, Generic[StateLike, ConfigLike]):
    """Protocol for branch callables."""

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> BranchResult: ...


class BranchResultModel(BaseModel):
    """Structured result from a branch evaluation."""

    next_node: Optional[str] = None
    send_objects: List[Send] = Field(default_factory=list)
    command_object: Optional[Command] = None
    output_mapping: Optional[Dict[str, str]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def is_send(self) -> bool:
        """Whether the result contains Send objects."""
        return len(self.send_objects) > 0

    @property
    def is_command(self) -> bool:
        """Whether the result contains a Command object."""
        return self.command_object is not None

    @property
    def has_mapping(self) -> bool:
        """Whether the result contains output mapping."""
        return self.output_mapping is not None
