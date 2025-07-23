from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools.base import BaseToolkit
from pydantic import BaseModel

from haive.core.engine.base import Engine

if TYPE_CHECKING:
    from haive.agents.base.agent import Agent

# ============================================================================
# Engine Protocol
# ============================================================================


class EngineAware(Protocol):
    """Protocol for objects that have an engine attribute."""

    engine: Engine | None


class ToolAware(Protocol):
    """Protocol for objects that have a tool attribute."""

    tools: Sequence[
        type[BaseTool]
        | type[BaseModel]
        | Callable
        | StructuredTool
        | BaseModel
        | BaseToolkit
    ]


class AgentAware(Protocol):
    """Protocol for objects that have an agent attribute."""

    agent: "Agent | None"
