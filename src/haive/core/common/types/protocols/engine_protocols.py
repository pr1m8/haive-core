"""Engine protocol definitions for type-safe engine interactions.

This module defines protocol interfaces for objects that work with engines in the Haive framework.
These protocols enable type-safe composition and dependency injection for engine-aware components.

The protocols in this module support:
- Engine-aware objects that maintain an engine reference
- Tool-aware objects that can work with LangChain tools
- Agent protocols for structured interactions

Example:
    Create an engine-aware component::

        class MyComponent(EngineAware):
            def __init__(self, engine: Engine):
                self.engine = engine

See Also:
    haive.core.engine.base: Core engine implementations
    haive.core.common.types.protocols.general_protocols: General protocol definitions
"""

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
