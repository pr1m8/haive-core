"""Agent engine module for the Haive framework.

This module provides the core architecture for all agent implementations in Haive.
It delivers consistent schema handling, execution flows, persistence management,
and extensibility through patterns.

The agent engine implements protocol-based interfaces to ensure all agent
implementations conform to consistent APIs, while providing flexibility for
custom agent behaviors and execution patterns.

Key Components:
    Agent: Base agent class with graph-based execution
    AgentConfig: Comprehensive configuration for agents
    AgentProtocol: Protocol interface for agent implementations
    PersistenceManager: State persistence and checkpointing
    Pattern: Reusable agent implementation patterns

The agent engine integrates tightly with LangGraph for execution flow management
and provides built-in support for:
- Dynamic state schema management
- Streaming outputs
- State persistence
- Graph visualization
- Tool integration
- Custom execution patterns

Examples:
    Basic agent creation::

        from haive.core.engine.agent import Agent, AgentConfig
        from haive.core.engine.aug_llm import AugLLMConfig

        agent = Agent(
            name="assistant",
            engine=AugLLMConfig(model="gpt-4")
        )

        result = agent.invoke("Hello!")

    Agent with persistence::

        from haive.core.engine.agent import Agent, AgentConfig
        from haive.core.engine.agent.persistence import CheckpointerConfig

        config = AgentConfig(
            name="persistent_agent",
            engine=AugLLMConfig(model="gpt-4"),
            checkpointer_config=CheckpointerConfig(
                type="sqlite",
                connection_string="agent_state.db"
            )
        )

        agent = Agent(config)

See Also:
    - Agent Protocol documentation: protocols.py
    - Persistence system: persistence/
    - Agent patterns: pattern.py
    - Configuration guide: config.py
"""

from haive.core.engine.agent.agent import AGENT_REGISTRY, Agent
from haive.core.engine.agent.config import AgentConfig
from haive.core.engine.agent.pattern import PatternConfig, PatternManager
from haive.core.engine.agent.protocols import (
    AgentProtocol,
    PersistentAgentProtocol,
    StreamingAgentProtocol,
)

__all__ = [
    # Core Classes
    "Agent",
    "AgentConfig",
    "PatternConfig",
    "PatternManager",
    # Protocols
    "AgentProtocol",
    "StreamingAgentProtocol",
    "PersistentAgentProtocol",
    # Registry
    "AGENT_REGISTRY",
]
