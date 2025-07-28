"""Integration engine module.

This module provides integration functionality for the Haive framework.

Functions:
    create_persistence_manager: Create Persistence Manager functionality.
    prepare_agent_run: Prepare Agent Run functionality.
    aprepare_agent_run: Aprepare Agent Run functionality.
"""

# src/haive/core/engine/agent/persistence/integration.py
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from haive.core.engine.agent.persistence.base import CheckpointerConfig
from haive.core.engine.agent.persistence.factory import load_checkpointer_config
from haive.core.engine.agent.persistence.manager import PersistenceManager

logger = logging.getLogger(__name__)


def create_persistence_manager(
    persistence_config: dict[str, Any] | CheckpointerConfig | None = None,
) -> PersistenceManager:
    """Create a PersistenceManager from a configuration.

    Args:
        persistence_config: Configuration for persistence

    Returns:
        Configured PersistenceManager
    """
    return PersistenceManager(persistence_config)


def prepare_agent_run(
    persistence_manager: PersistenceManager,
    thread_id: str | None = None,
    user_info: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[RunnableConfig, str]:
    """Prepare for an agent run with proper persistence setup.

    Args:
        persistence_manager: The persistence manager
        thread_id: Optional thread ID for persistence
        user_info: Optional user information dictionary
        **kwargs: Additional runtime configuration

    Returns:
        Tuple of (RunnableConfig, current_thread_id)
    """
    # Prepare for agent run with proper setup
    config, current_thread_id = persistence_manager.prepare_for_agent_run(
        thread_id=thread_id, user_info=user_info, **kwargs
    )

    logger.debug(f"Prepared for agent run with thread_id={current_thread_id}")
    return config, current_thread_id


async def aprepare_agent_run(
    persistence_manager: PersistenceManager,
    thread_id: str | None = None,
    user_info: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[RunnableConfig, str]:
    """Asynchronously prepare for an agent run with persistence setup.

    Args:
        persistence_manager: The persistence manager
        thread_id: Optional thread ID for persistence
        user_info: Optional user information dictionary
        **kwargs: Additional runtime configuration

    Returns:
        Tuple of (RunnableConfig, current_thread_id)
    """
    # Prepare for agent run asynchronously
    config, current_thread_id = await persistence_manager.aprepare_for_agent_run(
        thread_id=thread_id, user_info=user_info, **kwargs
    )

    logger.debug(
        f"Asynchronously prepared for agent run with thread_id={current_thread_id}"
    )
    return config, current_thread_id


def extract_persistence_config(agent_config: Any) -> CheckpointerConfig | None:
    """Extract persistence configuration from an agent configuration.

    Args:
        agent_config: Agent configuration

    Returns:
        Checkpointer configuration if found, None otherwise
    """
    if hasattr(agent_config, "persistence"):
        persistence = agent_config.persistence

        # Handle direct CheckpointerConfig
        if isinstance(persistence, CheckpointerConfig):
            return persistence

        # Handle dictionary
        if isinstance(persistence, dict):
            return load_checkpointer_config(persistence)

        # Try to convert from model
        if hasattr(persistence, "model_dump"):
            # Pydantic v2
            persistence_dict = persistence.model_dump()
            return load_checkpointer_config(persistence_dict)
        if hasattr(persistence, "dict"):
            # Pydantic v1
            persistence_dict = persistence.dict()
            return load_checkpointer_config(persistence_dict)

    return None
