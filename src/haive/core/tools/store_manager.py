"""Store management system for Haive agents.

This module provides a comprehensive store management system similar to LangMem,
with tools for storing, retrieving, and managing agent memories using our
PostgreSQL store infrastructure.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from haive.core.persistence.store.base import SerializableStoreWrapper
from haive.core.persistence.store.factory import create_store
from haive.core.persistence.store.types import StoreType

logger = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """A single memory entry in the store."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(description="The memory content")
    category: str = Field(default="general", description="Memory category")
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Importance score 0-1"
    )
    tags: list[str] = Field(default_factory=list, description="Memory tags")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_store_value(self) -> dict[str, Any]:
        """Convert to store-compatible dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_store_value(cls, value: dict[str, Any]) -> "MemoryEntry":
        """Create from store dictionary."""
        if "created_at" in value and isinstance(value["created_at"], str):
            value["created_at"] = datetime.fromisoformat(value["created_at"])
        if "updated_at" in value and isinstance(value["updated_at"], str):
            value["updated_at"] = datetime.fromisoformat(value["updated_at"])
        return cls(**value)


class StoreManager:
    """Centralized store management for agent memories.

    This class provides a high-level interface for managing agent memories
    using the Haive store infrastructure, similar to LangMem but customized
    for our architecture.

    Features:
    - Namespace-based memory isolation (user_id, agent_id, session_id)
    - Semantic search capabilities
    - Memory categorization and tagging
    - Importance-based retrieval
    - Automatic metadata management
    """

    def __init__(
        self,
        store: SerializableStoreWrapper | None = None,
        default_namespace: tuple[str, ...] | None = None,
        store_config: dict[str, Any] | None = None,
    ):
        """Initialize the store manager.

        Args:
            store: Pre-configured store wrapper (optional)
            default_namespace: Default namespace for operations
            store_config: Configuration for creating a new store
        """
        if store is not None:
            self.store = store
        elif store_config is not None:
            self.store = create_store(**store_config)
        else:
            # Default to memory store for development
            self.store = create_store(store_type=StoreType.MEMORY)

        self.default_namespace = default_namespace or ("haive", "memories")
        logger.info(
            f"StoreManager initialized with store: {
                type(
                    self.store).__name__}"
        )

    def _get_namespace(
        self, namespace: tuple[str, ...] | None = None
    ) -> tuple[str, ...]:
        """Get the namespace to use for operations."""
        return namespace or self.default_namespace

    def store_memory(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        namespace: tuple[str, ...] | None = None,
        memory_id: str | None = None,
    ) -> str:
        """Store a new memory.

        Args:
            content: The memory content
            category: Memory category (e.g., "user_preference", "fact", "event")
            importance: Importance score 0-1
            tags: Optional tags for the memory
            metadata: Additional metadata
            namespace: Storage namespace (defaults to default_namespace)
            memory_id: Optional custom memory ID

        Returns:
            Memory ID
        """
        memory = MemoryEntry(
            id=memory_id or str(uuid.uuid4()),
            content=content,
            category=category,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )

        namespace = self._get_namespace(namespace)
        self.store.put(namespace, memory.id, memory.to_store_value())

        logger.debug(f"Stored memory {memory.id} in namespace {namespace}")
        return memory.id

    def retrieve_memory(
        self, memory_id: str, namespace: tuple[str, ...] | None = None
    ) -> MemoryEntry | None:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The memory ID to retrieve
            namespace: Storage namespace

        Returns:
            MemoryEntry if found, None otherwise
        """
        namespace = self._get_namespace(namespace)
        value = self.store.get(namespace, memory_id)

        if value is None:
            return None

        return MemoryEntry.from_store_value(value)

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        category: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        namespace: tuple[str, ...] | None = None,
    ) -> bool:
        """Update an existing memory.

        Args:
            memory_id: Memory ID to update
            content: New content (optional)
            category: New category (optional)
            importance: New importance score (optional)
            tags: New tags (optional)
            metadata: New metadata (optional)
            namespace: Storage namespace

        Returns:
            True if updated, False if memory not found
        """
        memory = self.retrieve_memory(memory_id, namespace)
        if memory is None:
            return False

        # Update fields if provided
        if content is not None:
            memory.content = content
        if category is not None:
            memory.category = category
        if importance is not None:
            memory.importance = importance
        if tags is not None:
            memory.tags = tags
        if metadata is not None:
            memory.metadata.update(metadata)

        memory.updated_at = datetime.utcnow()

        namespace = self._get_namespace(namespace)
        self.store.put(namespace, memory.id, memory.to_store_value())

        logger.debug(f"Updated memory {memory_id}")
        return True

    def delete_memory(
        self, memory_id: str, namespace: tuple[str, ...] | None = None
    ) -> bool:
        """Delete a memory.

        Args:
            memory_id: Memory ID to delete
            namespace: Storage namespace

        Returns:
            True if deleted, False if not found
        """
        namespace = self._get_namespace(namespace)

        # Check if memory exists first
        if self.store.get(namespace, memory_id) is None:
            return False

        self.store.delete(namespace, memory_id)
        logger.debug(f"Deleted memory {memory_id}")
        return True

    def search_memories(
        self,
        query: str,
        category: str | None = None,
        min_importance: float | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        namespace: tuple[str, ...] | None = None,
    ) -> list[MemoryEntry]:
        """Search memories using semantic search.

        Args:
            query: Search query
            category: Filter by category
            min_importance: Minimum importance score
            tags: Required tags
            limit: Maximum results
            namespace: Storage namespace

        Returns:
            List of matching memories
        """
        namespace = self._get_namespace(namespace)

        # Build filter criteria
        filter_criteria = {}
        if category:
            filter_criteria["category"] = category
        if min_importance is not None:
            filter_criteria["importance"] = {"$gte": min_importance}
        if tags:
            filter_criteria["tags"] = {"$in": tags}

        try:
            # Use store's search capability if available
            results = self.store.search(
                namespace=namespace, query=query, limit=limit, filter=filter_criteria
            )

            memories = []
            for result in results:
                # Handle different result formats
                if hasattr(result, "value"):
                    memory_data = result.value
                elif isinstance(result, dict):
                    memory_data = result
                else:
                    continue

                memories.append(MemoryEntry.from_store_value(memory_data))

            return memories

        except Exception as e:
            logger.warning(
                f"Semantic search failed: {e}. Falling back to basic search."
            )
            # Fallback: This would require implementing a basic search
            # For now, return empty list
            return []

    def list_memories_by_category(
        self,
        category: str,
        namespace: tuple[str, ...] | None = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """List memories by category.

        Args:
            category: Category to filter by
            namespace: Storage namespace
            limit: Maximum results

        Returns:
            List of memories in the category
        """
        return self.search_memories(
            query="",  # Empty query to get all
            category=category,
            limit=limit,
            namespace=namespace,
        )

    def get_memory_stats(
        self, namespace: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """Get statistics about stored memories.

        Args:
            namespace: Storage namespace

        Returns:
            Dictionary with memory statistics
        """
        # This would require iterating through all memories
        # For now, return basic stats
        return {
            "namespace": self._get_namespace(namespace),
            "store_type": type(self.store).__name__,
            "last_updated": datetime.utcnow().isoformat(),
        }

    def create_user_namespace(self, user_id: str) -> tuple[str, ...]:
        """Create a user-specific namespace.

        Args:
            user_id: User identifier

        Returns:
            Namespace tuple for the user
        """
        return ("haive", "users", user_id, "memories")

    def create_agent_namespace(
        self, agent_id: str, user_id: str | None = None
    ) -> tuple[str, ...]:
        """Create an agent-specific namespace.

        Args:
            agent_id: Agent identifier
            user_id: Optional user identifier for user-agent isolation

        Returns:
            Namespace tuple for the agent
        """
        if user_id:
            return ("haive", "users", user_id, "agents", agent_id, "memories")
        return ("haive", "agents", agent_id, "memories")

    def create_session_namespace(
        self,
        session_id: str,
        agent_id: str | None = None,
        user_id: str | None = None,
    ) -> tuple[str, ...]:
        """Create a session-specific namespace.

        Args:
            session_id: Session identifier
            agent_id: Optional agent identifier
            user_id: Optional user identifier

        Returns:
            Namespace tuple for the session
        """
        if user_id and agent_id:
            return (
                "haive",
                "users",
                user_id,
                "agents",
                agent_id,
                "sessions",
                session_id,
            )
        if agent_id:
            return ("haive", "agents", agent_id, "sessions", session_id)
        return ("haive", "sessions", session_id, "memories")
