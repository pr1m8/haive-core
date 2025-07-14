"""Store tools for Haive agents.

This module provides LangChain-compatible tools that agents can use to
interact with the store system for memory management, similar to LangMem.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import Tool, tool
from pydantic import BaseModel, Field

from .store_manager import StoreManager

logger = logging.getLogger(__name__)


class StoreMemoryInput(BaseModel):
    """Input schema for storing memories."""

    content: str = Field(description="The memory content to store")
    category: str = Field(
        default="general",
        description="Category of the memory (e.g., user_preference, fact, event)",
    )
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Importance score from 0.0 to 1.0"
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Optional tags for the memory"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional additional metadata"
    )


class SearchMemoryInput(BaseModel):
    """Input schema for searching memories."""

    query: str = Field(description="Search query to find relevant memories")
    category: Optional[str] = Field(
        default=None, description="Filter by memory category"
    )
    min_importance: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum importance score"
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Required tags to filter by"
    )
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")


class RetrieveMemoryInput(BaseModel):
    """Input schema for retrieving specific memories."""

    memory_id: str = Field(description="The ID of the memory to retrieve")


class UpdateMemoryInput(BaseModel):
    """Input schema for updating memories."""

    memory_id: str = Field(description="The ID of the memory to update")
    content: Optional[str] = Field(
        default=None, description="New content for the memory"
    )
    category: Optional[str] = Field(
        default=None, description="New category for the memory"
    )
    importance: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="New importance score"
    )
    tags: Optional[List[str]] = Field(
        default=None, description="New tags for the memory"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata to merge"
    )


class DeleteMemoryInput(BaseModel):
    """Input schema for deleting memories."""

    memory_id: str = Field(description="The ID of the memory to delete")


def create_store_memory_tool(
    store_manager: StoreManager,
    namespace: Optional[Tuple[str, ...]] = None,
    tool_name: str = "store_memory",
) -> Tool:
    """Create a tool for storing memories.

    Args:
        store_manager: The store manager instance
        namespace: Optional namespace for operations
        tool_name: Name for the tool

    Returns:
        LangChain Tool for storing memories
    """

    @tool(tool_name, args_schema=StoreMemoryInput)
    def store_memory_func(
        content: str,
        category: str = "general",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store important information in memory for later retrieval. Use this to remember user preferences, facts, events, or any important information.

        Args:
            content: The memory content to store
            category: Category of memory (user_preference, fact, event, etc.)
            importance: How important this memory is (0.0 to 1.0)
            tags: Optional tags to help categorize the memory
            metadata: Optional additional metadata

        Returns:
            Memory ID of the stored memory
        """
        try:
            memory_id = store_manager.store_memory(
                content=content,
                category=category,
                importance=importance,
                tags=tags,
                metadata=metadata,
                namespace=namespace,
            )

            result = {
                "success": True,
                "memory_id": memory_id,
                "message": f"Successfully stored memory with ID: {memory_id}",
            }

            logger.debug(f"Stored memory: {memory_id}")
            return json.dumps(result)

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "message": "Failed to store memory",
            }
            logger.error(f"Failed to store memory: {e}")
            return json.dumps(error_result)

    return store_memory_func


def create_search_memory_tool(
    store_manager: StoreManager,
    namespace: Optional[Tuple[str, ...]] = None,
    tool_name: str = "search_memory",
) -> Tool:
    """Create a tool for searching memories.

    Args:
        store_manager: The store manager instance
        namespace: Optional namespace for operations
        tool_name: Name for the tool

    Returns:
        LangChain Tool for searching memories
    """

    @tool(tool_name, args_schema=SearchMemoryInput)
    def search_memory_func(
        query: str,
        category: Optional[str] = None,
        min_importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> str:
        """Search for relevant memories based on a query. Use this to recall information about users, facts, or past events.

        Args:
            query: Search query to find relevant memories
            category: Filter by memory category
            min_importance: Minimum importance score to filter by
            tags: Required tags to filter by
            limit: Maximum number of results to return

        Returns:
            JSON string with search results
        """
        try:
            memories = store_manager.search_memories(
                query=query,
                category=category,
                min_importance=min_importance,
                tags=tags,
                limit=limit,
                namespace=namespace,
            )

            results = []
            for memory in memories:
                results.append(
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "category": memory.category,
                        "importance": memory.importance,
                        "tags": memory.tags,
                        "created_at": memory.created_at.isoformat(),
                        "updated_at": memory.updated_at.isoformat(),
                    }
                )

            result = {
                "success": True,
                "memories": results,
                "count": len(results),
                "message": f"Found {len(results)} relevant memories",
            }

            logger.debug(f"Search returned {len(results)} memories for query: {query}")
            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "message": "Failed to search memories",
            }
            logger.error(f"Failed to search memories: {e}")
            return json.dumps(error_result)

    return search_memory_func


def create_retrieve_memory_tool(
    store_manager: StoreManager,
    namespace: Optional[Tuple[str, ...]] = None,
    tool_name: str = "retrieve_memory",
) -> Tool:
    """Create a tool for retrieving specific memories.

    Args:
        store_manager: The store manager instance
        namespace: Optional namespace for operations
        tool_name: Name for the tool

    Returns:
        LangChain Tool for retrieving memories
    """

    @tool(tool_name, args_schema=RetrieveMemoryInput)
    def retrieve_memory_func(memory_id: str) -> str:
        """Retrieve a specific memory by its ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            JSON string with the memory details
        """
        try:
            memory = store_manager.retrieve_memory(memory_id, namespace=namespace)

            if memory is None:
                result = {
                    "success": False,
                    "message": f"Memory with ID {memory_id} not found",
                }
            else:
                result = {
                    "success": True,
                    "memory": {
                        "id": memory.id,
                        "content": memory.content,
                        "category": memory.category,
                        "importance": memory.importance,
                        "tags": memory.tags,
                        "metadata": memory.metadata,
                        "created_at": memory.created_at.isoformat(),
                        "updated_at": memory.updated_at.isoformat(),
                    },
                    "message": f"Successfully retrieved memory {memory_id}",
                }

            logger.debug(f"Retrieved memory: {memory_id}")
            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "message": f"Failed to retrieve memory {memory_id}",
            }
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return json.dumps(error_result)

    return retrieve_memory_func


def create_update_memory_tool(
    store_manager: StoreManager,
    namespace: Optional[Tuple[str, ...]] = None,
    tool_name: str = "update_memory",
) -> Tool:
    """Create a tool for updating memories.

    Args:
        store_manager: The store manager instance
        namespace: Optional namespace for operations
        tool_name: Name for the tool

    Returns:
        LangChain Tool for updating memories
    """

    @tool(tool_name, args_schema=UpdateMemoryInput)
    def update_memory_func(
        memory_id: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Update an existing memory with new information.

        Args:
            memory_id: The ID of the memory to update
            content: New content for the memory
            category: New category for the memory
            importance: New importance score
            tags: New tags for the memory
            metadata: Additional metadata to merge

        Returns:
            JSON string with update status
        """
        try:
            success = store_manager.update_memory(
                memory_id=memory_id,
                content=content,
                category=category,
                importance=importance,
                tags=tags,
                metadata=metadata,
                namespace=namespace,
            )

            if success:
                result = {
                    "success": True,
                    "message": f"Successfully updated memory {memory_id}",
                }
            else:
                result = {
                    "success": False,
                    "message": f"Memory with ID {memory_id} not found",
                }

            logger.debug(f"Updated memory: {memory_id}, success: {success}")
            return json.dumps(result)

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "message": f"Failed to update memory {memory_id}",
            }
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return json.dumps(error_result)

    return update_memory_func


def create_delete_memory_tool(
    store_manager: StoreManager,
    namespace: Optional[Tuple[str, ...]] = None,
    tool_name: str = "delete_memory",
) -> Tool:
    """Create a tool for deleting memories.

    Args:
        store_manager: The store manager instance
        namespace: Optional namespace for operations
        tool_name: Name for the tool

    Returns:
        LangChain Tool for deleting memories
    """

    @tool(tool_name, args_schema=DeleteMemoryInput)
    def delete_memory_func(memory_id: str) -> str:
        """Delete a memory by its ID. Use with caution as this action cannot be undone.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            JSON string with deletion status
        """
        try:
            success = store_manager.delete_memory(memory_id, namespace=namespace)

            if success:
                result = {
                    "success": True,
                    "message": f"Successfully deleted memory {memory_id}",
                }
            else:
                result = {
                    "success": False,
                    "message": f"Memory with ID {memory_id} not found",
                }

            logger.debug(f"Deleted memory: {memory_id}, success: {success}")
            return json.dumps(result)

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "message": f"Failed to delete memory {memory_id}",
            }
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return json.dumps(error_result)

    return delete_memory_func


def create_memory_tools_suite(
    store_manager: StoreManager,
    namespace: Optional[Tuple[str, ...]] = None,
    include_tools: Optional[List[str]] = None,
) -> List[Tool]:
    """Create a complete suite of memory tools.

    Args:
        store_manager: The store manager instance
        namespace: Optional namespace for operations
        include_tools: Optional list of tools to include
                      (store, search, retrieve, update, delete)

    Returns:
        List of memory management tools
    """
    available_tools = {
        "store": lambda: create_store_memory_tool(store_manager, namespace),
        "search": lambda: create_search_memory_tool(store_manager, namespace),
        "retrieve": lambda: create_retrieve_memory_tool(store_manager, namespace),
        "update": lambda: create_update_memory_tool(store_manager, namespace),
        "delete": lambda: create_delete_memory_tool(store_manager, namespace),
    }

    if include_tools is None:
        include_tools = ["store", "search", "retrieve", "update", "delete"]

    tools = []
    for tool_name in include_tools:
        if tool_name in available_tools:
            tools.append(available_tools[tool_name]())
        else:
            logger.warning(f"Unknown tool requested: {tool_name}")

    logger.info(f"Created {len(tools)} memory tools: {include_tools}")
    return tools


# Convenience function similar to LangMem's API
def create_manage_memory_tool(
    store_manager: StoreManager, namespace: Optional[Tuple[str, ...]] = None
) -> Tool:
    """Create a manage memory tool (alias for store_memory_tool).

    This provides compatibility with LangMem-style naming.
    """
    return create_store_memory_tool(store_manager, namespace, "manage_memory")


def create_search_memory_tool_alias(
    store_manager: StoreManager, namespace: Optional[Tuple[str, ...]] = None
) -> Tool:
    """Create a search memory tool (alias for better naming consistency).

    This provides compatibility with LangMem-style naming.
    """
    return create_search_memory_tool(store_manager, namespace, "search_memory")
