# src/haive/core/engine/agent/persistence/utils.py

"""
Utilities for working with persistence backends.

This module provides helper functions for database operations
and state management across different persistence backends.
"""

import logging
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# Check if PostgreSQL is available
try:
    import psycopg
    from psycopg_pool import ConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

def generate_checkpoint_id() -> str:
    """
    Generate a unique checkpoint ID.
    
    Returns:
        A UUID string to use as checkpoint ID
    """
    return str(uuid.uuid4())

def serialize_metadata(metadata: Dict[str, Any]) -> str:
    """
    Serialize metadata to a JSON string.
    
    Args:
        metadata: Dictionary of metadata
        
    Returns:
        JSON string representation of metadata
    """
    def _serialize_item(item):
        """Helper to make items serializable."""
        if isinstance(item, (str, int, float, bool, type(None))):
            return item
        elif isinstance(item, dict):
            return {k: _serialize_item(v) for k, v in item.items()}
        elif isinstance(item, (list, tuple)):
            return [_serialize_item(i) for i in item]
        elif isinstance(item, datetime):
            return item.isoformat()
        elif isinstance(item, BaseModel):
            if hasattr(item, "model_dump"):
                return _serialize_item(item.model_dump())
            elif hasattr(item, "dict"):
                return _serialize_item(item.dict())
        # Default serialization
        return str(item)
    
    try:
        serializable_metadata = _serialize_item(metadata)
        return json.dumps(serializable_metadata)
    except Exception as e:
        logger.error(f"Error serializing metadata: {e}")
        # Return empty json object as fallback
        return "{}"

def deserialize_metadata(metadata_str: str) -> Dict[str, Any]:
    """
    Deserialize metadata from a JSON string.
    
    Args:
        metadata_str: JSON string representation of metadata
        
    Returns:
        Dictionary of metadata
    """
    try:
        if isinstance(metadata_str, dict):
            # Already deserialized
            return metadata_str
        return json.loads(metadata_str)
    except Exception as e:
        logger.error(f"Error deserializing metadata: {e}")
        return {}

def extract_thread_id(config: RunnableConfig) -> str:
    """
    Extract thread ID from a runnable config.
    
    Args:
        config: Runnable configuration
        
    Returns:
        Thread ID or a new UUID if not found
    """
    if not config:
        return str(uuid.uuid4())
    
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id")
    
    if not thread_id:
        thread_id = str(uuid.uuid4())
        
    return thread_id

def extract_checkpoint_id(config: RunnableConfig) -> Optional[str]:
    """
    Extract checkpoint ID from a runnable config.
    
    Args:
        config: Runnable configuration
        
    Returns:
        Checkpoint ID if found, None otherwise
    """
    if not config:
        return None
    
    configurable = config.get("configurable", {})
    return configurable.get("checkpoint_id")

def pg_execute_query(
    conn_string: str,
    query: str,
    params: Optional[Tuple] = None,
    fetch: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Execute a PostgreSQL query with connection management.
    
    Args:
        conn_string: PostgreSQL connection string
        query: SQL query to execute
        params: Query parameters
        fetch: Whether to fetch results
        
    Returns:
        Query results as a list of dictionaries if fetch=True, None otherwise
    """
    if not POSTGRES_AVAILABLE:
        logger.error("PostgreSQL dependencies not available")
        return None
    
    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
                cursor.execute(query, params or ())
                
                if fetch:
                    return cursor.fetchall()
                return None
    except Exception as e:
        logger.error(f"PostgreSQL query error: {e}")
        return None

def pg_list_threads(conn_string: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    List all threads in a PostgreSQL database.
    
    Args:
        conn_string: PostgreSQL connection string
        limit: Maximum number of threads to return
        
    Returns:
        List of thread dictionaries with metadata
    """
    query = """
    SELECT thread_id, created_at, last_access, metadata
    FROM threads
    ORDER BY last_access DESC
    LIMIT %s
    """
    
    results = pg_execute_query(conn_string, query, (limit,))
    if not results:
        return []
        
    # Process metadata
    for thread in results:
        if "metadata" in thread:
            thread["metadata"] = deserialize_metadata(thread["metadata"])
    
    return results

def pg_get_thread(conn_string: str, thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific thread from PostgreSQL.
    
    Args:
        conn_string: PostgreSQL connection string
        thread_id: Thread ID to retrieve
        
    Returns:
        Thread dictionary with metadata if found, None otherwise
    """
    query = """
    SELECT thread_id, created_at, last_access, metadata
    FROM threads
    WHERE thread_id = %s
    """
    
    results = pg_execute_query(conn_string, query, (thread_id,))
    if not results:
        return None
        
    thread = results[0]
    
    # Process metadata
    if "metadata" in thread:
        thread["metadata"] = deserialize_metadata(thread["metadata"])
    
    return thread

def pg_update_thread_metadata(
    conn_string: str,
    thread_id: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Update a thread's metadata in PostgreSQL.
    
    Args:
        conn_string: PostgreSQL connection string
        thread_id: Thread ID to update
        metadata: New metadata to set
        
    Returns:
        True if successful, False otherwise
    """
    query = """
    UPDATE threads
    SET metadata = %s, last_access = CURRENT_TIMESTAMP
    WHERE thread_id = %s
    """
    
    metadata_json = serialize_metadata(metadata)
    
    results = pg_execute_query(conn_string, query, (metadata_json, thread_id), fetch=False)
    return results is not None

def pg_delete_thread(conn_string: str, thread_id: str) -> bool:
    """
    Delete a thread and its checkpoints from PostgreSQL.
    
    Args:
        conn_string: PostgreSQL connection string
        thread_id: Thread ID to delete
        
    Returns:
        True if successful, False otherwise
    """
    # Delete checkpoints first due to foreign key constraints
    checkpoint_query = """
    DELETE FROM checkpoints
    WHERE thread_id = %s
    """
    
    thread_query = """
    DELETE FROM threads
    WHERE thread_id = %s
    """
    
    try:
        # Delete checkpoints
        pg_execute_query(conn_string, checkpoint_query, (thread_id,), fetch=False)
        
        # Delete thread
        pg_execute_query(conn_string, thread_query, (thread_id,), fetch=False)
        
        return True
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {e}")
        return False

def pg_list_checkpoints(
    conn_string: str,
    thread_id: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    List checkpoints for a thread in PostgreSQL.
    
    Args:
        conn_string: PostgreSQL connection string
        thread_id: Thread ID to list checkpoints for
        limit: Maximum number of checkpoints to return
        
    Returns:
        List of checkpoint dictionaries
    """
    query = """
    SELECT checkpoint_id, created_at, metadata
    FROM checkpoints
    WHERE thread_id = %s
    ORDER BY created_at DESC
    LIMIT %s
    """
    
    results = pg_execute_query(conn_string, query, (thread_id, limit))
    if not results:
        return []
        
    # Process metadata
    for checkpoint in results:
        if "metadata" in checkpoint:
            checkpoint["metadata"] = deserialize_metadata(checkpoint["metadata"])
    
    return results

def create_connection_string(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_pass: str,
    ssl_mode: str = "disable"
) -> str:
    """
    Create a PostgreSQL connection string from parameters.
    
    Args:
        db_host: Database host
        db_port: Database port
        db_name: Database name
        db_user: Database user
        db_pass: Database password
        ssl_mode: SSL mode
        
    Returns:
        Connection string for PostgreSQL
    """
    import urllib.parse
    encoded_pass = urllib.parse.quote_plus(str(db_pass))
    
    conn_string = f"postgresql://{db_user}:{encoded_pass}@{db_host}:{db_port}/{db_name}"
    
    if ssl_mode:
        conn_string += f"?sslmode={ssl_mode}"
        
    return conn_string