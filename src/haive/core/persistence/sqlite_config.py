# src/haive/core/engine/agent/persistence/sqlite_config.py

"""
SQLite-based checkpointer for agent state persistence.

This module provides a SQLite-based implementation of the checkpointer
interface, suitable for local development and testing.
"""

import os
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from .types import CheckpointerType
from .base import CheckpointerConfig
from .utils import serialize_metadata, deserialize_metadata

logger = logging.getLogger(__name__)

class SQLiteSaver:
    """
    A checkpointer implementation using SQLite.
    
    This class provides a LangGraph-compatible checkpointer that stores
    state in a SQLite database.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the SQLite saver.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self.setup()
        
    def _ensure_db_dir(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def setup(self):
        """
        Set up the SQLite database schema.
        
        Creates the necessary tables for storing checkpoints and threads.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create threads table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_access TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
            """)
            
            # Create checkpoints table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT,
                thread_id TEXT,
                checkpoint_ns TEXT DEFAULT '',
                parent_checkpoint_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                data TEXT,
                metadata TEXT DEFAULT '{}',
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id),
                FOREIGN KEY (thread_id) REFERENCES threads (thread_id) ON DELETE CASCADE
            )
            """)
            
            conn.commit()
    
    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get a checkpoint from the database.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            
        Returns:
            Checkpoint data if found, None otherwise
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if checkpoint_id:
                # Get specific checkpoint
                cursor.execute(
                    """
                    SELECT data FROM checkpoints 
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id)
                )
            else:
                # Get latest checkpoint
                cursor.execute(
                    """
                    SELECT data FROM checkpoints 
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (thread_id, checkpoint_ns)
                )
            
            row = cursor.fetchone()
            
            if row:
                return json.loads(row["data"])
            
            return None
    
    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None, new_versions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save a checkpoint to the database.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            checkpoint: The checkpoint data to save
            metadata: Optional metadata to associate with the checkpoint
            new_versions: Optional channel versions (ignored in SQLite implementation)
            
        Returns:
            Updated config with checkpoint_id
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        
        # Generate a checkpoint ID if not present in the data
        checkpoint_id = checkpoint.get("id")
        if not checkpoint_id:
            checkpoint_id = str(uuid.uuid4())
            checkpoint["id"] = checkpoint_id
        
        # Serialize data and metadata
        serialized_data = json.dumps(checkpoint)
        serialized_metadata = serialize_metadata(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Ensure thread exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO threads (thread_id, last_access)
                VALUES (?, CURRENT_TIMESTAMP)
                """,
                (thread_id,)
            )
            
            # Update last access time if thread already exists
            cursor.execute(
                """
                UPDATE threads SET last_access = CURRENT_TIMESTAMP
                WHERE thread_id = ?
                """,
                (thread_id,)
            )
            
            # Insert checkpoint
            cursor.execute(
                """
                INSERT INTO checkpoints 
                (checkpoint_id, thread_id, checkpoint_ns, parent_checkpoint_id, data, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (checkpoint_id, thread_id, checkpoint_ns, parent_checkpoint_id, 
                 serialized_data, serialized_metadata)
            )
            
            conn.commit()
        
        # Return updated config
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id
            }
        }
    
    def list(self, config: Dict[str, Any], limit: Optional[int] = None, filter: Optional[Dict[str, Any]] = None, before: Optional[Dict[str, Any]] = None) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        List checkpoints for a thread.
        
        Args:
            config: Configuration with thread_id
            limit: Optional maximum number of checkpoints to return
            filter: Optional filter conditions (not implemented)
            before: Optional checkpoint to start listing from (not implemented)
            
        Returns:
            List of (config, checkpoint) tuples
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
            SELECT checkpoint_id, parent_checkpoint_id, data, metadata
            FROM checkpoints 
            WHERE thread_id = ? AND checkpoint_ns = ?
            ORDER BY created_at DESC
            """
            
            params = [thread_id, checkpoint_ns]
            
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                checkpoint_data = json.loads(row["data"])
                checkpoint_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row["checkpoint_id"]
                    }
                }
                
                # Create parent config if available
                parent_config = None
                if row["parent_checkpoint_id"]:
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row["parent_checkpoint_id"]
                        }
                    }
                
                # Use a namedtuple-like structure to match LangGraph's API
                class CheckpointTuple:
                    def __init__(self, config, checkpoint, metadata, parent_config, writes=None):
                        self.config = config
                        self.checkpoint = checkpoint
                        self.metadata = metadata
                        self.parent_config = parent_config
                        self.writes = writes or []
                
                metadata = deserialize_metadata(row["metadata"])
                
                result.append(
                    CheckpointTuple(
                        checkpoint_config, 
                        checkpoint_data, 
                        metadata,
                        parent_config
                    )
                )
            
            return result

class SQLiteCheckpointerConfig(CheckpointerConfig):
    """
    Configuration for SQLite-based checkpointing.
    
    This class provides a configuration for using SQLite as a persistence
    backend for LangGraph state.
    """
    type: CheckpointerType = CheckpointerType.sqlite
    
    # SQLite configuration
    db_path: str = Field(
        default="./checkpoints.db", 
        description="Path to SQLite database file"
    )
    
    # Runtime settings
    setup_needed: bool = Field(
        default=True, 
        description="Whether to initialize DB tables on startup"
    )
    
    # Internal state (not serialized)
    checkpointer: Optional[Any] = Field(default=None, exclude=True)
    
    def create_checkpointer(self) -> Any:
        """
        Create a SQLite checkpointer with the specified configuration.
        
        Returns:
            A SQLiteSaver instance for use with LangGraph
        """
        if self.checkpointer is None:
            self.checkpointer = SQLiteSaver(self.db_path)
            
        return self.checkpointer
    
    def register_thread(self, thread_id: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a thread in the SQLite database.
        
        Args:
            thread_id: The thread ID to register
            name: Optional thread name
            metadata: Optional metadata dict
        """
        checkpointer = self.create_checkpointer()
        
        # Convert metadata to JSON string if provided
        metadata_json = serialize_metadata(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if thread already exists
            cursor.execute(
                "SELECT 1 FROM threads WHERE thread_id = ?", 
                (thread_id,)
            )
            
            thread_exists = cursor.fetchone() is not None
            
            if not thread_exists:
                # Insert new thread
                cursor.execute(
                    """
                    INSERT INTO threads (thread_id, metadata) 
                    VALUES (?, ?)
                    """,
                    (thread_id, metadata_json)
                )
                logger.info(f"Thread {thread_id} registered in SQLite")
            else:
                # Update last access time
                cursor.execute(
                    """
                    UPDATE threads 
                    SET last_access = CURRENT_TIMESTAMP 
                    WHERE thread_id = ?
                    """,
                    (thread_id,)
                )
                logger.debug(f"Thread {thread_id} already exists in SQLite")
    
    def put_checkpoint(self, config: Dict[str, Any], data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a checkpoint in the SQLite database.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            data: The checkpoint data to store
            metadata: Optional metadata to associate with the checkpoint
            
        Returns:
            Updated config with checkpoint_id
        """
        checkpointer = self.create_checkpointer()
        
        # Structure the data as expected
        checkpoint_data = {
            "id": config["configurable"].get("checkpoint_id", ""),  # Will be auto-generated if empty
            "channel_values": data
        }
        
        # Store the checkpoint
        return checkpointer.put(config, checkpoint_data, metadata)
    
    def get_checkpoint(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a checkpoint from the SQLite database.
        
        Args:
            config: Configuration with thread_id and optional checkpoint_id
            
        Returns:
            The checkpoint data if found, None otherwise
        """
        checkpointer = self.create_checkpointer()
        
        # Get the checkpoint
        result = checkpointer.get(config)
        
        # Extract and return channel_values if available
        if result and "channel_values" in result:
            return result["channel_values"]
        
        return result
    
    def list_checkpoints(self, config: Dict[str, Any], limit: Optional[int] = None) -> List[Tuple[Dict[str, Any], Any]]:
        """
        List checkpoints for a thread.
        
        Args:
            config: Configuration with thread_id
            limit: Optional maximum number of checkpoints to return
            
        Returns:
            List of (config, checkpoint) tuples
        """
        checkpointer = self.create_checkpointer()
        
        # List checkpoints
        checkpoint_tuples = checkpointer.list(config, limit=limit)
        
        # Extract and return (config, channel_values) tuples
        result = []
        for cp in checkpoint_tuples:
            checkpoint_data = cp.checkpoint
            channel_values = checkpoint_data.get("channel_values", checkpoint_data)
            result.append((cp.config, channel_values))
            
        return result
    
    def close(self) -> None:
        """Close any resources associated with this checkpointer."""
        # SQLite connections are closed automatically after each operation
        self.checkpointer = None