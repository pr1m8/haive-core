# src/haive/core/persistence/handlers.py

"""
Persistence handling utilities for checkpointing.

This module provides high-level functions for managing checkpoint operations,
including creation, recovery, and thread management.
"""
import json 
from pydantic import BaseModel
import logging
import asyncio
import uuid
from typing import Any, Optional, Dict, Union, Type, List, Tuple

from langchain_core.runnables import RunnableConfig

from .types import CheckpointerType, CheckpointStorageMode, CheckpointerMode
from .base import CheckpointerConfig
from .memory import MemoryCheckpointerConfig
from .postgres_config import PostgresCheckpointerConfig

logger = logging.getLogger(__name__)

def setup_checkpointer(config: Any) -> Any:
    """
    Set up the appropriate checkpointer based on persistence configuration.
    
    Args:
        config: Configuration containing persistence settings
        
    Returns:
        A configured checkpointer instance
    """
    # Default to memory checkpointer
    if not hasattr(config, 'persistence') or config.persistence is None:
        logger.info(f"No persistence config for {getattr(config, 'name', 'unnamed')}. Using memory checkpointer.")
        memory_config = MemoryCheckpointerConfig()
        return memory_config.create_checkpointer()
    
    # Handle already created checkpointer
    if hasattr(config.persistence, 'create_checkpointer'):
        # It's a CheckpointerConfig instance
        try:
            return config.persistence.create_checkpointer()
        except Exception as e:
            logger.error(f"Failed to create checkpointer: {e}")
            logger.warning(f"Falling back to memory checkpointer for {getattr(config, 'name', 'unnamed')}")
            memory_config = MemoryCheckpointerConfig()
            return memory_config.create_checkpointer()
    
    # Handle dictionary config
    if isinstance(config.persistence, dict):
        # Parse the persistence config
        persistence_type = config.persistence.get('type', 'memory')
        
        if persistence_type == 'memory':
            # Memory checkpointer
            memory_config = MemoryCheckpointerConfig()
            return memory_config.create_checkpointer()
        
        elif persistence_type == 'postgres':
            # PostgreSQL checkpointer
            try:
                # Get connection parameters
                use_shallow = config.persistence.get('shallow', False)
                use_async = config.persistence.get('async', False)
                
                # Create the config
                postgres_config = PostgresCheckpointerConfig(
                    checkpoint_mode=CheckpointStorageMode.shallow if use_shallow else CheckpointStorageMode.standard,
                    sync_mode=CheckpointerMode.async_ if use_async else CheckpointerMode.sync,
                    db_host=config.persistence.get('db_host', 'localhost'),
                    db_port=config.persistence.get('db_port', 5432),
                    db_name=config.persistence.get('db_name', 'postgres'),
                    db_user=config.persistence.get('db_user', 'postgres'),
                    db_pass=config.persistence.get('db_pass', 'postgres'),
                    ssl_mode=config.persistence.get('ssl_mode', 'disable'),
                    min_pool_size=config.persistence.get('min_pool_size', 1),
                    max_pool_size=config.persistence.get('max_pool_size', 5),
                    auto_commit=config.persistence.get('auto_commit', True),
                    prepare_threshold=config.persistence.get('prepare_threshold', 0),
                    setup_needed=config.persistence.get('setup_needed', True),
                    connection_string=config.persistence.get('connection_string'),
                    use_pipeline=config.persistence.get('use_pipeline', False)
                )
                
                return postgres_config.create_checkpointer()
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL checkpointer: {e}")
                logger.warning(f"Falling back to memory checkpointer for {getattr(config, 'name', 'unnamed')}")
                memory_config = MemoryCheckpointerConfig()
                return memory_config.create_checkpointer()
    
    # Default to memory checkpointer for any other case
    logger.info(f"Using memory checkpointer (default) for {getattr(config, 'name', 'unnamed')}")
    memory_config = MemoryCheckpointerConfig()
    return memory_config.create_checkpointer()

async def setup_async_checkpointer(config: Any) -> Any:
    """
    Set up the appropriate async checkpointer based on persistence configuration.
    
    Args:
        config: Configuration containing persistence settings
        
    Returns:
        A configured async checkpointer instance or context manager
    """
    # Default to memory checkpointer
    if not hasattr(config, 'persistence') or config.persistence is None:
        logger.info(f"No persistence config for {getattr(config, 'name', 'unnamed')}. Using memory checkpointer.")
        memory_config = MemoryCheckpointerConfig()
        return memory_config.create_checkpointer()
    
    # Handle already created checkpointer config
    if hasattr(config.persistence, 'is_async_mode') and callable(config.persistence.is_async_mode):
        # It's a CheckpointerConfig instance
        try:
            if config.persistence.is_async_mode():
                # For async modes, we need to initialize the checkpointer
                if hasattr(config.persistence, 'initialize_async_checkpointer'):
                    return await config.persistence.initialize_async_checkpointer()
                else:
                    # Fall back to sync mode
                    return config.persistence.create_checkpointer()
            else:
                # For sync modes, just create normally
                return config.persistence.create_checkpointer()
        except Exception as e:
            logger.error(f"Failed to create async checkpointer: {e}")
            logger.warning(f"Falling back to memory checkpointer for {getattr(config, 'name', 'unnamed')}")
            memory_config = MemoryCheckpointerConfig()
            return memory_config.create_checkpointer()
    
    # Handle dictionary config
    if isinstance(config.persistence, dict):
        # Parse the persistence config
        persistence_type = config.persistence.get('type', 'memory')
        
        if persistence_type == 'memory':
            # Memory checkpointer
            memory_config = MemoryCheckpointerConfig()
            return memory_config.create_checkpointer()
        
        elif persistence_type == 'postgres':
            # PostgreSQL checkpointer
            try:
                # Get connection parameters
                use_shallow = config.persistence.get('shallow', False)
                use_async = config.persistence.get('async', True)  # Default to async for async setup
                
                # Create the config
                postgres_config = PostgresCheckpointerConfig(
                    checkpoint_mode=CheckpointMode.shallow if use_shallow else CheckpointMode.standard,
                    sync_mode=SyncMode.async_ if use_async else SyncMode.sync,
                    db_host=config.persistence.get('db_host', 'localhost'),
                    db_port=config.persistence.get('db_port', 5432),
                    db_name=config.persistence.get('db_name', 'postgres'),
                    db_user=config.persistence.get('db_user', 'postgres'),
                    db_pass=config.persistence.get('db_pass', 'postgres'),
                    ssl_mode=config.persistence.get('ssl_mode', 'disable'),
                    min_pool_size=config.persistence.get('min_pool_size', 1),
                    max_pool_size=config.persistence.get('max_pool_size', 5),
                    auto_commit=config.persistence.get('auto_commit', True),
                    prepare_threshold=config.persistence.get('prepare_threshold', 0),
                    setup_needed=config.persistence.get('setup_needed', True),
                    connection_string=config.persistence.get('connection_string'),
                    use_pipeline=config.persistence.get('use_pipeline', False)
                )
                
                if use_async:
                    # For async mode, initialize the checkpointer
                    return await postgres_config.initialize_async_checkpointer()
                else:
                    # For sync mode, just create normally
                    return postgres_config.create_checkpointer()
            except Exception as e:
                logger.error(f"Failed to create async PostgreSQL checkpointer: {e}")
                logger.warning(f"Falling back to memory checkpointer for {getattr(config, 'name', 'unnamed')}")
                memory_config = MemoryCheckpointerConfig()
                return memory_config.create_checkpointer()
    
    # Default to memory checkpointer for any other case
    logger.info(f"Using memory checkpointer (default) for {getattr(config, 'name', 'unnamed')}")
    memory_config = MemoryCheckpointerConfig()
    return memory_config.create_checkpointer()

def ensure_pool_open(checkpointer: Any) -> Optional[Any]:
    """
    Ensure that any PostgreSQL connection pool is properly opened.
    
    This should be called before any operation that uses the checkpointer.
    
    Args:
        checkpointer: The checkpointer to check
        
    Returns:
        The opened pool if one was found and opened, None otherwise
    """
    opened_pool = None
    try:
        # Check for connection pools in the checkpointer
        if hasattr(checkpointer, 'conn'):
            conn = checkpointer.conn
            
            # Import here to avoid dependency issues
            try:
                from psycopg_pool.base import BaseConnectionPool
                
                # Check if it's a pool
                if isinstance(conn, BaseConnectionPool):
                    # Check if the pool is already open
                    try:
                        if hasattr(conn, 'is_open'):
                            is_open = conn.is_open()
                        else:
                            # Older versions might not have is_open()
                            is_open = getattr(conn, '_opened', False)
                        
                        # Open the pool if needed
                        if not is_open:
                            logger.info(f"Opening PostgreSQL connection pool")
                            try:
                                conn.open()
                                opened_pool = conn
                                logger.info(f"Successfully opened pool")
                            except Exception as e:
                                logger.error(f"Error opening pool: {e}")
                                
                                # Try a different approach with direct pool access
                                if hasattr(conn, '_pool'):
                                    logger.info("Trying alternative pool opening method")
                                    conn._pool = [] if not hasattr(conn, '_pool') or conn._pool is None else conn._pool
                                    conn._opened = True
                                    opened_pool = conn
                    except Exception as e:
                        logger.error(f"Error checking if pool is open: {e}")
                        # Last ditch effort - try direct attribute manipulation
                        if hasattr(conn, '_pool'):
                            conn._pool = [] if not hasattr(conn, '_pool') or conn._pool is None else conn._pool
                            conn._opened = True
                            opened_pool = conn
            except ImportError:
                logger.debug("psycopg_pool not available")
                
        # Additional check for other types of pools or connections
        if not opened_pool and hasattr(checkpointer, 'setup'):
            # If the checkpointer has a setup method but no connection was found,
            # just make sure tables are set up
            logger.debug("No pool found but checkpointer has setup method")
            try:
                checkpointer.setup()
            except Exception as e:
                logger.error(f"Error setting up checkpointer: {e}")
                
    except Exception as e:
        logger.error(f"Error ensuring pool is open: {e}")
        
    return opened_pool

async def ensure_async_pool_open(checkpointer: Any) -> Optional[Any]:
    """
    Ensure that any async PostgreSQL connection pool is properly opened.
    
    This should be called before any operation that uses the async checkpointer.
    
    Args:
        checkpointer: The checkpointer to check
        
    Returns:
        The opened pool if one was found and opened, None otherwise
    """
    opened_pool = None
    try:
        # Check for connection pools in the checkpointer
        if hasattr(checkpointer, 'conn'):
            conn = checkpointer.conn
            
            # Import here to avoid dependency issues
            try:
                from psycopg_pool.base import BaseConnectionPool
                
                # Check if it's a pool
                if isinstance(conn, BaseConnectionPool):
                    # Check if the pool is already open
                    try:
                        if hasattr(conn, 'is_open'):
                            is_open = await conn.is_open()
                        else:
                            # Older versions might not have is_open()
                            is_open = getattr(conn, '_opened', False)
                        
                        # Open the pool if needed
                        if not is_open:
                            logger.info(f"Opening async PostgreSQL connection pool")
                            try:
                                await conn.open()
                                opened_pool = conn
                                logger.info(f"Successfully opened async pool")
                            except Exception as e:
                                logger.error(f"Error opening async pool: {e}")
                                
                                # Try a different approach with direct pool access
                                if hasattr(conn, '_pool'):
                                    logger.info("Trying alternative pool opening method")
                                    conn._pool = [] if not hasattr(conn, '_pool') or conn._pool is None else conn._pool
                                    conn._opened = True
                                    opened_pool = conn
                    except Exception as e:
                        logger.error(f"Error checking if async pool is open: {e}")
                        # Last ditch effort - try direct attribute manipulation
                        if hasattr(conn, '_pool'):
                            conn._pool = [] if not hasattr(conn, '_pool') or conn._pool is None else conn._pool
                            conn._opened = True
                            opened_pool = conn
            except ImportError:
                logger.debug("psycopg_pool not available")
                
        # Additional check for other types of pools or connections
        if not opened_pool and hasattr(checkpointer, 'setup'):
            # If the checkpointer has a setup method but no connection was found,
            # just make sure tables are set up
            logger.debug("No pool found but checkpointer has setup method")
            try:
                await checkpointer.setup()
            except Exception as e:
                logger.error(f"Error setting up async checkpointer: {e}")
                
    except Exception as e:
        logger.error(f"Error ensuring async pool is open: {e}")
        
    return opened_pool

def close_pool_if_needed(checkpointer: Any, pool: Any = None) -> None:
    """
    Close a PostgreSQL connection pool if it was previously opened.
    
    This should be called in finally blocks after operations.
    
    Args:
        checkpointer: The checkpointer to check
        pool: The pool to close. If None, will try to find the pool 
            from the checkpointer.
    """
    if pool is None:
        # Try to find a pool from the checkpointer
        try:
            if hasattr(checkpointer, 'conn'):
                pool = checkpointer.conn
        except AttributeError:
            return
            
    # Close the pool if it's a ConnectionPool
    try:
        from psycopg_pool.pool import ConnectionPool
        if isinstance(pool, ConnectionPool) and pool.is_open():
            logger.debug(f"Closing PostgreSQL connection pool")
            # We don't actually close the pool - generally not recommended
            # unless you're sure you won't need it again
            # pool.close()
    except (ImportError, AttributeError):
        pass

async def close_async_pool_if_needed(checkpointer: Any, pool: Any = None) -> None:
    """
    Close an async PostgreSQL connection pool if it was previously opened.
    
    This should be called in finally blocks after operations.
    
    Args:
        checkpointer: The checkpointer to check
        pool: The pool to close. If None, will try to find the pool 
            from the checkpointer.
    """
    if pool is None:
        # Try to find a pool from the checkpointer
        try:
            if hasattr(checkpointer, 'conn'):
                pool = checkpointer.conn
        except AttributeError:
            return
            
    # Close the pool if it's an AsyncConnectionPool
    try:
        from psycopg_pool.pool import AsyncConnectionPool
        if isinstance(pool, AsyncConnectionPool) and await pool.is_open():
            logger.debug(f"Closing async PostgreSQL connection pool")
            # Similarly, we don't actually close the pool
            # await pool.close()
    except (ImportError, AttributeError):
        pass

def register_thread_if_needed(checkpointer: Any, thread_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Register a thread in the persistence system if needed.
    
    Args:
        checkpointer: The checkpointer to use
        thread_id: Thread ID to register
        metadata: Optional metadata to associate with the thread
    """
    # Skip for memory checkpointers
    if hasattr(checkpointer, '__class__') and checkpointer.__class__.__name__ == 'MemorySaver':
        return
        
    # Handle PostgreSQL checkpointers
    if hasattr(checkpointer, 'conn'):
        try:
            pool = checkpointer.conn
            if pool:
                # Ensure connection pool is usable
                pool_opened = ensure_pool_open(checkpointer)
                
                # Register the thread
                with pool.connection() as conn:
                    with conn.cursor() as cursor:
                        # Check if threads table exists
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'threads'
                            );
                        """)
                        table_exists = cursor.fetchone()[0]
                        
                        if not table_exists:
                            logger.debug("Creating threads table")
                            cursor.execute("""
                                CREATE TABLE IF NOT EXISTS threads (
                                    thread_id TEXT PRIMARY KEY,
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    metadata JSONB DEFAULT '{}'::jsonb,
                                    user_id TEXT,
                                    last_access TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                                );
                            """)
                            
                        # Convert metadata to JSON string if provided
                        metadata_json = '{}'
                        if metadata:
                            metadata_json = json.dumps(metadata)
                            
                        # Insert the thread if not exists
                        cursor.execute("""
                            INSERT INTO threads (thread_id, metadata, last_access) 
                            VALUES (%s, %s, CURRENT_TIMESTAMP) 
                            ON CONFLICT (thread_id) 
                            DO UPDATE SET last_access = CURRENT_TIMESTAMP
                        """, (thread_id, metadata_json))
                        
                        logger.info(f"Thread {thread_id} registered/updated in PostgreSQL")
        except Exception as e:
            logger.warning(f"Error registering thread: {e}")

async def register_async_thread_if_needed(checkpointer: Any, thread_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Register a thread in the persistence system asynchronously if needed.
    
    Args:
        checkpointer: The checkpointer to use
        thread_id: Thread ID to register
        metadata: Optional metadata to associate with the thread
    """
    # Skip for memory checkpointers
    if hasattr(checkpointer, '__class__') and checkpointer.__class__.__name__ == 'MemorySaver':
        return
        
    # Handle async PostgreSQL checkpointers
    if hasattr(checkpointer, 'conn'):
        try:
            pool = checkpointer.conn
            if pool:
                # Ensure connection pool is usable
                pool_opened = await ensure_async_pool_open(checkpointer)
                
                # Register the thread
                async with pool.connection() as conn:
                    async with conn.cursor() as cursor:
                        # Check if threads table exists
                        await cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'threads'
                            );
                        """)
                        table_exists = (await cursor.fetchone())[0]
                        
                        if not table_exists:
                            logger.debug("Creating threads table")
                            await cursor.execute("""
                                CREATE TABLE IF NOT EXISTS threads (
                                    thread_id TEXT PRIMARY KEY,
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    metadata JSONB DEFAULT '{}'::jsonb,
                                    user_id TEXT,
                                    last_access TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                                );
                            """)
                            
                        # Convert metadata to JSON string if provided
                        metadata_json = '{}'
                        if metadata:
                            metadata_json = json.dumps(metadata)
                            
                        # Insert the thread if not exists
                        await cursor.execute("""
                            INSERT INTO threads (thread_id, metadata, last_access) 
                            VALUES (%s, %s, CURRENT_TIMESTAMP) 
                            ON CONFLICT (thread_id) 
                            DO UPDATE SET last_access = CURRENT_TIMESTAMP
                        """, (thread_id, metadata_json))
                        
                        logger.info(f"Thread {thread_id} registered/updated asynchronously in PostgreSQL")
        except Exception as e:
            logger.warning(f"Error registering thread asynchronously: {e}")

def prepare_merged_input(
    input_data: Union[str, List[str], Dict[str, Any], BaseModel],
    previous_state: Optional[Any] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    input_schema=None,
    state_schema=None
) -> Any:
    """
    Process input data and merge with previous state if available.
    
    Args:
        input_data: Input data in various formats
        previous_state: Previous state from checkpointer
        runtime_config: Runtime configuration
        input_schema: Schema for input validation
        state_schema: Schema for state validation
        
    Returns:
        Processed input data merged with previous state
    """
    # Process the input
    processed_input = input_data
    
    # Return as is if no previous state
    if not previous_state:
        return processed_input
    
    # Extract values from StateSnapshot if needed
    previous_values = None
    
    if hasattr(previous_state, 'values'):
        # For StateSnapshot objects
        previous_values = previous_state.values
    elif hasattr(previous_state, 'channel_values') and previous_state.channel_values:
        # Alternative attribute name
        previous_values = previous_state.channel_values
    elif isinstance(previous_state, dict):
        # Dictionary state
        previous_values = previous_state
        
    # Return processed input if no valid previous values
    if not previous_values:
        return processed_input
    
    # Special handling for merging different input types
    if isinstance(processed_input, dict) and isinstance(previous_values, dict):
        # Merge dictionaries
        merged_input = dict(previous_values)
        
        # Update with new input values
        for key, value in processed_input.items():
            # Special case for messages - append rather than replace
            if key == "messages" and key in previous_values:
                # Start with all previous messages
                merged_messages = list(previous_values["messages"])
                
                # Add new messages if any
                if key in processed_input:
                    new_messages = processed_input[key]
                    if isinstance(new_messages, list):
                        merged_messages.extend(new_messages)
                    else:
                        merged_messages.append(new_messages)
                
                # Update merged_input with merged messages
                merged_input["messages"] = merged_messages
            else:
                # For other keys, just override
                merged_input[key] = value
        
        # Handle shared fields and reducers if using state_schema
        if state_schema:
            # Apply shared field values
            if hasattr(state_schema, '__shared_fields__'):
                for field in state_schema.__shared_fields__:
                    if field in previous_values and field not in processed_input:
                        merged_input[field] = previous_values[field]
            
            # Apply reducer functions
            if hasattr(state_schema, '__reducer_fields__'):
                for field, reducer in state_schema.__reducer_fields__.items():
                    if field in merged_input and field in previous_values:
                        try:
                            merged_input[field] = reducer(previous_values[field], merged_input[field])
                        except Exception as e:
                            logger.warning(f"Reducer for {field} failed: {e}")
        
        # Validate against schema if available
        if state_schema:
            try:
                validated = state_schema(**merged_input)
                # Convert back to dict
                if hasattr(validated, 'model_dump'):
                    return validated.model_dump()
                elif hasattr(validated, 'dict'):
                    return validated.dict()
                return validated
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")
        
        return merged_input
    
    # Return unmodified input as fallback
    return processed_input

def get_thread_id_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Extract thread_id from a RunnableConfig.
    
    Args:
        config: Configuration to extract from
        
    Returns:
        Thread ID if found, None otherwise
    """
    if not config:
        return None
        
    if isinstance(config, dict) and "configurable" in config:
        return config["configurable"].get("thread_id")
        
    return None