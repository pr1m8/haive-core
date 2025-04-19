# src/haive/core/engine/agent/persistence/handlers.py

"""Persistence handling utilities for agent checkpointing.

This module provides functions for managing agent state persistence
through checkpointers, with special support for PostgreSQL.
"""

import inspect
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from haive_core.engine.agent.persistence.types import CheckpointerType

logger = logging.getLogger(__name__)

# Check if PostgreSQL dependencies are available
try:
    import ormsgpack
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True

    # Add custom serializer function for MessagePack
    def _custom_msgpack_default(obj):
        """Custom serializer for MessagePack that handles functions."""
        # Enhanced debugging
        logger.debug(f"Serializing object of type: {type(obj).__name__}")

        if callable(obj) and inspect.isfunction(obj):
            logger.info(f"Converting function to serializable form: {obj.__name__} from {obj.__module__}")
            # Convert function to a serializable representation
            return {
                "__function__": True,
                "name": obj.__name__,
                "module": obj.__module__
            }

        # Handle callable objects (methods, lambdas, etc.)
        if callable(obj):
            logger.info(f"Converting callable object to serializable form: {obj}")
            return {
                "__callable__": True,
                "type": str(type(obj).__name__),
                "repr": repr(obj)
            }

        # Handle other special types that might be in the state
        if hasattr(obj, "__dict__"):
            logger.info(f"Converting object with __dict__ to serializable form: {type(obj).__name__}")
            return {
                "__object__": True,
                "type": str(type(obj).__name__),
                "attributes": {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            }

        # Warn about unhandled types
        logger.warning(f"Unserializable object type: {type(obj).__name__}, repr: {repr(obj)[:100]}")
        # Return a placeholder for unserializable objects
        return {"__unserializable__": str(type(obj)), "repr": repr(obj)[:100]}

    # Patch the msgpack serializer in LangGraph if possible
    try:
        from langgraph.checkpoint.serde.jsonplus import _msgpack_default

        # Back up the original default function
        original_msgpack_default = _msgpack_default

        # Create enhanced version that falls back to our custom handler
        def enhanced_msgpack_default(obj):
            try:
                return original_msgpack_default(obj)
            except TypeError as e:
                logger.debug(f"Original serializer failed with: {e}, using custom serializer")
                return _custom_msgpack_default(obj)

        # Apply the patch
        from langgraph.checkpoint.serde import jsonplus
        jsonplus._msgpack_default = enhanced_msgpack_default
        logger.info("Enhanced MessagePack serialization with function support")

        # Also patch the packb function to provide more context
        original_packb = ormsgpack.packb

        def debug_packb(obj, default=None, option=None):
            try:
                return original_packb(obj, default=default, option=option)
            except TypeError as e:
                logger.error(f"MessagePack serialization error: {e}")
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            original_packb(v)
                        except TypeError:
                            logger.error(f"Problem key: {k}, value type: {type(v).__name__}")
                return original_packb(obj, default=default, option=option)

        # Override the packb function
        ormsgpack.packb = debug_packb
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not patch MessagePack serializer: {e}")

except ImportError:
    POSTGRES_AVAILABLE = False

def setup_checkpointer(config: Any) -> BaseCheckpointSaver:
    """Set up the appropriate checkpointer based on persistence configuration.
    
    Args:
        config: Agent configuration containing persistence settings
        
    Returns:
        A configured checkpointer instance
    """
    # Default to memory checkpointer

    # Handle no persistence config case
    if not hasattr(config, "persistence") or config.persistence is None:
        logger.info(f"No persistence config for {config.name}. Using memory checkpointer.")
        return MemorySaver()

    # Handle dictionary config
    if isinstance(config.persistence, dict):
        # Check for explicit memory type
        if config.persistence.get("type", "memory") == "memory":
            logger.info(f"Using memory checkpointer (per config) for {config.name}")
            return MemorySaver()

        # Only continue with postgres if it's explicitly requested and available
        if config.persistence.get("type") == "postgres" and POSTGRES_AVAILABLE:
            try:
                # Import only when needed
                from langgraph.checkpoint.postgres import PostgresSaver
                from psycopg_pool import ConnectionPool

                # Get connection parameters
                db_host = config.persistence.get("db_host", "localhost")
                db_port = config.persistence.get("db_port", 5432)
                db_name = config.persistence.get("db_name", "postgres")
                db_user = config.persistence.get("db_user", "postgres")
                db_pass = config.persistence.get("db_pass", "postgres")
                ssl_mode = config.persistence.get("ssl_mode", "disable")

                # Create connection URI
                import urllib.parse
                encoded_pass = urllib.parse.quote_plus(str(db_pass))
                db_uri = f"postgresql://{db_user}:{encoded_pass}@{db_host}:{db_port}/{db_name}"
                if ssl_mode:
                    db_uri += f"?sslmode={ssl_mode}"

                # Create connection pool
                pool = ConnectionPool(
                    conninfo=db_uri,
                    min_size=config.persistence.get("min_pool_size", 1),
                    max_size=config.persistence.get("max_pool_size", 5),
                    kwargs={
                        "autocommit": config.persistence.get("auto_commit", True),
                        "prepare_threshold": config.persistence.get("prepare_threshold", 0),
                    },
                    open=True  # Explicitly open the pool to avoid "not open yet" errors
                )

                # Create checkpointer
                checkpointer = PostgresSaver(pool)

                # Setup tables if needed
                if config.persistence.get("setup_needed", True):
                    try:
                        checkpointer.setup()
                    except Exception as e:
                        logger.warning(f"Error during initial setup: {e}")

                logger.info(f"Using PostgreSQL checkpointer for {config.name}")
                return checkpointer

            except Exception as e:
                logger.error(f"Failed to setup PostgreSQL checkpointer: {e}")
                logger.warning(f"Falling back to memory checkpointer for {config.name}")

    # Handle CheckpointerConfig objects
    elif hasattr(config.persistence, "type"):
        if config.persistence.type == CheckpointerType.memory:
            logger.info(f"Using memory checkpointer for {config.name}")
            return MemorySaver()

        if config.persistence.type == CheckpointerType.postgres and POSTGRES_AVAILABLE:
            try:
                # Try to use create_checkpointer method if available
                if hasattr(config.persistence, "create_checkpointer"):
                    checkpointer = config.persistence.create_checkpointer()

                    # Setup tables if needed
                    if getattr(config.persistence, "setup_needed", True):
                        try:
                            checkpointer.setup()
                        except Exception as e:
                            logger.warning(f"Error during initial setup: {e}")

                    logger.info(f"Using PostgreSQL checkpointer for {config.name}")
                    return checkpointer
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL checkpointer: {e}")

    # Default to memory checkpointer for any other case
    logger.info(f"Using memory checkpointer (default) for {config.name}")
    return MemorySaver()

def ensure_pool_open(checkpointer: Any) -> Any | None:
    """Ensure that any PostgreSQL connection pool is properly opened.
    
    This should be called before any operation that uses the checkpointer.
    
    Args:
        checkpointer: The checkpointer to check
        
    Returns:
        The opened pool if one was found and opened, None otherwise
    """
    opened_pool = None
    try:
        # Check for connection pools in the checkpointer
        if hasattr(checkpointer, "conn"):
            conn = checkpointer.conn

            # Import here to avoid dependency issues
            try:
                from psycopg_pool.base import BaseConnectionPool

                # Check if it's a pool
                if isinstance(conn, BaseConnectionPool):
                    # Check if the pool is already open
                    try:
                        if hasattr(conn, "is_open"):
                            is_open = conn.is_open()
                        else:
                            # Older versions might not have is_open()
                            is_open = getattr(conn, "_opened", False)

                        # Open the pool if needed
                        if not is_open:
                            logger.info("Opening PostgreSQL connection pool")
                            try:
                                conn.open()
                                opened_pool = conn
                                logger.info("Successfully opened pool")
                            except Exception as e:
                                logger.error(f"Error opening pool: {e}")

                                # Try a different approach with direct pool access
                                if hasattr(conn, "_pool"):
                                    logger.info("Trying alternative pool opening method")
                                    conn._pool = [] if not hasattr(conn, "_pool") or conn._pool is None else conn._pool
                                    conn._opened = True
                                    opened_pool = conn
                    except Exception as e:
                        logger.error(f"Error checking if pool is open: {e}")
                        # Last ditch effort - try direct attribute manipulation
                        if hasattr(conn, "_pool"):
                            conn._pool = [] if not hasattr(conn, "_pool") or conn._pool is None else conn._pool
                            conn._opened = True
                            opened_pool = conn
            except ImportError:
                logger.debug("psycopg_pool not available")

        # Additional check for other types of pools or connections
        if not opened_pool and hasattr(checkpointer, "setup"):
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

def close_pool_if_needed(checkpointer: Any, pool: Any = None) -> None:
    """Close a PostgreSQL connection pool if it was previously opened.
    
    This should be called in finally blocks after operations.
    
    Args:
        checkpointer: The checkpointer to check
        pool: The pool to close. If None, will try to find the pool 
            from the checkpointer.
    """
    if pool is None:
        # Try to find a pool from the checkpointer
        try:
            if hasattr(checkpointer, "conn"):
                pool = checkpointer.conn
        except AttributeError:
            return

    # Close the pool if it's a ConnectionPool
    try:
        from psycopg_pool.pool import ConnectionPool
        if isinstance(pool, ConnectionPool) and pool.is_open():
            logger.debug("Closing PostgreSQL connection pool")
            # We don't actually close the pool - generally not recommended
            # unless you're sure you won't need it again
            # pool.close()
    except (ImportError, AttributeError):
        pass

    # Close the pool if it's an AsyncConnectionPool
    try:
        from psycopg_pool.pool import AsyncConnectionPool
        if isinstance(pool, AsyncConnectionPool) and pool.is_open():
            logger.debug("Closing async PostgreSQL connection pool")
            # Similarly, we don't actually close the pool
            # import asyncio
            # try:
            #     asyncio.run(pool.close())
            # except RuntimeError:
            #     loop = asyncio.get_event_loop()
            #     task = loop.create_task(pool.close())
    except (ImportError, AttributeError):
        pass

def register_thread_if_needed(checkpointer: Any, thread_id: str) -> None:
    """Register a thread in the PostgreSQL database if needed.
    
    Args:
        checkpointer: The checkpointer to use
        thread_id: Thread ID to register
    """
    if hasattr(checkpointer, "conn"):
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
                                    thread_id VARCHAR(255) PRIMARY KEY,
                                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                );
                            """)

                        # Insert the thread if not exists
                        cursor.execute("""
                            INSERT INTO threads (thread_id) 
                            VALUES (%s) 
                            ON CONFLICT (thread_id) 
                            DO NOTHING
                        """, (thread_id,))

                        logger.info(f"Thread {thread_id} registered/updated in PostgreSQL")
        except Exception as e:
            logger.warning(f"Error registering thread: {e}")

def prepare_merged_input(
    input_data: str | list[str] | dict[str, Any] | BaseModel,
    previous_state: Any | None = None,
    runtime_config: dict[str, Any] | None = None,
    input_schema=None,
    state_schema=None
) -> Any:
    """Process input data and merge with previous state if available.
    
    Args:
        input_data: Input data in various formats
        previous_state: Previous state from checkpointer
        runtime_config: Runtime configuration
        input_schema: Schema for input validation
        state_schema: Schema for state validation
        
    Returns:
        Processed input data merged with previous state
    """
    # Process the input based on schema
    processed_input = process_input(input_data, input_schema)

    # Return as is if no previous state
    if not previous_state:
        return processed_input

    # Extract values from StateSnapshot if needed
    previous_values = None

    if hasattr(previous_state, "values"):
        # For StateSnapshot objects
        previous_values = previous_state.values
    elif hasattr(previous_state, "channel_values") and previous_state.channel_values:
        # Alternative attribute name
        previous_values = previous_state.channel_values
    elif isinstance(previous_state, dict):
        # Dictionary state
        previous_values = previous_state

    # Return processed input if no valid previous values
    if not previous_values:
        return processed_input

    # Merge with previous state

    # Special handling for messages to append rather than replace
    if "messages" in processed_input and "messages" in previous_values:
        # Start with all previous messages
        merged_messages = list(previous_values["messages"])

        # Add new messages
        new_messages = processed_input["messages"]
        merged_messages.extend(new_messages)

        # Update processed input with merged messages
        processed_input["messages"] = merged_messages

        # Log the message count
        logger.debug(f"Merged messages: {len(merged_messages)} total")

    # Merge other fields, keeping processed_input as priority
    merged_input = dict(previous_values)

    # Update with new input values
    for key, value in processed_input.items():
        merged_input[key] = value

    # Handle shared fields and reducers if using StateSchema
    if state_schema and hasattr(state_schema, "__shared_fields__"):
        for field in state_schema.__shared_fields__:
            if field in previous_values and field not in processed_input:
                merged_input[field] = previous_values[field]

    if state_schema and hasattr(state_schema, "__reducer_fields__"):
        for field, reducer in state_schema.__reducer_fields__.items():
            if field in merged_input and field in previous_values:
                try:
                    merged_input[field] = reducer(previous_values[field], merged_input[field])
                except Exception as e:
                    logger.warning(f"Reducer for {field} failed: {e}")

    # Validate against state schema if available
    if state_schema:
        try:
            return state_schema(**merged_input)
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")

    return merged_input

def process_input(
    input_data: str | list[str] | dict[str, Any] | BaseModel,
    input_schema=None
) -> dict[str, Any]:
    """Process input for the agent based on the input schema.
    
    Args:
        input_data: Input in various formats
        input_schema: Schema for validation
        
    Returns:
        Processed input compatible with the graph
    """
    # Extract schema field information if available
    schema_fields = {}
    if input_schema and hasattr(input_schema, "model_fields"):
        schema_fields = input_schema.model_fields

    # Handle string input
    if isinstance(input_data, str):
        # Initialize with messages
        prepared_input = {"messages": [HumanMessage(content=input_data)]}

        # Add to other input fields based on schema
        for field_name, field_info in schema_fields.items():
            if field_name != "messages" and field_name != "__runnable_config__":
                # Only add to text fields
                field_type = field_info.annotation
                type_name = str(field_type)
                if "str" in type_name or "String" in type_name:
                    prepared_input[field_name] = input_data

        # Validate against schema if available
        if input_schema:
            try:
                return input_schema(**prepared_input)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")

        return prepared_input

    # Handle list of strings
    if isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
        # Create messages list
        messages = [HumanMessage(content=item) for item in input_data]
        prepared_input = {"messages": messages}

        # Join strings for other text fields
        joined_text = "\n".join(input_data)
        for field_name, field_info in schema_fields.items():
            if field_name != "messages" and field_name != "__runnable_config__":
                # Only add to text fields
                field_type = field_info.annotation
                type_name = str(field_type)
                if "str" in type_name or "String" in type_name:
                    prepared_input[field_name] = joined_text

        # Validate against schema
        if input_schema:
            try:
                return input_schema(**prepared_input)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")

        return prepared_input

    # Handle dictionary input
    if isinstance(input_data, dict):
        # Create a copy to avoid modifying the original
        input_dict = input_data.copy()

        # Ensure there's a messages field if not present and required
        if "messages" not in input_dict and "messages" in schema_fields:
            # Try to create messages from other fields
            for field in ["input", "query", "content", "text"]:
                if field in input_dict and isinstance(input_dict[field], str):
                    input_dict["messages"] = [HumanMessage(content=input_dict[field])]
                    break

        # Validate against schema if available
        if input_schema:
            try:
                return input_schema(**input_dict)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")

        return input_dict

    # Handle Pydantic model input
    if isinstance(input_data, BaseModel):
        # Convert to dict
        if hasattr(input_data, "model_dump"):
            # Pydantic v2
            model_dict = input_data.model_dump()
        elif hasattr(input_data, "dict"):
            # Pydantic v1
            model_dict = input_data.dict()
        else:
            # Manual extraction
            model_dict = {}
            for field in input_data.__annotations__:
                if hasattr(input_data, field):
                    model_dict[field] = getattr(input_data, field)

        # Ensure there's a messages field if needed by schema
        if "messages" not in model_dict and "messages" in schema_fields:
            # Try to create messages from other fields
            for field in ["input", "query", "content", "text"]:
                if field in model_dict and isinstance(model_dict[field], str):
                    model_dict["messages"] = [HumanMessage(content=model_dict[field])]
                    break

        # Validate against schema if available
        if input_schema:
            try:
                return input_schema(**model_dict)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")

        return model_dict

    # Fallback for other types - convert to string message
    fallback_input = {
        "messages": [HumanMessage(content=str(input_data))]
    }

    # Validate against schema if available
    if input_schema:
        try:
            return input_schema(**fallback_input)
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")

    return fallback_input
