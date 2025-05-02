import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def ensure_pool_open(checkpointer: Any) -> Optional[Any]:
    """
    Ensure that any PostgreSQL connection pool is properly opened.
    """
    if not hasattr(checkpointer, "conn"):
        return None

    conn = checkpointer.conn

    try:
        from psycopg_pool import ConnectionPool

        if isinstance(conn, ConnectionPool):
            # Check if opened using the _opened attribute
            is_open = getattr(conn, "_opened", False)
            if not is_open:
                logger.info("Opening PostgreSQL connection pool")
                conn.open()
                logger.info("Pool opened successfully")
            return conn
    except ImportError:
        logger.debug("psycopg_pool not available")

    # Make sure setup is called
    if hasattr(checkpointer, "setup"):
        try:
            checkpointer.setup()
        except Exception as e:
            logger.error(f"Error setting up checkpointer: {e}")

    return None


async def ensure_async_pool_open(checkpointer: Any) -> Optional[Any]:
    """
    Ensure that any async PostgreSQL connection pool is properly opened.

    Args:
        checkpointer: The async checkpointer to check

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
                from psycopg_pool.base import AsyncPool

                # Check if it's an async pool
                if isinstance(conn, AsyncPool):
                    # Check if the pool is already open
                    try:
                        if hasattr(conn, "is_open"):
                            is_open = conn.is_open()
                        else:
                            # Older versions might not have is_open()
                            is_open = getattr(conn, "_opened", False)

                        # Open the pool if needed
                        if not is_open:
                            logger.info("Opening async PostgreSQL connection pool")
                            try:
                                await conn.open()
                                opened_pool = conn
                                logger.info("Successfully opened async pool")
                            except Exception as e:
                                logger.error(f"Error opening async pool: {e}")
                    except Exception as e:
                        logger.error(f"Error checking if async pool is open: {e}")
            except ImportError:
                logger.debug("psycopg_pool AsyncPool not available")

        # Additional check for other types of pools or connections
        if not opened_pool and hasattr(checkpointer, "setup"):
            # If the checkpointer has a setup method but no connection was found,
            # just make sure tables are set up
            logger.debug("No async pool found but checkpointer has setup method")
            try:
                await checkpointer.setup()
            except Exception as e:
                logger.error(f"Error setting up async checkpointer: {e}")

    except Exception as e:
        logger.error(f"Error ensuring async pool is open: {e}")

    return opened_pool


# In utils.py
def register_thread(
    checkpointer: Any, thread_id: str, metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Register a thread in the PostgreSQL database if needed.
    """
    try:
        if hasattr(checkpointer, "conn"):
            pool = checkpointer.conn
            if pool:
                # Ensure connection pool is usable
                ensure_pool_open(checkpointer)

                # Register the thread
                with pool.connection() as conn:
                    with conn.cursor() as cursor:
                        # Check if threads table exists
                        cursor.execute(
                            """
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'threads'
                            );
                        """
                        )
                        table_exists = cursor.fetchone()[0]

                        if not table_exists:
                            logger.debug("Creating threads table")
                            cursor.execute(
                                """
                                CREATE TABLE IF NOT EXISTS threads (
                                    thread_id VARCHAR(255) PRIMARY KEY,
                                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                    last_access TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                    metadata JSONB DEFAULT '{}'::jsonb,
                                    user_id VARCHAR(255)
                                );
                            """
                            )

                        # Convert metadata to JSON string first
                        import json

                        metadata_json = json.dumps(metadata) if metadata else "{}"

                        # Insert the thread if not exists, or update last_access
                        cursor.execute(
                            """
                            INSERT INTO threads (thread_id, last_access, metadata) 
                            VALUES (%s, CURRENT_TIMESTAMP, %s::jsonb) 
                            ON CONFLICT (thread_id) 
                            DO UPDATE SET last_access = CURRENT_TIMESTAMP, metadata = %s::jsonb
                        """,
                            (thread_id, metadata_json, metadata_json),
                        )

                        logger.info(
                            f"Thread {thread_id} registered/updated in PostgreSQL"
                        )
                        return True
    except Exception as e:
        logger.warning(f"Error registering thread: {e}")

    return False


async def register_thread_async(
    checkpointer: Any, thread_id: str, metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Register a thread in the PostgreSQL database asynchronously.
    """
    if not hasattr(checkpointer, "conn"):
        return False

    try:
        import json

        metadata_json = json.dumps(metadata) if metadata else "{}"

        async with checkpointer.conn.connection() as conn:
            async with conn.cursor() as cursor:
                # Check if table exists
                await cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'threads'
                    );
                """
                )
                table_exists = (await cursor.fetchone())[0]

                if not table_exists:
                    await cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS threads (
                            thread_id VARCHAR(255) PRIMARY KEY,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            last_access TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB DEFAULT '{}'::jsonb,
                            user_id VARCHAR(255)
                        );
                    """
                    )

                # Use parameterized query with proper JSONB handling
                await cursor.execute(
                    """
                    INSERT INTO threads (thread_id, last_access, metadata) 
                    VALUES (%s, CURRENT_TIMESTAMP, %s::jsonb) 
                    ON CONFLICT (thread_id) 
                    DO UPDATE SET last_access = CURRENT_TIMESTAMP, metadata = %s::jsonb
                """,
                    (thread_id, metadata_json, metadata_json),
                )

                logger.info(
                    f"Thread {thread_id} registered/updated in PostgreSQL (async)"
                )
                return True
    except Exception as e:
        logger.warning(f"Error registering thread asynchronously: {e}")

    return False
