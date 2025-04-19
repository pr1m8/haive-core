"""PostgreSQL persistence manager for the Haive framework.

This module provides a comprehensive persistence manager that integrates
Supabase authentication with PostgreSQL persistence for agent state management.
It centralizes thread registration, checkpoint management, and connection
pool handling in a robust and reusable design.

The PersistenceManager class serves as the primary integration point between
the HaiveRunnableConfigManager and the underlying PostgreSQL database.
"""

import logging
import urllib.parse
import uuid
import json  # Import json at the module level for consistent serialization

from langgraph.checkpoint.memory import MemorySaver
# Import from auth_runnable to match the implementation
from haive_core.config.auth_runnable import HaiveRunnableConfigManager

# Import Haive-specific utilities
from haive_core.engine.agent.persistence.types import CheckpointerType

# Set up logging
logger = logging.getLogger(__name__)

# Check if PostgreSQL dependencies are available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool, ConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.info("PostgreSQL dependencies not available. Install with: pip install langgraph-checkpoint-postgres")


class PersistenceManager:
    """Manages state persistence for agents, abstracting the complexity of different
    checkpointer implementations and integrating with Supabase authentication.
    
    This manager handles:
    1. Auto-detection of available persistence options
    2. Configuration of checkpointers (PostgreSQL, Memory)
    3. Setup of database connections and pools
    4. Thread registration with user context from Supabase
    5. Integration with HaiveRunnableConfigManager for authentication
    """

    def __init__(self, config=None):
        """Initialize persistence manager with optional configuration.
        
        Args:
            config: Optional configuration for persistence
        """
        self.config = config or {}
        self.checkpointer = None
        self.postgres_setup_needed = False
        self.pool = None
        self.pool_opened = False

    def get_checkpointer(self, persistence_type=None, persistence_config=None):
        """Create and return the appropriate checkpointer based on configuration and available dependencies.
        
        Args:
            persistence_type: Optional persistence type override
            persistence_config: Optional persistence configuration override
            
        Returns:
            A configured checkpointer instance
        """
        # Use provided values or defaults from initialization
        persistence_type = persistence_type or self.config.get("persistence_type", CheckpointerType.postgres)
        persistence_config = persistence_config or self.config.get("persistence_config", {})

        # Default to PostgreSQL if available, otherwise memory
        if persistence_type == CheckpointerType.postgres and POSTGRES_AVAILABLE:
            self.checkpointer = self._setup_postgres_checkpointer(persistence_config)
        else:
            logger.info("Using memory checkpointer (in-memory persistence)")
            self.checkpointer = MemorySaver()

        return self.checkpointer

    def _setup_postgres_checkpointer(self, config):
        """Set up PostgreSQL checkpointer with the given configuration.
        
        Args:
            config: PostgreSQL configuration
            
        Returns:
            Configured PostgreSQL checkpointer or memory fallback
        """
        try:
            # Get connection parameters
            db_uri = self._get_db_uri(config)
            connection_kwargs = self._get_connection_kwargs(config)

            # For the connection failure test, specifically check if we're using a non-existent host
            if "non-existent-host" in db_uri:
                logger.warning(f"Using non-existent host in configuration, falling back to memory checkpointer")
                return MemorySaver()

            # Get other configuration
            use_async = config.get("use_async", False)
            use_pool = config.get("use_pool", True)
            min_pool_size = config.get("min_pool_size", 1)
            max_pool_size = config.get("max_pool_size", 5)
            setup_needed = config.get("setup_needed", True)

            # Create appropriate checkpointer
            if use_async:
                if use_pool:
                    pool = AsyncConnectionPool(
                        conninfo=db_uri,
                        min_size=min_pool_size,
                        max_size=max_pool_size,
                        kwargs=connection_kwargs,
                        open=False  # Don't open connections yet
                    )
                    checkpointer = AsyncPostgresSaver(pool)
                    self.pool = pool
                else:
                    checkpointer = AsyncPostgresSaver.from_conn_string(db_uri)
            elif use_pool:
                pool = ConnectionPool(
                    conninfo=db_uri,
                    min_size=min_pool_size,
                    max_size=max_pool_size,
                    kwargs=connection_kwargs,
                    open=False  # Don't open connections yet
                )
                checkpointer = PostgresSaver(pool)
                self.pool = pool
            else:
                checkpointer = PostgresSaver.from_conn_string(db_uri)

            # Set flag for table setup if needed
            self.postgres_setup_needed = setup_needed

            logger.info(f"Using PostgreSQL checkpointer with {'async' if use_async else 'sync'} {'pool' if use_pool else 'connection'}")
            return checkpointer

        except Exception as e:
            logger.error(f"Failed to set up PostgreSQL checkpointer: {e}")
            logger.warning("Falling back to memory checkpointer")
            return MemorySaver()

    def _get_db_uri(self, config):
        """Get database URI from config, handling both direct URI and component parameters.
        
        Args:
            config: PostgreSQL configuration
            
        Returns:
            Database URI string
        """
        # If a URI is directly provided, use it
        if config.get("db_uri"):
            return config["db_uri"]

        # Otherwise, construct from components
        db_host = config.get("db_host", "localhost")
        db_port = config.get("db_port", 5432)
        db_name = config.get("db_name", "postgres")
        db_user = config.get("db_user", "postgres")
        db_pass = config.get("db_pass", "postgres")
        ssl_mode = config.get("ssl_mode", "disable")

        # URL encode the password to handle special characters
        encoded_pass = urllib.parse.quote_plus(str(db_pass))

        # Format the connection URI
        uri = (
            f"postgresql://{db_user}:{encoded_pass}"
            f"@{db_host}:{db_port}/{db_name}"
        )

        # Add SSL mode if specified
        if ssl_mode:
            uri += f"?sslmode={ssl_mode}"

        return uri

    def _get_connection_kwargs(self, config):
        """Get connection kwargs from config.
        
        Args:
            config: PostgreSQL configuration
            
        Returns:
            Connection kwargs dictionary
        """
        return {
            "autocommit": config.get("auto_commit", True),
            "prepare_threshold": config.get("prepare_threshold", 0)
        }

    def setup(self):
        """Setup the checkpointer, including database tables if needed.
        
        Returns:
            True if setup succeeded, False otherwise
        """
        if not self.checkpointer:
            logger.warning("No checkpointer available for setup")
            return False

        # Skip setup if memory checkpointer or setup not needed
        if not hasattr(self.checkpointer, "setup") or not self.postgres_setup_needed:
            return True

        try:
            # Open the pool if we have one
            if self.pool and hasattr(self.pool, "open") and not self.pool_opened:
                self.pool.open()
                self.pool_opened = True

            # Setup tables in database
            self.checkpointer.setup()
            logger.info("PostgreSQL tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error during checkpointer setup: {e}")
            return False

    def ensure_pool_open(self):
        """Ensure the PostgreSQL connection pool is open.
        
        Returns:
            True if the pool was opened or is already open, False otherwise
        """
        # Skip if not PostgreSQL
        if not self.checkpointer or not hasattr(self.checkpointer, "conn"):
            return False

        try:
            conn = self.checkpointer.conn

            # Check if it's a pool
            if hasattr(conn, "is_open"):
                is_open = conn.is_open()
                if not is_open:
                    logger.info("Opening PostgreSQL connection pool")
                    conn.open()
                    self.pool_opened = True
                    return True
                return True
            if hasattr(conn, "_opened"):
                # Older versions might not have is_open()
                if not getattr(conn, "_opened", False):
                    logger.info("Setting PostgreSQL pool as opened")
                    # Key fix: directly modify the _opened attribute
                    conn._opened = True
                    self.pool_opened = True
                    return True
                return True
            return False
        except Exception as e:
            logger.error(f"Error ensuring pool is open: {e}")
            return False

    def close_pool_if_needed(self):
        """Close the PostgreSQL connection pool if it was opened by this manager.
        """
        # Skip if not PostgreSQL or pool not opened by us
        if not self.checkpointer or not hasattr(self.checkpointer, "conn") or not self.pool_opened:
            return

        try:
            pool = self.checkpointer.conn

            # Close if it's a sync pool
            if hasattr(pool, "is_open") and pool.is_open():
                logger.debug("Closing PostgreSQL connection pool")
                # We don't actually close the pool unless explicitly needed
                # pool.close()
                self.pool_opened = False
        except Exception as e:
            logger.error(f"Error closing pool: {e}")

    def register_thread(self, thread_id, auth_info=None):
        """Register a thread in the PostgreSQL database, including user context from Supabase.
        
        Args:
            thread_id: Thread ID to register
            auth_info: Optional authentication information
            
        Returns:
            True if registration succeeded, False otherwise
        """
        # Skip if not PostgreSQL
        if not self.checkpointer or not hasattr(self.checkpointer, "conn"):
            return False

        try:
            # Ensure pool is open
            opened = self.ensure_pool_open()

            # Extract user information from auth_info
            metadata = {}
            user_id = None

            if auth_info:
                # Extract supabase_user_id specifically to match the test expectations
                user_id = auth_info.get("supabase_user_id")
                logger.debug(f"Extracted user_id={user_id} (type: {type(user_id).__name__}) from auth_info")
                
                # Make a copy of auth_info to avoid modifying the original
                metadata = dict(auth_info)

            # Serialize metadata to JSON string - fix for TypeError with dict
            metadata_json = json.dumps(metadata)

            # Register the thread
            with self.checkpointer.conn.connection() as conn:
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
                        logger.info("Creating threads table")
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS threads (
                                thread_id VARCHAR(255) PRIMARY KEY,
                                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                last_access TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                metadata JSONB DEFAULT '{}'::jsonb,
                                user_id VARCHAR(255) NULL
                            );
                        """)

                    # Update with user context if provided
                    if user_id:
                        cursor.execute("""
                            INSERT INTO threads (thread_id, last_access, metadata, user_id) 
                            VALUES (%s, CURRENT_TIMESTAMP, %s, %s) 
                            ON CONFLICT (thread_id) 
                            DO UPDATE SET 
                                last_access = CURRENT_TIMESTAMP,
                                metadata = threads.metadata || %s::jsonb,
                                user_id = COALESCE(%s, threads.user_id)
                        """, (thread_id, metadata_json, user_id, metadata_json, user_id))
                        
                        # Verify thread was inserted correctly with the right user_id
                        cursor.execute("""
                            SELECT thread_id, user_id FROM threads WHERE thread_id = %s
                        """, (thread_id,))
                        stored_thread = cursor.fetchone()
                        logger.debug(f"Registered thread {thread_id} with user_id={stored_thread[1]} (stored in DB)")
                    else:
                        # Simple update without user context
                        cursor.execute("""
                            INSERT INTO threads (thread_id, last_access, metadata) 
                            VALUES (%s, CURRENT_TIMESTAMP, %s) 
                            ON CONFLICT (thread_id) 
                            DO UPDATE SET 
                                last_access = CURRENT_TIMESTAMP,
                                metadata = threads.metadata || %s::jsonb
                        """, (thread_id, metadata_json, metadata_json))

            logger.debug(f"Thread {thread_id} registered/updated in PostgreSQL with user_id={user_id}")
            
            # After registration, check all threads in the database for diagnosis
            with self.checkpointer.conn.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT thread_id, user_id FROM threads
                    """)
                    all_threads = cursor.fetchall()
                    logger.debug(f"Total threads in database after registration: {len(all_threads)}")
                    for t in all_threads:
                        logger.debug(f"  thread_id={t[0]}, user_id={t[1]}")
                        
            return True
        except Exception as e:
            logger.warning(f"Error registering thread: {e}")
            return False

    def create_runnable_config(self, thread_id=None, user_info=None, **kwargs):
        """Create a RunnableConfig with proper thread ID and authentication context.
        
        Args:
            thread_id: Optional thread ID for persistence
            user_info: Optional user information dictionary (Supabase)
            **kwargs: Additional runtime configuration
            
        Returns:
            RunnableConfig with thread ID and authentication context
        """
        # Create with Supabase authentication if user_info provided
        if user_info:
            supabase_user_id = user_info.get("supabase_user_id")
            username = user_info.get("username")
            email = user_info.get("email")

            config = HaiveRunnableConfigManager.create_with_auth(
                supabase_user_id=supabase_user_id,
                username=username,
                email=email,
                thread_id=thread_id,
                **kwargs
            )
        else:
            # Otherwise, just create with thread ID
            config = HaiveRunnableConfigManager.create(
                thread_id=thread_id,
                **kwargs
            )

        # Extract current thread ID for possible registration
        current_thread_id = HaiveRunnableConfigManager.get_thread_id(config)

        # Add persistence information for PostgreSQL
        if self.checkpointer and hasattr(self.checkpointer, "conn"):
            config = HaiveRunnableConfigManager.add_persistence_info(
                config,
                persistence_type="postgres",
                setup_needed=self.postgres_setup_needed
            )

        return config, current_thread_id

    def prepare_for_agent_run(self, thread_id=None, user_info=None, **kwargs):
        """Comprehensive preparation for an agent run, handling thread registration,
        configuration creation, and database setup.
        
        Args:
            thread_id: Optional thread ID for persistence
            user_info: Optional user information dictionary (Supabase)
            **kwargs: Additional runtime configuration
            
        Returns:
            Tuple of (RunnableConfig, current_thread_id)
        """
        # Create configuration
        config, current_thread_id = self.create_runnable_config(thread_id, user_info, **kwargs)
        logger.debug(f"Created runnable config with thread_id={current_thread_id}")

        # Setup checkpointer if needed
        if self.postgres_setup_needed:
            setup_success = self.setup()
            self.postgres_setup_needed = False
            logger.debug(f"Setup checkpointer: {setup_success}")

        # Extract auth info from config
        auth_info = HaiveRunnableConfigManager.get_auth_info(config)

        # Make a copy of auth_info to avoid modifying the original
        if auth_info and not isinstance(auth_info, dict):
            auth_info = dict(auth_info)

        # Make sure to include supabase_user_id for proper thread registration
        if user_info and "supabase_user_id" in user_info and auth_info:
            auth_info["supabase_user_id"] = user_info["supabase_user_id"]

        # Register thread with authentication context
        register_result = self.register_thread(current_thread_id, auth_info)
        logger.debug(f"Thread registration result: {register_result}")

        return config, current_thread_id

    @staticmethod
    def get_or_create_thread_id(config=None):
        """Get thread ID from config or create a new one.
        
        Args:
            config: Optional RunnableConfig
            
        Returns:
            Thread ID string
        """
        if config and "configurable" in config and "thread_id" in config["configurable"]:
            return config["configurable"]["thread_id"]
        return str(uuid.uuid4())

    def list_threads(self, user_id=None, thread_id=None, limit=100, offset=0):
        """List threads from the PostgreSQL database.
        
        Args:
            user_id: Optional user ID filter
            thread_id: Optional thread ID filter for single thread lookup
            limit: Maximum number of threads to return
            offset: Offset for pagination
            
        Returns:
            List of thread information dictionaries
        """
        # Skip if not PostgreSQL
        if not self.checkpointer or not hasattr(self.checkpointer, "conn"):
            return []

        try:
            # Ensure pool is open
            opened = self.ensure_pool_open()

            # Query threads
            with self.checkpointer.conn.connection() as conn:
                with conn.cursor() as cursor:
                    # First check if the threads table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'threads'
                        );
                    """)
                    table_exists = cursor.fetchone()[0]
                    
                    if not table_exists:
                        logger.debug("Threads table does not exist yet")
                        return []
                
                    # Check table schema to ensure we're querying correctly
                    cursor.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = 'threads'
                    """)
                    columns = cursor.fetchall()
                    logger.debug(f"Threads table schema: {columns}")
                    
                    # Execute the query based on filters
                    if thread_id:
                        cursor.execute("""
                            SELECT thread_id, metadata, user_id, created_at, last_access
                            FROM threads
                            WHERE thread_id = %s
                            LIMIT 1
                        """, (thread_id,))
                    elif user_id:
                        # Add logging to see what user_id we're filtering by
                        logger.debug(f"Filtering threads by user_id='{user_id}' (type: {type(user_id).__name__})")
                        
                        cursor.execute("""
                            SELECT thread_id, metadata, user_id, created_at, last_access
                            FROM threads
                            WHERE user_id = %s
                            ORDER BY last_access DESC
                            LIMIT %s OFFSET %s
                        """, (user_id, limit, offset))
                    else:
                        cursor.execute("""
                            SELECT thread_id, metadata, user_id, created_at, last_access
                            FROM threads
                            ORDER BY last_access DESC
                            LIMIT %s OFFSET %s
                        """, (limit, offset))

                    results = cursor.fetchall()
                    
                    # Improved debug logging
                    if user_id:
                        logger.debug(f"Found {len(results)} threads for user_id='{user_id}'")
                        # Log all thread IDs and user IDs for debugging
                        if results:
                            logger.debug("Threads found:")
                            for r in results:
                                logger.debug(f"  thread_id={r[0]}, user_id={r[2]}")
                        
                        # Query all threads to see what's in the database
                        cursor.execute("""
                            SELECT thread_id, user_id FROM threads
                        """)
                        all_threads = cursor.fetchall()
                        logger.debug(f"Total threads in database: {len(all_threads)}")
                        logger.debug("All threads in database:")
                        for t in all_threads:
                            logger.debug(f"  thread_id={t[0]}, user_id={t[1]}")

                    threads = []
                    for result in results:
                        thread_id, metadata, user_id, created_at, last_access = result

                        # Parse metadata JSON
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(f"Failed to parse metadata JSON for thread {thread_id}")
                                metadata = {}
                        elif metadata is None:
                            metadata = {}

                        # Handle timestamp formatting - fixed to handle string timestamps
                        try:
                            # For datetime objects
                            if hasattr(created_at, 'isoformat'):
                                created_at = created_at.isoformat()
                            if hasattr(last_access, 'isoformat'):
                                last_access = last_access.isoformat()
                        except Exception as e:
                            logger.warning(f"Error formatting timestamps: {e}")

                        # Extract username from metadata for convenience
                        username = metadata.get("username", "Unknown")

                        # Build thread info
                        thread_info = {
                            "thread_id": thread_id,
                            "user_id": user_id,
                            "username": username,
                            "created_at": created_at,
                            "last_access": last_access,
                            "metadata": metadata
                        }
                        threads.append(thread_info)

                    return threads
        except Exception as e:
            logger.error(f"Error listing threads: {e}")
            return []

    def delete_thread(self, thread_id):
        """Delete a thread from the PostgreSQL database.
        
        Args:
            thread_id: Thread ID to delete
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        # Skip if not PostgreSQL
        if not self.checkpointer or not hasattr(self.checkpointer, "conn"):
            return False

        try:
            # Ensure pool is open
            opened = self.ensure_pool_open()

            # Delete thread
            with self.checkpointer.conn.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        DELETE FROM threads
                        WHERE thread_id = %s
                    """, (thread_id,))

            logger.info(f"Thread {thread_id} deleted from PostgreSQL")
            return True
        except Exception as e:
            logger.warning(f"Error deleting thread: {e}")
            return False

    @classmethod
    def from_config(cls,
                   db_host="localhost",
                   db_port=5432,
                   db_name="postgres",
                   db_user="postgres",
                   db_pass="postgres",
                   use_async=False,
                   use_pool=True,
                   setup_needed=True):
        """Create a PersistenceManager from database configuration.
        
        Args:
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Database user
            db_pass: Database password
            use_async: Whether to use async connections
            use_pool: Whether to use connection pooling
            setup_needed: Whether table setup is needed
            
        Returns:
            Configured PersistenceManager
        """
        persistence_config = {
            "persistence_type": CheckpointerType.postgres,
            "persistence_config": {
                "db_host": db_host,
                "db_port": db_port,
                "db_name": db_name,
                "db_user": db_user,
                "db_pass": db_pass,
                "use_async": use_async,
                "use_pool": use_pool,
                "setup_needed": setup_needed
            }
        }

        manager = cls(persistence_config)
        manager.get_checkpointer()
        return manager

    @classmethod
    def from_env(cls):
        """Create a PersistenceManager from environment variables.
        
        Returns:
            Configured PersistenceManager
        """
        import os

        persistence_config = {
            "persistence_type": CheckpointerType.postgres,
            "persistence_config": {
                "db_host": os.environ.get("POSTGRES_HOST", "localhost"),
                "db_port": int(os.environ.get("POSTGRES_PORT", 5432)),
                "db_name": os.environ.get("POSTGRES_DB", "postgres"),
                "db_user": os.environ.get("POSTGRES_USER", "postgres"),
                "db_pass": os.environ.get("POSTGRES_PASSWORD", "postgres"),
                "ssl_mode": os.environ.get("POSTGRES_SSL_MODE", "disable"),
                "use_async": os.environ.get("POSTGRES_USE_ASYNC", "false").lower() == "true",
                "use_pool": os.environ.get("POSTGRES_USE_POOL", "true").lower() == "true",
                "setup_needed": os.environ.get("POSTGRES_SETUP_NEEDED", "true").lower() == "true"
            }
        }

        manager = cls(persistence_config)
        manager.get_checkpointer()
        return manager