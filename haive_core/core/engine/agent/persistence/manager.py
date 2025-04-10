# src/haive/core/engine/agent/persistence/manager.py

from typing import Optional, Dict, Any, Union
import logging
import uuid
import urllib.parse
from pydantic import BaseModel, Field

from src.haive.core.engine.agent.persistence.types import CheckpointerType
from langgraph.checkpoint.memory import MemorySaver

# Set up logging
logger = logging.getLogger(__name__)

# Check if PostgreSQL dependencies are available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import ConnectionPool, AsyncConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.info("PostgreSQL dependencies not available. Install with: pip install langgraph-checkpoint-postgres")


class PersistenceManager:
    """
    Manages state persistence for agents, abstracting the complexity of different
    checkpointer implementations.
    
    This manager handles:
    1. Auto-detection of available persistence options
    2. Configuration of checkpointers (PostgreSQL, Memory)
    3. Setup of database connections and pools
    4. Consistent thread ID management across invocations
    """
    
    def __init__(self, config=None):
        """
        Initialize persistence manager with optional configuration.
        
        Args:
            config: Optional configuration for persistence
        """
        self.config = config or {}
        self.checkpointer = None
        self.postgres_setup_needed = False
        
    def get_checkpointer(self, persistence_type=None, persistence_config=None):
        """
        Create and return the appropriate checkpointer based on configuration and available dependencies.
        
        Args:
            persistence_type: Optional persistence type override
            persistence_config: Optional persistence configuration override
            
        Returns:
            A configured checkpointer instance
        """
        # Use provided values or defaults from initialization
        persistence_type = persistence_type or self.config.get('persistence_type', CheckpointerType.postgres)
        persistence_config = persistence_config or self.config.get('persistence_config', {})
        
        # Default to PostgreSQL if available, otherwise memory
        if persistence_type == CheckpointerType.postgres and POSTGRES_AVAILABLE:
            return self._setup_postgres_checkpointer(persistence_config)
        else:
            logger.info("Using memory checkpointer (in-memory persistence)")
            return MemorySaver()
    
    def _setup_postgres_checkpointer(self, config):
        """
        Set up PostgreSQL checkpointer with the given configuration.
        
        Args:
            config: PostgreSQL configuration
            
        Returns:
            Configured PostgreSQL checkpointer or memory fallback
        """
        try:
            # Get connection parameters
            db_uri = self._get_db_uri(config)
            connection_kwargs = self._get_connection_kwargs(config)
            
            # Get other configuration
            use_async = config.get('use_async', False)
            use_pool = config.get('use_pool', True)
            min_pool_size = config.get('min_pool_size', 1)
            max_pool_size = config.get('max_pool_size', 5)
            setup_needed = config.get('setup_needed', False)
            
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
                else:
                    checkpointer = AsyncPostgresSaver.from_conn_string(db_uri)
            else:
                if use_pool:
                    pool = ConnectionPool(
                        conninfo=db_uri,
                        min_size=min_pool_size,
                        max_size=max_pool_size,
                        kwargs=connection_kwargs,
                        open=False  # Don't open connections yet
                    )
                    checkpointer = PostgresSaver(pool)
                else:
                    checkpointer = PostgresSaver.from_conn_string(db_uri)
            
            # Set flag for table setup if needed
            if setup_needed:
                self.postgres_setup_needed = True
                
            logger.info(f"Using PostgreSQL checkpointer with {'async' if use_async else 'sync'} {'pool' if use_pool else 'connection'}")
            return checkpointer
            
        except Exception as e:
            logger.error(f"Failed to set up PostgreSQL checkpointer: {e}")
            logger.warning("Falling back to memory checkpointer")
            return MemorySaver()
    
    def _get_db_uri(self, config):
        """
        Get database URI from config, handling both direct URI and component parameters.
        
        Args:
            config: PostgreSQL configuration
            
        Returns:
            Database URI string
        """
        # If a URI is directly provided, use it
        if config.get('db_uri'):
            return config['db_uri']
        
        # Otherwise, construct from components
        db_host = config.get('db_host', 'localhost')
        db_port = config.get('db_port', 5432)
        db_name = config.get('db_name', 'postgres')
        db_user = config.get('db_user', 'postgres')
        db_pass = config.get('db_pass', 'postgres')
        ssl_mode = config.get('ssl_mode', 'disable')
        
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
        """
        Get connection kwargs from config.
        
        Args:
            config: PostgreSQL configuration
            
        Returns:
            Connection kwargs dictionary
        """
        return {
            "autocommit": config.get('auto_commit', True),
            "prepare_threshold": config.get('prepare_threshold', 0)
        }
    
    @staticmethod
    def get_runtime_config(thread_id=None, **kwargs):
        """
        Get runtime configuration with proper thread ID handling.
        
        Args:
            thread_id: Optional thread ID for persistence
            **kwargs: Additional runtime configuration
            
        Returns:
            Runtime configuration dictionary with thread ID
        """
        # Start with basic configurable section
        runtime_config = {"configurable": {}}
        
        # Use provided thread ID or generate new one
        if thread_id:
            runtime_config["configurable"]["thread_id"] = thread_id
        else:
            runtime_config["configurable"]["thread_id"] = str(uuid.uuid4())
        
        # Add other kwargs
        for key, value in kwargs.items():
            # If it's a configurable param, add to configurable section
            if key.startswith("configurable_"):
                param_name = key.replace("configurable_", "")
                runtime_config["configurable"][param_name] = value
            else:
                # Otherwise add to top level
                runtime_config[key] = value
                
        return runtime_config