"""Test suite for the enhanced PersistenceManager.

This module provides comprehensive tests for the PersistenceManager class,
using a real PostgreSQL database for testing.
"""

import pytest
import os
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, Generator

from haive_core.engine.agent.persistence.manager import PersistenceManager
from haive_core.config.auth_runnable import HaiveRunnableConfigManager
from haive_core.engine.agent.persistence.types import CheckpointerType

from langgraph.checkpoint.memory import MemorySaver

# Check if PostgreSQL dependencies are available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool
    import psycopg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


# Test configuration for PostgreSQL
POSTGRES_TEST_CONFIG = {
    'db_host': os.environ.get('POSTGRES_HOST', 'localhost'),
    'db_port': int(os.environ.get('POSTGRES_PORT', '5432')),
    'db_name': os.environ.get('POSTGRES_DB', 'postgres'),
    'db_user': os.environ.get('POSTGRES_USER', 'postgres'),
    'db_pass': os.environ.get('POSTGRES_PASSWORD', 'postgres'),
    'ssl_mode': os.environ.get('POSTGRES_SSL_MODE', 'prefer'),
    'use_pool': True,
    'setup_needed': True
}

@pytest.fixture
def clean_database() -> Generator[None, None, None]:
    """Fixture to ensure a clean database for tests."""
    if not POSTGRES_AVAILABLE:
        yield None
        return
    
    # Get database URI
    db_config = POSTGRES_TEST_CONFIG
    user = db_config["db_user"]
    password = db_config["db_pass"]
    host = db_config["db_host"]
    port = db_config["db_port"]
    db = db_config["db_name"]
    ssl_mode = db_config["ssl_mode"]
    uri = f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode={ssl_mode}"
    
    # Clear ALL existing data before the test
    try:
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                # Check if the threads table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'threads'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    # First, let's check what's in the database
                    cur.execute("SELECT COUNT(*) FROM threads")
                    count = cur.fetchone()[0]
                    print(f"Clean database: Found {count} existing threads before test")
                    
                    # Delete ALL data from the threads table - we'll recreate what we need
                    cur.execute("DELETE FROM threads")
                    print(f"Clean database: Deleted all threads from database")
                    conn.commit()
    except Exception as e:
        print(f"Error cleaning database before test: {e}")
    
    # Run the test
    yield
    
    # Cleanup after the test again
    try:
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                # Check if the threads table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'threads'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    # Delete ALL data
                    cur.execute("DELETE FROM threads")
                    print(f"Clean database: Deleted all threads after test")
                    conn.commit()
    except Exception as e:
        print(f"Error cleaning database after test: {e}")


@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL dependencies not available")
class TestPersistenceManager:
    """Test class for PersistenceManager."""
    
    def test_initialization(self):
        """Test basic initialization of the PersistenceManager."""
        # Arrange & Act
        manager = PersistenceManager()
        
        # Assert
        assert manager.config == {}
        assert manager.checkpointer is None
        assert manager.postgres_setup_needed is False
        assert manager.pool is None
        assert manager.pool_opened is False
    
    def test_initialization_with_config(self):
        """Test initialization with configuration."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        
        # Act
        manager = PersistenceManager(config)
        
        # Assert
        assert manager.config == config
        assert manager.checkpointer is None
        assert manager.postgres_setup_needed is False
    
    def test_get_memory_checkpointer(self):
        """Test getting a memory checkpointer."""
        # Arrange
        manager = PersistenceManager()
        
        # Act
        checkpointer = manager.get_checkpointer(CheckpointerType.memory)
        
        # Assert
        assert isinstance(checkpointer, MemorySaver)
        assert manager.checkpointer is checkpointer
    
    def test_get_postgres_checkpointer(self):
        """Test getting a PostgreSQL checkpointer."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        manager = PersistenceManager(config)
        
        # Act
        checkpointer = manager.get_checkpointer()
        
        # Assert
        assert isinstance(checkpointer, PostgresSaver)
        assert manager.checkpointer is checkpointer
        assert manager.postgres_setup_needed is True
    
    def test_get_db_uri_with_direct_uri(self):
        """Test getting the database URI with a direct URI provided."""
        # Arrange
        manager = PersistenceManager()
        config = {'db_uri': 'postgresql://user:pass@host:5432/db'}
        
        # Act
        uri = manager._get_db_uri(config)
        
        # Assert
        assert uri == 'postgresql://user:pass@host:5432/db'
    
    def test_get_db_uri_with_components(self):
        """Test getting the database URI with component parameters."""
        # Arrange
        manager = PersistenceManager()
        config = {
            'db_host': 'localhost',
            'db_port': 5432,
            'db_name': 'testdb',
            'db_user': 'testuser',
            'db_pass': 'testpass',
            'ssl_mode': 'disable'
        }
        
        # Act
        uri = manager._get_db_uri(config)
        
        # Assert
        assert uri == 'postgresql://testuser:testpass@localhost:5432/testdb?sslmode=disable'
    
    def test_get_db_uri_with_special_chars_in_password(self):
        """Test getting the database URI with special characters in password."""
        # Arrange
        manager = PersistenceManager()
        config = {
            'db_host': 'localhost',
            'db_port': 5432,
            'db_name': 'testdb',
            'db_user': 'testuser',
            'db_pass': 'test@pass#!%',
            'ssl_mode': 'disable'
        }
        
        # Act
        uri = manager._get_db_uri(config)
        
        # Assert
        assert 'postgresql://testuser:' in uri
        assert '@localhost:5432/testdb' in uri
        assert '?sslmode=disable' in uri
    
    def test_get_connection_kwargs(self):
        """Test getting connection kwargs."""
        # Arrange
        manager = PersistenceManager()
        config = {
            'auto_commit': True,
            'prepare_threshold': 5
        }
        
        # Act
        kwargs = manager._get_connection_kwargs(config)
        
        # Assert
        assert kwargs == {'autocommit': True, 'prepare_threshold': 5}
    
    def test_get_connection_kwargs_defaults(self):
        """Test getting connection kwargs with defaults."""
        # Arrange
        manager = PersistenceManager()
        config = {}
        
        # Act
        kwargs = manager._get_connection_kwargs(config)
        
        # Assert
        assert kwargs == {'autocommit': True, 'prepare_threshold': 0}
    
    def test_setup_with_memory_checkpointer(self):
        """Test setup with a memory checkpointer."""
        # Arrange
        manager = PersistenceManager()
        manager.checkpointer = MemorySaver()
        
        # Act
        result = manager.setup()
        
        # Assert
        assert result is True
    
    def test_setup_with_postgres_checkpointer(self):
        """Test setup with a PostgreSQL checkpointer."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        manager = PersistenceManager(config)
        
        # Act
        checkpointer = manager.get_checkpointer()
        result = manager.setup()
        
        # Assert
        assert result is True
        assert manager.pool_opened is True
    
    @pytest.mark.usefixtures("clean_database")
    def test_register_thread(self):
        """Test registering a thread with a PostgreSQL database."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        manager = PersistenceManager(config)
        manager.get_checkpointer()
        manager.setup()
        
        # Generate a test thread ID
        thread_id = f"test-{str(uuid.uuid4())}"
        
        # Act
        result = manager.register_thread(thread_id)
        
        # Assert
        assert result is True
        
        # Verify the thread was registered in the database
        db_config = POSTGRES_TEST_CONFIG
        user = db_config["db_user"]
        password = db_config["db_pass"]
        host = db_config["db_host"]
        port = db_config["db_port"]
        db = db_config["db_name"]
        ssl_mode = db_config["ssl_mode"]
        uri = f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode={ssl_mode}"
        
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_id FROM threads WHERE thread_id = %s", (thread_id,))
                result = cur.fetchone()
                assert result is not None
                assert result[0] == thread_id
    
    @pytest.mark.usefixtures("clean_database")
    def test_register_thread_with_auth_info(self):
        """Test registering a thread with authentication information."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        manager = PersistenceManager(config)
        manager.get_checkpointer()
        manager.setup()
        
        # Generate a test thread ID and auth info
        thread_id = f"test-{str(uuid.uuid4())}"
        auth_info = {
            "supabase_user_id": "test-user-id",
            "username": "testuser",
            "email": "test@example.com"
        }
        
        # Act
        result = manager.register_thread(thread_id, auth_info)
        
        # Assert
        assert result is True
        
        # Verify the thread was registered in the database with user info
        db_config = POSTGRES_TEST_CONFIG
        user = db_config["db_user"]
        password = db_config["db_pass"]
        host = db_config["db_host"]
        port = db_config["db_port"]
        db = db_config["db_name"]
        ssl_mode = db_config["ssl_mode"]
        uri = f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode={ssl_mode}"
        
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT thread_id, user_id, metadata FROM threads WHERE thread_id = %s", 
                    (thread_id,)
                )
                result = cur.fetchone()
                assert result is not None
                assert result[0] == thread_id
                assert result[1] == "test-user-id"
                
                # Parse JSON metadata if it's a string, otherwise use directly if it's already a dict
                metadata = result[2]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                assert "username" in metadata
                assert metadata["username"] == "testuser"
                assert "email" in metadata
                assert metadata["email"] == "test@example.com"
    
    def test_create_runnable_config_without_user_info(self):
        """Test creating a runnable config without user information."""
        # Arrange
        manager = PersistenceManager()
        thread_id = str(uuid.uuid4())
        
        # Act
        config, current_thread_id = manager.create_runnable_config(thread_id)
        
        # Assert
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == thread_id
        assert current_thread_id == thread_id
    
    def test_create_runnable_config_with_user_info(self):
        """Test creating a runnable config with user information."""
        # Arrange
        manager = PersistenceManager()
        thread_id = str(uuid.uuid4())
        user_info = {
            "supabase_user_id": "auth0|1234567890",
            "username": "testuser",
            "email": "test@example.com"
        }
        
        # Act
        config, current_thread_id = manager.create_runnable_config(thread_id, user_info)
        
        # Assert
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == thread_id
        assert "auth" in config["configurable"]
        assert config["configurable"]["auth"]["supabase_user_id"] == "auth0|1234567890"
        assert config["configurable"]["auth"]["username"] == "testuser"
        assert config["configurable"]["auth"]["email"] == "test@example.com"
        assert current_thread_id == thread_id
    
    def test_create_runnable_config_with_postgres_checkpointer(self):
        """Test creating a runnable config with a PostgreSQL checkpointer."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        manager = PersistenceManager(config)
        manager.get_checkpointer()
        thread_id = str(uuid.uuid4())
        
        # Act
        config, current_thread_id = manager.create_runnable_config(thread_id)
        
        # Assert
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == thread_id
        assert "persistence" in config["configurable"]
        assert config["configurable"]["persistence"]["type"] == "postgres"
        assert config["configurable"]["persistence"]["setup_needed"] is True
        assert current_thread_id == thread_id
    
    @pytest.mark.usefixtures("clean_database")
    def test_prepare_for_agent_run(self):
        """Test preparing for an agent run using real PostgreSQL."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        manager = PersistenceManager(config)
        
        # Initialize and set up the checkpointer first
        manager.get_checkpointer()
        manager.setup()
        
        thread_id = f"test-{str(uuid.uuid4())}"
        user_info = {
            "supabase_user_id": "test-user-id",
            "username": "testuser",
            "email": "test@example.com"
        }
        
        # Act
        config, current_thread_id = manager.prepare_for_agent_run(thread_id, user_info)
        
        # Assert
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == thread_id
        assert "auth" in config["configurable"]
        assert current_thread_id == thread_id
        
        # Verify thread is registered in database
        db_config = POSTGRES_TEST_CONFIG
        uri = f"postgresql://{db_config['db_user']}:{db_config['db_pass']}@{db_config['db_host']}:{db_config['db_port']}/{db_config['db_name']}?sslmode={db_config['ssl_mode']}"
        
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_id FROM threads WHERE thread_id = %s", (thread_id,))
                result = cur.fetchone()
                assert result is not None, f"Thread {thread_id} was not found in the database"
                assert result[0] == thread_id
    
    def test_get_or_create_thread_id_with_config(self):
        """Test getting or creating a thread ID with a config."""
        # Arrange
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Act
        result = PersistenceManager.get_or_create_thread_id(config)
        
        # Assert
        assert result == thread_id
    
    def test_get_or_create_thread_id_without_config(self):
        """Test getting or creating a thread ID without a config."""
        # Act
        result = PersistenceManager.get_or_create_thread_id()
        
        # Assert
        assert uuid.UUID(result)  # Verify it's a valid UUID
    
    @pytest.mark.usefixtures("clean_database")
    def test_list_threads(self):
        """Test listing threads from the database."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        manager = PersistenceManager(config)
        manager.get_checkpointer()
        manager.setup()
        
        # Create a test thread
        thread_id = f"test-{str(uuid.uuid4())}"
        user_id = "test-user-id"
        auth_info = {
            "supabase_user_id": user_id,
            "username": "testuser",
            "email": "test@example.com"
        }
        manager.register_thread(thread_id, auth_info)
        
        # Act
        threads = manager.list_threads()
        
        # Filter to only find our test thread (there could be other threads in the database)
        test_threads = [t for t in threads if t["thread_id"] == thread_id]
        
        # Assert
        assert len(test_threads) == 1
        assert test_threads[0]["thread_id"] == thread_id
        assert test_threads[0]["user_id"] == user_id
        assert test_threads[0]["username"] == "testuser"
        assert "created_at" in test_threads[0]
        assert "last_access" in test_threads[0]
    
    @pytest.mark.usefixtures("clean_database")
    def test_list_threads_with_user_id(self):
        """Test listing threads with a user ID filter."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        manager = PersistenceManager(config)
        manager.get_checkpointer()
        manager.setup()
        
        # Check database is clean before we start
        db_config = POSTGRES_TEST_CONFIG
        uri = f"postgresql://{db_config['db_user']}:{db_config['db_pass']}@{db_config['db_host']}:{db_config['db_port']}/{db_config['db_name']}?sslmode={db_config['ssl_mode']}"
        
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_id, user_id FROM threads WHERE thread_id LIKE 'test-%'")
                existing_threads = cur.fetchall()
                print(f"Existing threads before test: {existing_threads}")
        
        # Create two test threads with different user IDs
        thread_id1 = f"test-{str(uuid.uuid4())}"
        user_id1 = "test-user-id-1"
        auth_info1 = {
            "supabase_user_id": user_id1,
            "username": "testuser1",
            "email": "test1@example.com"
        }
        result1 = manager.register_thread(thread_id1, auth_info1)
        print(f"Registration result for thread1: {result1}")
        
        thread_id2 = f"test-{str(uuid.uuid4())}"
        user_id2 = "test-user-id-2"
        auth_info2 = {
            "supabase_user_id": user_id2,
            "username": "testuser2",
            "email": "test2@example.com"
        }
        result2 = manager.register_thread(thread_id2, auth_info2)
        print(f"Registration result for thread2: {result2}")
        
        # Verify threads are in the database with the right user IDs
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_id, user_id FROM threads WHERE thread_id = %s OR thread_id = %s", 
                          (thread_id1, thread_id2))
                test_threads = cur.fetchall()
                print(f"Registered threads: {test_threads}")
        
        # Act - get all threads first to see what's in the database
        all_threads = manager.list_threads()
        print(f"All threads in database: {len(all_threads)}")
        for t in all_threads:
            print(f"  thread_id={t['thread_id']}, user_id={t['user_id']}")
        
        # Now list threads for user1
        threads = manager.list_threads(user_id=user_id1)
        print(f"Threads for user_id={user_id1}: {len(threads)}")
        for t in threads:
            print(f"  thread_id={t['thread_id']}, user_id={t['user_id']}")
        
        # Filter results to only include our test threads
        filtered_threads = [t for t in threads if t["thread_id"] in [thread_id1, thread_id2]]
        
        # Assert
        assert len(filtered_threads) == 1, f"Expected 1 thread, got {len(filtered_threads)}: {filtered_threads}"
        assert filtered_threads[0]["thread_id"] == thread_id1
        assert filtered_threads[0]["user_id"] == user_id1
        assert filtered_threads[0]["username"] == "testuser1"
    
    @pytest.mark.usefixtures("clean_database")
    def test_delete_thread(self):
        """Test deleting a thread from the database."""
        # Arrange
        config = {
            'persistence_type': CheckpointerType.postgres,
            'persistence_config': POSTGRES_TEST_CONFIG
        }
        manager = PersistenceManager(config)
        manager.get_checkpointer()
        manager.setup()
        
        # Create a test thread
        thread_id = f"test-{str(uuid.uuid4())}"
        manager.register_thread(thread_id)
        
        # Verify the thread exists
        db_config = POSTGRES_TEST_CONFIG
        uri = f"postgresql://{db_config['db_user']}:{db_config['db_pass']}@{db_config['db_host']}:{db_config['db_port']}/{db_config['db_name']}?sslmode={db_config['ssl_mode']}"
        
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_id FROM threads WHERE thread_id = %s", (thread_id,))
                assert cur.fetchone() is not None
        
        # Act - delete the thread
        result = manager.delete_thread(thread_id)
        
        # Assert
        assert result is True
        
        # Verify the thread no longer exists
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_id FROM threads WHERE thread_id = %s", (thread_id,))
                assert cur.fetchone() is None
    
    def test_from_config(self):
        """Test creating a PersistenceManager from configuration."""
        # Act
        manager = PersistenceManager.from_config(
            db_host=POSTGRES_TEST_CONFIG["db_host"],
            db_port=POSTGRES_TEST_CONFIG["db_port"],
            db_name=POSTGRES_TEST_CONFIG["db_name"],
            db_user=POSTGRES_TEST_CONFIG["db_user"],
            db_pass=POSTGRES_TEST_CONFIG["db_pass"],
            use_async=False,
            use_pool=True,
            setup_needed=True
        )
        
        # Assert
        assert manager.config['persistence_type'] == CheckpointerType.postgres
        assert manager.config['persistence_config']['db_host'] == POSTGRES_TEST_CONFIG["db_host"]
        assert manager.config['persistence_config']['db_port'] == POSTGRES_TEST_CONFIG["db_port"]
        assert manager.config['persistence_config']['db_name'] == POSTGRES_TEST_CONFIG["db_name"]
        assert manager.config['persistence_config']['db_user'] == POSTGRES_TEST_CONFIG["db_user"]
        assert manager.config['persistence_config']['db_pass'] == POSTGRES_TEST_CONFIG["db_pass"]
        assert manager.config['persistence_config']['use_async'] is False
        assert manager.config['persistence_config']['use_pool'] is True
        assert manager.config['persistence_config']['setup_needed'] is True
        assert isinstance(manager.checkpointer, PostgresSaver)
    
    def test_from_env(self):
        """Test creating a PersistenceManager from environment variables."""
        # Arrange
        env_vars = {
            "POSTGRES_HOST": POSTGRES_TEST_CONFIG["db_host"],
            "POSTGRES_PORT": str(POSTGRES_TEST_CONFIG["db_port"]),
            "POSTGRES_DB": POSTGRES_TEST_CONFIG["db_name"],
            "POSTGRES_USER": POSTGRES_TEST_CONFIG["db_user"],
            "POSTGRES_PASSWORD": POSTGRES_TEST_CONFIG["db_pass"],
            "POSTGRES_SSL_MODE": POSTGRES_TEST_CONFIG["ssl_mode"],
            "POSTGRES_USE_ASYNC": "false",
            "POSTGRES_USE_POOL": "true",
            "POSTGRES_SETUP_NEEDED": "true"
        }
        
        # Save current environment
        old_env = {}
        for key in env_vars:
            if key in os.environ:
                old_env[key] = os.environ[key]
        
        # Set new environment
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            # Act
            manager = PersistenceManager.from_env()
            
            # Assert
            assert manager.config['persistence_type'] == CheckpointerType.postgres
            assert manager.config['persistence_config']['db_host'] == POSTGRES_TEST_CONFIG["db_host"]
            assert manager.config['persistence_config']['db_port'] == POSTGRES_TEST_CONFIG["db_port"]
            assert manager.config['persistence_config']['db_name'] == POSTGRES_TEST_CONFIG["db_name"]
            assert manager.config['persistence_config']['db_user'] == POSTGRES_TEST_CONFIG["db_user"]
            assert manager.config['persistence_config']['db_pass'] == POSTGRES_TEST_CONFIG["db_pass"]
            assert manager.config['persistence_config']['ssl_mode'] == POSTGRES_TEST_CONFIG["ssl_mode"]
            assert manager.config['persistence_config']['use_async'] is False
            assert manager.config['persistence_config']['use_pool'] is True
            assert manager.config['persistence_config']['setup_needed'] is True
            assert isinstance(manager.checkpointer, PostgresSaver)
        finally:
            # Restore environment
            for key in env_vars:
                if key in old_env:
                    os.environ[key] = old_env[key]
                else:
                    os.environ.pop(key, None)