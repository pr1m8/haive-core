#!/usr/bin/env python3
"""Test PostgreSQL persistence functionality.

This test verifies PostgreSQL checkpointer creation and basic functionality.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv
import pytest
import psycopg

from haive.core.persistence.postgres_config import PostgresCheckpointerConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_postgres_checkpointer_creation():
    """Test PostgreSQL checkpointer creation."""
    # Load environment variables
    load_dotenv()
    
    # Check environment variable
    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if not conn_string:
        pytest.skip("POSTGRES_CONNECTION_STRING not found")
    
    # Create PostgreSQL checkpointer config
    postgres_config = PostgresCheckpointerConfig(
        connection_string=conn_string
    )
    assert postgres_config is not None
    
    # Test creating checkpointer
    checkpointer = postgres_config.create_checkpointer()
    assert checkpointer is not None
    logger.info(f"✅ PostgreSQL checkpointer created: {type(checkpointer).__name__}")


def test_postgres_config_creation():
    """Test PostgreSQL config creation with connection string."""
    # Load environment variables
    load_dotenv()
    
    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if not conn_string:
        pytest.skip("POSTGRES_CONNECTION_STRING not found")
    
    # Create config
    postgres_config = PostgresCheckpointerConfig(
        connection_string=conn_string
    )
    
    # Verify config properties
    assert postgres_config.connection_string == conn_string
    assert postgres_config.get_connection_uri() == conn_string
    
    logger.info("✅ PostgreSQL config created successfully")


def test_direct_postgres_connection():
    """Test direct PostgreSQL connection."""
    # Load environment variables
    load_dotenv()
    
    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if not conn_string:
        pytest.skip("POSTGRES_CONNECTION_STRING not found")
    
    # Test direct connection
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 as test")
            result = cur.fetchone()
            assert result[0] == 1
            
            # Check for langgraph tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%checkpoint%'
            """)
            tables = cur.fetchall()
            
            logger.info(f"Found checkpoint tables: {[t[0] for t in tables] if tables else 'None (will be created on first use)'}")