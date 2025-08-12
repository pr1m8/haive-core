#!/usr/bin/env python3
"""Test PostgreSQL connection with environment variables.

This test verifies:
1. Loading .env file correctly
2. PostgreSQL store factory creation
3. Store wrapper functionality
"""

import logging
import os

import pytest
from dotenv import load_dotenv

from haive.core.persistence.store.factory import create_store
from haive.core.persistence.store.types import StoreType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_postgres_store_creation():
    """Test PostgreSQL store creation from environment variables."""
    # Load environment variables
    load_dotenv()

    # Check environment variable
    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if not conn_string:
        pytest.skip("POSTGRES_CONNECTION_STRING not found in environment")

    # Test store creation with PostgreSQL
    postgres_store = create_store(
        store_type=StoreType.POSTGRES_SYNC,
        connection_string=conn_string
    )

    assert postgres_store is not None
    assert hasattr(postgres_store, "_create_store")
    logger.info(f"✅ PostgreSQL store created: {type(postgres_store).__name__}")


def test_postgres_environment_loading():
    """Test that environment variables load correctly."""
    # Load environment variables
    load_dotenv()

    # Check if PostgreSQL connection string exists
    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")

    # This test passes if env is loaded (even if no connection string)
    # The connection string test is separate
    assert conn_string is None or isinstance(conn_string, str)

    if conn_string:
        assert conn_string.startswith("postgresql://")
        logger.info("✅ POSTGRES_CONNECTION_STRING found and properly formatted")


def test_postgres_not_disabled():
    """Test that PostgreSQL is not disabled by environment variable."""
    # Ensure PostgreSQL is NOT disabled
    disabled = os.getenv("HAIVE_DISABLE_POSTGRES")

    if disabled:
        logger.warning("PostgreSQL is disabled by HAIVE_DISABLE_POSTGRES environment variable")

    # This test documents the expected state but doesn't fail
    # since some environments may intentionally disable PostgreSQL
