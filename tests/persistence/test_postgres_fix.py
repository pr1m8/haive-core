#!/usr/bin/env python3
"""Test script to verify the PostgreSQL thread duplicate key fix."""

import asyncio
import logging
import os

from psycopg_pool import ConnectionPool

from haive.core.persistence.postgres_saver_with_thread_creation import (
    AsyncPostgresSaverWithThreadCreation,
    PostgresSaverWithThreadCreation,
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database connection string (adjust as needed)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/haive_dev")


def test_sync_saver():
    """Test the synchronous PostgresSaver with thread creation."""
    logger.info("Testing synchronous PostgresSaver...")

    # Create connection pool
    with ConnectionPool(DATABASE_URL, min_size=1, max_size=10) as pool:
        # Create saver
        saver = PostgresSaverWithThreadCreation(pool)

        # Test thread creation
        thread_id = "test_thread_sync_123"

        # First call - should create thread
        logger.info(f"First call to ensure thread {thread_id} exists...")
        saver._ensure_thread_exists(thread_id)
        logger.info("First call succeeded!")

        # Second call - should update existing thread
        logger.info(f"Second call to ensure thread {thread_id} exists...")
        saver._ensure_thread_exists(thread_id)
        logger.info("Second call succeeded!")

        # Clear cache and try again
        saver._thread_creation_cache.clear()
        logger.info(f"Third call (cache cleared) to ensure thread {thread_id} exists...")
        saver._ensure_thread_exists(thread_id)
        logger.info("Third call succeeded!")


async def test_async_saver():
    """Test the asynchronous PostgresSaver with thread creation."""
    logger.info("Testing asynchronous PostgresSaver...")

    # Import async psycopg
    from psycopg_pool import AsyncConnectionPool

    # Create async connection pool
    async with AsyncConnectionPool(DATABASE_URL, min_size=1, max_size=10) as pool:
        # Create saver
        saver = AsyncPostgresSaverWithThreadCreation(pool)

        # Test thread creation
        thread_id = "test_thread_async_456"

        # First call - should create thread
        logger.info(f"First call to ensure thread {thread_id} exists...")
        await saver._ensure_thread_exists(thread_id)
        logger.info("First call succeeded!")

        # Second call - should update existing thread
        logger.info(f"Second call to ensure thread {thread_id} exists...")
        await saver._ensure_thread_exists(thread_id)
        logger.info("Second call succeeded!")

        # Clear cache and try again
        saver._thread_creation_cache.clear()
        logger.info(f"Third call (cache cleared) to ensure thread {thread_id} exists...")
        await saver._ensure_thread_exists(thread_id)
        logger.info("Third call succeeded!")


def main():
    """Run all tests."""
    logger.info("Starting PostgreSQL thread duplicate key fix tests...")

    try:
        # Test synchronous version
        test_sync_saver()
        logger.info("✅ Synchronous tests passed!")

        # Test asynchronous version
        asyncio.run(test_async_saver())
        logger.info("✅ Asynchronous tests passed!")

        logger.info("✅ All tests passed! The fix is working correctly.")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
