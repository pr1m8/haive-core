"""Configuration for store tests."""

import os
from typing import Optional

import pytest


@pytest.fixture(scope="session")
def postgres_connection_string() -> Optional[str]:
    """Get PostgreSQL connection string from environment.

    Returns:
        Connection string if available, None otherwise
    """
    return os.getenv("POSTGRES_CONNECTION_STRING")


@pytest.fixture(scope="session")
def skip_postgres_if_unavailable(postgres_connection_string):
    """Skip PostgreSQL tests if connection string not available."""
    if not postgres_connection_string:
        pytest.skip(
            "PostgreSQL tests require POSTGRES_CONNECTION_STRING environment variable"
        )


@pytest.fixture
def unique_namespace():
    """Generate unique namespace for each test to avoid conflicts."""
    import uuid

    test_id = str(uuid.uuid4())[:8]
    return ("test", "store", test_id)
