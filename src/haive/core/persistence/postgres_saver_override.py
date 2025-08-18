"""PostgreSQL persistence utilities with Pydantic support.

This module provides utilities for handling Pydantic models in PostgreSQL
persistence. The main functionality is the JSON encoder configuration that
ensures Pydantic models are properly serialized to JSONB columns.

The override classes are kept for backward compatibility and as a fallback
when using connection strings directly. However, the preferred approach is
to configure the connection pool with the configure parameter.
"""

import json
from datetime import datetime

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver as BasePostgresSaver
from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver as BaseAsyncPostgresSaver,
)
from psycopg.rows import dict_row
from psycopg.types.json import set_json_dumps
from pydantic import BaseModel


def pydantic_aware_json_dumps(obj):
    """JSON encoder that handles Pydantic models.

    This encoder ensures that Pydantic models are properly serialized
    when stored in PostgreSQL JSONB columns.
    """

    class PydanticEncoder(json.JSONEncoder):
        def default(self, o):
            """Default.

            Args:
                o: [TODO: Add description]
            """
            if isinstance(o, BaseModel):
                return o.model_dump()
            if isinstance(o, datetime):
                return o.isoformat()
            return super().default(o)

    return json.dumps(obj, cls=PydanticEncoder)


def configure_postgres_json(connection):
    """Configure a PostgreSQL connection to handle Pydantic JSON serialization.

    Args:
        connection: A psycopg connection object
    """

    set_json_dumps(pydantic_aware_json_dumps, context=connection)


# Override classes for backward compatibility and direct connection string usage


class PostgresSaverNoPreparedStatements(BasePostgresSaver):
    """PostgresSaver that disables prepared statements and handles Pydantic models.

    This class is kept for backward compatibility and for cases where you need
    to use from_conn_string directly. The preferred approach is to configure
    the connection pool with the configure parameter in postgres_config.py.
    """

    @classmethod
    def from_conn_string(cls, conn_string: str) -> "PostgresSaverNoPreparedStatements":
        """Create a PostgresSaver with proper configuration."""
        conn = psycopg.connect(
            conn_string,
            autocommit=True,
            prepare_threshold=None,
            row_factory=dict_row,
        )
        configure_postgres_json(conn)
        return cls(conn)


class AsyncPostgresSaverNoPreparedStatements(BaseAsyncPostgresSaver):
    """Async PostgresSaver with proper configuration.

    This class is kept for backward compatibility and for cases where you need
    to use from_conn_string directly. The preferred approach is to configure
    the connection pool with the configure parameter in postgres_config.py.
    """

    @classmethod
    async def from_conn_string(
        cls, conn_string: str
    ) -> "AsyncPostgresSaverNoPreparedStatements":
        """Create an AsyncPostgresSaver with proper configuration."""
        conn = await psycopg.AsyncConnection.connect(
            conn_string,
            autocommit=True,
            prepare_threshold=None,
            row_factory=dict_row,
        )
        configure_postgres_json(conn)
        return cls(conn)
