"""Override for LangGraph's PostgresSaver to properly disable prepared statements."""

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver as BasePostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class PostgresSaverNoPreparedStatements(BasePostgresSaver):
    """PostgresSaver that properly disables prepared statements."""

    @classmethod
    def from_conn_string(
        cls,
        conn_string: str,
    ) -> "PostgresSaverNoPreparedStatements":
        """Create a PostgresSaver from a connection string.

        Overrides the base implementation to properly disable prepared statements.
        """
        # Create connection with prepare_threshold=None to disable prepared statements
        conn = psycopg.connect(
            conn_string,
            autocommit=True,
            prepare_threshold=None,  # Disable prepared statements
            row_factory=dict_row,
        )
        return cls(conn)

    def __init__(
        self,
        conn: psycopg.Connection | ConnectionPool,
        pipe: psycopg.Pipeline | None = None,
        serde=None,
    ) -> None:
        """Initialize with connection that has prepared statements disabled."""
        # If it's a raw connection, ensure prepare_threshold is None
        if isinstance(conn, psycopg.Connection):
            # Can't modify existing connection, but we can warn
            if (
                hasattr(conn, "prepare_threshold")
                and conn.prepare_threshold is not None
            ):
                import logging

                logging.getLogger(__name__).warning(
                    "Connection has prepare_threshold enabled. This may cause conflicts."
                )

        # Call parent init with all parameters
        super().__init__(conn, pipe=pipe, serde=serde)

        # CRITICAL: Override jsonplus_serde to use our secure serializer for metadata
        if serde is not None:
            self.jsonplus_serde = serde
            import logging

            logging.getLogger(__name__).info(
                f"Set jsonplus_serde to secure serializer: {type(serde).__name__}"
            )


import psycopg
from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver as BaseAsyncPostgresSaver,
)
from psycopg_pool import AsyncConnectionPool


class AsyncPostgresSaverNoPreparedStatements(BaseAsyncPostgresSaver):
    """Async PostgresSaver that properly disables prepared statements."""

    @classmethod
    async def from_conn_string(
        cls,
        conn_string: str,
    ) -> "AsyncPostgresSaverNoPreparedStatements":
        """Create an AsyncPostgresSaver from a connection string.

        Overrides the base implementation to properly disable prepared statements.
        """
        # Create async connection with prepare_threshold=None to disable prepared statements
        conn = await psycopg.AsyncConnection.connect(
            conn_string,
            autocommit=True,
            prepare_threshold=None,  # Disable prepared statements
            row_factory=dict_row,
        )
        return cls(conn)

    def __init__(
        self,
        conn: psycopg.AsyncConnection | AsyncConnectionPool,
        pipe: psycopg.AsyncPipeline | None = None,
        serde=None,
    ) -> None:
        """Initialize with async connection that has prepared statements disabled."""
        # If it's a raw connection, ensure prepare_threshold is None
        if hasattr(conn, "prepare_threshold") and conn.prepare_threshold is not None:
            import logging

            logging.getLogger(__name__).warning(
                "Connection has prepare_threshold enabled. This may cause conflicts."
            )

        # Call parent init with all parameters
        super().__init__(conn, pipe=pipe, serde=serde)

        # CRITICAL: Override jsonplus_serde to use our secure serializer for metadata
        if serde is not None:
            self.jsonplus_serde = serde
            import logging

            logging.getLogger(__name__).info(
                f"Set jsonplus_serde to secure serializer: {type(serde).__name__}"
            )
