from enum import Enum
from typing import Dict


class CheckpointerType(str, Enum):
    """Types of checkpointers available in the system."""

    MEMORY = "memory"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"


class CheckpointerMode(str, Enum):
    """Operational modes for checkpointers."""

    SYNC = "sync"
    ASYNC = "async"

    # Special handling for async mode
    # This is needed because 'async' is a Python keyword
    async_ = "async"

    def __eq__(self, other):
        """Custom equality to handle async_ special case."""
        if isinstance(other, str):
            if self.value == "async" and other == "async_":
                return True
            if self.value == "async_" and other == "async":
                return True
        return super().__eq__(other)


class CheckpointStorageMode(str, Enum):
    """Storage modes for checkpointers."""

    FULL = "full"  # Store full history
    SHALLOW = "shallow"  # Store only the most recent state
    CHAIN = "chain"  # Store checkpoints in a chain/linked list

    # For compatibility with earlier versions
    STANDARD = "full"

    def __eq__(self, other):
        """Custom equality to handle compatibility modes."""
        if isinstance(other, str):
            if self.value == "full" and other == "standard":
                return True
            if self.value == "standard" and other == "full":
                return True
        return super().__eq__(other)


class ConnectionOptions:
    """Common connection options for database-backed checkpointers."""

    @staticmethod
    def get_postgres_ssl_modes() -> Dict[str, str]:
        """
        Get valid PostgreSQL SSL modes.

        Returns:
            Dictionary of SSL mode names to descriptions
        """
        return {
            "disable": "No SSL",
            "allow": "Allow SSL connection but don't require",
            "prefer": "Prefer SSL connection but accept non-SSL",
            "require": "Require SSL connection",
            "verify-ca": "Verify server certificate",
            "verify-full": "Verify server certificate and hostname",
        }

    @staticmethod
    def get_postgres_dsn(
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "postgres",
        user: str = "postgres",
        password: str = "postgres",
        ssl_mode: str = "disable",
    ) -> str:
        """
        Create a PostgreSQL DSN connection string.

        Args:
            host: Database host
            port: Database port
            dbname: Database name
            user: Username
            password: Password
            ssl_mode: SSL mode

        Returns:
            PostgreSQL connection string
        """
        import urllib.parse

        encoded_pass = urllib.parse.quote_plus(password)
        return f"postgresql://{user}:{encoded_pass}@{host}:{port}/{dbname}?sslmode={ssl_mode}"
