"""Types engine module.

This module provides types functionality for the Haive framework.

Classes:
    DatabaseSourceType: DatabaseSourceType implementation.
"""

from enum import Enum


class DatabaseSourceType(Enum, str):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLSERVER = "sqlserver"
    ORACLE = "oracle"
    MONGODB = "mongodb"
    ASTRADB = "astradb"
    GRAPHQL = "graphql"
    NEO4J = "neo4j"
    REDIS = "redis"
    CASSANDRA = "cassandra"
    COUCHBASE = "couchbase"
