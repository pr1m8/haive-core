from enum import Enum


class DatabaseSourceType(str, Enum):
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
