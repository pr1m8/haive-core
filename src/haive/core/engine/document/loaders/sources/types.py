"""Document loader source types and enums.

This module provides common types and enums used across document loader sources.
"""

from enum import Enum

from haive.core.engine.document.loaders.sources.local.types import (
    CodeFileType,
    LocalSourceFileType,
    ProgrammingLanguage,
)
from haive.core.engine.document.loaders.sources.database.types import DatabaseSourceType


class SourceType(str, Enum):
    """High-level source type classification."""

    LOCAL = "local"
    WEB = "web"
    DATABASE = "database"
    CLOUD = "cloud"
    API = "api"


__all__ = [
    "CodeFileType",
    "DatabaseSourceType",
    "LocalSourceFileType",
    "ProgrammingLanguage",
    "SourceType",
]


