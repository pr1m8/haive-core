"""Document sources module.

Sources represent the data about where documents come from.
They are data models that hold configuration and metadata.
"""

from .registry import register_source, source_registry
from .source_base import (
    BaseSource,
    CloudSource,
    DatabaseSource,
    LocalSource,
    RemoteSource,
)

__all__ = [
    "BaseSource",
    "CloudSource",
    "DatabaseSource",
    "LocalSource",
    "RemoteSource",
    "register_source",
    "source_registry",
]
