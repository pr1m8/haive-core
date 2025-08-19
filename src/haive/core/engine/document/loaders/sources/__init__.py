"""Document sources module.

Sources represent the data about where documents come from.
They are data models that hold configuration and metadata.
"""

from haive.core.engine.document.loaders.sources.registry import register_source, source_registry
from haive.core.engine.document.loaders.sources.source_base import (
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
