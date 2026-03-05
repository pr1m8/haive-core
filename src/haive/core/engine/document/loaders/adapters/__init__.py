"""Document loader adapters."""

from haive.core.engine.document.loaders.adapters.base import LoaderAdapter

# Alias for backwards compatibility
BaseAdapter = LoaderAdapter
create_loader = None  # placeholder

__all__ = [
    "BaseAdapter",
    "LoaderAdapter",
]
