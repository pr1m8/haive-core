"""Document loader adapters for the Haive framework.

This module provides adapter implementations that bridge different document
loading approaches and normalize interfaces between various loader types.

Adapters handle the translation between different loader interfaces, data
formats, and execution patterns, ensuring consistent behavior across the
document loading system.

Key Components:
    BaseAdapter: Abstract base class for all loader adapters
    LocalAdapter: Adapter for local file system loaders

Features:
    - Interface normalization between loader types
    - Data format translation and conversion
    - Error handling and retry logic
    - Performance optimization through caching
    - Consistent metadata handling

Examples:
    Using a local file adapter::

        from haive.core.engine.document.loaders.adapters import LocalAdapter

        # Create adapter for local files
        adapter = LocalAdapter()

        # Load document through adapter
        documents = adapter.load("document.pdf")

See Also:
    - Document loaders base classes
    - Source implementations
    - Loader registry system
"""

from haive.core.engine.document.loaders.adapters.base import BaseAdapter
from haive.core.engine.document.loaders.adapters.local import LocalAdapter

__all__ = [
    "BaseAdapter",
    "LocalAdapter",
]
