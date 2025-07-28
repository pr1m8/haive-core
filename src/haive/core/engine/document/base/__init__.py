"""Base package.

This package provides base functionality for the Haive framework.

Modules:
    schema: Schema implementation.
"""

from haive.core.engine.document.base.schema import (
    DocumentEngineInputSchema,
    DocumentEngineOutputSchema,
)

__all__ = ["DocumentEngineInputSchema", "DocumentEngineOutputSchema"]
