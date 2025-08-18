"""Runtime extension system for the Haive framework.

This module provides the extension system for runtime components, allowing
for customization and enhancement of runtime behavior.
"""

from .base import Extension
from .protocols import ExtensionProtocol

__all__ = [
    "Extension",
    "ExtensionProtocol",
]
