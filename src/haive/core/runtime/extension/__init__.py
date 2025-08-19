"""Runtime extension system for the Haive framework.

This module provides the extension system for runtime components, allowing
for customization and enhancement of runtime behavior.
"""

from haive.core.runtime.extension.base import Extension
from haive.core.runtime.extension.protocols import ExtensionProtocol

__all__ = [
    "Extension",
    "ExtensionProtocol",
]

