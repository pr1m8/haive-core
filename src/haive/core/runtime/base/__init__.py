"""Runtime base components for the Haive framework.

This module provides the base classes and protocols for runtime execution components.
"""

from haive.core.runtime.base.base import RuntimeComponent
from haive.core.runtime.base.protocols import RuntimeComponentProtocol

__all__ = [
    "RuntimeComponent",
    "RuntimeComponentProtocol",
]

