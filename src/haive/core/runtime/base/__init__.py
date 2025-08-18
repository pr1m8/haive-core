"""Runtime base components for the Haive framework.

This module provides the base classes and protocols for runtime execution components.
"""

from .base import RuntimeComponent
from .protocols import RuntimeComponentProtocol

__all__ = [
    "RuntimeComponent",
    "RuntimeComponentProtocol",
]
