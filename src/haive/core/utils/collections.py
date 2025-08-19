"""Collections utilities - compatibility alias.

This module provides backward compatibility for imports that expect
haive.core.utils.collections but the actual implementation is in
haive_collections.py
"""

# Import all from the actual collections module
from haive.core.utils.haive_collections import *

# Also make sure the main classes are available
from haive.core.utils.haive_collections import NamedDict

__all__ = ["NamedDict"]

