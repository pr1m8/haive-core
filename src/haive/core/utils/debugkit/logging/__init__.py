"""
Logging utilities submodule.

This submodule provides structured logging capabilities with
correlation IDs, context management, and rich output formatting.
"""

try:
    from haive.core.utils.debugkit.logging.structured import StructuredLog
except ImportError:
    from haive.core.utils.debugkit.fallbacks import FallbackLog as StructuredLog

# Create default log instance
log = StructuredLog()

__all__ = [
    "StructuredLog",
    "log",
]
