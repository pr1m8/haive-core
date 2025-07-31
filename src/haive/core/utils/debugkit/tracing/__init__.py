"""
Tracing utilities submodule.

This submodule provides execution tracing capabilities including
call tracing, performance measurement, and distributed tracing support.
"""

try:
    from haive.core.utils.debugkit.tracing.execution import ExecutionTrace
except ImportError:
    from haive.core.utils.debugkit.fallbacks import FallbackTrace as ExecutionTrace

# Create default trace instance
trace = ExecutionTrace()

__all__ = [
    "trace",
    "ExecutionTrace",
]
