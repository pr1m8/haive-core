"""
Debug Decorators

Provides decorators for automatic debugging on exceptions and call tracing.
"""

import time
from collections.abc import Callable
from functools import wraps

try:
    import birdseye

    HAS_BIRDSEYE = True
except ImportError:
    HAS_BIRDSEYE = False


class DebugDecorators:
    """Decorators for automatic debugging and tracing."""

    def __init__(self):
        self.debug_enabled = True

    def breakpoint_on_exception(self, func: Callable) -> Callable:
        """Decorator to automatically break into debugger on exceptions."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                if self.debug_enabled:
                    # Import here to avoid circular imports
                    from haive.core.utils.dev.debug_interactive import (
                        interactive_debugger,
                    )

                    interactive_debugger.pdb()
                raise

        return wrapper

    def trace_calls(self, func: Callable) -> Callable:
        """Decorator to trace function calls with birdseye if available."""
        if HAS_BIRDSEYE and self.debug_enabled:
            return birdseye.eye(func)

        # Fallback implementation
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.debug_enabled:
                return func(*args, **kwargs)

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                time.time() - start_time
                return result
            except Exception:
                time.time() - start_time
                raise

        return wrapper

    def enable(self) -> None:
        """Enable debug decorators."""
        self.debug_enabled = True

    def disable(self) -> None:
        """Disable debug decorators."""
        self.debug_enabled = False


# Global instance for easy import
debug_decorators = DebugDecorators()
