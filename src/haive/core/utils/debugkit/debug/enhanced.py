"""
Enhanced Debugging Utilities

Provides icecream-style debugging with beautiful output and context detection.
"""

import inspect
from pathlib import Path
from typing import Any

try:
    import icecream

    HAS_ICECREAM = True
except ImportError:
    HAS_ICECREAM = False

try:
    import rich  # noqa: F401

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def fallback_ice(*args):
    """Fallback icecream implementation when icecream is not available."""
    if args:
        for _arg in args:
            pass
    else:
        pass


class EnhancedDebugger:
    """Enhanced debugging with icecream-style output."""

    def __init__(self):
        self.debug_enabled = True
        self.debug_history: list[dict[str, Any]] = []

    def ice(self, *args, **kwargs) -> Any:
        """Enhanced print debugging with context (icecream replacement)."""
        if not self.debug_enabled:
            return

        # Get caller information
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name

        # Record in history
        self.debug_history.append(
            {
                "type": "ice",
                "file": filename,
                "line": line_number,
                "function": function_name,
                "args": args,
                "kwargs": kwargs,
            }
        )

        # Use icecream if available, otherwise fallback
        if HAS_ICECREAM:
            if args or kwargs:
                combined_args = list(args)
                for key, value in kwargs.items():
                    combined_args.append(f"{key}={value}")
                return icecream.ic(*combined_args)
            else:
                return icecream.ic()
        else:
            # Fallback implementation
            f"{Path(filename).name}:{line_number} in {function_name}()"

            if args:
                for _arg in args:
                    pass
            else:
                pass

            if kwargs:
                for key, value in kwargs.items():
                    pass

    def enable(self) -> None:
        """Enable enhanced debugging."""
        self.debug_enabled = True

    def disable(self) -> None:
        """Disable enhanced debugging."""
        self.debug_enabled = False

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get debug history."""
        history = self.debug_history[-limit:] if limit else self.debug_history
        return history

    def clear_history(self) -> None:
        """Clear debug history."""
        self.debug_history.clear()


# Global instance for easy import
enhanced_debugger = EnhancedDebugger()
