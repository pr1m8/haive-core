"""
Debug Inspection Utilities

Provides utilities for inspecting stack traces, local/global variables, and runtime state.
"""

import inspect
import traceback
from typing import Any


class DebugInspector:
    """Utilities for inspecting runtime state and variables."""

    def __init__(self):
        self.debug_enabled = True

    def stack_trace(self, limit: int | None = None) -> str:
        """Get formatted stack trace."""
        if not self.debug_enabled:
            return ""

        stack = traceback.format_stack(limit=limit)
        formatted = "📚 Call Stack:\n" + "".join(stack)
        return formatted

    def locals_inspect(self) -> dict[str, Any]:
        """Inspect local variables in the calling frame."""
        if not self.debug_enabled:
            return {}

        frame = inspect.currentframe().f_back
        locals_dict = frame.f_locals.copy()

        # Filter out private variables
        filtered_locals = {
            name: value
            for name, value in locals_dict.items()
            if not name.startswith("_")
        }

        return filtered_locals

    def globals_inspect(self) -> dict[str, Any]:
        """Inspect global variables in the calling frame."""
        if not self.debug_enabled:
            return {}

        frame = inspect.currentframe().f_back
        globals_dict = frame.f_globals.copy()

        # Filter to only user-defined globals
        filtered_globals = {
            name: value
            for name, value in globals_dict.items()
            if not name.startswith("_") and not inspect.ismodule(value)
        }

        return filtered_globals

    def frame_info(self) -> dict[str, Any]:
        """Get detailed information about the current frame."""
        if not self.debug_enabled:
            return {}

        frame = inspect.currentframe().f_back
        frame_info = inspect.getframeinfo(frame)

        return {
            "filename": frame_info.filename,
            "lineno": frame_info.lineno,
            "function": frame_info.function,
            "code_context": frame_info.code_context,
            "locals_count": len(frame.f_locals),
            "globals_count": len(frame.f_globals),
        }

    def get_caller_chain(self, depth: int = 5) -> list[dict[str, Any]]:
        """Get information about the call chain."""
        if not self.debug_enabled:
            return []

        chain = []
        frame = inspect.currentframe()

        try:
            for i in range(depth + 1):  # +1 to skip this function
                if frame is None:
                    break
                frame = frame.f_back
                if frame is None:
                    break

                frame_info = inspect.getframeinfo(frame)
                chain.append(
                    {
                        "depth": i,
                        "filename": frame_info.filename,
                        "lineno": frame_info.lineno,
                        "function": frame_info.function,
                        "code_context": (
                            frame_info.code_context[0].strip()
                            if frame_info.code_context
                            else None
                        ),
                    }
                )
        except Exception:
            pass  # Handle edge cases gracefully

        return chain

    def enable(self) -> None:
        """Enable debug inspection."""
        self.debug_enabled = True

    def disable(self) -> None:
        """Disable debug inspection."""
        self.debug_enabled = False


# Global instance for easy import
debug_inspector = DebugInspector()
