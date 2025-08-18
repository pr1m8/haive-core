"""
Enhanced Debugging Utilities

A unified interface for all debugging capabilities including enhanced print debugging,
interactive debugging, decorators, and inspection utilities.

Examples:
    Enhanced debugging (icecream replacement):
        >>> from haive.core.utils.dev.debugging import debug
        >>> debug.ice("Hello", variable=42)

    Interactive debugging:
        >>> debug.pdb()  # Enhanced pdb
        >>> debug.web(port=8080)  # Web-based debugging
        >>> debug.visual()  # Visual debugging with pudb

    Automatic exception debugging:
        >>> @debug.breakpoint_on_exception
        ... def risky_function():
        ...     # Will auto-debug on exceptions
        ...     pass

    Call tracing:
        >>> @debug.trace_calls
        ... def tracked_function():
        ...     # Will show call trace
        ...     pass

    Variable inspection:
        >>> debug.locals_inspect()  # See local variables
        >>> debug.stack_trace()     # See call stack
"""

from haive.core.utils.debugkit.debug.decorators import debug_decorators
from haive.core.utils.debugkit.debug.enhanced import enhanced_debugger
from haive.core.utils.debugkit.debug.inspection import debug_inspector
from haive.core.utils.debugkit.debug.interactive import interactive_debugger


class DebugUtilities:
    """Unified interface for all debugging utilities."""

    def __init__(self):
        self.enhanced = enhanced_debugger
        self.interactive = interactive_debugger
        self.decorators = debug_decorators
        self.inspector = debug_inspector

    # Enhanced debugging methods
    def ice(self, *args, **kwargs):
        """Enhanced print debugging with context (icecream replacement)."""
        return self.enhanced.ice(*args, **kwargs)

    # Interactive debugging methods
    def pdb(self, condition: bool = True) -> None:
        """Start interactive debugging session."""
        self.interactive.pdb(condition)

    def web(self, port: int = 5555, condition: bool = True) -> None:
        """Start web-based debugging session."""
        self.interactive.web(port, condition)

    def visual(self, condition: bool = True) -> None:
        """Start visual debugging with pudb."""
        self.interactive.visual(condition)

    # Decorators
    def breakpoint_on_exception(self, func):
        """Decorator to automatically break into debugger on exceptions."""
        return self.decorators.breakpoint_on_exception(func)

    def trace_calls(self, func):
        """Decorator to trace function calls."""
        return self.decorators.trace_calls(func)

    # Inspection methods
    def stack_trace(self, limit: int | None = None) -> str:
        """Get formatted stack trace."""
        return self.inspector.stack_trace(limit)

    def locals_inspect(self) -> dict:
        """Inspect local variables in the calling frame."""
        return self.inspector.locals_inspect()

    def globals_inspect(self) -> dict:
        """Inspect global variables in the calling frame."""
        return self.inspector.globals_inspect()

    # Management methods
    def enable(self) -> None:
        """Enable all debugging utilities."""
        self.enhanced.enable()
        self.interactive.enable()
        self.decorators.enable()
        self.inspector.enable()

    def disable(self) -> None:
        """Disable all debugging utilities."""
        self.enhanced.disable()
        self.interactive.disable()
        self.decorators.disable()
        self.inspector.disable()

    def history(self, limit: int | None = None) -> list:
        """Get debug history."""
        return self.enhanced.get_history(limit)

    def clear_history(self) -> None:
        """Clear debug history."""
        self.enhanced.clear_history()

    def status(self) -> dict:
        """Get status of available debugging tools."""
        return self.interactive.status()


# Global instance for easy import
debug = DebugUtilities()
