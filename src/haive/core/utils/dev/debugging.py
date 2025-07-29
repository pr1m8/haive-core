"""
Enhanced Debugging Utilities

Provides powerful debugging tools including enhanced print debugging,
interactive debugging, and visual debugging capabilities.
"""

import inspect
import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Try to import icecream for enhanced print debugging
try:
    from icecream import ic

    ic.configureOutput(includeContext=True, contextAbsPath=False)
    HAS_ICECREAM = True
except ImportError:
    # Fallback to print if icecream not available
    def ic(*args):
        if args:
            for arg in args:
                print(f"ic| {arg}")
        else:
            print(f"ic| {inspect.stack()[1].filename}:{inspect.stack()[1].lineno}")

    HAS_ICECREAM = False

# Try to import enhanced pdb variants
try:
    import pdbpp as pdb

    HAS_PDBPP = True
except ImportError:
    try:
        import ipdb as pdb

        HAS_IPDB = True
    except ImportError:
        import pdb

        HAS_PDBPP = False
        HAS_IPDB = False

# Try to import web-pdb for remote debugging
try:
    import web_pdb

    HAS_WEB_PDB = True
except ImportError:
    HAS_WEB_PDB = False

# Try to import pudb for visual debugging
try:
    import pudb

    HAS_PUDB = True
except ImportError:
    HAS_PUDB = False

# Try to import birdseye for visual execution tracing
try:
    import birdseye

    HAS_BIRDSEYE = True
except ImportError:
    HAS_BIRDSEYE = False


class DebugUtilities:
    """Enhanced debugging utilities with multiple debugging backends."""

    def __init__(self):
        self.debug_enabled = True
        self.debug_history: List[Dict[str, Any]] = []

    def ice(self, *args, **kwargs) -> Any:
        """Enhanced print debugging using icecream or fallback."""
        if not self.debug_enabled:
            return

        # Store debug info
        frame = inspect.currentframe().f_back
        self.debug_history.append(
            {
                "type": "icecream",
                "file": frame.f_code.co_filename,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
                "args": args,
                "kwargs": kwargs,
            }
        )

        if HAS_ICECREAM:
            return ic(*args, **kwargs)
        else:
            # Enhanced fallback
            frame_info = inspect.getframeinfo(frame)
            filename = Path(frame_info.filename).name
            context = f"{filename}:{frame_info.lineno} in {frame_info.function}()"

            if args:
                for arg in args:
                    print(f"🍦 {context} | {arg}")
            else:
                print(f"🍦 {context}")

            if kwargs:
                for key, value in kwargs.items():
                    print(f"🍦 {context} | {key}={value}")

    def pdb(self, condition: bool = True) -> None:
        """Enhanced pdb debugging with best available debugger."""
        if not self.debug_enabled or not condition:
            return

        frame = inspect.currentframe().f_back
        self.debug_history.append(
            {
                "type": "pdb",
                "file": frame.f_code.co_filename,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
            }
        )

        print(
            f"🐛 Debugger started at {Path(frame.f_code.co_filename).name}:{frame.f_lineno}"
        )

        if HAS_PDBPP:
            print("Using pdb++ (enhanced debugger)")
        elif HAS_IPDB:
            print("Using ipdb (IPython debugger)")
        else:
            print("Using standard pdb")

        pdb.set_trace()

    def web(self, port: int = 5555, condition: bool = True) -> None:
        """Web-based debugging interface."""
        if not self.debug_enabled or not condition:
            return

        if not HAS_WEB_PDB:
            print("⚠️  web-pdb not available, falling back to standard pdb")
            self.pdb(condition)
            return

        frame = inspect.currentframe().f_back
        self.debug_history.append(
            {
                "type": "web_pdb",
                "file": frame.f_code.co_filename,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
                "port": port,
            }
        )

        print(f"🌐 Web debugger started at http://localhost:{port}")
        print(f"📍 Location: {Path(frame.f_code.co_filename).name}:{frame.f_lineno}")
        web_pdb.set_trace(port=port)

    def visual(self, condition: bool = True) -> None:
        """Visual debugging with pudb."""
        if not self.debug_enabled or not condition:
            return

        if not HAS_PUDB:
            print("⚠️  pudb not available, falling back to enhanced pdb")
            self.pdb(condition)
            return

        frame = inspect.currentframe().f_back
        self.debug_history.append(
            {
                "type": "pudb",
                "file": frame.f_code.co_filename,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
            }
        )

        print(
            f"👁️  Visual debugger started at {Path(frame.f_code.co_filename).name}:{frame.f_lineno}"
        )
        pudb.set_trace()

    def breakpoint_on_exception(self, func: Callable) -> Callable:
        """Decorator to automatically break on exceptions."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"💥 Exception in {func.__name__}: {e}")
                print(f"📍 Breaking into debugger...")
                self.pdb()
                raise

        return wrapper

    def trace_calls(self, func: Callable) -> Callable:
        """Decorator to trace function calls with birdseye if available."""
        if HAS_BIRDSEYE:
            return birdseye.eye(func)
        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                self.ice(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
                result = func(*args, **kwargs)
                self.ice(f"{func.__name__} returned: {result}")
                return result

            return wrapper

    def stack_trace(self, limit: Optional[int] = None) -> str:
        """Get formatted stack trace."""
        stack = traceback.format_stack(limit=limit)
        formatted = "📚 Call Stack:\n" + "".join(stack)
        print(formatted)
        return formatted

    def locals_inspect(self) -> Dict[str, Any]:
        """Inspect local variables in the calling frame."""
        frame = inspect.currentframe().f_back
        locals_dict = frame.f_locals.copy()

        print("🔍 Local Variables:")
        for name, value in locals_dict.items():
            if not name.startswith("_"):
                print(f"  {name} = {repr(value)}")

        return locals_dict

    def globals_inspect(self) -> Dict[str, Any]:
        """Inspect global variables in the calling frame."""
        frame = inspect.currentframe().f_back
        globals_dict = {
            k: v
            for k, v in frame.f_globals.items()
            if not k.startswith("_") and not inspect.ismodule(v)
        }

        print("🌍 Global Variables:")
        for name, value in globals_dict.items():
            print(f"  {name} = {repr(value)}")

        return globals_dict

    def enable(self) -> None:
        """Enable debugging utilities."""
        self.debug_enabled = True
        print("✅ Debug utilities enabled")

    def disable(self) -> None:
        """Disable debugging utilities."""
        self.debug_enabled = False
        print("❌ Debug utilities disabled")

    def history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get debug history."""
        history = self.debug_history[-limit:] if limit else self.debug_history

        print("📝 Debug History:")
        for i, entry in enumerate(history, 1):
            file_name = Path(entry["file"]).name
            print(
                f"  {i}. {entry['type']} at {file_name}:{entry['line']} in {entry['function']}()"
            )

        return history

    def clear_history(self) -> None:
        """Clear debug history."""
        self.debug_history.clear()
        print("🧹 Debug history cleared")

    def status(self) -> Dict[str, Any]:
        """Get status of available debugging tools."""
        status = {
            "debug_enabled": self.debug_enabled,
            "tools_available": {
                "icecream": HAS_ICECREAM,
                "pdbpp": HAS_PDBPP,
                "ipdb": HAS_IPDB,
                "web_pdb": HAS_WEB_PDB,
                "pudb": HAS_PUDB,
                "birdseye": HAS_BIRDSEYE,
            },
            "debug_history_count": len(self.debug_history),
        }

        print("🔧 Debug Tools Status:")
        print(f"  Debug enabled: {'✅' if status['debug_enabled'] else '❌'}")
        print(f"  History entries: {status['debug_history_count']}")
        print("  Available tools:")
        for tool, available in status["tools_available"].items():
            print(f"    {tool}: {'✅' if available else '❌'}")

        return status


# Create global debug instance
debug = DebugUtilities()
