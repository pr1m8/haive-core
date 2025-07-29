"""
Code Tracing and Inspection Utilities

Provides powerful tracing capabilities for understanding code execution flow,
function calls, variable changes, and performance bottlenecks.
"""

import functools
import inspect
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Try to import tracing tools
try:
    import hunter

    HAS_HUNTER = True
except ImportError:
    HAS_HUNTER = False

try:
    import pysnooper

    HAS_PYSNOOPER = True
except ImportError:
    HAS_PYSNOOPER = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class CallTracker:
    """Track function calls and execution flow."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.call_stack: list[str] = []
        self.enabled = False
        self.filters: set[str] = set()
        self.console = Console() if HAS_RICH else None

    def enable(self) -> None:
        """Enable call tracking."""
        self.enabled = True

    def disable(self) -> None:
        """Disable call tracking."""
        self.enabled = False

    def add_filter(self, pattern: str) -> None:
        """Add a filter pattern for calls to track."""
        self.filters.add(pattern)

    def remove_filter(self, pattern: str) -> None:
        """Remove a filter pattern."""
        self.filters.discard(pattern)

    def should_track(self, function_name: str, filename: str) -> bool:
        """Check if a call should be tracked based on filters."""
        if not self.filters:
            return True

        return any(
            pattern in function_name or pattern in filename for pattern in self.filters
        )

    def track_call(self, func: Callable) -> Callable:
        """Decorator to track function calls."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            function_name = func.__name__
            filename = Path(func.__code__.co_filename).name

            if not self.should_track(function_name, filename):
                return func(*args, **kwargs)

            # Record call start
            start_time = time.time()
            call_info = {
                "function": function_name,
                "file": filename,
                "line": func.__code__.co_firstlineno,
                "args": args,
                "kwargs": kwargs,
                "start_time": start_time,
                "depth": len(self.call_stack),
            }

            indent = "  " * len(self.call_stack)
            call_signature = f"{function_name}({len(args)} args, {len(kwargs)} kwargs)"

            if HAS_RICH and self.console:
                self.console.print(f"{indent}→ {call_signature}", style="blue")
            else:
                pass

            self.call_stack.append(function_name)

            try:
                result = func(*args, **kwargs)

                # Record successful return
                end_time = time.time()
                duration = end_time - start_time
                call_info.update(
                    {"result": result, "duration": duration, "status": "success"}
                )

                if HAS_RICH and self.console:
                    self.console.print(
                        f"{indent}← {function_name} ({duration:.3f}s)", style="green"
                    )
                else:
                    pass

                return result

            except Exception as e:
                # Record exception
                end_time = time.time()
                duration = end_time - start_time
                call_info.update(
                    {"exception": str(e), "duration": duration, "status": "error"}
                )

                if HAS_RICH and self.console:
                    self.console.print(
                        f"{indent}✗ {function_name} ({duration:.3f}s): {e}", style="red"
                    )
                else:
                    pass

                raise
            finally:
                self.calls.append(call_info)
                self.call_stack.pop()

        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get call statistics."""
        if not self.calls:
            return {"total_calls": 0}

        total_calls = len(self.calls)
        successful_calls = len([c for c in self.calls if c.get("status") == "success"])
        failed_calls = len([c for c in self.calls if c.get("status") == "error"])

        durations = [c.get("duration", 0) for c in self.calls if "duration" in c]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        function_counts = {}
        for call in self.calls:
            func_name = call.get("function", "unknown")
            function_counts[func_name] = function_counts.get(func_name, 0) + 1

        stats = {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "average_duration": avg_duration,
            "max_duration": max_duration,
            "function_counts": function_counts,
        }

        if HAS_RICH and self.console:
            table = Table(title="Call Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Calls", str(total_calls))
            table.add_row("Successful", str(successful_calls))
            table.add_row("Failed", str(failed_calls))
            table.add_row("Avg Duration", f"{avg_duration:.3f}s")
            table.add_row("Max Duration", f"{max_duration:.3f}s")

            self.console.print(table)
        else:
            for key, _value in stats.items():
                if key != "function_counts":
                    pass

        return stats

    def clear(self) -> None:
        """Clear call history."""
        self.calls.clear()
        self.call_stack.clear()


class VariableTracker:
    """Track variable changes during execution."""

    def __init__(self):
        self.tracked_vars: dict[str, Any] = {}
        self.changes: list[dict[str, Any]] = []
        self.enabled = False

    def enable(self) -> None:
        """Enable variable tracking."""
        self.enabled = True

    def disable(self) -> None:
        """Disable variable tracking."""
        self.enabled = False

    def track(self, name: str, value: Any) -> None:
        """Track a variable change."""
        if not self.enabled:
            return

        old_value = self.tracked_vars.get(name)

        if old_value != value:
            change_info = {
                "name": name,
                "old_value": old_value,
                "new_value": value,
                "timestamp": time.time(),
                "caller": self._get_caller_info(),
            }

            self.changes.append(change_info)
            self.tracked_vars[name] = value

    def _get_caller_info(self) -> dict[str, Any]:
        """Get information about the caller."""
        frame = inspect.currentframe().f_back.f_back
        if frame:
            return {
                "file": Path(frame.f_code.co_filename).name,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
            }
        return {}

    def get_history(self, var_name: str | None = None) -> list[dict[str, Any]]:
        """Get change history for a variable or all variables."""
        if var_name:
            return [c for c in self.changes if c["name"] == var_name]
        return self.changes.copy()

    def clear(self) -> None:
        """Clear tracking history."""
        self.tracked_vars.clear()
        self.changes.clear()


class TracingUtilities:
    """Enhanced tracing utilities with multiple backends."""

    def __init__(self):
        self.call_tracker = CallTracker()
        self.var_tracker = VariableTracker()
        self.console = Console() if HAS_RICH else None

    def calls(
        self, func: Callable | None = None, filters: list[str] | None = None
    ) -> Callable:
        """Trace function calls."""
        if filters:
            for pattern in filters:
                self.call_tracker.add_filter(pattern)

        self.call_tracker.enable()

        if func:
            return self.call_tracker.track_call(func)
        else:
            return self.call_tracker.track_call

    def snoop(self, func: Callable | None = None, **kwargs) -> Callable:
        """Trace function execution with pysnooper if available."""
        if not HAS_PYSNOOPER:
            return self.calls(func)

        if func:
            return pysnooper.snoop(**kwargs)(func)
        else:
            return pysnooper.snoop(**kwargs)

    def hunt(self, condition: str = "call", **kwargs) -> None:
        """Use hunter for advanced tracing if available."""
        if not HAS_HUNTER:
            return

        hunter.trace(condition, **kwargs)

    def stack(self, limit: int | None = None) -> str:
        """Get formatted call stack."""
        stack = traceback.format_stack(limit=limit)
        formatted = "📚 Call Stack:\n" + "".join(stack)

        if HAS_RICH and self.console:
            from rich.syntax import Syntax

            syntax = Syntax(formatted, "python", theme="monokai")
            panel = Panel(syntax, title="Call Stack", border_style="blue")
            self.console.print(panel)
        else:
            pass

        return formatted

    def vars(self, **kwargs) -> None:
        """Track variable changes."""
        frame = inspect.currentframe().f_back
        local_vars = frame.f_locals

        self.var_tracker.enable()

        for name, value in kwargs.items():
            self.var_tracker.track(name, value)

        # Track all local variables if no specific ones provided
        if not kwargs:
            for name, value in local_vars.items():
                if not name.startswith("_"):
                    self.var_tracker.track(name, value)

    @contextmanager
    def trace_context(self, name: str = "trace"):
        """Context manager for tracing a block of code."""
        start_time = time.time()

        try:
            yield self
        except Exception:
            raise
        finally:
            time.time() - start_time

    def profile_calls(self, func: Callable) -> Callable:
        """Profile function calls with detailed timing."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Get memory usage if possible
            memory_before = self._get_memory_usage()

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                end_time - start_time
                memory_after = self._get_memory_usage()
                memory_delta = (
                    memory_after - memory_before
                    if memory_after and memory_before
                    else None
                )

                if memory_delta:
                    pass

                return result

            except Exception:
                end_time = time.time()
                end_time - start_time
                raise

        return wrapper

    def _get_memory_usage(self) -> float | None:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None

    def stats(self) -> dict[str, Any]:
        """Get comprehensive tracing statistics."""
        call_stats = self.call_tracker.get_stats()
        var_changes = len(self.var_tracker.changes)

        stats = {
            "call_stats": call_stats,
            "variable_changes": var_changes,
            "tracked_variables": len(self.var_tracker.tracked_vars),
        }

        if HAS_RICH and self.console:
            table = Table(title="Tracing Statistics")
            table.add_column("Category", style="cyan")
            table.add_column("Count", style="green")

            table.add_row("Total Calls", str(call_stats.get("total_calls", 0)))
            table.add_row("Variable Changes", str(var_changes))
            table.add_row("Tracked Variables", str(len(self.var_tracker.tracked_vars)))

            self.console.print(table)

        return stats

    def clear(self) -> None:
        """Clear all tracing data."""
        self.call_tracker.clear()
        self.var_tracker.clear()

    def report(self, filename: str | None = None) -> str:
        """Generate a comprehensive tracing report."""
        report_lines = [
            "# Tracing Report",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Call Statistics",
        ]

        call_stats = self.call_tracker.get_stats()
        for key, value in call_stats.items():
            if key != "function_counts":
                report_lines.append(f"- {key}: {value}")

        if "function_counts" in call_stats:
            report_lines.extend(["", "## Function Call Counts"])
            for func, count in call_stats["function_counts"].items():
                report_lines.append(f"- {func}: {count}")

        report_lines.extend(
            [
                "",
                f"## Variable Changes: {len(self.var_tracker.changes)}",
                f"## Tracked Variables: {len(self.var_tracker.tracked_vars)}",
            ]
        )

        report = "\n".join(report_lines)

        if filename:
            Path(filename).write_text(report)
        else:
            pass

        return report


# Create global trace instance
trace = TracingUtilities()
