"""
Enhanced Logging Utilities

Provides beautiful, structured logging with Rich integration,
context awareness, and development-friendly features.
"""

import inspect
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Try to import rich for beautiful logging
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import track
    from rich.table import Table
    from rich.text import Text
    from rich.traceback import install as install_rich_traceback

    HAS_RICH = True

    # Install rich traceback handling
    install_rich_traceback(show_locals=True)
except ImportError:
    HAS_RICH = False

# Try to import structlog for structured logging
try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class RichFormatter(logging.Formatter):
    """Custom formatter for non-Rich environments."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        if HAS_RICH:
            return super().format(record)

        # Add colors in non-Rich environments
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Format message
        message = super().format(record)

        return f"{color}[{timestamp}] {record.levelname:8s}{reset} {record.name}: {message}"


class DevLogger:
    """Enhanced development logger with rich formatting and context."""

    def __init__(self, name: str = "haive.dev"):
        self.name = name
        self.console = Console() if HAS_RICH else None
        self.context_stack: list[dict[str, Any]] = []
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger with Rich handler if available."""
        logger = logging.getLogger(self.name)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        if HAS_RICH:
            handler = RichHandler(
                console=self.console,
                show_path=True,
                show_time=True,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(RichFormatter())

        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        return logger

    def _get_caller_info(self) -> dict[str, Any]:
        """Get information about the calling code."""
        frame = inspect.currentframe()
        try:
            # Go back 3 frames: _get_caller_info -> log method -> actual caller
            caller_frame = frame.f_back.f_back.f_back
            if caller_frame:
                return {
                    "file": Path(caller_frame.f_code.co_filename).name,
                    "line": caller_frame.f_lineno,
                    "function": caller_frame.f_code.co_name,
                }
        finally:
            del frame
        return {}

    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with context and kwargs."""
        parts = [message]

        # Add context if available
        if self.context_stack:
            context = " → ".join(
                [ctx.get("name", "context") for ctx in self.context_stack]
            )
            parts.append(f"[{context}]")

        # Add extra data
        if kwargs:
            extra_parts = []
            for key, value in kwargs.items():
                if isinstance(value, dict | list):
                    extra_parts.append(f"{key}={json.dumps(value, default=str)}")
                else:
                    extra_parts.append(f"{key}={value}")
            if extra_parts:
                parts.append(f"({', '.join(extra_parts)})")

        return " ".join(parts)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context."""
        formatted = self._format_message(message, **kwargs)
        caller = self._get_caller_info()

        if HAS_RICH and self.console:
            self.console.print(f"🐛 {formatted}", style="dim cyan")
        else:
            self.logger.debug(formatted, extra=caller)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with context."""
        formatted = self._format_message(message, **kwargs)
        caller = self._get_caller_info()

        if HAS_RICH and self.console:
            self.console.print(f"ℹ️  {formatted}", style="bold blue")
        else:
            self.logger.info(formatted, extra=caller)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context."""
        formatted = self._format_message(message, **kwargs)
        caller = self._get_caller_info()

        if HAS_RICH and self.console:
            self.console.print(f"⚠️  {formatted}", style="bold yellow")
        else:
            self.logger.warning(formatted, extra=caller)

    def error(self, message: str, exception: Exception | None = None, **kwargs) -> None:
        """Log error message with optional exception."""
        formatted = self._format_message(message, **kwargs)
        caller = self._get_caller_info()

        if HAS_RICH and self.console:
            self.console.print(f"❌ {formatted}", style="bold red")
            if exception:
                self.console.print_exception()
        else:
            self.logger.error(formatted, extra=caller, exc_info=exception)

    def critical(
        self, message: str, exception: Exception | None = None, **kwargs
    ) -> None:
        """Log critical message with optional exception."""
        formatted = self._format_message(message, **kwargs)
        caller = self._get_caller_info()

        if HAS_RICH and self.console:
            self.console.print(f"🚨 {formatted}", style="bold red on white")
            if exception:
                self.console.print_exception()
        else:
            self.logger.critical(formatted, extra=caller, exc_info=exception)

    def success(self, message: str, **kwargs) -> None:
        """Log success message."""
        formatted = self._format_message(message, **kwargs)

        if HAS_RICH and self.console:
            self.console.print(f"✅ {formatted}", style="bold green")
        else:
            self.logger.info(f"SUCCESS: {formatted}")

    def progress(self, message: str, **kwargs) -> None:
        """Log progress message."""
        formatted = self._format_message(message, **kwargs)

        if HAS_RICH and self.console:
            self.console.print(f"🔄 {formatted}", style="bold blue")
        else:
            self.logger.info(f"PROGRESS: {formatted}")

    @contextmanager
    def context(self, name: str, **kwargs):
        """Context manager for adding context to logs."""
        ctx = {"name": name, **kwargs}
        self.context_stack.append(ctx)

        if HAS_RICH and self.console:
            self.console.print(f"📍 Entering: {name}", style="dim green")
        else:
            self.logger.debug(f"Entering context: {name}")

        try:
            yield self
        except Exception as e:
            self.error(f"Exception in context '{name}'", exception=e)
            raise
        finally:
            self.context_stack.pop()
            if HAS_RICH and self.console:
                self.console.print(f"📍 Exiting: {name}", style="dim green")
            else:
                self.logger.debug(f"Exiting context: {name}")

    def table(self, data: list[dict[str, Any]], title: str = "Data") -> None:
        """Log data as a formatted table."""
        if not data:
            self.info(f"Empty table: {title}")
            return

        if HAS_RICH and self.console:
            table = Table(title=title)

            # Add columns
            for key in data[0]:
                table.add_column(str(key), style="cyan")

            # Add rows
            for row in data:
                table.add_row(*[str(value) for value in row.values()])

            self.console.print(table)
        else:
            self.info(f"Table: {title}")
            for i, row in enumerate(data):
                self.info(f"  Row {i+1}: {row}")

    def json(self, data: Any, title: str = "JSON Data") -> None:
        """Log data as formatted JSON."""
        try:
            json_str = json.dumps(data, indent=2, default=str)

            if HAS_RICH and self.console:
                from rich.syntax import Syntax

                syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
                panel = Panel(syntax, title=title, border_style="blue")
                self.console.print(panel)
            else:
                self.info(f"{title}:\n{json_str}")
        except Exception as e:
            self.error(f"Failed to serialize data as JSON: {e}")

    def panel(self, message: str, title: str = "Message", style: str = "blue") -> None:
        """Log message in a panel."""
        if HAS_RICH and self.console:
            panel = Panel(message, title=title, border_style=style)
            self.console.print(panel)
        else:
            self.info(f"[{title}] {message}")

    def divider(self, text: str = "", style: str = "blue") -> None:
        """Print a divider line."""
        if HAS_RICH and self.console:
            from rich.rule import Rule

            self.console.print(Rule(text, style=style))
        else:
            line = "=" * 50
            if text:
                self.info(f"{line} {text} {line}")
            else:
                self.info(line)

    def metrics(self, metrics: dict[str, int | float], title: str = "Metrics") -> None:
        """Log metrics in a formatted way."""
        if HAS_RICH and self.console:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in metrics.items():
                table.add_row(str(key), str(value))

            self.console.print(table)
        else:
            self.info(f"Metrics: {title}")
            for key, value in metrics.items():
                self.info(f"  {key}: {value}")

    def timer_start(self, name: str) -> None:
        """Start a timer."""
        if not hasattr(self, "_timers"):
            self._timers = {}

        self._timers[name] = datetime.now()
        self.progress(f"Timer started: {name}")

    def timer_end(self, name: str) -> float:
        """End a timer and log duration."""
        if not hasattr(self, "_timers") or name not in self._timers:
            self.error(f"Timer '{name}' not found")
            return 0.0

        start_time = self._timers.pop(name)
        duration = (datetime.now() - start_time).total_seconds()

        self.success(f"Timer '{name}' completed in {duration:.3f}s")
        return duration

    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        self.timer_start(name)
        try:
            yield
        finally:
            self.timer_end(name)

    def set_level(self, level: str | int) -> None:
        """Set logging level."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
        self.info(f"Log level set to {logging.getLevelName(level)}")


# Create global log instance
log = DevLogger()
