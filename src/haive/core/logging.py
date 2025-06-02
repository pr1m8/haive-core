"""
Centralized logging management for the haive framework.

This module provides a unified interface for managing logging across all haive packages,
including engine.agent, dynamic graphs, games, and more. It allows easy configuration
of log levels, formats, and outputs.

Usage:
    from haive.core.logging import haive_logger

    # Configure globally
    haive_logger.configure(level="INFO", format="simple")

    # Get logger for a module
    logger = haive_logger.get_logger("my_module")
    logger.info("This is a log message")

    # Temporarily change log level
    with haive_logger.log_level("DEBUG"):
        # Debug logs will be shown here
        pass
"""

import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Union

# Try to import rich for enhanced output
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.traceback import install as install_rich_traceback

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


class LogLevel:
    """Standard log levels"""

    TRACE = 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    SILENT = 100  # Suppresses all logs


class HaiveLoggingManager:
    """
    Centralized logging manager for the entire haive framework.

    Features:
    - Global configuration for all haive modules
    - Easy log level switching
    - Multiple output formats
    - Performance tracking
    - Error traceback formatting
    - Log filtering by module
    """

    def __init__(self):
        """Initialize the logging manager"""
        self._loggers: Dict[str, logging.Logger] = {}
        self._default_level = LogLevel.INFO
        self._default_format = "auto"  # auto, simple, detailed, json, rich
        self._log_to_file = False
        self._log_file = None
        self._log_dir = Path.home() / ".haive" / "logs"
        self._module_levels: Dict[str, int] = {}  # Module-specific levels
        self._filters: List[str] = []  # Module name filters
        self._initialized = False

        # Performance tracking
        self._performance_tracking = False
        self._operation_times: Dict[str, List[float]] = {}

        # Initialize on first use
        self._initialize()

    def _initialize(self):
        """Initialize the logging system"""
        if self._initialized:
            return

        # Create log directory if needed
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.TRACE)  # Set to lowest level

        # Remove existing handlers
        root_logger.handlers = []

        # Add custom log level
        logging.addLevelName(LogLevel.TRACE, "TRACE")

        # Install rich traceback if available
        if RICH_AVAILABLE:
            install_rich_traceback(show_locals=True, suppress=[])

        # Load configuration from environment or config file
        self._load_configuration()

        self._initialized = True

    def _load_configuration(self):
        """Load configuration from environment variables or config file"""
        # Environment variables
        self._default_level = getattr(
            LogLevel, os.getenv("HAIVE_LOG_LEVEL", "INFO").upper(), LogLevel.INFO
        )
        self._default_format = os.getenv("HAIVE_LOG_FORMAT", "auto")
        self._log_to_file = os.getenv("HAIVE_LOG_TO_FILE", "false").lower() == "true"
        self._performance_tracking = (
            os.getenv("HAIVE_LOG_PERF", "false").lower() == "true"
        )

        # Module-specific levels (comma-separated module:level pairs)
        module_levels = os.getenv("HAIVE_LOG_MODULES", "")
        if module_levels:
            for pair in module_levels.split(","):
                if ":" in pair:
                    module, level = pair.split(":", 1)
                    self._module_levels[module] = getattr(
                        LogLevel, level.upper(), LogLevel.INFO
                    )

        # Filters (comma-separated module names to include)
        filters = os.getenv("HAIVE_LOG_FILTER", "")
        if filters:
            self._filters = [f.strip() for f in filters.split(",") if f.strip()]

    def configure(
        self,
        level: Union[str, int] = None,
        format: str = None,
        log_to_file: bool = None,
        log_file: str = None,
        performance_tracking: bool = None,
        module_levels: Dict[str, Union[str, int]] = None,
        filters: List[str] = None,
    ):
        """
        Configure global logging settings.

        Args:
            level: Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT)
            format: Log format (auto, simple, detailed, json, rich)
            log_to_file: Whether to log to file
            log_file: Custom log file path
            performance_tracking: Enable performance tracking
            module_levels: Module-specific log levels
            filters: List of module names to include (empty = all)
        """
        if level is not None:
            if isinstance(level, str):
                self._default_level = getattr(LogLevel, level.upper(), LogLevel.INFO)
            else:
                self._default_level = level

        if format is not None:
            self._default_format = format

        if log_to_file is not None:
            self._log_to_file = log_to_file

        if log_file is not None:
            self._log_file = Path(log_file)

        if performance_tracking is not None:
            self._performance_tracking = performance_tracking

        if module_levels is not None:
            for module, mod_level in module_levels.items():
                if isinstance(mod_level, str):
                    self._module_levels[module] = getattr(
                        LogLevel, mod_level.upper(), LogLevel.INFO
                    )
                else:
                    self._module_levels[module] = mod_level

        if filters is not None:
            self._filters = filters

        # Reconfigure existing loggers
        for name, logger in self._loggers.items():
            self._configure_logger(logger, name)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific module.

        Args:
            name: Module name (e.g., "haive.core.engine.agent")

        Returns:
            Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]

        # Create new logger
        logger = logging.getLogger(name)
        self._configure_logger(logger, name)
        self._loggers[name] = logger

        return logger

    def _configure_logger(self, logger: logging.Logger, name: str):
        """Configure a specific logger"""
        # Clear existing handlers
        logger.handlers = []
        logger.propagate = False

        # Set level
        level = self._get_level_for_module(name)
        logger.setLevel(level)

        # Check filters
        if self._filters and not any(name.startswith(f) for f in self._filters):
            # Module is filtered out
            logger.setLevel(LogLevel.SILENT)
            return

        # Determine format
        format_type = self._default_format
        if format_type == "auto":
            format_type = "rich" if RICH_AVAILABLE else "simple"

        # Console handler
        if format_type == "rich" and RICH_AVAILABLE:
            handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            handler = logging.StreamHandler(sys.stdout)

            if format_type == "simple":
                formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
            elif format_type == "detailed":
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            elif format_type == "json":
                # Custom JSON formatter
                handler.setFormatter(self._get_json_formatter())
            else:
                formatter = logging.Formatter("%(message)s")

            if format_type != "json":
                handler.setFormatter(formatter)

        logger.addHandler(handler)

        # File handler if enabled
        if self._log_to_file:
            file_path = self._get_log_file_path(name)
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)

    def _get_level_for_module(self, name: str) -> int:
        """Get the appropriate log level for a module"""
        # Check for exact match first
        if name in self._module_levels:
            return self._module_levels[name]

        # Check for parent module matches
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            parent = ".".join(parts[:i])
            if parent in self._module_levels:
                return self._module_levels[parent]

        return self._default_level

    def _get_log_file_path(self, name: str) -> Path:
        """Get the log file path for a module"""
        if self._log_file:
            return self._log_file

        # Generate filename based on date and module
        date_str = datetime.now().strftime("%Y%m%d")
        safe_name = name.replace(".", "_")
        return self._log_dir / f"haive_{safe_name}_{date_str}.log"

    def _get_json_formatter(self):
        """Get a JSON formatter"""

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "module": record.name,
                    "function": record.funcName,
                    "line": record.lineno,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_data)

        return JSONFormatter()

    @contextmanager
    def log_level(self, level: Union[str, int], modules: Optional[List[str]] = None):
        """
        Temporarily change log level.

        Args:
            level: Temporary log level
            modules: Specific modules to affect (None = all)

        Example:
            with haive_logger.log_level("DEBUG"):
                # Debug logs will be shown here
                pass
        """
        if isinstance(level, str):
            level = getattr(LogLevel, level.upper(), LogLevel.INFO)

        # Save current levels
        if modules:
            old_levels = {
                m: self._module_levels.get(m, self._default_level) for m in modules
            }
            for m in modules:
                self._module_levels[m] = level
        else:
            old_level = self._default_level
            self._default_level = level

        # Reconfigure loggers
        for name, logger in self._loggers.items():
            if modules is None or any(name.startswith(m) for m in modules):
                self._configure_logger(logger, name)

        try:
            yield
        finally:
            # Restore levels
            if modules:
                for m, old_lvl in old_levels.items():
                    if old_lvl == self._default_level:
                        self._module_levels.pop(m, None)
                    else:
                        self._module_levels[m] = old_lvl
            else:
                self._default_level = old_level

            # Reconfigure loggers again
            for name, logger in self._loggers.items():
                if modules is None or any(name.startswith(m) for m in modules):
                    self._configure_logger(logger, name)

    @contextmanager
    def performance_tracking(
        self, operation: str, logger: Optional[logging.Logger] = None
    ):
        """
        Track performance of an operation.

        Args:
            operation: Name of the operation
            logger: Logger to use for output

        Example:
            with haive_logger.performance_tracking("database_query"):
                # Code to time
                pass
        """
        import time

        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time

            if self._performance_tracking:
                # Store timing
                if operation not in self._operation_times:
                    self._operation_times[operation] = []
                self._operation_times[operation].append(duration)

                # Log if logger provided
                if logger:
                    logger.debug(f"Performance: {operation} took {duration:.3f}s")

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for tracked operations"""
        stats = {}
        for operation, times in self._operation_times.items():
            if times:
                stats[operation] = {
                    "count": len(times),
                    "total": sum(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
        return stats

    def show_configuration(self):
        """Display current logging configuration"""
        if RICH_AVAILABLE:
            table = Table(title="Haive Logging Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="yellow")

            table.add_row("Default Level", logging.getLevelName(self._default_level))
            table.add_row("Format", self._default_format)
            table.add_row("Log to File", str(self._log_to_file))
            table.add_row("Performance Tracking", str(self._performance_tracking))

            if self._module_levels:
                module_info = "\n".join(
                    [
                        f"{m}: {logging.getLevelName(l)}"
                        for m, l in self._module_levels.items()
                    ]
                )
                table.add_row("Module Levels", module_info)

            if self._filters:
                table.add_row("Filters", ", ".join(self._filters))

            console.print(table)
        else:
            print("=== Haive Logging Configuration ===")
            print(f"Default Level: {logging.getLevelName(self._default_level)}")
            print(f"Format: {self._default_format}")
            print(f"Log to File: {self._log_to_file}")
            print(f"Performance Tracking: {self._performance_tracking}")

            if self._module_levels:
                print("Module Levels:")
                for m, l in self._module_levels.items():
                    print(f"  {m}: {logging.getLevelName(l)}")

            if self._filters:
                print(f"Filters: {', '.join(self._filters)}")

    def suppress_module(self, module: str):
        """Suppress all logs from a specific module"""
        self._module_levels[module] = LogLevel.SILENT

        # Reconfigure affected loggers
        for name, logger in self._loggers.items():
            if name.startswith(module):
                self._configure_logger(logger, name)

    def trace(self, logger_name: str, message: str, **kwargs):
        """Log a trace message (most verbose)"""
        logger = self.get_logger(logger_name)
        logger.log(LogLevel.TRACE, message, **kwargs)


# Global instance
haive_logger = HaiveLoggingManager()


# Convenience decorators
def log_performance(operation: Optional[str] = None):
    """
    Decorator to track performance of a function.

    Args:
        operation: Operation name (defaults to function name)

    Example:
        @log_performance()
        def my_function():
            pass
    """

    def decorator(func):
        op_name = operation or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = haive_logger.get_logger(func.__module__)
            with haive_logger.performance_tracking(op_name, logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_errors(logger_name: Optional[str] = None, reraise: bool = True):
    """
    Decorator to log errors from a function.

    Args:
        logger_name: Logger to use (defaults to function module)
        reraise: Whether to re-raise the exception

    Example:
        @log_errors()
        def my_function():
            pass
    """

    def decorator(func):
        log_name = logger_name or func.__module__

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = haive_logger.get_logger(log_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                if reraise:
                    raise

        return wrapper

    return decorator


# Quick access functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module"""
    return haive_logger.get_logger(name)


def configure(**kwargs):
    """Configure logging settings"""
    haive_logger.configure(**kwargs)


def set_level(level: Union[str, int], modules: Optional[List[str]] = None):
    """Set log level for all or specific modules"""
    if modules:
        haive_logger.configure(module_levels={m: level for m in modules})
    else:
        haive_logger.configure(level=level)


def suppress(module: str):
    """Suppress logs from a module"""
    haive_logger.suppress_module(module)


def show_config():
    """Show current configuration"""
    haive_logger.show_configuration()


# Environment variable reference
"""
Environment Variables:
- HAIVE_LOG_LEVEL: Global log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT)
- HAIVE_LOG_FORMAT: Output format (auto, simple, detailed, json, rich)
- HAIVE_LOG_TO_FILE: Whether to log to file (true/false)
- HAIVE_LOG_PERF: Enable performance tracking (true/false)
- HAIVE_LOG_MODULES: Module-specific levels (module1:DEBUG,module2:ERROR)
- HAIVE_LOG_FILTER: Comma-separated list of modules to include

Examples:
    # Set global level to DEBUG
    export HAIVE_LOG_LEVEL=DEBUG
    
    # Set specific module levels
    export HAIVE_LOG_MODULES="haive.core.engine:DEBUG,haive.games:INFO"
    
    # Only show logs from specific modules
    export HAIVE_LOG_FILTER="haive.core.engine,haive.core.models"
    
    # Enable performance tracking
    export HAIVE_LOG_PERF=true
"""
