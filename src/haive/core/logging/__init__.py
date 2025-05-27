# src/haive/core/logging/__init__.py

"""
Haive Rich Logging System

A comprehensive logging system built around Rich UI with extensive formatting,
file logging, and beautiful console output.
"""

from haive.core.logging.decorators import log_calls, log_errors, log_performance
from haive.core.logging.formatters import FileFormatter, RichFormatter
from haive.core.logging.handlers import RichConsoleHandler, RotatingFileHandler
from haive.core.logging.logging_mixin import LoggingMixin, RichLoggerMixin
from haive.core.logging.manager import LoggingManager
from haive.core.logging.utils import get_logger, log_exception, setup_project_logging

__all__ = [
    "LoggingManager",
    "LoggingMixin",
    "RichLoggerMixin",
    "RichFormatter",
    "FileFormatter",
    "RichConsoleHandler",
    "RotatingFileHandler",
    "get_logger",
    "setup_project_logging",
    "log_exception",
    "log_calls",
    "log_performance",
    "log_errors",
]
