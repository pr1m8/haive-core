"""
Haive Development Utilities

A unified development utilities suite for enhanced debugging, logging,
tracing, profiling, and benchmarking in Python applications.

This module provides a powerful yet easy-to-use toolkit that replaces and enhances
traditional debugging methods with modern, rich-featured alternatives.

Examples:
    Basic usage with single import:

        >>> from haive.core.utils.dev import debug, log, trace, profile, benchmark

        # Enhanced debugging (icecream replacement)
        >>> debug.ice("Hello", variable=42)

        # Rich logging with context
        >>> with log.context("my_operation"):
        ...     log.info("Starting process...")
        ...     log.success("Complete!")

        # Performance profiling
        >>> @profile.time
        ... def my_function():
        ...     return sum(i**2 for i in range(10000))

    Advanced usage patterns:

        >>> @debug.breakpoint_on_exception  # Auto-debug on errors
        >>> @trace.calls                    # Track all function calls
        >>> @profile.time                   # Measure performance
        >>> def complex_algorithm(data):
        ...     with log.context("algorithm"):
        ...         debug.ice("Processing", data_size=len(data))
        ...         result = process(data)
        ...         log.success("Complete!")
        ...         return result

Components:
    debug: Enhanced debugging with icecream-style output, web debugging, visual debugging
    log: Structured logging with rich formatting and context managers
    trace: Function call and variable tracking with detailed analysis
    profile: Performance profiling including timing, memory, and CPU analysis
    benchmark: Benchmarking and stress testing with statistical analysis

Note:
    All utilities provide smart fallbacks when optional dependencies are missing
    and can be safely disabled for production use with minimal overhead.
"""

from haive.core.utils.dev.benchmarking import benchmark
from haive.core.utils.dev.debugging import debug
from haive.core.utils.dev.logging import log
from haive.core.utils.dev.profiling import profile
from haive.core.utils.dev.tracing import trace

__all__ = ["benchmark", "debug", "log", "profile", "trace"]

# Version information
__version__ = "1.0.0"
__author__ = "Haive Development Team"
