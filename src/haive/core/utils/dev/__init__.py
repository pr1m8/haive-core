"""
Development Utilities Module for Haive

This module provides a comprehensive suite of debugging, logging, tracing,
performance profiling, and benchmarking tools for development excellence.

Usage:
    from haive.core.utils.dev import debug, log, trace, profile, benchmark

    # Enhanced debugging
    debug.ice("Variable value")  # icecream replacement
    debug.pdb()  # enhanced pdb
    debug.web()  # web-based debugging

    # Rich logging
    log.info("Message")
    log.debug("Debug info", extra={"key": "value"})
    log.error("Error occurred")

    # Code tracing
    trace.calls()  # trace function calls
    trace.stack()  # analyze call stack
    trace.hunt()   # track variable changes

    # Performance profiling
    profile.line()    # line-by-line profiling
    profile.memory()  # memory usage analysis
    profile.cpu()     # CPU profiling

    # Benchmarking
    benchmark.time_it(func)  # time execution
    benchmark.compare(funcs)  # compare functions
"""

from .benchmarking import benchmark
from .debugging import debug
from .logging import log
from .profiling import profile
from .tracing import trace

__all__ = ["debug", "log", "trace", "profile", "benchmark"]
