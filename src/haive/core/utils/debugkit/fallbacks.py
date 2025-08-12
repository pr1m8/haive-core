"""
Fallback implementations for development utilities when dependencies are missing.

This module provides minimal fallback implementations that allow the development
utilities to function even when optional dependencies (like rich, icecream, etc.)
are not available. The fallbacks maintain the same API but with reduced functionality.
"""

import functools
import sys
import time
from collections.abc import Callable
from typing import Any


class FallbackDebug:
    """Minimal debug implementation when icecream and other debuggers are unavailable.

    Provides basic debugging functionality using standard print statements
    with enhanced formatting to approximate the full debug experience.

    Examples:
        Basic debugging output::

            debug = FallbackDebug()
            debug.ice("Processing", count=10, status="active")
            # Output: DEBUG: Processing | count=10, status=active
    """

    def __init__(self):
        """Initialize fallback debugger."""
        self.enabled = True

    def ice(self, *args: Any, **kwargs: Any) -> Any:
        """Fallback implementation of icecream-style debugging.

        Args:
            *args: Positional arguments to debug
            **kwargs: Named arguments to debug

        Returns:
            The first positional argument for chaining

        Examples:
            Debug with return value::

                result = debug.ice(expensive_computation())
                debug.ice("Status", processing=True, count=len(items))
        """
        if not self.enabled:
            return args[0] if args else None

        parts = []

        # Handle positional arguments
        for arg in args:
            if isinstance(arg, str):
                parts.append(arg)
            else:
                parts.append(repr(arg))

        # Handle keyword arguments
        for key, value in kwargs.items():
            parts.append(f"{key}={value!r}")

        output = " | ".join(parts) if parts else "DEBUG"
        print(f"DEBUG: {output}", file=sys.stderr)

        return args[0] if args else None

    def pdb(self) -> None:
        """Start Python debugger.

        Falls back to standard pdb when enhanced debuggers are unavailable.
        """
        import pdb

        pdb.set_trace()

    def breakpoint_on_exception(self, func: Callable) -> Callable:
        """Decorator to start debugger on exceptions.

        Args:
            func: Function to wrap with exception debugging

        Returns:
            Wrapped function that debugs on exceptions
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                print("Exception occurred, starting debugger...", file=sys.stderr)
                import pdb

                pdb.post_mortem()
                raise

        return wrapper

    def trace_calls(self, func: Callable) -> Callable:
        """Decorator to trace function calls.

        Args:
            func: Function to trace

        Returns:
            Wrapped function with call tracing
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"TRACE: Calling {func.__name__}", file=sys.stderr)
            try:
                result = func(*args, **kwargs)
                print(
                    f"TRACE: {func.__name__} returned {type(result).__name__}",
                    file=sys.stderr,
                )
                return result
            except Exception as e:
                print(
                    f"TRACE: {func.__name__} raised {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                raise

        return wrapper


class FallbackLog:
    """Minimal logging implementation when rich and loguru are unavailable.

    Provides structured logging functionality using standard Python logging
    with enhanced formatting and context management capabilities.

    Examples:
        Basic logging::

            log = FallbackLog()
            log.info("Process started", user_id=123)
            log.success("Operation completed successfully")
            log.error("Failed to process", error="Connection timeout")
    """

    def __init__(self):
        """Initialize fallback logger."""
        self._correlation_id: str | None = None
        self.enabled = True

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for distributed tracing.

        Args:
            correlation_id: Unique identifier for request correlation
        """
        self._correlation_id = correlation_id

    def _format_message(self, level: str, message: str, **kwargs: Any) -> str:
        """Format log message with context.

        Args:
            level: Log level (INFO, DEBUG, etc.)
            message: Primary log message
            **kwargs: Additional context fields

        Returns:
            Formatted log message string
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        correlation = f"[{self._correlation_id}] " if self._correlation_id else ""

        context_parts = []
        for key, value in kwargs.items():
            if (
                key != "correlation_id"
            ):  # Skip correlation_id as it's handled separately
                context_parts.append(f"{key}={value!r}")

        context_str = f" | {', '.join(context_parts)}" if context_parts else ""

        return f"{timestamp} {level} {correlation}{message}{context_str}"

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            message: Debug message
            **kwargs: Additional context fields
        """
        if self.enabled:
            formatted = self._format_message("DEBUG", message, **kwargs)
            print(formatted, file=sys.stderr)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.

        Args:
            message: Info message
            **kwargs: Additional context fields
        """
        if self.enabled:
            formatted = self._format_message("INFO", message, **kwargs)
            print(formatted)

    def success(self, message: str, **kwargs: Any) -> None:
        """Log success message.

        Args:
            message: Success message
            **kwargs: Additional context fields
        """
        if self.enabled:
            formatted = self._format_message("SUCCESS", message, **kwargs)
            print(formatted)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            message: Warning message
            **kwargs: Additional context fields
        """
        if self.enabled:
            formatted = self._format_message("WARNING", message, **kwargs)
            print(formatted, file=sys.stderr)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message.

        Args:
            message: Error message
            **kwargs: Additional context fields
        """
        if self.enabled:
            formatted = self._format_message("ERROR", message, **kwargs)
            print(formatted, file=sys.stderr)

    def context(self, name: str) -> "FallbackLogContext":
        """Create logging context.

        Args:
            name: Context name

        Returns:
            Context manager for structured logging
        """
        return FallbackLogContext(self, name)


class FallbackLogContext:
    """Context manager for fallback logging.

    Provides basic context management for logging operations,
    tracking timing and nested context levels.
    """

    def __init__(self, logger: FallbackLog, name: str):
        """Initialize log context.

        Args:
            logger: Parent logger instance
            name: Context name
        """
        self.logger = logger
        self.name = name
        self.start_time: float | None = None

    def __enter__(self) -> "FallbackLogContext":
        """Enter logging context."""
        self.start_time = time.time()
        self.logger.debug(f"Entering context: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit logging context."""
        duration = time.time() - self.start_time if self.start_time else 0.0

        if exc_type is None:
            self.logger.debug(
                f"Exiting context: {self.name}", duration=f"{duration:.3f}s"
            )
        else:
            self.logger.error(
                f"Context failed: {self.name}",
                duration=f"{duration:.3f}s",
                error=str(exc_val),
            )

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message within context."""
        self.logger.debug(f"[{self.name}] {message}", **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message within context."""
        self.logger.info(f"[{self.name}] {message}", **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        """Log success message within context."""
        self.logger.success(f"[{self.name}] {message}", **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message within context."""
        self.logger.error(f"[{self.name}] {message}", **kwargs)


class FallbackTrace:
    """Minimal tracing implementation when advanced tracers are unavailable.

    Provides basic function call tracing and variable tracking using
    standard Python mechanisms without external dependencies.

    Examples:
        Function call tracing::

            trace = FallbackTrace()

            @trace.calls
            def my_function():
                return "result"
    """

    def __init__(self):
        """Initialize fallback tracer."""
        self._contexts: list[str] = []
        self._correlation_id: str | None = None
        self.enabled = True

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for distributed tracing.

        Args:
            correlation_id: Unique identifier for request correlation
        """
        self._correlation_id = correlation_id

    def push_context(self, name: str, correlation_id: str) -> None:
        """Push a new tracing context.

        Args:
            name: Context name
            correlation_id: Correlation identifier
        """
        self._contexts.append(name)
        self._correlation_id = correlation_id

        if self.enabled:
            context_path = " -> ".join(self._contexts)
            correlation = f"[{correlation_id}] " if correlation_id else ""
            print(f"TRACE: {correlation}Push context: {context_path}", file=sys.stderr)

    def pop_context(self) -> None:
        """Pop the current tracing context."""
        if self._contexts:
            context_name = self._contexts.pop()

            if self.enabled:
                context_path = " -> ".join(self._contexts) if self._contexts else "root"
                correlation = (
                    f"[{self._correlation_id}] " if self._correlation_id else ""
                )
                print(
                    f"TRACE: {correlation}Pop context: {context_name} -> {context_path}",
                    file=sys.stderr,
                )

    def mark(self, name: str, value: Any) -> None:
        """Mark a trace point with a value.

        Args:
            name: Trace point name
            value: Value to record
        """
        if self.enabled:
            correlation = f"[{self._correlation_id}] " if self._correlation_id else ""
            context_path = " -> ".join(self._contexts) if self._contexts else "root"
            print(
                f"TRACE: {correlation}{context_path}: {name} = {value!r}",
                file=sys.stderr,
            )

    def calls(self, func: Callable) -> Callable:
        """Decorator to trace function calls.

        Args:
            func: Function to trace

        Returns:
            Wrapped function with call tracing
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.enabled:
                correlation = (
                    f"[{self._correlation_id}] " if self._correlation_id else ""
                )
                print(f"TRACE: {correlation}Calling {func.__name__}", file=sys.stderr)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                if self.enabled:
                    print(
                        f"TRACE: {correlation}Completed {func.__name__} "
                        f"in {duration:.3f}s -> {type(result).__name__}",
                        file=sys.stderr,
                    )

                return result
            except Exception as e:
                duration = time.time() - start_time

                if self.enabled:
                    print(
                        f"TRACE: {correlation}Failed {func.__name__} "
                        f"after {duration:.3f}s: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )

                raise

        return wrapper


class FallbackProfile:
    """Minimal profiling implementation when advanced profilers are unavailable.

    Provides basic timing and memory profiling using standard Python
    mechanisms without external dependencies.

    Examples:
        Function timing::

            profile = FallbackProfile()

            @profile.time
            def slow_function():
                time.sleep(1)
                return "done"
    """

    def __init__(self):
        """Initialize fallback profiler."""
        self.enabled = True
        self._active_contexts: dict[str, dict[str, Any]] = {}

    def start_context(self, name: str) -> dict[str, Any]:
        """Start profiling context.

        Args:
            name: Context name

        Returns:
            Context data for stopping profiler
        """
        context = {
            "name": name,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
        }

        self._active_contexts[name] = context
        return context

    def stop_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Stop profiling context and return statistics.

        Args:
            context: Context data from start_context

        Returns:
            Profiling statistics
        """
        end_time = time.time()
        end_memory = self._get_memory_usage()

        stats = {
            "name": context["name"],
            "duration": end_time - context["start_time"],
            "start_time": context["start_time"],
            "end_time": end_time,
            "memory_start": context["start_memory"],
            "memory_end": end_memory,
            "memory_delta": end_memory - context["start_memory"],
        }

        # Remove from active contexts
        if context["name"] in self._active_contexts:
            del self._active_contexts[context["name"]]

        if self.enabled:
            print(
                f"PROFILE: {context['name']} completed in {stats['duration']:.3f}s "
                f"(memory: {stats['memory_delta']:+.1f}MB)",
                file=sys.stderr,
            )

        return stats

    def time(self, func: Callable) -> Callable:
        """Decorator to time function execution.

        Args:
            func: Function to time

        Returns:
            Wrapped function with timing
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            start_time = time.time()
            start_memory = self._get_memory_usage()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                memory_delta = self._get_memory_usage() - start_memory

                print(
                    f"PROFILE: {func.__name__} took {duration:.3f}s "
                    f"(memory: {memory_delta:+.1f}MB)",
                    file=sys.stderr,
                )

                return result
            except Exception:
                duration = time.time() - start_time
                print(
                    f"PROFILE: {func.__name__} failed after {duration:.3f}s",
                    file=sys.stderr,
                )
                raise

        return wrapper

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback to basic memory estimation
            import resource

            return (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            )  # KB to MB on Linux


class FallbackBenchmark:
    """Minimal benchmarking implementation when advanced benchmarking tools are unavailable.

    Provides basic performance measurement and comparison capabilities
    using standard Python timing mechanisms.

    Examples:
        Basic benchmarking::

            benchmark = FallbackBenchmark()

            def test_function():
                return sum(range(1000))

            stats = benchmark.measure(test_function, iterations=100)
            print(f"Average time: {stats['average']:.3f}s")
    """

    def __init__(self):
        """Initialize fallback benchmarker."""
        self.enabled = True

    def measure(
        self, func: Callable, iterations: int = 100, warmup: int = 10
    ) -> dict[str, Any]:
        """Measure function performance over multiple iterations.

        Args:
            func: Function to benchmark
            iterations: Number of measurement iterations
            warmup: Number of warmup iterations

        Returns:
            Performance statistics dictionary

        Examples:
            Benchmark with custom iterations::

                def computation():
                    return [x**2 for x in range(1000)]

                stats = benchmark.measure(computation, iterations=50)
                print(f"Min: {stats['min']:.3f}s, Max: {stats['max']:.3f}s")
        """
        if not self.enabled:
            return {"average": 0.0, "min": 0.0, "max": 0.0, "iterations": 0}

        # Warmup iterations
        for _ in range(warmup):
            try:
                func()
            except Exception:
                pass  # Ignore warmup errors

        # Measurement iterations
        times = []
        for _ in range(iterations):
            start_time = time.time()
            try:
                func()
                duration = time.time() - start_time
                times.append(duration)
            except Exception as e:
                # Record failed iteration
                duration = time.time() - start_time
                times.append(duration)
                if self.enabled:
                    print(f"BENCHMARK: Iteration failed: {e}", file=sys.stderr)

        if not times:
            return {
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "iterations": 0,
                "error": "No successful iterations",
            }

        stats = {
            "average": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "iterations": len(times),
            "total_time": sum(times),
            "successful_iterations": len(times),
            "failed_iterations": iterations - len(times),
        }

        if self.enabled:
            print(
                f"BENCHMARK: {func.__name__} - "
                f"avg: {stats['average']:.3f}s, "
                f"min: {stats['min']:.3f}s, "
                f"max: {stats['max']:.3f}s "
                f"({stats['successful_iterations']}/{iterations} successful)",
                file=sys.stderr,
            )

        return stats

    def compare(
        self, functions: dict[str, Callable], iterations: int = 100
    ) -> dict[str, dict[str, Any]]:
        """Compare performance of multiple functions.

        Args:
            functions: Dictionary of name -> function pairs
            iterations: Number of iterations per function

        Returns:
            Comparison results dictionary

        Examples:
            Compare implementations::

                results = benchmark.compare({
                    'list_comp': lambda: [x**2 for x in range(1000)],
                    'map_pow': lambda: list(map(lambda x: x**2, range(1000)))
                })

                fastest = min(results.items(), key=lambda x: x[1]['average'])
                print(f"Fastest: {fastest[0]}")
        """
        results = {}

        for name, func in functions.items():
            if self.enabled:
                print(f"BENCHMARK: Measuring {name}...", file=sys.stderr)

            stats = self.measure(func, iterations)
            results[name] = stats

        if self.enabled and len(results) > 1:
            # Find fastest and slowest
            fastest = min(results.items(), key=lambda x: x[1]["average"])
            slowest = max(results.items(), key=lambda x: x[1]["average"])

            speedup = (
                slowest[1]["average"] / fastest[1]["average"]
                if fastest[1]["average"] > 0
                else float("inf")
            )

            print(
                f"BENCHMARK: Fastest: {fastest[0]} ({fastest[1]['average']:.3f}s), "
                f"Slowest: {slowest[0]} ({slowest[1]['average']:.3f}s), "
                f"Speedup: {speedup:.2f}x",
                file=sys.stderr,
            )

        return results
