"""
Performance Profiling Utilities

Provides comprehensive performance analysis including line-by-line profiling,
memory analysis, CPU profiling, and execution optimization insights.
"""

import functools
import subprocess
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from pathlib import Path

# Try to import profiling tools
try:
    import line_profiler

    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False

try:
    import memory_profiler

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import pyinstrument

    HAS_PYINSTRUMENT = True
except ImportError:
    HAS_PYINSTRUMENT = False

try:
    HAS_SCALENE = True
except ImportError:
    HAS_SCALENE = False

try:
    HAS_PY_SPY = True
except ImportError:
    HAS_PY_SPY = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from rich.console import Console
    from rich.progress import track
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class TimingProfiler:
    """Simple timing profiler for function execution."""

    def __init__(self):
        self.timings: dict[str, list[float]] = {}
        self.console = Console() if HAS_RICH else None

    def time_function(self, func: Callable) -> Callable:
        """Decorator to time function execution."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time

                func_name = f"{func.__module__}.{func.__name__}"
                if func_name not in self.timings:
                    self.timings[func_name] = []
                self.timings[func_name].append(duration)

                if HAS_RICH and self.console:
                    self.console.print(f"⏱️  {func_name}: {duration:.6f}s", style="blue")
                else:
                    pass

        return wrapper

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get timing statistics for all functions."""
        stats = {}

        for func_name, times in self.timings.items():
            if times:
                stats[func_name] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }

        if HAS_RICH and self.console:
            table = Table(title="Timing Statistics")
            table.add_column("Function", style="cyan")
            table.add_column("Count", style="green")
            table.add_column("Total (s)", style="yellow")
            table.add_column("Mean (s)", style="magenta")
            table.add_column("Min (s)", style="blue")
            table.add_column("Max (s)", style="red")

            for func_name, stat in stats.items():
                table.add_row(
                    func_name,
                    str(stat["count"]),
                    f"{stat['total']:.6f}",
                    f"{stat['mean']:.6f}",
                    f"{stat['min']:.6f}",
                    f"{stat['max']:.6f}",
                )

            self.console.print(table)

        return stats

    def clear(self) -> None:
        """Clear timing data."""
        self.timings.clear()


class MemoryProfiler:
    """Memory usage profiler."""

    def __init__(self):
        self.baseline_memory = self._get_memory_usage()
        self.console = Console() if HAS_RICH else None

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        else:
            # Fallback using memory_profiler if available
            if HAS_MEMORY_PROFILER:
                return memory_profiler.memory_usage()[0]
            return 0.0

    def profile_memory(self, func: Callable) -> Callable:
        """Decorator to profile memory usage."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get memory before
            memory_before = self._get_memory_usage()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Get memory after
                memory_after = self._get_memory_usage()
                memory_delta = memory_after - memory_before

                func_name = f"{func.__module__}.{func.__name__}"

                if HAS_RICH and self.console:
                    color = "red" if memory_delta > 0 else "green"
                    self.console.print(
                        f"💾 {func_name}: {memory_delta:+.2f} MB (total: {memory_after:.2f} MB)",
                        style=color,
                    )
                else:
                    pass

        return wrapper

    def memory_line_by_line(self, func: Callable) -> Callable:
        """Profile memory usage line by line using memory_profiler."""
        if not HAS_MEMORY_PROFILER:
            return self.profile_memory(func)

        # Add @profile decorator for memory_profiler
        return memory_profiler.profile(func)

    def get_current_usage(self) -> dict[str, float]:
        """Get current memory usage statistics."""
        current = self._get_memory_usage()
        delta_from_baseline = current - self.baseline_memory

        stats = {
            "current_mb": current,
            "baseline_mb": self.baseline_memory,
            "delta_mb": delta_from_baseline,
        }

        if HAS_RICH and self.console:
            table = Table(title="Memory Usage")
            table.add_column("Metric", style="cyan")
            table.add_column("Value (MB)", style="green")

            table.add_row("Current", f"{current:.2f}")
            table.add_row("Baseline", f"{self.baseline_memory:.2f}")
            table.add_row("Delta", f"{delta_from_baseline:+.2f}")

            self.console.print(table)
        else:
            pass

        return stats


class LineProfiler:
    """Line-by-line performance profiler."""

    def __init__(self):
        self.profiler = line_profiler.LineProfiler() if HAS_LINE_PROFILER else None
        self.console = Console() if HAS_RICH else None

    def profile_lines(self, func: Callable) -> Callable:
        """Profile function line by line."""
        if not HAS_LINE_PROFILER:
            return func

        self.profiler.add_function(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.profiler.enable_by_count()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.profiler.disable_by_count()

        return wrapper

    def show_stats(self, filename: str | None = None) -> None:
        """Show line profiling statistics."""
        if not self.profiler:
            return

        if filename:
            with open(filename, "w") as f:
                self.profiler.print_stats(stream=f)
        else:
            self.profiler.print_stats()


class CPUProfiler:
    """CPU profiling utilities."""

    def __init__(self):
        self.console = Console() if HAS_RICH else None

    def profile_cpu(self, func: Callable, duration: int = 10) -> Callable:
        """Profile CPU usage with pyinstrument."""
        if not HAS_PYINSTRUMENT:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = pyinstrument.Profiler()
            profiler.start()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.stop()

                if HAS_RICH and self.console:
                    self.console.print("🔥 CPU Profiling Results:", style="bold red")

        return wrapper

    def profile_with_scalene(
        self, script_path: str, output_dir: str = "scalene_profiles"
    ) -> None:
        """Profile a script with Scalene."""
        if not HAS_SCALENE:
            return

        Path(output_dir).mkdir(exist_ok=True)
        output_file = Path(output_dir) / f"profile_{int(time.time())}.html"

        cmd = ["scalene", "--html", "--outfile", str(output_file), script_path]

        with suppress(subprocess.CalledProcessError):
            subprocess.run(cmd, check=True)


class ProfilingUtilities:
    """Comprehensive profiling utilities."""

    def __init__(self):
        self.timing_profiler = TimingProfiler()
        self.memory_profiler = MemoryProfiler()
        self.line_profiler = LineProfiler()
        self.cpu_profiler = CPUProfiler()
        self.console = Console() if HAS_RICH else None

    def time(self, func: Callable | None = None) -> Callable:
        """Time function execution."""
        if func:
            return self.timing_profiler.time_function(func)
        else:
            return self.timing_profiler.time_function

    def memory(
        self, func: Callable | None = None, line_by_line: bool = False
    ) -> Callable:
        """Profile memory usage."""
        if line_by_line:
            profiler_func = self.memory_profiler.memory_line_by_line
        else:
            profiler_func = self.memory_profiler.profile_memory

        if func:
            return profiler_func(func)
        else:
            return profiler_func

    def line(self, func: Callable | None = None) -> Callable:
        """Profile line-by-line execution."""
        if func:
            return self.line_profiler.profile_lines(func)
        else:
            return self.line_profiler.profile_lines

    def cpu(self, func: Callable | None = None) -> Callable:
        """Profile CPU usage."""
        if func:
            return self.cpu_profiler.profile_cpu(func)
        else:
            return self.cpu_profiler.profile_cpu

    def comprehensive(self, func: Callable) -> Callable:
        """Apply comprehensive profiling (time + memory + CPU)."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if HAS_RICH and self.console:
                self.console.print(
                    f"🔬 Starting comprehensive profiling of {func.__name__}",
                    style="bold blue",
                )
            else:
                pass

            # Apply multiple profilers
            profiled_func = self.timing_profiler.time_function(
                self.memory_profiler.profile_memory(self.cpu_profiler.profile_cpu(func))
            )

            return profiled_func(*args, **kwargs)

        return wrapper

    @contextmanager
    def profile_context(
        self,
        name: str = "profile",
        include_memory: bool = True,
        include_cpu: bool = True,
    ):
        """Context manager for profiling a block of code."""
        start_time = time.perf_counter()
        start_memory = (
            self.memory_profiler._get_memory_usage() if include_memory else None
        )

        cpu_profiler = None
        if include_cpu and HAS_PYINSTRUMENT:
            cpu_profiler = pyinstrument.Profiler()
            cpu_profiler.start()

        if HAS_RICH and self.console:
            self.console.print(f"🏁 Starting profile context: {name}", style="blue")
        else:
            pass

        try:
            yield self
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time

            if cpu_profiler:
                cpu_profiler.stop()

            # Report results
            if HAS_RICH and self.console:
                table = Table(title=f"Profile Results: {name}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Duration", f"{duration:.6f}s")

                if start_memory:
                    end_memory = self.memory_profiler._get_memory_usage()
                    memory_delta = end_memory - start_memory
                    table.add_row("Memory Delta", f"{memory_delta:+.2f} MB")
                    table.add_row("Final Memory", f"{end_memory:.2f} MB")

                self.console.print(table)

                if cpu_profiler:
                    self.console.print("🔥 CPU Profile:", style="bold red")
            else:
                if start_memory:
                    end_memory = self.memory_profiler._get_memory_usage()
                    memory_delta = end_memory - start_memory

                if cpu_profiler:
                    pass

    def benchmark(
        self, func: Callable, iterations: int = 1000, warmup: int = 100
    ) -> dict[str, float]:
        """Benchmark a function with multiple iterations."""
        if HAS_RICH and self.console:
            self.console.print(
                f"🏃 Benchmarking {func.__name__} ({iterations} iterations)",
                style="bold green",
            )
        else:
            pass

        # Warmup
        for _ in range(warmup):
            func()

        # Actual benchmark
        times = []

        if HAS_RICH:
            iterations_range = track(range(iterations), description="Benchmarking...")
        else:
            iterations_range = range(iterations)

        for _ in iterations_range:
            start_time = time.perf_counter()
            func()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        # Calculate statistics
        total_time = sum(times)
        mean_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)

        # Calculate percentiles
        sorted_times = sorted(times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        stats = {
            "iterations": iterations,
            "total_time": total_time,
            "mean_time": mean_time,
            "min_time": min_time,
            "max_time": max_time,
            "p50_time": p50,
            "p95_time": p95,
            "p99_time": p99,
            "ops_per_second": iterations / total_time,
        }

        if HAS_RICH and self.console:
            table = Table(title=f"Benchmark Results: {func.__name__}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Iterations", str(iterations))
            table.add_row("Total Time", f"{total_time:.6f}s")
            table.add_row("Mean Time", f"{mean_time:.6f}s")
            table.add_row("Min Time", f"{min_time:.6f}s")
            table.add_row("Max Time", f"{max_time:.6f}s")
            table.add_row("P50 Time", f"{p50:.6f}s")
            table.add_row("P95 Time", f"{p95:.6f}s")
            table.add_row("P99 Time", f"{p99:.6f}s")
            table.add_row("Ops/Second", f"{stats['ops_per_second']:.2f}")

            self.console.print(table)
        else:
            for _key, _value in stats.items():
                pass

        return stats

    def compare(
        self, funcs: list[Callable], iterations: int = 1000
    ) -> dict[str, dict[str, float]]:
        """Compare performance of multiple functions."""
        results = {}

        for func in funcs:
            results[func.__name__] = self.benchmark(func, iterations)

        # Show comparison
        if HAS_RICH and self.console:
            table = Table(title="Performance Comparison")
            table.add_column("Function", style="cyan")
            table.add_column("Mean Time", style="green")
            table.add_column("Ops/Second", style="yellow")
            table.add_column("Relative Speed", style="magenta")

            # Find fastest function
            fastest_ops = max(result["ops_per_second"] for result in results.values())

            for func_name, result in results.items():
                relative_speed = result["ops_per_second"] / fastest_ops
                table.add_row(
                    func_name,
                    f"{result['mean_time']:.6f}s",
                    f"{result['ops_per_second']:.2f}",
                    f"{relative_speed:.2f}x",
                )

            self.console.print(table)

        return results

    def stats(self) -> None:
        """Show all profiling statistics."""
        if HAS_RICH and self.console:
            self.console.print("📊 Profiling Statistics", style="bold blue")
        else:
            pass

        self.timing_profiler.get_stats()
        self.memory_profiler.get_current_usage()

        if HAS_LINE_PROFILER:
            self.line_profiler.show_stats()

    def clear(self) -> None:
        """Clear all profiling data."""
        self.timing_profiler.clear()

    def status(self) -> dict[str, bool]:
        """Get status of available profiling tools."""
        status = {
            "line_profiler": HAS_LINE_PROFILER,
            "memory_profiler": HAS_MEMORY_PROFILER,
            "pyinstrument": HAS_PYINSTRUMENT,
            "scalene": HAS_SCALENE,
            "py_spy": HAS_PY_SPY,
            "psutil": HAS_PSUTIL,
            "rich": HAS_RICH,
        }

        if HAS_RICH and self.console:
            table = Table(title="Profiling Tools Status")
            table.add_column("Tool", style="cyan")
            table.add_column("Available", style="green")

            for tool, available in status.items():
                table.add_row(tool, "✅" if available else "❌")

            self.console.print(table)
        else:
            for tool, available in status.items():
                pass

        return status


# Create global profile instance
profile = ProfilingUtilities()
