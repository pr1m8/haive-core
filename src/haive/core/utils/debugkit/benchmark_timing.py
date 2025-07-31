"""
Timing Benchmark Utilities

High-precision timing benchmarks and function comparisons.
"""

import statistics
import time
from collections.abc import Callable

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class TimingBenchmark:
    """High-precision timing benchmark utilities."""

    def __init__(self):
        self.results: dict[str, list[float]] = {}
        self.console = Console() if HAS_RICH else None

    def time_it(
        self, func: Callable, *args, iterations: int = 1000, warmup: int = 100, **kwargs
    ) -> dict[str, float]:
        """Time a function execution with statistical analysis."""
        func_name = (
            f"{func.__module__}.{func.__name__}"
            if hasattr(func, "__module__") and hasattr(func, "__name__")
            else str(func)
        )

        if HAS_RICH and self.console:
            self.console.print(
                f"⏱️  Timing {func_name} ({iterations} iterations)", style="blue"
            )

        # Warmup runs
        for _ in range(warmup):
            try:
                func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors

        # Actual timing
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            except Exception:
                # Skip failed iterations
                continue

        if not times:
            raise RuntimeError(f"All iterations failed for {func_name}")

        # Store results
        self.results[func_name] = times

        # Calculate statistics
        stats = {
            "function": func_name,
            "iterations": len(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "ops_per_second": 1.0 / statistics.mean(times),
            "p10": (
                statistics.quantiles(times, n=10)[0] if len(times) >= 10 else min(times)
            ),
            "p25": (
                statistics.quantiles(times, n=4)[0] if len(times) >= 4 else min(times)
            ),
            "p75": (
                statistics.quantiles(times, n=4)[2] if len(times) >= 4 else max(times)
            ),
            "p90": (
                statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times)
            ),
            "p95": (
                statistics.quantiles(times, n=20)[18]
                if len(times) >= 20
                else max(times)
            ),
            "p99": (
                statistics.quantiles(times, n=100)[98]
                if len(times) >= 100
                else max(times)
            ),
        }

        self._display_timing_results(stats)
        return stats

    def _display_timing_results(self, stats: dict[str, any]) -> None:
        """Display timing results in a formatted table."""
        if HAS_RICH and self.console:
            table = Table(title=f"⏱️  Timing Results: {stats['function']}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Iterations", str(stats["iterations"]))
            table.add_row("Mean", f"{stats['mean']:.6f}s")
            table.add_row("Median", f"{stats['median']:.6f}s")
            table.add_row("Std Dev", f"{stats['stdev']:.6f}s")
            table.add_row("Min", f"{stats['min']:.6f}s")
            table.add_row("Max", f"{stats['max']:.6f}s")
            table.add_row("Ops/Second", f"{stats['ops_per_second']:.2f}")
            table.add_row("95th percentile", f"{stats['p95']:.6f}s")
            table.add_row("99th percentile", f"{stats['p99']:.6f}s")

            self.console.print(table)

    def compare_functions(
        self, functions: list[Callable], *args, iterations: int = 1000, **kwargs
    ) -> dict[str, dict[str, float]]:
        """Compare multiple functions performance."""
        results = {}

        for func in functions:
            result = self.time_it(func, *args, iterations=iterations, **kwargs)
            results[result["function"]] = result

        self._display_comparison(results)
        return results

    def _display_comparison(self, results: dict[str, dict[str, float]]) -> None:
        """Display comparison results."""
        if not results:
            return

        if HAS_RICH and self.console:
            table = Table(title="🏁 Performance Comparison")
            table.add_column("Function", style="cyan")
            table.add_column("Mean (s)", style="magenta")
            table.add_column("Ops/Sec", style="green")
            table.add_column("Relative", style="yellow")

            # Sort by mean time (fastest first)
            sorted_results = sorted(results.items(), key=lambda x: x[1]["mean"])
            fastest_mean = sorted_results[0][1]["mean"]

            for func_name, result in sorted_results:
                relative = result["mean"] / fastest_mean
                table.add_row(
                    func_name,
                    f"{result['mean']:.6f}",
                    f"{result['ops_per_second']:.2f}",
                    f"{relative:.2f}x",
                )

            self.console.print(table)

    def clear(self) -> None:
        """Clear all timing results."""
        self.results.clear()


# Global instance for easy import
timing_benchmark = TimingBenchmark()
