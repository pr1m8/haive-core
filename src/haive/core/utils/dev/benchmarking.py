"""
Benchmarking and Performance Testing Utilities

Provides comprehensive benchmarking capabilities including timing comparisons,
stress testing, load testing, and performance regression detection.
"""

import asyncio
import functools
import json
import multiprocessing
import pickle
import statistics
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from rich.bar import Bar
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, track
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TimingBenchmark:
    """High-precision timing benchmark utilities."""

    def __init__(self):
        self.results: Dict[str, List[float]] = {}
        self.console = Console() if HAS_RICH else None

    def time_it(
        self, func: Callable, *args, iterations: int = 1000, warmup: int = 100, **kwargs
    ) -> Dict[str, float]:
        """Time a function execution with statistical analysis."""
        func_name = (
            f"{func.__module__}.{func.__name__}"
            if hasattr(func, "__module__")
            else str(func)
        )

        if HAS_RICH and self.console:
            self.console.print(
                f"⏱️  Timing {func_name} ({iterations} iterations)", style="blue"
            )
        else:
            print(f"⏱️  Timing {func_name} ({iterations} iterations)")

        # Warmup runs
        for _ in range(warmup):
            func(*args, **kwargs)

        # Actual timing runs
        times = []

        if HAS_RICH:
            iterations_range = track(range(iterations), description="Benchmarking...")
        else:
            iterations_range = range(iterations)

        for _ in iterations_range:
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        # Statistical analysis
        stats = {
            "function": func_name,
            "iterations": iterations,
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "mode": (
                statistics.mode(times) if len(set(times)) < len(times) / 2 else None
            ),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "variance": statistics.variance(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "range": max(times) - min(times),
            "total": sum(times),
            "ops_per_second": iterations / sum(times),
            "percentiles": {
                "p10": (
                    statistics.quantiles(times, n=10)[0]
                    if len(times) >= 10
                    else min(times)
                ),
                "p25": (
                    statistics.quantiles(times, n=4)[0]
                    if len(times) >= 4
                    else min(times)
                ),
                "p50": statistics.median(times),
                "p75": (
                    statistics.quantiles(times, n=4)[2]
                    if len(times) >= 4
                    else max(times)
                ),
                "p90": (
                    statistics.quantiles(times, n=10)[8]
                    if len(times) >= 10
                    else max(times)
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
            },
        }

        # Store results
        self.results[func_name] = times

        self._display_timing_results(stats)
        return stats

    def _display_timing_results(self, stats: Dict[str, Any]) -> None:
        """Display timing results in a formatted table."""
        if HAS_RICH and self.console:
            table = Table(title=f"Timing Results: {stats['function']}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Iterations", str(stats["iterations"]))
            table.add_row("Mean Time", f"{stats['mean']:.6f}s")
            table.add_row("Median Time", f"{stats['median']:.6f}s")
            table.add_row("Std Deviation", f"{stats['stdev']:.6f}s")
            table.add_row("Min Time", f"{stats['min']:.6f}s")
            table.add_row("Max Time", f"{stats['max']:.6f}s")
            table.add_row("Ops/Second", f"{stats['ops_per_second']:.2f}")
            table.add_row("P50", f"{stats['percentiles']['p50']:.6f}s")
            table.add_row("P95", f"{stats['percentiles']['p95']:.6f}s")
            table.add_row("P99", f"{stats['percentiles']['p99']:.6f}s")

            self.console.print(table)
        else:
            print(f"\n📊 Timing Results for {stats['function']}:")
            print(f"  Iterations: {stats['iterations']}")
            print(f"  Mean: {stats['mean']:.6f}s")
            print(f"  Median: {stats['median']:.6f}s")
            print(f"  Std Dev: {stats['stdev']:.6f}s")
            print(f"  Min/Max: {stats['min']:.6f}s / {stats['max']:.6f}s")
            print(f"  Ops/Second: {stats['ops_per_second']:.2f}")

    def compare_functions(
        self, functions: List[Callable], *args, iterations: int = 1000, **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple functions performance."""
        results = {}

        for func in functions:
            results[func.__name__] = self.time_it(
                func, *args, iterations=iterations, **kwargs
            )

        self._display_comparison(results)
        return results

    def _display_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """Display comparison results."""
        if not results:
            return

        if HAS_RICH and self.console:
            table = Table(title="Performance Comparison")
            table.add_column("Function", style="cyan")
            table.add_column("Mean Time", style="green")
            table.add_column("Ops/Second", style="yellow")
            table.add_column("Relative Speed", style="magenta")
            table.add_column("Winner", style="bold red")

            # Find fastest function
            fastest_ops = max(r["ops_per_second"] for r in results.values())
            fastest_func = next(
                name
                for name, r in results.items()
                if r["ops_per_second"] == fastest_ops
            )

            for func_name, result in results.items():
                relative_speed = result["ops_per_second"] / fastest_ops
                is_winner = func_name == fastest_func

                table.add_row(
                    func_name,
                    f"{result['mean']:.6f}s",
                    f"{result['ops_per_second']:.2f}",
                    f"{relative_speed:.2f}x",
                    "🏆" if is_winner else "",
                )

            self.console.print(table)
        else:
            print("\n🏁 Performance Comparison:")
            for func_name, result in results.items():
                print(
                    f"  {func_name}: {result['mean']:.6f}s ({result['ops_per_second']:.2f} ops/s)"
                )


class StressTester:
    """Stress testing utilities for load and concurrency testing."""

    def __init__(self):
        self.console = Console() if HAS_RICH else None

    def load_test(
        self,
        func: Callable,
        *args,
        concurrent_users: int = 10,
        duration_seconds: int = 60,
        ramp_up_seconds: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform load testing with concurrent users."""
        if HAS_RICH and self.console:
            self.console.print(f"🔥 Load Testing {func.__name__}", style="bold red")
            self.console.print(
                f"Users: {concurrent_users}, Duration: {duration_seconds}s",
                style="blue",
            )
        else:
            print(f"🔥 Load Testing {func.__name__}")
            print(f"Users: {concurrent_users}, Duration: {duration_seconds}s")

        results = {
            "function": func.__name__,
            "concurrent_users": concurrent_users,
            "duration_seconds": duration_seconds,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": [],
            "throughput": 0,
            "avg_response_time": 0,
            "percentiles": {},
        }

        start_time = time.time()
        end_time = start_time + duration_seconds

        def worker():
            """Worker function for concurrent execution."""
            while time.time() < end_time:
                request_start = time.perf_counter()
                try:
                    func(*args, **kwargs)
                    request_end = time.perf_counter()
                    response_time = request_end - request_start

                    results["total_requests"] += 1
                    results["successful_requests"] += 1
                    results["response_times"].append(response_time)

                except Exception as e:
                    results["total_requests"] += 1
                    results["failed_requests"] += 1
                    results["errors"].append(str(e))

                # Small delay to prevent overwhelming
                time.sleep(0.001)

        # Start concurrent workers
        threads = []
        for _ in range(concurrent_users):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)

            # Ramp up delay
            if ramp_up_seconds > 0:
                time.sleep(ramp_up_seconds / concurrent_users)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Calculate final statistics
        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["throughput"] = results["successful_requests"] / duration_seconds
            results["percentiles"] = {
                "p50": statistics.median(results["response_times"]),
                "p95": (
                    statistics.quantiles(results["response_times"], n=20)[18]
                    if len(results["response_times"]) >= 20
                    else max(results["response_times"])
                ),
                "p99": (
                    statistics.quantiles(results["response_times"], n=100)[98]
                    if len(results["response_times"]) >= 100
                    else max(results["response_times"])
                ),
            }

        self._display_load_test_results(results)
        return results

    def _display_load_test_results(self, results: Dict[str, Any]) -> None:
        """Display load test results."""
        if HAS_RICH and self.console:
            table = Table(title=f"Load Test Results: {results['function']}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Requests", str(results["total_requests"]))
            table.add_row("Successful", str(results["successful_requests"]))
            table.add_row("Failed", str(results["failed_requests"]))
            table.add_row(
                "Success Rate",
                f"{(results['successful_requests']/max(results['total_requests'], 1)*100):.2f}%",
            )
            table.add_row("Throughput", f"{results['throughput']:.2f} req/s")
            table.add_row("Avg Response Time", f"{results['avg_response_time']:.6f}s")

            if results["percentiles"]:
                table.add_row(
                    "P50 Response Time", f"{results['percentiles']['p50']:.6f}s"
                )
                table.add_row(
                    "P95 Response Time", f"{results['percentiles']['p95']:.6f}s"
                )
                table.add_row(
                    "P99 Response Time", f"{results['percentiles']['p99']:.6f}s"
                )

            self.console.print(table)
        else:
            print(f"\n🔥 Load Test Results for {results['function']}:")
            print(f"  Total Requests: {results['total_requests']}")
            print(f"  Successful: {results['successful_requests']}")
            print(f"  Failed: {results['failed_requests']}")
            print(f"  Throughput: {results['throughput']:.2f} req/s")

    def spike_test(
        self,
        func: Callable,
        *args,
        base_users: int = 5,
        spike_users: int = 50,
        spike_duration: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform spike testing with sudden load increases."""
        if HAS_RICH and self.console:
            self.console.print(f"⚡ Spike Testing {func.__name__}", style="bold yellow")
        else:
            print(f"⚡ Spike Testing {func.__name__}")

        # Phase 1: Base load
        print(f"Phase 1: Base load ({base_users} users)")
        base_results = self.load_test(
            func, *args, concurrent_users=base_users, duration_seconds=30, **kwargs
        )

        # Phase 2: Spike load
        print(f"Phase 2: Spike load ({spike_users} users)")
        spike_results = self.load_test(
            func,
            *args,
            concurrent_users=spike_users,
            duration_seconds=spike_duration,
            ramp_up_seconds=5,
            **kwargs,
        )

        # Phase 3: Recovery
        print(f"Phase 3: Recovery ({base_users} users)")
        recovery_results = self.load_test(
            func, *args, concurrent_users=base_users, duration_seconds=30, **kwargs
        )

        return {
            "base_load": base_results,
            "spike_load": spike_results,
            "recovery": recovery_results,
        }


class AsyncBenchmark:
    """Benchmarking utilities for async functions."""

    def __init__(self):
        self.console = Console() if HAS_RICH else None

    async def time_async(
        self,
        async_func: Callable,
        *args,
        iterations: int = 1000,
        warmup: int = 100,
        **kwargs,
    ) -> Dict[str, float]:
        """Time async function execution."""
        func_name = f"{async_func.__module__}.{async_func.__name__}"

        if HAS_RICH and self.console:
            self.console.print(
                f"⏱️  Timing async {func_name} ({iterations} iterations)", style="blue"
            )
        else:
            print(f"⏱️  Timing async {func_name} ({iterations} iterations)")

        # Warmup
        for _ in range(warmup):
            await async_func(*args, **kwargs)

        # Timing runs
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            await async_func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

            if HAS_RICH and self.console and i % 100 == 0:
                # Update progress occasionally
                pass

        # Calculate statistics
        stats = {
            "function": func_name,
            "iterations": iterations,
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "total": sum(times),
            "ops_per_second": iterations / sum(times),
        }

        return stats

    async def concurrent_benchmark(
        self,
        async_func: Callable,
        *args,
        concurrent_tasks: int = 10,
        iterations_per_task: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """Benchmark async function with concurrent execution."""
        if HAS_RICH and self.console:
            self.console.print(
                f"🔄 Concurrent async benchmark: {async_func.__name__}", style="green"
            )
        else:
            print(f"🔄 Concurrent async benchmark: {async_func.__name__}")

        async def task_worker():
            times = []
            for _ in range(iterations_per_task):
                start = time.perf_counter()
                await async_func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            return times

        # Run concurrent tasks
        start_time = time.perf_counter()
        tasks = [task_worker() for _ in range(concurrent_tasks)]
        all_results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        # Combine all timing results
        all_times = []
        for task_times in all_results:
            all_times.extend(task_times)

        total_operations = concurrent_tasks * iterations_per_task
        total_duration = end_time - start_time

        stats = {
            "function": async_func.__name__,
            "concurrent_tasks": concurrent_tasks,
            "iterations_per_task": iterations_per_task,
            "total_operations": total_operations,
            "total_duration": total_duration,
            "mean_per_operation": statistics.mean(all_times),
            "median_per_operation": statistics.median(all_times),
            "operations_per_second": total_operations / total_duration,
            "concurrent_throughput": total_operations / total_duration,
        }

        return stats


class BenchmarkSuite:
    """Comprehensive benchmarking suite."""

    def __init__(self):
        self.timing_benchmark = TimingBenchmark()
        self.stress_tester = StressTester()
        self.async_benchmark = AsyncBenchmark()
        self.console = Console() if HAS_RICH else None

        self.suite_results: Dict[str, Any] = {}
        self.benchmark_history: List[Dict[str, Any]] = []

    def add_benchmark(
        self, name: str, func: Callable, *args, benchmark_type: str = "timing", **kwargs
    ) -> None:
        """Add a benchmark to the suite."""
        benchmark_config = {
            "name": name,
            "function": func,
            "args": args,
            "kwargs": kwargs,
            "type": benchmark_type,
        }

        if name not in self.suite_results:
            self.suite_results[name] = []

        self.suite_results[name].append(benchmark_config)

        if HAS_RICH and self.console:
            self.console.print(
                f"➕ Added benchmark: {name} ({benchmark_type})", style="green"
            )
        else:
            print(f"➕ Added benchmark: {name} ({benchmark_type})")

    def run_suite(self, save_results: bool = True) -> Dict[str, Any]:
        """Run all benchmarks in the suite."""
        if HAS_RICH and self.console:
            self.console.print("🏃 Running benchmark suite...", style="bold blue")
        else:
            print("🏃 Running benchmark suite...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {},
            "summary": {},
        }

        for name, configs in self.suite_results.items():
            if HAS_RICH and self.console:
                self.console.print(f"Running: {name}", style="blue")

            benchmark_results = []

            for config in configs:
                if config["type"] == "timing":
                    result = self.timing_benchmark.time_it(
                        config["function"], *config["args"], **config["kwargs"]
                    )
                elif config["type"] == "load":
                    result = self.stress_tester.load_test(
                        config["function"], *config["args"], **config["kwargs"]
                    )
                elif config["type"] == "spike":
                    result = self.stress_tester.spike_test(
                        config["function"], *config["args"], **config["kwargs"]
                    )
                else:
                    result = {"error": f"Unknown benchmark type: {config['type']}"}

                benchmark_results.append(result)

            results["benchmarks"][name] = benchmark_results

        # Generate summary
        results["summary"] = self._generate_summary(results["benchmarks"])

        if save_results:
            self._save_results(results)

        self.benchmark_history.append(results)

        if HAS_RICH and self.console:
            self.console.print("✅ Benchmark suite completed!", style="bold green")
        else:
            print("✅ Benchmark suite completed!")

        return results

    def _generate_summary(
        self, benchmarks: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Generate summary statistics for the benchmark suite."""
        summary = {
            "total_benchmarks": len(benchmarks),
            "fastest_function": None,
            "slowest_function": None,
            "average_ops_per_second": 0,
        }

        all_ops_per_second = []
        fastest_ops = 0
        slowest_ops = float("inf")

        for name, results in benchmarks.items():
            for result in results:
                if "ops_per_second" in result:
                    ops = result["ops_per_second"]
                    all_ops_per_second.append(ops)

                    if ops > fastest_ops:
                        fastest_ops = ops
                        summary["fastest_function"] = name

                    if ops < slowest_ops:
                        slowest_ops = ops
                        summary["slowest_function"] = name

        if all_ops_per_second:
            summary["average_ops_per_second"] = statistics.mean(all_ops_per_second)

        return summary

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"

        Path("benchmark_results").mkdir(exist_ok=True)
        filepath = Path("benchmark_results") / filename

        try:
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)

            if HAS_RICH and self.console:
                self.console.print(f"💾 Results saved to {filepath}", style="green")
            else:
                print(f"💾 Results saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

    def compare_with_history(self, benchmark_name: str) -> Dict[str, Any]:
        """Compare current benchmark with historical results."""
        if not self.benchmark_history:
            return {"error": "No historical data available"}

        current = self.benchmark_history[-1]["benchmarks"].get(benchmark_name)
        if not current:
            return {"error": f"Benchmark {benchmark_name} not found in current results"}

        historical_results = []
        for run in self.benchmark_history[:-1]:
            if benchmark_name in run["benchmarks"]:
                historical_results.append(run["benchmarks"][benchmark_name])

        if not historical_results:
            return {"error": "No historical data for this benchmark"}

        # Calculate trend analysis
        comparison = {
            "benchmark_name": benchmark_name,
            "current_performance": current,
            "historical_count": len(historical_results),
            "trend": "stable",  # Will be calculated
            "improvement_percentage": 0,
        }

        return comparison

    def plot_results(self, benchmark_name: str, save_plot: bool = True) -> None:
        """Plot benchmark results over time."""
        if not HAS_MATPLOTLIB:
            print("⚠️  matplotlib not available for plotting")
            return

        # Extract data for plotting
        timestamps = []
        ops_per_second = []

        for run in self.benchmark_history:
            if benchmark_name in run["benchmarks"]:
                timestamps.append(run["timestamp"])
                # Get ops_per_second from first result
                result = run["benchmarks"][benchmark_name][0]
                if "ops_per_second" in result:
                    ops_per_second.append(result["ops_per_second"])

        if not ops_per_second:
            print(f"❌ No data available for plotting {benchmark_name}")
            return

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(ops_per_second)), ops_per_second, marker="o")
        plt.title(f"Performance Trend: {benchmark_name}")
        plt.xlabel("Benchmark Run")
        plt.ylabel("Operations per Second")
        plt.grid(True, alpha=0.3)

        if save_plot:
            Path("benchmark_plots").mkdir(exist_ok=True)
            filename = f"benchmark_plots/{benchmark_name}_trend.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"📊 Plot saved to {filename}")

        plt.show()


# Create global benchmark instance
benchmark = BenchmarkSuite()
