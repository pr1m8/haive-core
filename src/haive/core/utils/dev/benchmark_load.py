"""
Load Testing and Stress Testing Utilities

Provides concurrent load testing and spike testing capabilities.
"""

import statistics
import threading
import time
from collections.abc import Callable

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class LoadTester:
    """Load testing with concurrent users."""

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
    ) -> dict[str, any]:
        """Perform load testing with concurrent users."""
        if HAS_RICH and self.console:
            self.console.print(
                f"🔥 Load Testing {func.__name__} "
                f"({concurrent_users} users, {duration_seconds}s)",
                style="bold red",
            )

        results = {
            "function": func.__name__,
            "concurrent_users": concurrent_users,
            "duration_seconds": duration_seconds,
            "response_times": [],
            "successful_requests": 0,
            "failed_requests": 0,
            "start_time": time.time(),
        }

        # Shared data structures
        response_times = []
        successful_count = 0
        failed_count = 0
        lock = threading.Lock()

        def worker():
            """Worker function for each concurrent user."""
            nonlocal successful_count, failed_count

            user_start_time = time.time()
            while time.time() - user_start_time < duration_seconds:
                start_time = time.perf_counter()
                try:
                    func(*args, **kwargs)
                    end_time = time.perf_counter()
                    response_time = end_time - start_time

                    with lock:
                        response_times.append(response_time)
                        successful_count += 1
                except Exception:
                    with lock:
                        failed_count += 1

        # Start threads with ramp-up
        threads = []
        ramp_up_delay = (
            ramp_up_seconds / concurrent_users if concurrent_users > 0 else 0
        )

        for i in range(concurrent_users):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
            if i < concurrent_users - 1:  # Don't delay after last thread
                time.sleep(ramp_up_delay)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Calculate final results
        end_time = time.time()
        total_duration = end_time - results["start_time"]
        total_requests = successful_count + failed_count

        results.update(
            {
                "response_times": response_times,
                "successful_requests": successful_count,
                "failed_requests": failed_count,
                "total_requests": total_requests,
                "throughput": (
                    total_requests / total_duration if total_duration > 0 else 0
                ),
                "error_rate": (
                    failed_count / total_requests if total_requests > 0 else 0
                ),
            }
        )

        if response_times:
            results["stats"] = {
                "mean_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "p95": (
                    statistics.quantiles(response_times, n=20)[18]
                    if len(response_times) >= 20
                    else max(response_times)
                ),
                "p99": (
                    statistics.quantiles(response_times, n=100)[98]
                    if len(response_times) >= 100
                    else max(response_times)
                ),
            }

        self._display_load_test_results(results)
        return results

    def _display_load_test_results(self, results: dict[str, any]) -> None:
        """Display load test results."""
        if HAS_RICH and self.console:
            table = Table(title=f"🔥 Load Test Results: {results['function']}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Concurrent Users", str(results["concurrent_users"]))
            table.add_row("Duration", f"{results['duration_seconds']}s")
            table.add_row("Total Requests", str(results["total_requests"]))
            table.add_row("Successful", str(results["successful_requests"]))
            table.add_row("Failed", str(results["failed_requests"]))
            table.add_row("Throughput", f"{results['throughput']:.2f} req/s")
            table.add_row("Error Rate", f"{results['error_rate']:.2%}")

            if "stats" in results:
                stats = results["stats"]
                table.add_row("Mean Response", f"{stats['mean_response_time']:.3f}s")
                table.add_row("95th Percentile", f"{stats['p95']:.3f}s")
                table.add_row("99th Percentile", f"{stats['p99']:.3f}s")

            self.console.print(table)

    def spike_test(
        self,
        func: Callable,
        *args,
        base_users: int = 5,
        spike_users: int = 50,
        spike_duration: int = 30,
        **kwargs,
    ) -> dict[str, any]:
        """Perform spike testing with sudden load increases."""
        if HAS_RICH and self.console:
            self.console.print(f"⚡ Spike Testing {func.__name__}", style="bold yellow")

        # Phase 1: Base load
        base_results = self.load_test(
            func, *args, concurrent_users=base_users, duration_seconds=30, **kwargs
        )

        # Phase 2: Spike load
        spike_results = self.load_test(
            func,
            *args,
            concurrent_users=spike_users,
            duration_seconds=spike_duration,
            ramp_up_seconds=5,
            **kwargs,
        )

        # Phase 3: Recovery
        recovery_results = self.load_test(
            func, *args, concurrent_users=base_users, duration_seconds=30, **kwargs
        )

        return {
            "function": func.__name__,
            "base_load": base_results,
            "spike_load": spike_results,
            "recovery": recovery_results,
            "performance_degradation": (
                spike_results["stats"]["mean_response_time"]
                / base_results["stats"]["mean_response_time"]
                if "stats" in spike_results and "stats" in base_results
                else None
            ),
        }


# Global instance for easy import
load_tester = LoadTester()
