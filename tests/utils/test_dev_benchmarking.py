"""
Tests for the benchmarking utilities module.

Tests timing benchmarks, stress testing, load testing, async benchmarking,
and the comprehensive benchmark suite functionality.
"""

import asyncio
import io
import time
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from haive.core.utils.dev.benchmarking import (
    AsyncBenchmark,
    BenchmarkSuite,
    StressTester,
    TimingBenchmark,
    benchmark,
)


class TestTimingBenchmark:
    """Test the TimingBenchmark class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = TimingBenchmark()

    def teardown_method(self):
        """Clean up after tests."""
        self.benchmark.results.clear()

    def test_initialization(self):
        """Test TimingBenchmark initialization."""
        assert len(self.benchmark.results) == 0
        assert hasattr(self.benchmark, "console")

    def test_time_it_basic(self):
        """Test basic timing functionality."""

        def test_func():
            time.sleep(0.001)
            return "result"

        with redirect_stdout(io.StringIO()) as f:
            stats = self.benchmark.time_it(test_func, iterations=5, warmup=1)

        assert isinstance(stats, dict)
        assert "function" in stats
        assert "iterations" in stats
        assert "mean" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert "ops_per_second" in stats
        assert "percentiles" in stats

        assert stats["iterations"] == 5
        assert stats["mean"] > 0
        assert stats["ops_per_second"] > 0

        # Check that results were stored
        func_name = f"{test_func.__module__}.{test_func.__name__}"
        assert func_name in self.benchmark.results

        output = f.getvalue()
        assert "Timing" in output or func_name in output

    def test_time_it_with_args(self):
        """Test timing with function arguments."""

        def test_func(x, y, multiplier=1):
            time.sleep(0.001)
            return (x + y) * multiplier

        with redirect_stdout(io.StringIO()):
            stats = self.benchmark.time_it(test_func, 5, 3, multiplier=2, iterations=3)

        assert stats["iterations"] == 3
        assert stats["mean"] > 0

    def test_statistical_analysis(self):
        """Test statistical analysis in timing results."""

        def consistent_func():
            time.sleep(0.001)  # Consistent timing

        with redirect_stdout(io.StringIO()):
            stats = self.benchmark.time_it(consistent_func, iterations=10)

        # Statistical measures should be reasonable
        assert stats["mean"] > 0
        assert stats["median"] > 0
        assert stats["stdev"] >= 0
        assert stats["variance"] >= 0
        assert stats["min"] <= stats["median"] <= stats["max"]

        # Percentiles should be ordered
        percentiles = stats["percentiles"]
        assert percentiles["p10"] <= percentiles["p50"]
        assert percentiles["p50"] <= percentiles["p90"]
        assert percentiles["p90"] <= percentiles["p99"]

    def test_compare_functions(self):
        """Test comparing multiple functions."""

        def fast_func():
            return sum(range(10))

        def slow_func():
            time.sleep(0.001)
            return sum(range(10))

        functions = [fast_func, slow_func]

        with redirect_stdout(io.StringIO()) as f:
            results = self.benchmark.compare_functions(functions, iterations=3)

        assert isinstance(results, dict)
        assert "fast_func" in results
        assert "slow_func" in results

        # Fast function should have higher ops/second
        fast_ops = results["fast_func"]["ops_per_second"]
        slow_ops = results["slow_func"]["ops_per_second"]
        assert fast_ops > slow_ops

        output = f.getvalue()
        assert "Performance Comparison" in output or "fast_func" in output


class TestStressTester:
    """Test the StressTester class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tester = StressTester()

    def test_initialization(self):
        """Test StressTester initialization."""
        assert hasattr(self.tester, "console")

    def test_load_test_basic(self):
        """Test basic load testing functionality."""
        call_count = 0

        def test_func():
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)  # Small delay
            return "success"

        with redirect_stdout(io.StringIO()) as f:
            results = self.tester.load_test(
                test_func, concurrent_users=2, duration_seconds=1, ramp_up_seconds=0
            )

        assert isinstance(results, dict)
        assert "function" in results
        assert "concurrent_users" in results
        assert "duration_seconds" in results
        assert "total_requests" in results
        assert "successful_requests" in results
        assert "failed_requests" in results
        assert "throughput" in results

        assert results["concurrent_users"] == 2
        assert results["duration_seconds"] == 1
        assert results["total_requests"] > 0
        assert results["successful_requests"] > 0
        assert results["failed_requests"] == 0
        assert results["throughput"] > 0

        # Function should have been called multiple times
        assert call_count > 0

        output = f.getvalue()
        assert "Load Testing" in output

    def test_load_test_with_failures(self):
        """Test load testing with function failures."""
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise ValueError("Simulated failure")
            return "success"

        with redirect_stdout(io.StringIO()):
            results = self.tester.load_test(
                failing_func, concurrent_users=1, duration_seconds=1, ramp_up_seconds=0
            )

        assert results["total_requests"] > 0
        assert results["failed_requests"] > 0
        assert results["successful_requests"] > 0
        assert len(results["errors"]) > 0

        # Check that error messages are captured
        assert any("Simulated failure" in error for error in results["errors"])

    def test_spike_test(self):
        """Test spike testing functionality."""

        def test_func():
            time.sleep(0.001)
            return "spike_result"

        with redirect_stdout(io.StringIO()) as f:
            results = self.tester.spike_test(
                test_func, base_users=1, spike_users=3, spike_duration=1
            )

        assert isinstance(results, dict)
        assert "base_load" in results
        assert "spike_load" in results
        assert "recovery" in results

        # Each phase should have results
        for phase in ["base_load", "spike_load", "recovery"]:
            phase_results = results[phase]
            assert "total_requests" in phase_results
            assert "successful_requests" in phase_results

        # Spike phase should have higher concurrency
        assert results["spike_load"]["concurrent_users"] > results["base_load"]["concurrent_users"]

        output = f.getvalue()
        assert "Spike Testing" in output

    @patch("haive.core.utils.dev.benchmarking.threading.Thread")
    def test_load_test_threading(self, mock_thread):
        """Test that load testing properly uses threading."""

        def test_func():
            return "threaded"

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        with redirect_stdout(io.StringIO()):
            self.tester.load_test(test_func, concurrent_users=3, duration_seconds=1)

        # Should create 3 threads for 3 concurrent users
        assert mock_thread.call_count == 3
        assert mock_thread_instance.start.call_count == 3
        assert mock_thread_instance.join.call_count == 3


class TestAsyncBenchmark:
    """Test the AsyncBenchmark class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = AsyncBenchmark()

    def test_initialization(self):
        """Test AsyncBenchmark initialization."""
        assert hasattr(self.benchmark, "console")

    @pytest.mark.asyncio
    async def test_time_async_basic(self):
        """Test basic async function timing."""

        async def async_func():
            await asyncio.sleep(0.001)
            return "async_result"

        with redirect_stdout(io.StringIO()) as f:
            stats = await self.benchmark.time_async(async_func, iterations=3, warmup=1)

        assert isinstance(stats, dict)
        assert "function" in stats
        assert "iterations" in stats
        assert "mean" in stats
        assert "median" in stats
        assert "ops_per_second" in stats

        assert stats["iterations"] == 3
        assert stats["mean"] > 0
        assert stats["ops_per_second"] > 0

        output = f.getvalue()
        assert "async" in output.lower()

    @pytest.mark.asyncio
    async def test_time_async_with_args(self):
        """Test async timing with arguments."""

        async def async_func_with_args(x, y=1):
            await asyncio.sleep(0.001)
            return x + y

        with redirect_stdout(io.StringIO()):
            stats = await self.benchmark.time_async(async_func_with_args, 5, y=3, iterations=2)

        assert stats["iterations"] == 2
        assert stats["mean"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_benchmark(self):
        """Test concurrent async benchmarking."""

        async def async_func():
            await asyncio.sleep(0.001)
            return "concurrent"

        with redirect_stdout(io.StringIO()) as f:
            stats = await self.benchmark.concurrent_benchmark(
                async_func, concurrent_tasks=3, iterations_per_task=2
            )

        assert isinstance(stats, dict)
        assert "function" in stats
        assert "concurrent_tasks" in stats
        assert "iterations_per_task" in stats
        assert "total_operations" in stats
        assert "concurrent_throughput" in stats

        assert stats["concurrent_tasks"] == 3
        assert stats["iterations_per_task"] == 2
        assert stats["total_operations"] == 6  # 3 * 2
        assert stats["concurrent_throughput"] > 0

        output = f.getvalue()
        assert "Concurrent async benchmark" in output or "concurrent" in output


class TestBenchmarkSuite:
    """Test the BenchmarkSuite class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.suite = BenchmarkSuite()

    def teardown_method(self):
        """Clean up after tests."""
        self.suite.suite_results.clear()
        self.suite.benchmark_history.clear()

    def test_initialization(self):
        """Test BenchmarkSuite initialization."""
        assert hasattr(self.suite, "timing_benchmark")
        assert hasattr(self.suite, "stress_tester")
        assert hasattr(self.suite, "async_benchmark")
        assert len(self.suite.suite_results) == 0
        assert len(self.suite.benchmark_history) == 0

    def test_add_benchmark(self):
        """Test adding benchmarks to the suite."""

        def test_func():
            return "test"

        with redirect_stdout(io.StringIO()) as f:
            self.suite.add_benchmark(
                "test_benchmark",
                test_func,
                "arg1",
                benchmark_type="timing",
                key="value",
            )

        assert "test_benchmark" in self.suite.suite_results
        benchmarks = self.suite.suite_results["test_benchmark"]
        assert len(benchmarks) == 1

        benchmark_config = benchmarks[0]
        assert benchmark_config["name"] == "test_benchmark"
        assert benchmark_config["function"] == test_func
        assert benchmark_config["args"] == ("arg1",)
        assert benchmark_config["kwargs"] == {"key": "value"}
        assert benchmark_config["type"] == "timing"

        output = f.getvalue()
        assert "Added benchmark" in output

    def test_add_multiple_benchmarks(self):
        """Test adding multiple benchmarks with same name."""

        def test_func1():
            return "test1"

        def test_func2():
            return "test2"

        with redirect_stdout(io.StringIO()):
            self.suite.add_benchmark("multi_test", test_func1, benchmark_type="timing")
            self.suite.add_benchmark("multi_test", test_func2, benchmark_type="load")

        benchmarks = self.suite.suite_results["multi_test"]
        assert len(benchmarks) == 2
        assert benchmarks[0]["type"] == "timing"
        assert benchmarks[1]["type"] == "load"

    def test_run_suite_timing(self):
        """Test running suite with timing benchmarks."""

        def fast_func():
            return sum(range(10))

        def slow_func():
            time.sleep(0.001)
            return sum(range(10))

        with redirect_stdout(io.StringIO()):
            self.suite.add_benchmark("fast", fast_func, iterations=3, benchmark_type="timing")
            self.suite.add_benchmark("slow", slow_func, iterations=3, benchmark_type="timing")

        with redirect_stdout(io.StringIO()) as f:
            results = self.suite.run_suite(save_results=False)

        assert isinstance(results, dict)
        assert "timestamp" in results
        assert "benchmarks" in results
        assert "summary" in results

        benchmarks = results["benchmarks"]
        assert "fast" in benchmarks
        assert "slow" in benchmarks

        # Check benchmark results
        fast_result = benchmarks["fast"][0]
        slow_result = benchmarks["slow"][0]
        assert "ops_per_second" in fast_result
        assert "ops_per_second" in slow_result
        assert fast_result["ops_per_second"] > slow_result["ops_per_second"]

        # Check summary
        summary = results["summary"]
        assert "total_benchmarks" in summary
        assert "fastest_function" in summary
        assert "slowest_function" in summary
        assert summary["fastest_function"] == "fast"
        assert summary["slowest_function"] == "slow"

        # Should be added to history
        assert len(self.suite.benchmark_history) == 1

        output = f.getvalue()
        assert "Running benchmark suite" in output
        assert "completed" in output

    def test_run_suite_unknown_type(self):
        """Test running suite with unknown benchmark type."""

        def test_func():
            return "test"

        with redirect_stdout(io.StringIO()):
            self.suite.add_benchmark("unknown", test_func, benchmark_type="unknown_type")

        with redirect_stdout(io.StringIO()):
            results = self.suite.run_suite(save_results=False)

        benchmark_result = results["benchmarks"]["unknown"][0]
        assert "error" in benchmark_result
        assert "Unknown benchmark type" in benchmark_result["error"]

    def test_summary_generation(self):
        """Test benchmark summary generation."""
        benchmarks = {
            "fast": [{"ops_per_second": 1000}],
            "medium": [{"ops_per_second": 500}],
            "slow": [{"ops_per_second": 100}],
        }

        summary = self.suite._generate_summary(benchmarks)

        assert summary["total_benchmarks"] == 3
        assert summary["fastest_function"] == "fast"
        assert summary["slowest_function"] == "slow"
        assert summary["average_ops_per_second"] == 533.33  # (1000+500+100)/3

    @patch("haive.core.utils.dev.benchmarking.json.dump")
    @patch("haive.core.utils.dev.benchmarking.Path")
    def test_save_results(self, mock_path, mock_json_dump):
        """Test saving benchmark results."""
        MagicMock()
        mock_path.return_value.mkdir.return_value = None
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        results = {"test": "data"}

        with redirect_stdout(io.StringIO()) as f:
            self.suite._save_results(results)

        mock_path.return_value.mkdir.assert_called_with(exist_ok=True)
        mock_json_dump.assert_called_once()

        output = f.getvalue()
        assert "Results saved" in output

    def test_compare_with_history_no_history(self):
        """Test comparison with no historical data."""
        result = self.suite.compare_with_history("test_benchmark")

        assert "error" in result
        assert "No historical data available" in result["error"]

    def test_compare_with_history_missing_benchmark(self):
        """Test comparison with missing benchmark in current results."""
        # Add some history without the benchmark we're looking for
        self.suite.benchmark_history.append(
            {"benchmarks": {"other_benchmark": [{"ops_per_second": 100}]}}
        )

        result = self.suite.compare_with_history("missing_benchmark")

        assert "error" in result
        assert "not found in current results" in result["error"]

    @patch("haive.core.utils.dev.benchmarking.HAS_MATPLOTLIB", False)
    def test_plot_results_no_matplotlib(self):
        """Test plotting when matplotlib is not available."""
        with redirect_stdout(io.StringIO()) as f:
            self.suite.plot_results("test_benchmark")

        output = f.getvalue()
        assert "matplotlib not available" in output

    @patch("haive.core.utils.dev.benchmarking.HAS_MATPLOTLIB", True)
    @patch("haive.core.utils.dev.benchmarking.plt")
    def test_plot_results_no_data(self, mock_plt):
        """Test plotting with no data available."""
        with redirect_stdout(io.StringIO()) as f:
            self.suite.plot_results("nonexistent_benchmark")

        output = f.getvalue()
        assert "No data available for plotting" in output
        mock_plt.figure.assert_not_called()


class TestGlobalBenchmarkInstance:
    """Test the global benchmark instance."""

    def test_global_benchmark_instance(self):
        """Test that the global benchmark instance works correctly."""
        assert hasattr(benchmark, "timing_benchmark")
        assert hasattr(benchmark, "stress_tester")
        assert hasattr(benchmark, "async_benchmark")
        assert hasattr(benchmark, "add_benchmark")
        assert hasattr(benchmark, "run_suite")

    def test_global_benchmark_functionality(self):
        """Test basic functionality of global benchmark instance."""

        def test_func():
            return "global_test"

        with redirect_stdout(io.StringIO()) as f:
            benchmark.add_benchmark("global_test", test_func, iterations=2)

        output = f.getvalue()
        assert "Added benchmark" in output

        # Clean up
        benchmark.suite_results.clear()


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_benchmark_with_exception(self):
        """Test benchmarking function that raises exceptions."""

        def failing_func():
            raise ValueError("Test exception")

        suite = BenchmarkSuite()

        with redirect_stdout(io.StringIO()):
            suite.add_benchmark("failing", failing_func, iterations=1)

        # This should not crash the suite
        with redirect_stdout(io.StringIO()):
            results = suite.run_suite(save_results=False)

        # The benchmark should still be recorded, but with error handling
        assert "failing" in results["benchmarks"]

    def test_stress_test_with_resource_constraints(self):
        """Test stress testing under resource constraints."""
        call_count = 0
        max_calls = 50  # Limit to prevent runaway

        def resource_limited_func():
            nonlocal call_count
            call_count += 1
            if call_count > max_calls:
                raise RuntimeError("Resource limit exceeded")
            time.sleep(0.001)
            return "limited"

        tester = StressTester()

        with redirect_stdout(io.StringIO()):
            results = tester.load_test(
                resource_limited_func,
                concurrent_users=2,
                duration_seconds=1,
                ramp_up_seconds=0,
            )

        # Should handle some failures gracefully
        assert results["total_requests"] > 0
        # May have some failures due to resource constraints
        assert results["successful_requests"] >= 0

    def test_concurrent_benchmarking_thread_safety(self):
        """Test thread safety in concurrent benchmarking."""
        shared_counter = {"value": 0}

        def thread_safe_func():
            # Simulate some work that modifies shared state
            current = shared_counter["value"]
            time.sleep(0.001)  # Simulate processing time
            shared_counter["value"] = current + 1
            return shared_counter["value"]

        tester = StressTester()

        with redirect_stdout(io.StringIO()):
            results = tester.load_test(
                thread_safe_func,
                concurrent_users=3,
                duration_seconds=1,
                ramp_up_seconds=0,
            )

        # Should complete without deadlocks
        assert results["total_requests"] > 0
        assert results["successful_requests"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
