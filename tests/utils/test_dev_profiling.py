"""
Tests for the profiling utilities module.

Tests timing profiling, memory profiling, CPU profiling, and benchmarking
functionality with proper mocking for dependencies that may not be available.
"""

import io
import time
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from haive.core.utils.dev.profiling import (
    CPUProfiler,
    LineProfiler,
    MemoryProfiler,
    ProfilingUtilities,
    TimingProfiler,
    profile,
)


class TestTimingProfiler:
    """Test the TimingProfiler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = TimingProfiler()

    def teardown_method(self):
        """Clean up after tests."""
        self.profiler.clear()

    def test_initialization(self):
        """Test TimingProfiler initialization."""
        assert len(self.profiler.timings) == 0
        assert hasattr(self.profiler, "console")

    def test_time_function_decorator(self):
        """Test timing function decorator."""

        @self.profiler.time_function
        def test_func():
            time.sleep(0.01)
            return "result"

        with redirect_stdout(io.StringIO()) as f:
            result = test_func()

        assert result == "result"
        output = f.getvalue()
        assert "test_func" in output

        # Check that timing was recorded
        func_name = f"{test_func.__module__}.{test_func.__name__}"
        assert func_name in self.profiler.timings
        assert len(self.profiler.timings[func_name]) == 1
        assert self.profiler.timings[func_name][0] > 0

    def test_get_stats(self):
        """Test getting timing statistics."""

        @self.profiler.time_function
        def test_func():
            time.sleep(0.001)

        # Run function multiple times
        with redirect_stdout(io.StringIO()):
            for _ in range(3):
                test_func()

        with redirect_stdout(io.StringIO()) as f:
            stats = self.profiler.get_stats()

        func_name = f"{test_func.__module__}.{test_func.__name__}"
        assert func_name in stats

        func_stats = stats[func_name]
        assert func_stats["count"] == 3
        assert func_stats["total"] > 0
        assert func_stats["mean"] > 0
        assert func_stats["min"] <= func_stats["max"]

        output = f.getvalue()
        assert "Timing Statistics" in output or func_name in output

    def test_clear_timings(self):
        """Test clearing timing data."""

        @self.profiler.time_function
        def test_func():
            pass

        with redirect_stdout(io.StringIO()):
            test_func()

        assert len(self.profiler.timings) > 0

        with redirect_stdout(io.StringIO()):
            self.profiler.clear()

        assert len(self.profiler.timings) == 0


class TestMemoryProfiler:
    """Test the MemoryProfiler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = MemoryProfiler()

    def test_initialization(self):
        """Test MemoryProfiler initialization."""
        assert hasattr(self.profiler, "baseline_memory")
        assert hasattr(self.profiler, "console")

    @patch("haive.core.utils.dev.profiling.HAS_PSUTIL", True)
    @patch("haive.core.utils.dev.profiling.psutil")
    def test_memory_usage_with_psutil(self, mock_psutil):
        """Test memory usage calculation with psutil."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_psutil.Process.return_value = mock_process

        memory_mb = self.profiler._get_memory_usage()
        assert memory_mb == 100.0  # Should convert to MB

    @patch("haive.core.utils.dev.profiling.HAS_PSUTIL", False)
    @patch("haive.core.utils.dev.profiling.HAS_MEMORY_PROFILER", True)
    @patch("haive.core.utils.dev.profiling.memory_profiler")
    def test_memory_usage_fallback(self, mock_memory_profiler):
        """Test memory usage fallback to memory_profiler."""
        mock_memory_profiler.memory_usage.return_value = [50.0]

        memory_mb = self.profiler._get_memory_usage()
        assert memory_mb == 50.0

    @patch("haive.core.utils.dev.profiling.HAS_PSUTIL", False)
    @patch("haive.core.utils.dev.profiling.HAS_MEMORY_PROFILER", False)
    def test_memory_usage_no_tools(self):
        """Test memory usage when no tools are available."""
        memory_mb = self.profiler._get_memory_usage()
        assert memory_mb == 0.0

    def test_profile_memory_decorator(self):
        """Test memory profiling decorator."""

        @self.profiler.profile_memory
        def test_func():
            # Simulate some memory usage
            data = [0] * 1000
            return len(data)

        with redirect_stdout(io.StringIO()) as f:
            result = test_func()

        assert result == 1000
        output = f.getvalue()
        assert "test_func" in output
        # Should show memory delta (may be positive or negative)
        assert "MB" in output

    @patch("haive.core.utils.dev.profiling.HAS_MEMORY_PROFILER", True)
    @patch("haive.core.utils.dev.profiling.memory_profiler")
    def test_memory_line_by_line_available(self, mock_memory_profiler):
        """Test line-by-line memory profiling when memory_profiler is available."""
        mock_memory_profiler.profile.return_value = lambda x: x

        @self.profiler.memory_line_by_line
        def test_func():
            return "profiled"

        result = test_func()
        assert result == "profiled"
        mock_memory_profiler.profile.assert_called_once()

    @patch("haive.core.utils.dev.profiling.HAS_MEMORY_PROFILER", False)
    def test_memory_line_by_line_fallback(self):
        """Test line-by-line memory profiling fallback."""
        with redirect_stdout(io.StringIO()) as f:
            decorated_func = self.profiler.memory_line_by_line(lambda: "fallback")

        output = f.getvalue()
        assert "memory_profiler not available" in output

        result = decorated_func()
        assert result == "fallback"

    def test_get_current_usage(self):
        """Test getting current memory usage statistics."""
        with redirect_stdout(io.StringIO()) as f:
            stats = self.profiler.get_current_usage()

        assert isinstance(stats, dict)
        assert "current_mb" in stats
        assert "baseline_mb" in stats
        assert "delta_mb" in stats

        output = f.getvalue()
        assert "Memory" in output


class TestLineProfiler:
    """Test the LineProfiler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = LineProfiler()

    @patch("haive.core.utils.dev.profiling.HAS_LINE_PROFILER", True)
    @patch("haive.core.utils.dev.profiling.line_profiler")
    def test_line_profiler_available(self, mock_line_profiler):
        """Test line profiler when line_profiler is available."""
        mock_profiler_instance = MagicMock()
        mock_line_profiler.LineProfiler.return_value = mock_profiler_instance

        profiler = LineProfiler()

        @profiler.profile_lines
        def test_func():
            return "profiled"

        result = test_func()
        assert result == "profiled"

        mock_profiler_instance.add_function.assert_called_once()
        mock_profiler_instance.enable_by_count.assert_called()
        mock_profiler_instance.disable_by_count.assert_called()

    @patch("haive.core.utils.dev.profiling.HAS_LINE_PROFILER", False)
    def test_line_profiler_not_available(self):
        """Test line profiler when line_profiler is not available."""
        profiler = LineProfiler()

        with redirect_stdout(io.StringIO()) as f:
            decorated_func = profiler.profile_lines(lambda: "original")

        output = f.getvalue()
        assert "line_profiler not available" in output

        result = decorated_func()
        assert result == "original"

    @patch("haive.core.utils.dev.profiling.HAS_LINE_PROFILER", True)
    @patch("haive.core.utils.dev.profiling.line_profiler")
    def test_show_stats(self, mock_line_profiler):
        """Test showing line profiler statistics."""
        mock_profiler_instance = MagicMock()
        mock_line_profiler.LineProfiler.return_value = mock_profiler_instance

        profiler = LineProfiler()

        with redirect_stdout(io.StringIO()) as f:
            profiler.show_stats()

        mock_profiler_instance.print_stats.assert_called_once()

        # Test with filename
        with redirect_stdout(io.StringIO()):
            profiler.show_stats("test_output.txt")

        output = f.getvalue()
        assert "test_output.txt" in output


class TestCPUProfiler:
    """Test the CPUProfiler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = CPUProfiler()

    @patch("haive.core.utils.dev.profiling.HAS_PYINSTRUMENT", True)
    @patch("haive.core.utils.dev.profiling.pyinstrument")
    def test_cpu_profiler_available(self, mock_pyinstrument):
        """Test CPU profiler when pyinstrument is available."""
        mock_profiler_instance = MagicMock()
        mock_profiler_instance.output_text.return_value = "Profile results"
        mock_pyinstrument.Profiler.return_value = mock_profiler_instance

        @self.profiler.profile_cpu
        def test_func():
            return "cpu_profiled"

        with redirect_stdout(io.StringIO()) as f:
            result = test_func()

        assert result == "cpu_profiled"
        output = f.getvalue()
        assert "Profile results" in output

        mock_profiler_instance.start.assert_called_once()
        mock_profiler_instance.stop.assert_called_once()

    @patch("haive.core.utils.dev.profiling.HAS_PYINSTRUMENT", False)
    def test_cpu_profiler_not_available(self):
        """Test CPU profiler when pyinstrument is not available."""
        with redirect_stdout(io.StringIO()) as f:
            decorated_func = self.profiler.profile_cpu(lambda: "original")

        output = f.getvalue()
        assert "pyinstrument not available" in output

        result = decorated_func()
        assert result == "original"

    @patch("haive.core.utils.dev.profiling.HAS_SCALENE", True)
    @patch("haive.core.utils.dev.profiling.subprocess")
    @patch("haive.core.utils.dev.profiling.Path")
    def test_scalene_profiling(self, mock_path, mock_subprocess):
        """Test Scalene profiling."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        with redirect_stdout(io.StringIO()) as f:
            self.profiler.profile_with_scalene("test_script.py")

        mock_subprocess.run.assert_called_once()
        output = f.getvalue()
        assert "Scalene profile saved" in output or "test_script.py" in output

    @patch("haive.core.utils.dev.profiling.HAS_SCALENE", False)
    def test_scalene_not_available(self):
        """Test Scalene profiling when not available."""
        with redirect_stdout(io.StringIO()) as f:
            self.profiler.profile_with_scalene("test_script.py")

        output = f.getvalue()
        assert "scalene not available" in output


class TestProfilingUtilities:
    """Test the main ProfilingUtilities class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = ProfilingUtilities()

    def teardown_method(self):
        """Clean up after tests."""
        self.profiler.clear()

    def test_initialization(self):
        """Test ProfilingUtilities initialization."""
        assert hasattr(self.profiler, "timing_profiler")
        assert hasattr(self.profiler, "memory_profiler")
        assert hasattr(self.profiler, "line_profiler")
        assert hasattr(self.profiler, "cpu_profiler")

    def test_time_profiling(self):
        """Test time profiling wrapper."""

        @self.profiler.time
        def test_func():
            time.sleep(0.001)
            return "timed"

        with redirect_stdout(io.StringIO()) as f:
            result = test_func()

        assert result == "timed"
        output = f.getvalue()
        assert "test_func" in output

    def test_memory_profiling(self):
        """Test memory profiling wrapper."""

        @self.profiler.memory
        def test_func():
            data = [0] * 100
            return len(data)

        with redirect_stdout(io.StringIO()) as f:
            result = test_func()

        assert result == 100
        output = f.getvalue()
        assert "test_func" in output

    def test_memory_profiling_line_by_line(self):
        """Test line-by-line memory profiling."""
        with redirect_stdout(io.StringIO()):
            decorated_func = self.profiler.memory(line_by_line=True)

        assert callable(decorated_func)

    def test_comprehensive_profiling(self):
        """Test comprehensive profiling (all profilers)."""

        @self.profiler.comprehensive
        def test_func():
            time.sleep(0.001)
            return "comprehensive"

        with redirect_stdout(io.StringIO()) as f:
            result = test_func()

        assert result == "comprehensive"
        output = f.getvalue()
        assert "comprehensive profiling" in output.lower()

    def test_profile_context_manager(self):
        """Test profiling context manager."""
        with redirect_stdout(io.StringIO()) as f:
            with self.profiler.profile_context("test_context"):
                time.sleep(0.001)

        output = f.getvalue()
        assert "test_context" in output
        assert "Duration" in output or "completed" in output

    def test_profile_context_memory_only(self):
        """Test profiling context with memory only."""
        with redirect_stdout(io.StringIO()) as f:
            with self.profiler.profile_context("memory_test", include_cpu=False):
                pass

        output = f.getvalue()
        assert "memory_test" in output

    def test_profile_context_cpu_only(self):
        """Test profiling context with CPU only."""
        with redirect_stdout(io.StringIO()) as f:
            with self.profiler.profile_context("cpu_test", include_memory=False):
                time.sleep(0.001)

        output = f.getvalue()
        assert "cpu_test" in output

    def test_benchmark_function(self):
        """Test benchmarking a function."""

        def test_func():
            time.sleep(0.001)

        with redirect_stdout(io.StringIO()) as f:
            stats = self.profiler.benchmark(test_func, iterations=10, warmup=2)

        assert isinstance(stats, dict)
        assert "iterations" in stats
        assert "mean_time" in stats
        assert "ops_per_second" in stats
        assert stats["iterations"] == 10

        output = f.getvalue()
        assert "Benchmarking" in output

    def test_compare_functions(self):
        """Test comparing multiple functions."""

        def fast_func():
            return sum([1, 2, 3])

        def slow_func():
            time.sleep(0.001)
            return sum([1, 2, 3])

        with redirect_stdout(io.StringIO()) as f:
            results = self.profiler.compare([fast_func, slow_func], iterations=5)

        assert isinstance(results, dict)
        assert "fast_func" in results
        assert "slow_func" in results

        output = f.getvalue()
        assert "Performance Comparison" in output or "fast_func" in output

    def test_stats_display(self):
        """Test statistics display."""

        # Generate some data first
        @self.profiler.time
        def test_func():
            pass

        with redirect_stdout(io.StringIO()):
            test_func()

        with redirect_stdout(io.StringIO()) as f:
            self.profiler.stats()

        output = f.getvalue()
        assert "Statistics" in output or "test_func" in output

    def test_clear_profiling_data(self):
        """Test clearing profiling data."""

        # Generate some data first
        @self.profiler.time
        def test_func():
            pass

        with redirect_stdout(io.StringIO()):
            test_func()

        # Clear data
        with redirect_stdout(io.StringIO()) as f:
            self.profiler.clear()

        output = f.getvalue()
        assert "cleared" in output.lower()

    def test_status_reporting(self):
        """Test status reporting of available tools."""
        with redirect_stdout(io.StringIO()) as f:
            status = self.profiler.status()

        assert isinstance(status, dict)
        assert "line_profiler" in status
        assert "memory_profiler" in status
        assert "pyinstrument" in status
        assert "scalene" in status
        assert "psutil" in status
        assert "rich" in status

        output = f.getvalue()
        assert "Profiling Tools Status" in output


class TestGlobalProfileInstance:
    """Test the global profile instance."""

    def test_global_profile_instance(self):
        """Test that the global profile instance works correctly."""
        assert hasattr(profile, "time")
        assert hasattr(profile, "memory")
        assert hasattr(profile, "cpu")
        assert hasattr(profile, "benchmark")
        assert hasattr(profile, "compare")

    def test_global_profile_functionality(self):
        """Test basic functionality of global profile instance."""

        @profile.time
        def test_func():
            return "global_test"

        with redirect_stdout(io.StringIO()) as f:
            result = test_func()

        assert result == "global_test"
        output = f.getvalue()
        assert "test_func" in output

    def test_global_profile_context(self):
        """Test global profile context manager."""
        with redirect_stdout(io.StringIO()) as f, profile.profile_context(
            "global_context"
        ):
            time.sleep(0.001)

        output = f.getvalue()
        assert "global_context" in output


class TestErrorHandling:
    """Test error handling in profiling utilities."""

    def test_function_decorator_with_exception(self):
        """Test that decorators handle exceptions properly."""
        profiler = ProfilingUtilities()

        @profiler.time
        def failing_func():
            raise ValueError("Test exception")

        with pytest.raises(ValueError, match="Test exception"):
            with redirect_stdout(io.StringIO()):
                failing_func()

        # Should still record timing data even with exception
        func_name = f"{failing_func.__module__}.{failing_func.__name__}"
        assert func_name in profiler.timing_profiler.timings

    def test_context_manager_with_exception(self):
        """Test that context managers handle exceptions properly."""
        profiler = ProfilingUtilities()

        with pytest.raises(ValueError, match="Context exception"):
            with redirect_stdout(io.StringIO()) as f:
                with profiler.profile_context("error_context"):
                    raise ValueError("Context exception")

        output = f.getvalue()
        assert "error_context" in output


if __name__ == "__main__":
    pytest.main([__file__])
