"""
Tests for the tracing utilities module.

Tests call tracking, variable tracking, and the main TracingUtilities
functionality with proper mocking for optional dependencies.
"""

import io
import time
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from haive.core.utils.dev.tracing import (
    CallTracker,
    TracingUtilities,
    VariableTracker,
    trace,
)


class TestCallTracker:
    """Test the CallTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = CallTracker()
        self.tracker.clear()

    def teardown_method(self):
        """Clean up after tests."""
        self.tracker.clear()

    def test_initialization(self):
        """Test CallTracker initialization."""
        assert len(self.tracker.calls) == 0
        assert len(self.tracker.call_stack) == 0
        assert self.tracker.enabled is False
        assert len(self.tracker.filters) == 0

    def test_enable_disable(self):
        """Test enabling and disabling call tracking."""
        assert self.tracker.enabled is False

        with redirect_stdout(io.StringIO()) as f:
            self.tracker.enable()

        assert self.tracker.enabled is True
        output = f.getvalue()
        assert "Call tracking enabled" in output

        with redirect_stdout(io.StringIO()) as f:
            self.tracker.disable()

        assert self.tracker.enabled is False
        output = f.getvalue()
        assert "Call tracking disabled" in output

    def test_filter_management(self):
        """Test adding and removing filters."""
        with redirect_stdout(io.StringIO()) as f:
            self.tracker.add_filter("test_pattern")

        assert "test_pattern" in self.tracker.filters
        output = f.getvalue()
        assert "Added filter" in output

        with redirect_stdout(io.StringIO()) as f:
            self.tracker.remove_filter("test_pattern")

        assert "test_pattern" not in self.tracker.filters
        output = f.getvalue()
        assert "Removed filter" in output

    def test_should_track_no_filters(self):
        """Test should_track with no filters (should track everything)."""
        assert self.tracker.should_track("any_function", "any_file.py") is True

    def test_should_track_with_filters(self):
        """Test should_track with filters."""
        self.tracker.add_filter("test_func")
        self.tracker.add_filter("utils")

        assert self.tracker.should_track("test_function", "main.py") is True
        assert self.tracker.should_track("other_function", "test_utils.py") is True
        assert self.tracker.should_track("unrelated", "main.py") is False

    def test_track_call_decorator_disabled(self):
        """Test track_call decorator when tracking is disabled."""

        @self.tracker.track_call
        def test_func():
            return "result"

        # Should not track when disabled
        result = test_func()
        assert result == "result"
        assert len(self.tracker.calls) == 0

    def test_track_call_decorator_enabled(self):
        """Test track_call decorator when tracking is enabled."""
        self.tracker.enable()

        @self.tracker.track_call
        def test_func():
            time.sleep(0.001)
            return "tracked"

        with redirect_stdout(io.StringIO()) as f:
            result = test_func()

        assert result == "tracked"
        assert len(self.tracker.calls) == 1

        call_info = self.tracker.calls[0]
        assert call_info["function"] == "test_func"
        assert call_info["status"] == "success"
        assert call_info["duration"] > 0
        assert "result" in call_info

        output = f.getvalue()
        assert "test_func" in output

    def test_track_call_with_exception(self):
        """Test track_call decorator with function that raises exception."""
        self.tracker.enable()

        @self.tracker.track_call
        def failing_func():
            raise ValueError("Test exception")

        with pytest.raises(ValueError, match="Test exception"):
            with redirect_stdout(io.StringIO()) as f:
                failing_func()

        assert len(self.tracker.calls) == 1
        call_info = self.tracker.calls[0]
        assert call_info["function"] == "failing_func"
        assert call_info["status"] == "error"
        assert "exception" in call_info
        assert call_info["exception"] == "Test exception"

        output = f.getvalue()
        assert "failing_func" in output
        assert "Test exception" in output

    def test_track_call_filtered_out(self):
        """Test track_call decorator with function filtered out."""
        self.tracker.enable()
        self.tracker.add_filter("allowed")

        @self.tracker.track_call
        def filtered_func():
            return "not_tracked"

        result = filtered_func()
        assert result == "not_tracked"
        assert len(self.tracker.calls) == 0  # Should be filtered out

    def test_nested_calls(self):
        """Test tracking nested function calls."""
        self.tracker.enable()

        @self.tracker.track_call
        def outer_func():
            return inner_func()

        @self.tracker.track_call
        def inner_func():
            return "nested_result"

        with redirect_stdout(io.StringIO()) as f:
            result = outer_func()

        assert result == "nested_result"
        assert len(self.tracker.calls) == 2

        # Check call stack depths
        outer_call = next(c for c in self.tracker.calls if c["function"] == "outer_func")
        inner_call = next(c for c in self.tracker.calls if c["function"] == "inner_func")

        assert outer_call["depth"] == 0
        assert inner_call["depth"] == 1

        output = f.getvalue()
        assert "outer_func" in output
        assert "inner_func" in output

    def test_get_stats(self):
        """Test getting call statistics."""
        self.tracker.enable()

        @self.tracker.track_call
        def fast_func():
            return "fast"

        @self.tracker.track_call
        def slow_func():
            time.sleep(0.001)
            return "slow"

        # Generate some calls
        with redirect_stdout(io.StringIO()):
            fast_func()
            fast_func()
            slow_func()

        with redirect_stdout(io.StringIO()) as f:
            stats = self.tracker.get_stats()

        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 3
        assert stats["failed_calls"] == 0
        assert stats["average_duration"] > 0
        assert stats["max_duration"] > 0

        function_counts = stats["function_counts"]
        assert function_counts["fast_func"] == 2
        assert function_counts["slow_func"] == 1

        output = f.getvalue()
        assert "Call Statistics" in output or "Total Calls" in output

    def test_clear_calls(self):
        """Test clearing call history."""
        self.tracker.enable()

        @self.tracker.track_call
        def test_func():
            return "test"

        with redirect_stdout(io.StringIO()):
            test_func()

        assert len(self.tracker.calls) == 1

        with redirect_stdout(io.StringIO()) as f:
            self.tracker.clear()

        assert len(self.tracker.calls) == 0
        assert len(self.tracker.call_stack) == 0
        output = f.getvalue()
        assert "Call history cleared" in output


class TestVariableTracker:
    """Test the VariableTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = VariableTracker()
        self.tracker.clear()

    def teardown_method(self):
        """Clean up after tests."""
        self.tracker.clear()

    def test_initialization(self):
        """Test VariableTracker initialization."""
        assert len(self.tracker.tracked_vars) == 0
        assert len(self.tracker.changes) == 0
        assert self.tracker.enabled is False

    def test_enable_disable(self):
        """Test enabling and disabling variable tracking."""
        with redirect_stdout(io.StringIO()) as f:
            self.tracker.enable()

        assert self.tracker.enabled is True
        output = f.getvalue()
        assert "Variable tracking enabled" in output

        with redirect_stdout(io.StringIO()) as f:
            self.tracker.disable()

        assert self.tracker.enabled is False
        output = f.getvalue()
        assert "Variable tracking disabled" in output

    def test_track_variable_disabled(self):
        """Test tracking variables when disabled."""
        self.tracker.track("test_var", "value")

        assert len(self.tracker.changes) == 0
        assert len(self.tracker.tracked_vars) == 0

    def test_track_variable_enabled(self):
        """Test tracking variables when enabled."""
        self.tracker.enable()

        with redirect_stdout(io.StringIO()) as f:
            self.tracker.track("test_var", "initial_value")

        assert len(self.tracker.tracked_vars) == 1
        assert len(self.tracker.changes) == 1
        assert self.tracker.tracked_vars["test_var"] == "initial_value"

        change = self.tracker.changes[0]
        assert change["name"] == "test_var"
        assert change["old_value"] is None
        assert change["new_value"] == "initial_value"
        assert "timestamp" in change
        assert "caller" in change

        output = f.getvalue()
        assert "test_var" in output
        assert "initial_value" in output

    def test_track_variable_changes(self):
        """Test tracking variable value changes."""
        self.tracker.enable()

        with redirect_stdout(io.StringIO()) as f:
            self.tracker.track("counter", 0)
            self.tracker.track("counter", 1)
            self.tracker.track("counter", 2)

        assert len(self.tracker.changes) == 3
        assert self.tracker.tracked_vars["counter"] == 2

        # Check change progression
        changes = self.tracker.changes
        assert changes[0]["old_value"] is None
        assert changes[0]["new_value"] == 0
        assert changes[1]["old_value"] == 0
        assert changes[1]["new_value"] == 1
        assert changes[2]["old_value"] == 1
        assert changes[2]["new_value"] == 2

        output = f.getvalue()
        assert "counter" in output

    def test_track_no_change(self):
        """Test tracking when variable value doesn't change."""
        self.tracker.enable()

        with redirect_stdout(io.StringIO()):
            self.tracker.track("static_var", "same_value")
            self.tracker.track("static_var", "same_value")  # No change

        # Should only record the first change
        assert len(self.tracker.changes) == 1

    def test_get_history_all(self):
        """Test getting all variable change history."""
        self.tracker.enable()

        with redirect_stdout(io.StringIO()):
            self.tracker.track("var1", "value1")
            self.tracker.track("var2", "value2")
            self.tracker.track("var1", "value1_updated")

        history = self.tracker.get_history()
        assert len(history) == 3

        # Check that all changes are included
        var1_changes = [c for c in history if c["name"] == "var1"]
        var2_changes = [c for c in history if c["name"] == "var2"]
        assert len(var1_changes) == 2
        assert len(var2_changes) == 1

    def test_get_history_specific_variable(self):
        """Test getting history for a specific variable."""
        self.tracker.enable()

        with redirect_stdout(io.StringIO()):
            self.tracker.track("target_var", "initial")
            self.tracker.track("other_var", "other")
            self.tracker.track("target_var", "updated")

        target_history = self.tracker.get_history("target_var")
        assert len(target_history) == 2

        for change in target_history:
            assert change["name"] == "target_var"

    def test_clear_tracking(self):
        """Test clearing variable tracking data."""
        self.tracker.enable()

        with redirect_stdout(io.StringIO()):
            self.tracker.track("test_var", "value")

        assert len(self.tracker.tracked_vars) == 1
        assert len(self.tracker.changes) == 1

        with redirect_stdout(io.StringIO()) as f:
            self.tracker.clear()

        assert len(self.tracker.tracked_vars) == 0
        assert len(self.tracker.changes) == 0
        output = f.getvalue()
        assert "Variable tracking history cleared" in output


class TestTracingUtilities:
    """Test the main TracingUtilities class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = TracingUtilities()
        self.tracer.clear()

    def teardown_method(self):
        """Clean up after tests."""
        self.tracer.clear()

    def test_initialization(self):
        """Test TracingUtilities initialization."""
        assert hasattr(self.tracer, "call_tracker")
        assert hasattr(self.tracer, "var_tracker")
        assert hasattr(self.tracer, "console")

    def test_calls_method_with_function(self):
        """Test calls method with direct function."""

        def test_func():
            return "traced"

        decorated_func = self.tracer.calls(test_func)

        with redirect_stdout(io.StringIO()) as f:
            result = decorated_func()

        assert result == "traced"
        assert self.tracer.call_tracker.enabled is True
        assert len(self.tracer.call_tracker.calls) == 1

        output = f.getvalue()
        assert "test_func" in output

    def test_calls_method_as_decorator(self):
        """Test calls method as decorator."""

        @self.tracer.calls
        def test_func():
            return "decorated"

        with redirect_stdout(io.StringIO()):
            result = test_func()

        assert result == "decorated"
        assert len(self.tracer.call_tracker.calls) == 1

    def test_calls_method_with_filters(self):
        """Test calls method with filters."""

        def allowed_func():
            return "allowed"

        def filtered_func():
            return "filtered"

        # Add filter and trace functions
        decorated_allowed = self.tracer.calls(allowed_func, filters=["allowed"])
        decorated_filtered = self.tracer.calls(filtered_func)

        with redirect_stdout(io.StringIO()):
            decorated_allowed()
            decorated_filtered()  # Should be filtered out

        # Only the allowed function should be tracked
        assert len(self.tracer.call_tracker.calls) == 1
        assert self.tracer.call_tracker.calls[0]["function"] == "allowed_func"

    @patch("haive.core.utils.dev.tracing.HAS_PYSNOOPER", True)
    @patch("haive.core.utils.dev.tracing.pysnooper")
    def test_snoop_available(self, mock_pysnooper):
        """Test snoop method when pysnooper is available."""
        mock_pysnooper.snoop.return_value = lambda x: x

        def test_func():
            return "snooped"

        decorated_func = self.tracer.snoop(test_func)
        result = decorated_func()

        assert result == "snooped"
        mock_pysnooper.snoop.assert_called_once()

    @patch("haive.core.utils.dev.tracing.HAS_PYSNOOPER", False)
    def test_snoop_fallback(self):
        """Test snoop method fallback when pysnooper is not available."""

        def test_func():
            return "fallback"

        with redirect_stdout(io.StringIO()) as f:
            decorated_func = self.tracer.snoop(test_func)

        output = f.getvalue()
        assert "pysnooper not available" in output

        # Should fall back to basic call tracing
        with redirect_stdout(io.StringIO()):
            result = decorated_func()

        assert result == "fallback"
        assert len(self.tracer.call_tracker.calls) == 1

    @patch("haive.core.utils.dev.tracing.HAS_HUNTER", True)
    @patch("haive.core.utils.dev.tracing.hunter")
    def test_hunt_available(self, mock_hunter):
        """Test hunt method when hunter is available."""
        with redirect_stdout(io.StringIO()) as f:
            self.tracer.hunt("call", action="return")

        mock_hunter.trace.assert_called_once_with("call", action="return")
        output = f.getvalue()
        assert "Starting hunter trace" in output

    @patch("haive.core.utils.dev.tracing.HAS_HUNTER", False)
    def test_hunt_not_available(self):
        """Test hunt method when hunter is not available."""
        with redirect_stdout(io.StringIO()) as f:
            self.tracer.hunt("call")

        output = f.getvalue()
        assert "hunter not available" in output

    def test_stack_method(self):
        """Test stack trace method."""
        with redirect_stdout(io.StringIO()) as f:
            result = self.tracer.stack(limit=5)

        assert isinstance(result, str)
        assert "Call Stack" in result

        output = f.getvalue()
        assert "Call Stack" in output

    def test_vars_method_with_kwargs(self):
        """Test vars method with specific variables."""
        test_var = "test_value"

        with redirect_stdout(io.StringIO()) as f:
            self.tracer.vars(test_var=test_var, number=42)

        assert self.tracer.var_tracker.enabled is True
        assert len(self.tracer.var_tracker.changes) == 2

        changes = self.tracer.var_tracker.changes
        change_names = [c["name"] for c in changes]
        assert "test_var" in change_names
        assert "number" in change_names

        output = f.getvalue()
        assert "test_var" in output

    def test_vars_method_without_kwargs(self):
        """Test vars method without specific variables (should track locals)."""

        with redirect_stdout(io.StringIO()):
            self.tracer.vars()

        # Should have tracked local variables
        assert self.tracer.var_tracker.enabled is True
        assert len(self.tracer.var_tracker.changes) > 0

        # Check that our local variables were tracked
        tracked_names = [c["name"] for c in self.tracer.var_tracker.changes]
        assert "local_var1" in tracked_names
        assert "local_var2" in tracked_names

    def test_trace_context_manager(self):
        """Test trace context manager."""
        with redirect_stdout(io.StringIO()) as f, self.tracer.trace_context("test_trace"):
            time.sleep(0.001)

        output = f.getvalue()
        assert "Starting trace: test_trace" in output
        assert "completed" in output

    def test_trace_context_with_exception(self):
        """Test trace context manager with exception."""
        with pytest.raises(ValueError, match="Context exception"):
            with redirect_stdout(io.StringIO()) as f:
                with self.tracer.trace_context("error_trace"):
                    raise ValueError("Context exception")

        output = f.getvalue()
        assert "Starting trace: error_trace" in output
        assert "Exception in error_trace" in output

    def test_profile_calls_decorator(self):
        """Test profile_calls decorator."""

        @self.tracer.profile_calls
        def timed_func():
            time.sleep(0.001)
            return "profiled"

        with redirect_stdout(io.StringIO()) as f:
            result = timed_func()

        assert result == "profiled"
        output = f.getvalue()
        assert "timed_func" in output
        # Should show timing information
        assert any(char in output for char in "⏱️🔄")

    def test_stats_method(self):
        """Test stats method."""

        # Generate some tracing data
        @self.tracer.calls
        def test_func():
            return "test"

        with redirect_stdout(io.StringIO()):
            test_func()
            self.tracer.vars(test_var="value")

        with redirect_stdout(io.StringIO()) as f:
            stats = self.tracer.stats()

        assert isinstance(stats, dict)
        assert "call_stats" in stats
        assert "variable_changes" in stats
        assert "tracked_variables" in stats

        output = f.getvalue()
        assert "Tracing Statistics" in output or "Total Calls" in output

    def test_clear_method(self):
        """Test clear method."""

        # Generate some data
        @self.tracer.calls
        def test_func():
            return "test"

        with redirect_stdout(io.StringIO()):
            test_func()
            self.tracer.vars(test_var="value")

        assert len(self.tracer.call_tracker.calls) > 0
        assert len(self.tracer.var_tracker.changes) > 0

        with redirect_stdout(io.StringIO()) as f:
            self.tracer.clear()

        assert len(self.tracer.call_tracker.calls) == 0
        assert len(self.tracer.var_tracker.changes) == 0

        output = f.getvalue()
        assert "All tracing data cleared" in output

    def test_report_generation(self):
        """Test report generation."""

        # Generate some tracing data
        @self.tracer.calls
        def test_func():
            return "test"

        with redirect_stdout(io.StringIO()):
            test_func()
            test_func()  # Call twice
            self.tracer.vars(var1="value1", var2="value2")

        with redirect_stdout(io.StringIO()) as f:
            report = self.tracer.report()

        assert isinstance(report, str)
        assert "# Tracing Report" in report
        assert "Generated at:" in report
        assert "## Call Statistics" in report
        assert "## Variable Changes:" in report

        output = f.getvalue()
        # Report should be printed to stdout
        assert "Tracing Report" in output

    def test_report_save_to_file(self):
        """Test saving report to file."""
        # Generate minimal data
        with redirect_stdout(io.StringIO()):
            self.tracer.vars(test_var="value")

        with patch("haive.core.utils.dev.tracing.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance

            with redirect_stdout(io.StringIO()) as f:
                self.tracer.report("test_report.md")

            mock_path_instance.write_text.assert_called_once()
            output = f.getvalue()
            assert "Report saved to test_report.md" in output


class TestGlobalTraceInstance:
    """Test the global trace instance."""

    def setup_method(self):
        """Set up test fixtures."""
        trace.clear()

    def teardown_method(self):
        """Clean up after tests."""
        trace.clear()

    def test_global_trace_instance(self):
        """Test that the global trace instance works correctly."""
        assert hasattr(trace, "calls")
        assert hasattr(trace, "snoop")
        assert hasattr(trace, "hunt")
        assert hasattr(trace, "stack")
        assert hasattr(trace, "vars")

    def test_global_trace_functionality(self):
        """Test basic functionality of global trace instance."""

        @trace.calls
        def test_func():
            return "global_test"

        with redirect_stdout(io.StringIO()) as f:
            result = test_func()

        assert result == "global_test"
        assert len(trace.call_tracker.calls) == 1

        output = f.getvalue()
        assert "test_func" in output

    def test_global_trace_context(self):
        """Test global trace context manager."""
        with redirect_stdout(io.StringIO()) as f, trace.trace_context("global_context"):
            time.sleep(0.001)

        output = f.getvalue()
        assert "global_context" in output


class TestIntegrationScenarios:
    """Test integration scenarios and complex use cases."""

    def test_combined_call_and_variable_tracking(self):
        """Test using both call and variable tracking together."""
        tracer = TracingUtilities()

        @tracer.calls
        def complex_func(initial_value):
            tracer.vars(input_value=initial_value)

            processed = initial_value * 2
            tracer.vars(processed_value=processed)

            return processed

        with redirect_stdout(io.StringIO()) as f:
            result = complex_func(5)

        assert result == 10
        assert len(tracer.call_tracker.calls) == 1
        assert len(tracer.var_tracker.changes) == 2

        # Check that variables were tracked correctly
        changes = tracer.var_tracker.changes
        input_change = next(c for c in changes if c["name"] == "input_value")
        processed_change = next(c for c in changes if c["name"] == "processed_value")

        assert input_change["new_value"] == 5
        assert processed_change["new_value"] == 10

        output = f.getvalue()
        assert "complex_func" in output

    def test_recursive_function_tracing(self):
        """Test tracing recursive functions."""
        tracer = TracingUtilities()

        @tracer.calls
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        with redirect_stdout(io.StringIO()) as f:
            result = fibonacci(4)

        assert result == 3  # fibonacci(4) = 3

        # Should have multiple calls due to recursion
        assert len(tracer.call_tracker.calls) > 1

        # Check that different recursion depths were recorded
        depths = [call["depth"] for call in tracer.call_tracker.calls]
        assert max(depths) > min(depths)

        output = f.getvalue()
        assert "fibonacci" in output

    def test_exception_handling_in_traced_functions(self):
        """Test exception handling in traced functions."""
        tracer = TracingUtilities()

        @tracer.calls
        def error_prone_func(should_fail):
            tracer.vars(should_fail=should_fail)

            if should_fail:
                raise RuntimeError("Intentional failure")
            return "success"

        # Test successful case
        with redirect_stdout(io.StringIO()):
            result = error_prone_func(False)

        assert result == "success"

        # Test failure case
        with pytest.raises(RuntimeError, match="Intentional failure"):
            with redirect_stdout(io.StringIO()):
                error_prone_func(True)

        # Should have recorded both calls
        assert len(tracer.call_tracker.calls) == 2

        # Check that one succeeded and one failed
        statuses = [call["status"] for call in tracer.call_tracker.calls]
        assert "success" in statuses
        assert "error" in statuses


if __name__ == "__main__":
    pytest.main([__file__])
