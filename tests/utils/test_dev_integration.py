"""
Integration tests for the dev utilities module.

Tests the main module imports, integration between components,
and end-to-end workflows using the dev utilities.
"""

import io
import time
from contextlib import redirect_stdout

import pytest


class TestDevUtilitiesImports:
    """Test that all dev utilities can be imported correctly."""

    def test_main_module_imports(self):
        """Test importing from the main dev module."""
        from haive.core.utils.dev import benchmark, debug, log, profile, trace

        # Check that all main instances are available
        assert debug is not None
        assert log is not None
        assert trace is not None
        assert profile is not None
        assert benchmark is not None

    def test_individual_module_imports(self):
        """Test importing individual modules."""
        # These should not raise ImportError
        from haive.core.utils.dev import (
            benchmarking,
            debugging,
            logging,
            profiling,
            tracing,
        )

        assert debugging is not None
        assert logging is not None
        assert tracing is not None
        assert profiling is not None
        assert benchmarking is not None

    def test_class_imports(self):
        """Test importing specific classes."""
        from haive.core.utils.dev.benchmarking import BenchmarkSuite
        from haive.core.utils.dev.debugging import DebugUtilities
        from haive.core.utils.dev.logging import DevLogger
        from haive.core.utils.dev.profiling import ProfilingUtilities
        from haive.core.utils.dev.tracing import TracingUtilities

        # All classes should be importable
        assert DebugUtilities is not None
        assert DevLogger is not None
        assert TracingUtilities is not None
        assert ProfilingUtilities is not None
        assert BenchmarkSuite is not None


class TestBasicWorkflows:
    """Test basic workflows using the dev utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        from haive.core.utils.dev import benchmark, debug, log, profile, trace

        self.debug = debug
        self.log = log
        self.trace = trace
        self.profile = profile
        self.benchmark = benchmark

        # Clear any existing state
        self.debug.clear_history()
        self.trace.clear()

    def teardown_method(self):
        """Clean up after tests."""
        self.debug.clear_history()
        self.trace.clear()

    def test_debug_workflow(self):
        """Test basic debugging workflow."""

        def test_function(x, y):
            self.debug.ice(f"Inputs: x={x}, y={y}")
            result = x + y
            self.debug.ice(f"Result: {result}")
            return result

        with redirect_stdout(io.StringIO()) as f:
            result = test_function(5, 3)

        assert result == 8
        output = f.getvalue()
        assert "Inputs" in output or "🍦" in output
        assert "Result" in output or "8" in output

        # Check debug history
        history = self.debug.history()
        assert len(history) == 2

    def test_logging_workflow(self):
        """Test basic logging workflow."""
        with redirect_stdout(io.StringIO()) as f:
            with self.log.context("test_operation"):
                self.log.info("Starting process")
                self.log.progress("Processing data", step=1)
                self.log.success("Process completed")

        output = f.getvalue()
        assert "test_operation" in output
        assert any(marker in output for marker in ["Starting process", "ℹ️"])
        assert any(marker in output for marker in ["Processing data", "🔄"])
        assert any(marker in output for marker in ["completed", "✅"])

    def test_tracing_workflow(self):
        """Test basic tracing workflow."""

        @self.trace.calls
        def traced_function(value):
            self.trace.vars(input_value=value)
            processed = value * 2
            self.trace.vars(processed_value=processed)
            return processed

        with redirect_stdout(io.StringIO()) as f:
            result = traced_function(10)

        assert result == 20
        output = f.getvalue()
        assert "traced_function" in output

        # Check tracing data
        stats = self.trace.stats()
        assert stats["call_stats"]["total_calls"] == 1
        assert stats["variable_changes"] == 2

    def test_profiling_workflow(self):
        """Test basic profiling workflow."""

        @self.profile.time
        def timed_function():
            time.sleep(0.001)
            return "profiled"

        with redirect_stdout(io.StringIO()) as f:
            result = timed_function()

        assert result == "profiled"
        output = f.getvalue()
        assert "timed_function" in output
        assert any(char in output for char in "⏱️")

    def test_benchmarking_workflow(self):
        """Test basic benchmarking workflow."""

        def simple_function():
            return sum(range(100))

        with redirect_stdout(io.StringIO()) as f:
            self.benchmark.add_benchmark(
                "simple_sum", simple_function, iterations=5, benchmark_type="timing"
            )
            results = self.benchmark.run_suite(save_results=False)

        output = f.getvalue()
        assert "simple_sum" in output
        assert "benchmarking" in output.lower() or "timing" in output.lower()

        assert "simple_sum" in results["benchmarks"]
        benchmark_result = results["benchmarks"]["simple_sum"][0]
        assert "ops_per_second" in benchmark_result


class TestIntegratedWorkflows:
    """Test workflows that use multiple dev utilities together."""

    def setup_method(self):
        """Set up test fixtures."""
        from haive.core.utils.dev import benchmark, debug, log, profile, trace

        self.debug = debug
        self.log = log
        self.trace = trace
        self.profile = profile
        self.benchmark = benchmark

        # Clear state
        self.debug.clear_history()
        self.trace.clear()

    def teardown_method(self):
        """Clean up after tests."""
        self.debug.clear_history()
        self.trace.clear()

    def test_debug_and_log_integration(self):
        """Test using debug and log utilities together."""

        def complex_operation(data):
            self.log.info("Starting complex operation", data_size=len(data))

            with self.log.context("processing"):
                for i, item in enumerate(data):
                    self.debug.ice(f"Processing item {i}: {item}")
                    # Simulate processing
                    item * 2

                    if i % 2 == 0:
                        self.log.progress(f"Processed {i+1}/{len(data)} items")

            self.log.success("Complex operation completed")
            return [item * 2 for item in data]

        test_data = [1, 2, 3, 4, 5]

        with redirect_stdout(io.StringIO()) as f:
            result = complex_operation(test_data)

        assert result == [2, 4, 6, 8, 10]
        output = f.getvalue()

        # Should contain both logging and debug output
        assert "complex operation" in output.lower()
        assert "processing" in output.lower()
        assert len(self.debug.history()) > 0

    def test_trace_and_profile_integration(self):
        """Test using trace and profile utilities together."""

        @self.trace.calls
        @self.profile.time
        def fibonacci(n):
            self.trace.vars(n=n)

            if n <= 1:
                return n

            result = fibonacci(n - 1) + fibonacci(n - 2)
            self.trace.vars(result=result)
            return result

        with redirect_stdout(io.StringIO()) as f:
            result = fibonacci(5)

        assert result == 5  # fibonacci(5) = 5
        output = f.getvalue()
        assert "fibonacci" in output

        # Check that both tracing and profiling data was collected
        trace_stats = self.trace.stats()
        assert trace_stats["call_stats"]["total_calls"] > 1  # Due to recursion
        assert trace_stats["variable_changes"] > 0

    def test_full_dev_utilities_workflow(self):
        """Test a complete workflow using all dev utilities."""

        def data_processor(dataset):
            """Process a dataset with full debugging and monitoring."""

            # Start with logging
            self.log.info("Data processing started", dataset_size=len(dataset))

            with self.log.context("data_processing"):
                # Debug the input
                self.debug.ice("Input dataset", dataset)

                # Trace the processing function
                @self.trace.calls
                @self.profile.time
                def process_batch(batch):
                    self.trace.vars(batch_size=len(batch))

                    processed = []
                    for item in batch:
                        # Debug each item
                        self.debug.ice(f"Processing: {item}")

                        # Simulate some computation
                        if isinstance(item, (int, float)):
                            processed_item = item**2
                        else:
                            processed_item = len(str(item))

                        processed.append(processed_item)

                    self.trace.vars(processed_batch=processed)
                    return processed

                # Process in batches
                batch_size = 3
                results = []

                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i : i + batch_size]
                    self.log.progress(f"Processing batch {i//batch_size + 1}")

                    batch_result = process_batch(batch)
                    results.extend(batch_result)

                self.log.success("Data processing completed")
                return results

        # Test the complete workflow
        test_dataset = [1, 2, 3, "hello", 4.5, "world", 6]

        with redirect_stdout(io.StringIO()) as f:
            result = data_processor(test_dataset)

        # Verify results
        expected = [1, 4, 9, 5, 20.25, 5, 36]  # squares for numbers, length for strings
        assert result == expected

        output = f.getvalue()

        # Verify that all utilities were used
        assert "Data processing started" in output
        assert "process_batch" in output  # From tracing
        assert "Processing:" in output or "🍦" in output  # From debug
        assert "completed" in output or "✅" in output  # From logging

        # Check collected data
        debug_history = self.debug.history()
        trace_stats = self.trace.stats()

        assert len(debug_history) > 0
        assert trace_stats["call_stats"]["total_calls"] > 0
        assert trace_stats["variable_changes"] > 0


class TestErrorHandlingIntegration:
    """Test error handling across dev utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        from haive.core.utils.dev import debug, log, profile, trace

        self.debug = debug
        self.log = log
        self.trace = trace
        self.profile = profile

        self.debug.clear_history()
        self.trace.clear()

    def teardown_method(self):
        """Clean up after tests."""
        self.debug.clear_history()
        self.trace.clear()

    def test_exception_handling_across_utilities(self):
        """Test that exceptions are handled properly across all utilities."""

        @self.debug.breakpoint_on_exception
        @self.trace.calls
        @self.profile.time
        def error_function(should_fail=False):
            self.debug.ice("Function called", should_fail=should_fail)
            self.trace.vars(should_fail=should_fail)

            if should_fail:
                raise ValueError("Intentional test error")

            return "success"

        # Test successful case
        with redirect_stdout(io.StringIO()):
            result = error_function(False)

        assert result == "success"

        # Test error case - should not interfere with dev utilities
        with pytest.raises(ValueError, match="Intentional test error"):
            with redirect_stdout(io.StringIO()):
                error_function(True)

        # Verify that all utilities recorded the calls
        trace_stats = self.trace.stats()
        debug_history = self.debug.history()

        assert trace_stats["call_stats"]["total_calls"] == 2
        assert len(debug_history) > 0

        # Check that error was recorded in tracing
        failed_calls = [
            call
            for call in self.trace.call_tracker.calls
            if call.get("status") == "error"
        ]
        assert len(failed_calls) == 1

    def test_utility_resilience(self):
        """Test that dev utilities are resilient to various inputs."""
        # Test with None values
        with redirect_stdout(io.StringIO()):
            self.debug.ice(None)
            self.log.info("Test with None", value=None)
            self.trace.vars(none_var=None)

        # Test with complex objects
        class TestObject:
            def __init__(self, value):
                self.value = value

            def __repr__(self):
                return f"TestObject({self.value})"

        test_obj = TestObject("test")

        with redirect_stdout(io.StringIO()):
            self.debug.ice("Complex object", test_obj)
            self.log.json({"object": test_obj}, title="Object Test")
            self.trace.vars(complex_obj=test_obj)

        # All should complete without errors
        assert len(self.debug.history()) > 0
        assert len(self.trace.var_tracker.changes) > 0


class TestPerformanceIntegration:
    """Test performance aspects of dev utilities integration."""

    def test_minimal_overhead_when_disabled(self):
        """Test that disabled utilities have minimal overhead."""
        from haive.core.utils.dev import debug, trace

        # Disable utilities
        debug.disable()
        trace.call_tracker.disable()
        trace.var_tracker.disable()

        def test_function():
            debug.ice("This should not be processed")
            trace.vars(test_var="not_tracked")
            return "result"

        # Time the function execution
        start_time = time.perf_counter()
        for _ in range(100):
            result = test_function()
        end_time = time.perf_counter()

        assert result == "result"

        # Should have minimal overhead
        total_time = end_time - start_time
        assert total_time < 0.1  # Should be very fast when disabled

        # Verify nothing was tracked
        assert len(debug.debug_history) == 0
        assert len(trace.call_tracker.calls) == 0
        assert len(trace.var_tracker.changes) == 0

        # Re-enable for cleanup
        debug.enable()

    def test_reasonable_overhead_when_enabled(self):
        """Test that enabled utilities have reasonable overhead."""
        from haive.core.utils.dev import debug, trace

        debug.clear_history()
        trace.clear()

        @trace.calls
        def test_function():
            debug.ice("Processing")
            trace.vars(step=1)
            return "result"

        # Time the function execution
        start_time = time.perf_counter()
        for _ in range(10):  # Fewer iterations when enabled
            result = test_function()
        end_time = time.perf_counter()

        assert result == "result"

        # Should still be reasonably fast
        total_time = end_time - start_time
        assert total_time < 1.0  # Should complete within 1 second

        # Verify data was collected
        assert len(debug.debug_history) > 0
        assert len(trace.call_tracker.calls) > 0
        assert len(trace.var_tracker.changes) > 0


if __name__ == "__main__":
    pytest.main([__file__])
