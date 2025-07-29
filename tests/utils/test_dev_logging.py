"""
Tests for the logging utilities module.

Tests DevLogger functionality including Rich integration, context managers,
timing utilities, and various log levels.
"""

import io
import logging
import time
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from haive.core.utils.dev.logging import DevLogger, log


class TestDevLogger:
    """Test the DevLogger class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = DevLogger(name="test.logger")
        self.logger.context_stack.clear()

    def teardown_method(self):
        """Clean up after tests."""
        self.logger.context_stack.clear()

    def test_initialization(self):
        """Test that DevLogger initializes correctly."""
        assert self.logger.name == "test.logger"
        assert len(self.logger.context_stack) == 0
        assert self.logger.logger is not None

    def test_debug_logging(self):
        """Test debug level logging."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.debug("Test debug message", key="value")

        output = f.getvalue()
        # Should contain debug info and emoji/styling
        assert "debug message" in output.lower() or "🐛" in output

    def test_info_logging(self):
        """Test info level logging."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.info("Test info message", count=42)

        output = f.getvalue()
        assert "info message" in output.lower() or "ℹ️" in output

    def test_warning_logging(self):
        """Test warning level logging."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.warning("Test warning message", level="high")

        output = f.getvalue()
        assert "warning message" in output.lower() or "⚠️" in output

    def test_error_logging(self):
        """Test error level logging."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.error("Test error message", code=500)

        output = f.getvalue()
        assert "error message" in output.lower() or "❌" in output

    def test_error_logging_with_exception(self):
        """Test error logging with exception."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            with redirect_stdout(io.StringIO()) as f:
                self.logger.error("Error occurred", exception=e)

            output = f.getvalue()
            assert "error occurred" in output.lower() or "❌" in output

    def test_critical_logging(self):
        """Test critical level logging."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.critical("Test critical message")

        output = f.getvalue()
        assert "critical message" in output.lower() or "🚨" in output

    def test_success_logging(self):
        """Test success level logging."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.success("Operation completed successfully")

        output = f.getvalue()
        assert "success" in output.lower() or "✅" in output

    def test_progress_logging(self):
        """Test progress level logging."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.progress("Processing data", step=5, total=10)

        output = f.getvalue()
        assert "progress" in output.lower() or "🔄" in output

    def test_context_manager(self):
        """Test logging context manager."""
        with redirect_stdout(io.StringIO()) as f:
            with self.logger.context("test_operation", operation_id=123):
                self.logger.info("Inside context")

        output = f.getvalue()
        assert "test_operation" in output
        assert "Inside context" in output or "ℹ️" in output

        # Context should be cleared after exiting
        assert len(self.logger.context_stack) == 0

    def test_nested_context(self):
        """Test nested context managers."""
        with redirect_stdout(io.StringIO()) as f:
            with self.logger.context("outer_context"):
                with self.logger.context("inner_context"):
                    self.logger.info("Nested message")

        output = f.getvalue()
        assert "outer_context" in output
        assert "inner_context" in output

    def test_context_with_exception(self):
        """Test context manager behavior with exceptions."""
        with pytest.raises(ValueError):
            with redirect_stdout(io.StringIO()) as f:
                with self.logger.context("error_context"):
                    raise ValueError("Test error in context")

        output = f.getvalue()
        assert "error_context" in output
        assert len(self.logger.context_stack) == 0  # Should be cleaned up

    def test_table_logging_with_data(self):
        """Test table logging with data."""
        test_data = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "London"},
            {"name": "Charlie", "age": 35, "city": "Tokyo"},
        ]

        with redirect_stdout(io.StringIO()) as f:
            self.logger.table(test_data, title="User Data")

        output = f.getvalue()
        # Should contain table data or fall back to text
        assert "Alice" in output
        assert "Bob" in output
        assert "Charlie" in output

    def test_table_logging_empty(self):
        """Test table logging with empty data."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.table([], title="Empty Table")

        output = f.getvalue()
        assert "empty" in output.lower()

    def test_json_logging(self):
        """Test JSON data logging."""
        test_data = {
            "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "total": 2,
            "success": True,
        }

        with redirect_stdout(io.StringIO()) as f:
            self.logger.json(test_data, title="API Response")

        output = f.getvalue()
        assert "Alice" in output
        assert "Bob" in output
        # Should contain JSON formatting or readable text

    def test_json_logging_serialization_error(self):
        """Test JSON logging with non-serializable data."""

        class NonSerializable:
            def __init__(self):
                pass

        test_data = {"object": NonSerializable()}

        with redirect_stdout(io.StringIO()) as f:
            self.logger.json(test_data)

        output = f.getvalue()
        # Should handle serialization gracefully
        assert output  # Should produce some output

    def test_panel_logging(self):
        """Test panel message logging."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.panel("Important message", title="Alert", style="red")

        output = f.getvalue()
        assert "Important message" in output
        assert "Alert" in output

    def test_divider_logging(self):
        """Test divider logging."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.divider("Section Break")

        output = f.getvalue()
        assert "Section Break" in output
        # Should contain some form of divider
        assert "=" in output or "─" in output or output.strip()

    def test_metrics_logging(self):
        """Test metrics logging."""
        metrics = {
            "requests_per_second": 1250.5,
            "error_rate": 0.02,
            "response_time_ms": 45.8,
            "active_users": 1024,
        }

        with redirect_stdout(io.StringIO()) as f:
            self.logger.metrics(metrics, title="Performance Metrics")

        output = f.getvalue()
        assert "1250.5" in output
        assert "0.02" in output
        assert "Performance Metrics" in output

    def test_timer_functionality(self):
        """Test timer start/end functionality."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.timer_start("test_operation")
            time.sleep(0.01)  # Small delay
            duration = self.logger.timer_end("test_operation")

        output = f.getvalue()
        assert "test_operation" in output
        assert "Timer started" in output
        assert "completed" in output
        assert isinstance(duration, float)
        assert duration > 0

    def test_timer_context_manager(self):
        """Test timer context manager."""
        with redirect_stdout(io.StringIO()) as f:
            with self.logger.timer("context_timer"):
                time.sleep(0.01)  # Small delay

        output = f.getvalue()
        assert "context_timer" in output
        assert "Timer started" in output
        assert "completed" in output

    def test_timer_not_found(self):
        """Test timer end with non-existent timer."""
        with redirect_stdout(io.StringIO()) as f:
            duration = self.logger.timer_end("nonexistent_timer")

        output = f.getvalue()
        assert "not found" in output.lower()
        assert duration == 0.0

    def test_log_level_setting(self):
        """Test setting log levels."""
        with redirect_stdout(io.StringIO()) as f:
            self.logger.set_level("WARNING")

        output = f.getvalue()
        assert "WARNING" in output

        # Test with integer level
        with redirect_stdout(io.StringIO()) as f:
            self.logger.set_level(logging.DEBUG)

        output = f.getvalue()
        assert "DEBUG" in output

    def test_message_formatting_with_context(self):
        """Test message formatting with context stack."""
        with self.logger.context("operation1"):
            with self.logger.context("operation2"):
                formatted = self.logger._format_message("test message", key="value")

        assert "test message" in formatted
        assert "operation1" in formatted
        assert "operation2" in formatted
        assert "key=value" in formatted

    def test_message_formatting_with_kwargs(self):
        """Test message formatting with various kwargs."""
        formatted = self.logger._format_message(
            "test message",
            string_val="hello",
            int_val=42,
            dict_val={"nested": "data"},
            list_val=[1, 2, 3],
        )

        assert "test message" in formatted
        assert "string_val=hello" in formatted
        assert "int_val=42" in formatted
        # JSON serialized complex types
        assert "nested" in formatted

    def test_caller_info_capture(self):
        """Test that caller information is captured correctly."""

        def test_function():
            return self.logger._get_caller_info()

        caller_info = test_function()

        assert isinstance(caller_info, dict)
        if caller_info:  # May be empty in some test environments
            assert "file" in caller_info
            assert "line" in caller_info
            assert "function" in caller_info


class TestGlobalLogInstance:
    """Test the global log instance."""

    def setup_method(self):
        """Set up test fixtures."""
        log.context_stack.clear()

    def teardown_method(self):
        """Clean up after tests."""
        log.context_stack.clear()

    def test_global_log_instance(self):
        """Test that the global log instance works correctly."""
        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
        assert hasattr(log, "success")

        # Test basic functionality
        with redirect_stdout(io.StringIO()) as f:
            log.info("Global log test")

        output = f.getvalue()
        assert "Global log test" in output or "ℹ️" in output

    def test_global_log_context(self):
        """Test global log context manager."""
        with redirect_stdout(io.StringIO()) as f:
            with log.context("global_context"):
                log.info("Context test")

        output = f.getvalue()
        assert "global_context" in output
        assert "Context test" in output or "ℹ️" in output

    def test_global_log_utilities(self):
        """Test global log utility methods."""
        # Test table
        test_data = [{"id": 1, "name": "test"}]
        with redirect_stdout(io.StringIO()) as f:
            log.table(test_data)

        output = f.getvalue()
        assert "test" in output

        # Test metrics
        metrics = {"test_metric": 123}
        with redirect_stdout(io.StringIO()) as f:
            log.metrics(metrics)

        output = f.getvalue()
        assert "123" in output


@patch("haive.core.utils.dev.logging.HAS_RICH", False)
class TestFallbackBehavior:
    """Test logging behavior when Rich is not available."""

    def test_fallback_logging(self):
        """Test that logging works without Rich."""
        logger = DevLogger(name="fallback.test")

        with redirect_stdout(io.StringIO()) as f:
            logger.info("Fallback test message")

        output = f.getvalue()
        assert "Fallback test message" in output

    def test_fallback_table(self):
        """Test table fallback without Rich."""
        logger = DevLogger(name="fallback.test")
        test_data = [{"name": "Alice", "age": 30}]

        with redirect_stdout(io.StringIO()) as f:
            logger.table(test_data)

        output = f.getvalue()
        assert "Alice" in output
        assert "30" in output

    def test_fallback_divider(self):
        """Test divider fallback without Rich."""
        logger = DevLogger(name="fallback.test")

        with redirect_stdout(io.StringIO()) as f:
            logger.divider("Test Section")

        output = f.getvalue()
        assert "Test Section" in output
        assert "=" in output  # Should use text-based divider


if __name__ == "__main__":
    pytest.main([__file__])
