"""
Tests for the debugging utilities module.

Tests both functionality when dependencies are available and fallback behavior
when dependencies are missing.
"""

import io
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from haive.core.utils.dev.debugging import DebugUtilities, debug


class TestDebugUtilities:
    """Test the DebugUtilities class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.debug_utils = DebugUtilities()
        self.debug_utils.clear_history()
        self.debug_utils.enable()

    def teardown_method(self):
        """Clean up after tests."""
        self.debug_utils.clear_history()

    def test_initialization(self):
        """Test that DebugUtilities initializes correctly."""
        assert self.debug_utils.debug_enabled is True
        assert len(self.debug_utils.debug_history) == 0

    def test_enable_disable(self):
        """Test enabling and disabling debug utilities."""
        self.debug_utils.disable()
        assert self.debug_utils.debug_enabled is False

        self.debug_utils.enable()
        assert self.debug_utils.debug_enabled is True

    def test_ice_with_args(self):
        """Test icecream-style debugging with arguments."""
        with redirect_stdout(io.StringIO()) as f:
            self.debug_utils.ice("test_value", 42, {"key": "value"})

        output = f.getvalue()
        assert "test_value" in output or "🍦" in output

        # Check history
        assert len(self.debug_utils.debug_history) == 1
        history_entry = self.debug_utils.debug_history[0]
        assert history_entry["type"] == "icecream"
        assert "args" in history_entry

    def test_ice_without_args(self):
        """Test icecream-style debugging without arguments (location only)."""
        with redirect_stdout(io.StringIO()) as f:
            self.debug_utils.ice()

        output = f.getvalue()
        assert output  # Should print something (file:line info)

        # Check history
        assert len(self.debug_utils.debug_history) == 1

    def test_ice_disabled(self):
        """Test that ice does nothing when debugging is disabled."""
        self.debug_utils.disable()

        with redirect_stdout(io.StringIO()) as f:
            self.debug_utils.ice("should_not_print")

        output = f.getvalue()
        assert output == ""  # Should print nothing
        assert len(self.debug_utils.debug_history) == 0

    @patch("haive.core.utils.dev.debugging.pdb.set_trace")
    def test_pdb_called(self, mock_set_trace):
        """Test that pdb is called correctly."""
        with redirect_stdout(io.StringIO()):
            self.debug_utils.pdb()

        mock_set_trace.assert_called_once()

        # Check history
        assert len(self.debug_utils.debug_history) == 1
        history_entry = self.debug_utils.debug_history[0]
        assert history_entry["type"] == "pdb"

    @patch("haive.core.utils.dev.debugging.pdb.set_trace")
    def test_pdb_with_condition_true(self, mock_set_trace):
        """Test pdb with condition that evaluates to True."""
        with redirect_stdout(io.StringIO()):
            self.debug_utils.pdb(condition=True)

        mock_set_trace.assert_called_once()

    @patch("haive.core.utils.dev.debugging.pdb.set_trace")
    def test_pdb_with_condition_false(self, mock_set_trace):
        """Test pdb with condition that evaluates to False."""
        with redirect_stdout(io.StringIO()):
            self.debug_utils.pdb(condition=False)

        mock_set_trace.assert_not_called()
        assert len(self.debug_utils.debug_history) == 0

    @patch("haive.core.utils.dev.debugging.HAS_WEB_PDB", True)
    @patch("haive.core.utils.dev.debugging.web_pdb")
    def test_web_debugger_available(self, mock_web_pdb):
        """Test web debugger when web-pdb is available."""
        with redirect_stdout(io.StringIO()) as f:
            self.debug_utils.web(port=5556)

        mock_web_pdb.set_trace.assert_called_once_with(port=5556)

        output = f.getvalue()
        assert "5556" in output

        # Check history
        assert len(self.debug_utils.debug_history) == 1
        history_entry = self.debug_utils.debug_history[0]
        assert history_entry["type"] == "web_pdb"
        assert history_entry["port"] == 5556

    @patch("haive.core.utils.dev.debugging.HAS_WEB_PDB", False)
    @patch("haive.core.utils.dev.debugging.pdb.set_trace")
    def test_web_debugger_fallback(self, mock_set_trace):
        """Test web debugger fallback when web-pdb is not available."""
        with redirect_stdout(io.StringIO()) as f:
            self.debug_utils.web()

        output = f.getvalue()
        assert "web-pdb not available" in output
        mock_set_trace.assert_called_once()

    @patch("haive.core.utils.dev.debugging.HAS_PUDB", True)
    @patch("haive.core.utils.dev.debugging.pudb")
    def test_visual_debugger_available(self, mock_pudb):
        """Test visual debugger when pudb is available."""
        with redirect_stdout(io.StringIO()) as f:
            self.debug_utils.visual()

        mock_pudb.set_trace.assert_called_once()

        output = f.getvalue()
        assert "Visual debugger started" in output

        # Check history
        assert len(self.debug_utils.debug_history) == 1
        history_entry = self.debug_utils.debug_history[0]
        assert history_entry["type"] == "pudb"

    @patch("haive.core.utils.dev.debugging.HAS_PUDB", False)
    @patch("haive.core.utils.dev.debugging.pdb.set_trace")
    def test_visual_debugger_fallback(self, mock_set_trace):
        """Test visual debugger fallback when pudb is not available."""
        with redirect_stdout(io.StringIO()) as f:
            self.debug_utils.visual()

        output = f.getvalue()
        assert "pudb not available" in output
        mock_set_trace.assert_called_once()

    def test_breakpoint_on_exception_decorator_success(self):
        """Test breakpoint_on_exception decorator with successful function."""

        @self.debug_utils.breakpoint_on_exception
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"
        assert len(self.debug_utils.debug_history) == 0

    @patch("haive.core.utils.dev.debugging.pdb.set_trace")
    def test_breakpoint_on_exception_decorator_with_exception(self, mock_set_trace):
        """Test breakpoint_on_exception decorator with function that raises exception."""

        @self.debug_utils.breakpoint_on_exception
        def test_func():
            raise ValueError("Test exception")

        with pytest.raises(ValueError, match="Test exception"):
            with redirect_stdout(io.StringIO()):
                test_func()

        mock_set_trace.assert_called_once()

    @patch("haive.core.utils.dev.debugging.HAS_BIRDSEYE", True)
    @patch("haive.core.utils.dev.debugging.birdseye")
    def test_trace_calls_with_birdseye(self, mock_birdseye):
        """Test trace_calls decorator when birdseye is available."""
        mock_birdseye.eye.return_value = lambda x: x

        @self.debug_utils.trace_calls
        def test_func():
            return "traced"

        result = test_func()
        assert result == "traced"
        mock_birdseye.eye.assert_called_once()

    @patch("haive.core.utils.dev.debugging.HAS_BIRDSEYE", False)
    def test_trace_calls_fallback(self):
        """Test trace_calls decorator fallback when birdseye is not available."""

        @self.debug_utils.trace_calls
        def test_func(arg1, arg2=None):
            return f"{arg1}_{arg2}"

        with redirect_stdout(io.StringIO()) as f:
            result = test_func("hello", arg2="world")

        assert result == "hello_world"
        output = f.getvalue()
        # Should contain call and return info
        assert "test_func" in output

    def test_stack_trace(self):
        """Test stack trace functionality."""
        with redirect_stdout(io.StringIO()) as f:
            result = self.debug_utils.stack_trace(limit=5)

        assert isinstance(result, str)
        assert "Call Stack" in result
        output = f.getvalue()
        assert "Call Stack" in output

    def test_locals_inspect(self):
        """Test local variables inspection."""

        with redirect_stdout(io.StringIO()) as f:
            result = self.debug_utils.locals_inspect()

        assert isinstance(result, dict)
        # Should contain our local variables
        assert "test_var" in result
        assert "another_var" in result
        assert result["test_var"] == "test_value"
        assert result["another_var"] == 42

        output = f.getvalue()
        assert "Local Variables" in output
        assert "test_var" in output

    def test_globals_inspect(self):
        """Test global variables inspection."""
        with redirect_stdout(io.StringIO()) as f:
            result = self.debug_utils.globals_inspect()

        assert isinstance(result, dict)
        # Should not contain private variables or modules
        for key in result.keys():
            assert not key.startswith("_")

        output = f.getvalue()
        assert "Global Variables" in output

    def test_history_management(self):
        """Test debug history tracking and management."""
        # Generate some history
        with redirect_stdout(io.StringIO()):
            self.debug_utils.ice("test1")
            self.debug_utils.ice("test2")
            self.debug_utils.ice("test3")

        # Test history retrieval
        history = self.debug_utils.history(limit=2)
        assert len(history) == 2

        # Test full history
        full_history = self.debug_utils.history()
        assert len(full_history) == 3

        # Test clear history
        self.debug_utils.clear_history()
        assert len(self.debug_utils.debug_history) == 0

    def test_status_reporting(self):
        """Test status reporting functionality."""
        with redirect_stdout(io.StringIO()) as f:
            status = self.debug_utils.status()

        assert isinstance(status, dict)
        assert "debug_enabled" in status
        assert "tools_available" in status
        assert "debug_history_count" in status

        assert status["debug_enabled"] is True
        assert isinstance(status["tools_available"], dict)

        output = f.getvalue()
        assert "Debug Tools Status" in output


class TestGlobalDebugInstance:
    """Test the global debug instance."""

    def setup_method(self):
        """Set up test fixtures."""
        debug.clear_history()
        debug.enable()

    def teardown_method(self):
        """Clean up after tests."""
        debug.clear_history()

    def test_global_debug_instance(self):
        """Test that the global debug instance works correctly."""
        assert hasattr(debug, "ice")
        assert hasattr(debug, "pdb")
        assert hasattr(debug, "web")
        assert hasattr(debug, "visual")

        # Test basic functionality
        with redirect_stdout(io.StringIO()) as f:
            debug.ice("global_test")

        output = f.getvalue()
        assert "global_test" in output or "🍦" in output

        assert len(debug.debug_history) == 1

    def test_global_debug_decorators(self):
        """Test global debug decorators."""

        @debug.trace_calls
        def test_func():
            return "decorated"

        result = test_func()
        assert result == "decorated"

        @debug.breakpoint_on_exception
        def safe_func():
            return "safe"

        result = safe_func()
        assert result == "safe"


if __name__ == "__main__":
    pytest.main([__file__])
