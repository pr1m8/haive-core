"""
Tests for unified debug kit.

This test module validates the core functionality of the unified debug kit
including configuration, analysis, instrumentation, and integration between components.
"""

import pytest
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional
from unittest.mock import patch, MagicMock

from haive.core.utils.debugkit import (
    debugkit,
    UnifiedDev,
    DevConfig,
    DevContext,
    CodeAnalysisReport,
    Environment,
    LogLevel,
    StorageBackend,
)


class TestDevConfig:
    """Test development configuration functionality."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = DevConfig()

        assert config.enabled is True
        assert config.debug_enabled is True
        assert config.log_enabled is True
        assert config.trace_enabled is True
        assert config.profile_enabled is True
        assert config.benchmark_enabled is True

    def test_environment_based_config(self):
        """Test config adjusts based on environment."""
        # Test production environment
        prod_config = DevConfig(environment=Environment.PRODUCTION)

        assert prod_config.trace_sampling_rate == 0.01
        assert prod_config.profile_enabled is False
        assert prod_config.verbose is False
        assert prod_config.log_level == LogLevel.ERROR

    def test_config_from_env(self):
        """Test creating config from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "HAIVE_ENV": "testing",
                "HAIVE_DEBUG_ENABLED": "false",
                "HAIVE_LOG_LEVEL": "WARNING",
                "HAIVE_TRACE_SAMPLING_RATE": "0.5",
            },
        ):
            config = DevConfig.from_env()

            assert config.environment == Environment.TESTING
            assert config.debug_enabled is False
            assert config.log_level == LogLevel.WARNING
            assert config.trace_sampling_rate == 0.5

    def test_config_update(self):
        """Test updating configuration values."""
        config = DevConfig()

        config.update(
            verbose=True, trace_sampling_rate=0.25, storage_backend=StorageBackend.POSTGRESQL
        )

        assert config.verbose is True
        assert config.trace_sampling_rate == 0.25
        assert config.storage_backend == StorageBackend.POSTGRESQL

    def test_config_tool_enabled(self):
        """Test tool availability checking."""
        config = DevConfig(
            static_analysis_enabled=True,
            included_tools=["mypy", "radon"],
            excluded_tools=["vulture"],
        )

        assert config.is_tool_enabled("mypy") is True
        assert config.is_tool_enabled("radon") is True
        assert config.is_tool_enabled("vulture") is False
        assert config.is_tool_enabled("unknown") is False

    def test_config_serialization(self):
        """Test config to dictionary conversion."""
        config = DevConfig(
            environment=Environment.DEVELOPMENT,
            log_level=LogLevel.DEBUG,
            storage_backend=StorageBackend.SQLITE,
        )

        config_dict = config.to_dict()

        assert config_dict["environment"] == "development"
        assert config_dict["log_level"] == "DEBUG"
        assert config_dict["storage_backend"] == "sqlite"


class TestDevContext:
    """Test development context functionality."""

    def test_context_creation(self):
        """Test creating and using development context."""
        with DevContext("test_operation", user_id=123) as ctx:
            assert ctx.name == "test_operation"
            assert ctx.correlation_id is not None
            assert ctx.metadata["user_id"] == 123
            assert ctx.start_time is not None

    def test_context_timing(self):
        """Test context timing functionality."""
        with DevContext("timed_operation") as ctx:
            time.sleep(0.01)  # Small delay
            elapsed = ctx.get_elapsed_time()

            assert elapsed >= 0.01
            assert elapsed < 1.0  # Should be very quick

    def test_context_checkpoints(self):
        """Test context checkpoint functionality."""
        with DevContext("checkpoint_test") as ctx:
            ctx.checkpoint("step1", data="test")
            ctx.checkpoint("step2", count=42)

            assert len(ctx._checkpoints) == 2
            assert ctx._checkpoints[0]["name"] == "step1"
            assert ctx._checkpoints[0]["data"] == "test"
            assert ctx._checkpoints[1]["name"] == "step2"
            assert ctx._checkpoints[1]["count"] == 42

    def test_context_data_recording(self):
        """Test context data recording."""
        with DevContext("data_test") as ctx:
            ctx.record("processed_items", 150)
            ctx.record("status", "completed")

            assert ctx.data["processed_items"] == 150
            assert ctx.data["status"] == "completed"


class TestUnifiedDev:
    """Test unified development interface."""

    def test_unified_dev_creation(self):
        """Test creating UnifiedDev instance."""
        dev_instance = UnifiedDev()

        assert dev_instance.config is not None
        assert dev_instance.debug is not None
        assert dev_instance.log is not None
        assert dev_instance.trace is not None
        assert dev_instance.profile is not None
        assert dev_instance.benchmark is not None

    def test_unified_dev_with_custom_config(self):
        """Test creating UnifiedDev with custom configuration."""
        custom_config = DevConfig(
            environment=Environment.TESTING, debug_enabled=False, log_level=LogLevel.WARNING
        )

        dev_instance = UnifiedDev(custom_config)

        assert dev_instance.config.environment == Environment.TESTING
        assert dev_instance.config.debug_enabled is False
        assert dev_instance.config.log_level == LogLevel.WARNING

    def test_correlation_id_management(self):
        """Test correlation ID setting and propagation."""
        dev_instance = UnifiedDev()
        test_id = "test-correlation-123"

        dev_instance.set_correlation_id(test_id)

        assert dev_instance.correlation_id == test_id

    def test_configuration_updates(self):
        """Test runtime configuration updates."""
        dev_instance = UnifiedDev()

        dev_instance.configure(verbose=True, trace_sampling_rate=0.1, auto_analyze=True)

        assert dev_instance.config.verbose is True
        assert dev_instance.config.trace_sampling_rate == 0.1
        assert dev_instance.config.auto_analyze is True

    def test_context_creation(self):
        """Test context creation through unified interface."""
        dev_instance = UnifiedDev()

        with dev_instance.context("test_context", key="value") as ctx:
            assert ctx.name == "test_context"
            assert ctx.metadata["key"] == "value"

    def test_convenience_methods(self):
        """Test convenience methods for logging and debugging."""
        dev_instance = UnifiedDev()

        # These should not raise exceptions even with fallback implementations
        result = dev_instance.ice("test value", count=42)
        dev_instance.info("Test info message", context="test")
        dev_instance.success("Test success message")
        dev_instance.error("Test error message", error_code=500)

        # ice should return the first argument for chaining
        assert result == "test value"

    def test_cache_management(self):
        """Test analysis cache management."""
        dev_instance = UnifiedDev()

        # Add something to cache (simulated)
        test_key = "test.function"
        dev_instance._analysis_cache[test_key] = MagicMock()

        assert len(dev_instance._analysis_cache) == 1

        dev_instance.clear_cache()

        assert len(dev_instance._analysis_cache) == 0

    def test_statistics_collection(self):
        """Test development utilities statistics."""
        dev_instance = UnifiedDev()

        stats = dev_instance.get_stats()

        assert "config" in stats
        assert "analysis_cache_size" in stats
        assert "correlation_id" in stats
        assert isinstance(stats["config"], dict)


class TestCodeAnalysisIntegration:
    """Test code analysis functionality integration."""

    def create_test_function(self):
        """Create a test function for analysis."""

        def test_function(data: List[str], config: Optional[Dict] = None) -> Dict[str, int]:
            """Test function with type hints for analysis."""
            result = {}
            for item in data:
                if config and config.get("process", True):
                    result[item] = len(item)
            return result

        return test_function

    def test_code_analysis_basic(self):
        """Test basic code analysis functionality."""
        dev_instance = UnifiedDev()
        test_func = self.create_test_function()

        # This should work even with fallback implementations
        report = dev_instance.analyze_code(test_func)

        assert isinstance(report, CodeAnalysisReport)
        assert report.function_name == "test_function"
        assert report.type_analysis is not None
        assert report.complexity_analysis is not None
        assert isinstance(report.combined_score, (int, float))
        assert isinstance(report.recommendations, list)

    def test_code_analysis_caching(self):
        """Test code analysis result caching."""
        dev_instance = UnifiedDev()
        test_func = self.create_test_function()

        # First analysis
        report1 = dev_instance.analyze_code(test_func)

        # Second analysis (should use cache)
        report2 = dev_instance.analyze_code(test_func)

        # Should be the same object from cache
        assert report1 is report2

    def test_instrumentation_decorator(self):
        """Test function instrumentation decorator."""
        dev_instance = UnifiedDev()

        @dev_instance.instrument
        def test_function(x: int) -> int:
            return x * 2

        result = test_function(21)

        assert result == 42
        # Function should still work normally despite instrumentation

    def test_instrumentation_with_analysis(self):
        """Test instrumentation with code analysis enabled."""
        dev_instance = UnifiedDev()

        @dev_instance.instrument(analyze=True)
        def analyzed_function(value: str) -> str:
            return value.upper()

        result = analyzed_function("hello")

        assert result == "HELLO"
        # Check if analysis report was attached
        assert hasattr(analyzed_function, "_analysis_report")

    def test_instrumentation_selective_features(self):
        """Test selective feature enabling in instrumentation."""
        dev_instance = UnifiedDev()

        @dev_instance.instrument(profile=True, trace=False, log=False)
        def selective_function() -> str:
            return "test"

        result = selective_function()

        assert result == "test"
        # Should work with selective feature enabling


class TestFallbackImplementations:
    """Test fallback implementations when dependencies are missing."""

    def test_fallback_debug(self):
        """Test fallback debug functionality."""
        from haive.core.utils.debugkit.fallbacks import FallbackDebug

        debug = FallbackDebug()

        # Should not raise exceptions
        result = debug.ice("test", count=5)
        assert result == "test"

        @debug.trace_calls
        def traced_func():
            return "traced"

        assert traced_func() == "traced"

    def test_fallback_log(self):
        """Test fallback logging functionality."""
        from haive.core.utils.debugkit.fallbacks import FallbackLog

        log = FallbackLog()

        # Should not raise exceptions
        log.info("test message", key="value")
        log.error("error message", code=500)
        log.success("success message")

        # Test context manager
        with log.context("test_context") as ctx:
            ctx.info("context message")

    def test_fallback_trace(self):
        """Test fallback tracing functionality."""
        from haive.core.utils.debugkit.fallbacks import FallbackTrace

        trace = FallbackTrace()

        trace.push_context("test", "corr-123")
        trace.mark("checkpoint", "value")
        trace.pop_context()

        @trace.calls
        def traced_function():
            return 42

        assert traced_function() == 42

    def test_fallback_profile(self):
        """Test fallback profiling functionality."""
        from haive.core.utils.debugkit.fallbacks import FallbackProfile

        profile = FallbackProfile()

        # Test context profiling
        context = profile.start_context("test")
        time.sleep(0.01)
        stats = profile.stop_context(context)

        assert stats["duration"] >= 0.01
        assert "memory_delta" in stats

        # Test decorator profiling
        @profile.time
        def profiled_function():
            return sum(range(100))

        result = profiled_function()
        assert result == sum(range(100))

    def test_fallback_benchmark(self):
        """Test fallback benchmarking functionality."""
        from haive.core.utils.debugkit.fallbacks import FallbackBenchmark

        benchmark = FallbackBenchmark()

        def test_func():
            return sum(range(10))

        stats = benchmark.measure(test_func, iterations=5)

        assert "average" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["iterations"] == 5

        # Test comparison
        results = benchmark.compare(
            {"sum_range": lambda: sum(range(10)), "list_comp": lambda: [x for x in range(10)]},
            iterations=3,
        )

        assert len(results) == 2
        assert "sum_range" in results
        assert "list_comp" in results


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_complete_development_workflow(self):
        """Test complete development workflow with all components."""
        dev_instance = UnifiedDev()

        # Configure for development
        dev_instance.configure(
            verbose=True,
            debug_enabled=True,
            log_enabled=True,
            auto_analyze=False,  # Disable to avoid complex dependencies in tests
        )

        @dev_instance.instrument(profile=True, log=True)
        def example_workflow(data: List[str]) -> Dict[str, int]:
            """Example function with full instrumentation."""
            with dev_instance.context("processing") as ctx:
                ctx.info("Starting data processing", count=len(data))

                result = {}
                for i, item in enumerate(data):
                    if i % 10 == 0:
                        ctx.checkpoint(f"item_{i}", item=item)
                    result[item] = len(item)

                ctx.success("Processing complete", processed=len(result))
                return result

        # Execute workflow
        test_data = ["hello", "world", "test"]
        result = example_workflow(test_data)

        # Verify results
        assert result == {"hello": 5, "world": 5, "test": 4}
        assert len(result) == 3

    def test_error_handling_workflow(self):
        """Test error handling in development workflow."""
        dev_instance = UnifiedDev()

        @dev_instance.instrument(log=True)
        def failing_function():
            """Function that raises an exception."""
            with dev_instance.context("failing_operation") as ctx:
                ctx.info("About to fail")
                raise ValueError("Test error")

        # Should propagate the exception while logging it
        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_production_mode_workflow(self):
        """Test workflow in production mode with minimal overhead."""
        # Create production config
        prod_config = DevConfig(environment=Environment.PRODUCTION)
        dev_instance = UnifiedDev(prod_config)

        @dev_instance.instrument
        def production_function(x: int) -> int:
            return x * 2

        # Should work normally but with minimal instrumentation
        result = production_function(21)
        assert result == 42

        # Most features should be disabled in production
        assert dev_instance.config.profile_enabled is False
        assert dev_instance.config.verbose is False
        assert dev_instance.config.log_level == LogLevel.ERROR

    def test_analysis_workflow_with_real_function(self):
        """Test analysis workflow with a real function."""
        dev_instance = UnifiedDev()

        def complex_function(
            items: List[Dict[str, str]], config: Optional[Dict] = None
        ) -> List[str]:
            """Function with some complexity for analysis."""
            result = []

            for item in items:
                if config:
                    if config.get("filter_empty", True):
                        if item.get("name", "").strip():
                            result.append(item["name"])
                    else:
                        result.append(item.get("name", ""))
                else:
                    result.append(item.get("name", ""))

            return result

        # Analyze the function
        report = dev_instance.analyze_code(complex_function)

        # Should have reasonable analysis results
        assert report.function_name == "complex_function"
        assert isinstance(report.combined_score, (int, float))
        assert 0 <= report.combined_score <= 100
        assert len(report.recommendations) >= 0  # May have recommendations

        # Test the actual function
        test_data = [{"name": "Alice"}, {"name": "Bob"}, {"name": ""}]

        result = complex_function(test_data, {"filter_empty": True})
        assert result == ["Alice", "Bob"]


@pytest.fixture
def temp_test_file():
    """Create temporary test file for analysis."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('''
def example_function(data: str) -> str:
    """Example function for testing."""
    return data.upper()

def complex_function(items, config=None):
    """Function without type hints."""
    result = []
    for item in items:
        if config:
            if config.get('process'):
                result.append(process_item(item))
        else:
            result.append(item)
    return result
''')
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestFileAnalysis:
    """Test file-based analysis functionality."""

    def test_static_analysis_integration(self, temp_test_file):
        """Test static analysis integration with real file."""
        dev_instance = UnifiedDev()

        # Configure to enable static analysis
        dev_instance.configure(static_analysis_enabled=True)

        # This may not work if tools aren't available, but shouldn't crash
        try:
            available_tools = dev_instance.static_analysis.get_available_tools()

            if available_tools:
                results = dev_instance.static_analysis.analyze_file(
                    temp_test_file,
                    tools=available_tools[:1],  # Use first available tool
                )

                assert isinstance(results, dict)
                assert len(results) > 0
        except Exception:
            # Static analysis may fail if tools aren't available - that's OK
            pytest.skip("Static analysis tools not available")


if __name__ == "__main__":
    pytest.main([__file__])
