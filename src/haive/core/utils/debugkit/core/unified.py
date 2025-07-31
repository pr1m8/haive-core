"""Unified development interface components.

This module contains the main UnifiedDev class and CodeAnalysisReport
that were previously in the main __init__.py file.
"""

import asyncio
import functools
import inspect
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from haive.core.utils.debugkit.analysis import (
    get_complexity_analyzer,
    get_static_orchestrator,
    get_type_analyzer,
)
from haive.core.utils.debugkit.config import config
from haive.core.utils.debugkit.core.context import DevContext

if TYPE_CHECKING:
    from haive.core.utils.debugkit.analysis.complexity import (
        ComplexityAnalyzer,
        ComplexityReport,
    )
    from haive.core.utils.debugkit.analysis.static import (
        AnalysisResult,
        StaticAnalysisOrchestrator,
    )
    from haive.core.utils.debugkit.analysis.types import (
        FunctionTypeAnalysis,
        TypeAnalyzer,
    )


class CodeAnalysisReport:
    """Complete code analysis report combining multiple analysis types.

    This class provides a unified view of code analysis results including
    type analysis, complexity analysis, and static analysis findings.

    Attributes:
        function_name: Name of the analyzed function
        type_analysis: Type analysis results
        complexity_analysis: Complexity analysis results
        static_analysis: Static analysis results (if available)
        combined_score: Overall code quality score
        recommendations: Combined recommendations from all analyses
        analysis_timestamp: When the analysis was performed

    Examples:
        Review comprehensive analysis::

            report = debugkit.analyze_code(my_function)

            print(f"Function: {report.function_name}")
            print(f"Overall score: {report.combined_score:.1f}/100")
            print(f"Type coverage: {report.type_analysis.type_coverage:.1%}")
            print(f"Complexity grade: {report.complexity_analysis.complexity_grade}")

            if report.combined_score < 70:
                print("Recommendations:")
                for rec in report.recommendations:
                    print(f"  - {rec}")
    """

    def __init__(
        self,
        function_name: str,
        type_analysis: "FunctionTypeAnalysis",
        complexity_analysis: "ComplexityReport",
        static_analysis: Optional[Dict[str, "AnalysisResult"]] = None,
    ):
        """Initialize code analysis report.

        Args:
            function_name: Name of the analyzed function
            type_analysis: Type analysis results
            complexity_analysis: Complexity analysis results
            static_analysis: Static analysis results
        """
        self.function_name = function_name
        self.type_analysis = type_analysis
        self.complexity_analysis = complexity_analysis
        self.static_analysis = static_analysis or {}
        self.analysis_timestamp = time.time()

        # Calculate combined metrics
        self.combined_score = self._calculate_combined_score()
        self.recommendations = self._generate_combined_recommendations()

    def _calculate_combined_score(self) -> float:
        """Calculate overall code quality score.

        Returns:
            float: Combined score from 0-100
        """
        # Type analysis contribution (0-40 points)
        type_score = self.type_analysis.type_safety_score * 0.4

        # Complexity analysis contribution (0-40 points)
        complexity_score = (100 - self.complexity_analysis.risk_score) * 0.4

        # Static analysis contribution (0-20 points)
        static_score = 20.0
        if self.static_analysis:
            # Deduct points for findings
            total_findings = sum(
                len(result.findings) for result in self.static_analysis.values()
            )
            static_score = max(0, 20 - (total_findings * 2))

        return type_score + complexity_score + static_score

    def _generate_combined_recommendations(self) -> List[str]:
        """Generate combined recommendations from all analyses.

        Returns:
            List[str]: Combined recommendations
        """
        recommendations = []

        # Add type analysis recommendations
        recommendations.extend(self.type_analysis.recommendations)

        # Add complexity analysis recommendations
        recommendations.extend(self.complexity_analysis.refactoring_suggestions)

        # Add static analysis recommendations
        for result in self.static_analysis.values():
            recommendations.extend(result.suggestions)

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the analysis report.

        Returns:
            Dict[str, Any]: Summary statistics
        """
        return {
            "function_name": self.function_name,
            "combined_score": self.combined_score,
            "type_coverage": self.type_analysis.type_coverage,
            "type_safety_score": self.type_analysis.type_safety_score,
            "complexity_grade": self.complexity_analysis.complexity_grade.value,
            "complexity_risk_score": self.complexity_analysis.risk_score,
            "static_analysis_tools": len(self.static_analysis),
            "total_static_findings": sum(
                len(r.findings) for r in self.static_analysis.values()
            ),
            "recommendation_count": len(self.recommendations),
            "needs_attention": self.combined_score < 70,
        }


class UnifiedDev:
    """Unified development utilities interface.

    This class provides a single entry point for all development utilities
    including debugging, logging, tracing, profiling, and code analysis.
    It manages configuration, coordinates between different tools, and
    provides both simple and advanced interfaces for development tasks.

    Attributes:
        config: Development configuration
        debug: Debug utilities interface
        log: Logging utilities interface
        trace: Tracing utilities interface
        profile: Profiling utilities interface
        benchmark: Benchmarking utilities interface

    Examples:
        Basic usage::

            debugkit_instance = UnifiedDev()

            # Quick debugging
            debugkit_instance.ice("Debug info", variable=value)
            debugkit_instance.info("Process started")

            # Context management
            with debugkit_instance.context("operation") as ctx:
                ctx.debug("Step 1")
                ctx.checkpoint("validation")
                ctx.success("Complete")

        Advanced analysis::

            # Comprehensive code analysis
            @debugkit.instrument(analyze=True, profile=True)
            def complex_function(data):
                return process_data(data)

            # Static analysis
            results = debugkit.static_analysis.analyze_file(Path("module.py"))
            report = debugkit.static_analysis.generate_report(results)

        Configuration::

            debugkit.configure(
                verbose=True,
                trace_enabled=True,
                static_analysis_enabled=True
            )
    """

    def __init__(self, custom_config: Optional["DevConfig"] = None):
        """Initialize unified development utilities.

        Args:
            custom_config: Custom configuration (uses global config if None)
        """
        from haive.core.utils.debugkit.benchmarking import benchmark
        from haive.core.utils.debugkit.config import DevConfig
        from haive.core.utils.debugkit.debug import debug
        from haive.core.utils.debugkit.logging import log
        from haive.core.utils.debugkit.profiling import profile
        from haive.core.utils.debugkit.tracing import trace

        self.config = custom_config or config

        # Component interfaces
        self.debug = debug
        self.log = log
        self.trace = trace
        self.profile = profile
        self.benchmark = benchmark

        # Analysis interfaces (lazy loaded)
        self._type_analyzer: Optional["TypeAnalyzer"] = None
        self._complexity_analyzer: Optional["ComplexityAnalyzer"] = None
        self._static_orchestrator: Optional["StaticAnalysisOrchestrator"] = None

        # State
        self._correlation_id: Optional[str] = None
        self._analysis_cache: Dict[str, CodeAnalysisReport] = {}

    @property
    def type_analyzer(self) -> "TypeAnalyzer":
        """Get type analyzer instance."""
        if self._type_analyzer is None:
            self._type_analyzer = get_type_analyzer()
        return self._type_analyzer

    @property
    def complexity_analyzer(self) -> "ComplexityAnalyzer":
        """Get complexity analyzer instance."""
        if self._complexity_analyzer is None:
            self._complexity_analyzer = get_complexity_analyzer()
        return self._complexity_analyzer

    @property
    def static_analysis(self) -> "StaticAnalysisOrchestrator":
        """Get static analysis orchestrator instance."""
        if self._static_orchestrator is None:
            self._static_orchestrator = get_static_orchestrator()
        return self._static_orchestrator

    @property
    def correlation_id(self) -> Optional[str]:
        """Get current correlation ID.

        Returns:
            Optional[str]: Current correlation ID
        """
        return self._correlation_id

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for distributed tracing.

        Args:
            correlation_id: Correlation ID to set
        """
        self._correlation_id = correlation_id

        # Propagate to components
        if hasattr(self.log, "set_correlation_id"):
            self.log.set_correlation_id(correlation_id)
        if hasattr(self.trace, "set_correlation_id"):
            self.trace.set_correlation_id(correlation_id)

    def configure(self, **kwargs: Any) -> None:
        """Update development configuration.

        Args:
            **kwargs: Configuration values to update

        Examples:
            Update configuration::

                debugkit.configure(
                    verbose=True,
                    trace_sampling_rate=0.1,
                    static_analysis_enabled=False
                )
        """
        self.config.update(**kwargs)

    def context(self, name: str, **metadata: Any) -> DevContext:
        """Create a development context.

        Args:
            name: Context name
            **metadata: Additional context metadata

        Returns:
            DevContext: Context manager for unified operations

        Examples:
            Create context::

                with debugkit.context("user_registration", user_id=123) as ctx:
                    ctx.debug("Starting registration")
                    # ... work ...
                    ctx.success("Registration complete")
        """
        return DevContext(name, correlation_id=self._correlation_id, **metadata)

    def analyze_code(self, func: Callable[..., Any]) -> CodeAnalysisReport:
        """Perform comprehensive code analysis on a function.

        Combines type analysis, complexity analysis, and optional static
        analysis to provide a complete picture of code quality.

        Args:
            func: Function to analyze

        Returns:
            CodeAnalysisReport: Complete analysis results

        Examples:
            Analyze function quality::

                def complex_function(data, config=None):
                    # Complex implementation
                    pass

                report = debugkit.analyze_code(complex_function)

                if report.combined_score < 70:
                    print("Function needs improvement:")
                    for rec in report.recommendations:
                        print(f"  - {rec}")
        """
        func_name = getattr(func, "__name__", str(func))

        # Check cache
        cache_key = (
            f"{func.__module__}.{func_name}"
            if hasattr(func, "__module__")
            else func_name
        )
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        # Perform type analysis
        type_analysis = self.type_analyzer.analyze_function(func)

        # Perform complexity analysis
        complexity_analysis = self.complexity_analyzer.analyze_function(func)

        # Perform static analysis if enabled
        static_analysis = {}
        if self.config.static_analysis_enabled:
            try:
                # Get source file for static analysis
                source_file = inspect.getsourcefile(func)
                if source_file:
                    static_analysis = self.static_analysis.analyze_file(
                        Path(source_file),
                        tools=(
                            ["mypy", "pyflakes"]
                            if self.config.is_tool_enabled("mypy")
                            else ["pyflakes"]
                        ),
                    )
            except Exception:
                # Static analysis failed, continue without it
                pass

        # Create combined report
        report = CodeAnalysisReport(
            function_name=func_name,
            type_analysis=type_analysis,
            complexity_analysis=complexity_analysis,
            static_analysis=static_analysis,
        )

        # Cache result
        self._analysis_cache[cache_key] = report

        return report

    def instrument(
        self,
        func: Optional[Callable] = None,
        *,
        analyze: bool = False,
        profile: bool = None,
        trace: bool = None,
        log: bool = None,
        **options: Any,
    ) -> Union[Callable, Callable[[Callable], Callable]]:
        """Decorator for comprehensive function instrumentation.

        Adds logging, tracing, profiling, and optional analysis to functions.
        The decorator respects global configuration while allowing per-function overrides.

        Args:
            func: Function to instrument (when used without parentheses)
            analyze: Whether to perform code analysis
            profile: Whether to enable profiling (None uses config default)
            trace: Whether to enable tracing (None uses config default)
            log: Whether to enable logging (None uses config default)
            **options: Additional instrumentation options

        Returns:
            Decorated function or decorator

        Examples:
            Simple instrumentation::

                @debugkit.instrument
                def my_function():
                    return "result"

            Selective instrumentation::

                @debugkit.instrument(analyze=True, profile=True, log=False)
                def performance_critical():
                    return heavy_computation()

            Analysis with custom options::

                @debugkit.instrument(
                    analyze=True,
                    profile=True,
                    complexity_threshold=15
                )
                def complex_algorithm():
                    return algorithm_implementation()
        """

        def decorator(f: Callable) -> Callable:
            # Determine what to enable based on config and overrides
            enable_log = log if log is not None else self.config.log_enabled
            enable_trace = trace if trace is not None else self.config.trace_enabled
            enable_profile = (
                profile if profile is not None else self.config.profile_enabled
            )

            # Perform analysis if requested
            analysis_report = None
            if analyze or self.config.auto_analyze:
                try:
                    analysis_report = self.analyze_code(f)

                    # Log analysis results if requested
                    if enable_log and analysis_report.combined_score < 70:
                        self.log.warning(
                            f"Function {f.__name__} has low quality score: {analysis_report.combined_score:.1f}",
                            recommendations=analysis_report.recommendations[
                                :3
                            ],  # Top 3
                        )
                except Exception as e:
                    if enable_log:
                        self.log.warning(
                            f"Code analysis failed for {f.__name__}: {str(e)}"
                        )

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                func_name = f.__name__

                # Create execution context
                with self.context(f"function.{func_name}") as ctx:
                    # Log function entry
                    if enable_log:
                        ctx.debug(
                            f"Calling {func_name}",
                            args_count=len(args),
                            kwargs_keys=list(kwargs.keys()),
                        )

                    # Apply tracing
                    if enable_trace and hasattr(self.trace, "calls"):
                        ctx.checkpoint("trace_start")

                    # Apply profiling
                    profiler_context = None
                    if enable_profile and hasattr(self.profile, "start_context"):
                        profiler_context = self.profile.start_context(func_name)
                        ctx.checkpoint("profile_start")

                    try:
                        # Execute function
                        result = f(*args, **kwargs)

                        # Log success
                        if enable_log:
                            ctx.success(
                                f"Completed {func_name}",
                                result_type=type(result).__name__,
                            )

                        return result

                    except Exception as e:
                        # Log error
                        if enable_log:
                            ctx.error(
                                f"Failed {func_name}",
                                error=str(e),
                                error_type=type(e).__name__,
                            )
                        raise

                    finally:
                        # Stop profiling
                        if profiler_context and hasattr(self.profile, "stop_context"):
                            profile_stats = self.profile.stop_context(profiler_context)
                            ctx.record("profile_stats", profile_stats)
                            ctx.checkpoint("profile_complete")

            # Attach analysis report to wrapper for inspection
            if analysis_report:
                wrapper._analysis_report = analysis_report

            return wrapper

        # Handle both @debugkit.instrument and @debugkit.instrument(...) usage
        if func is None:
            return decorator
        else:
            return decorator(func)

    # Convenience methods for common operations

    def ice(self, *args, **kwargs) -> Any:
        """Enhanced debugging output (icecream style).

        Args:
            *args: Values to debug
            **kwargs: Named values to debug

        Returns:
            The first argument (for chaining)

        Examples:
            Debug variables::

                result = debugkit.ice(process_data(input_data))
                debugkit.ice("Processing", count=len(items), status="active")
        """
        if self.config.debug_enabled:
            return self.debug.ice(*args, **kwargs)
        return args[0] if args else None

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.

        Args:
            message: Info message
            **kwargs: Additional context
        """
        if self.config.log_enabled:
            self.log.info(message, **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        """Log success message.

        Args:
            message: Success message
            **kwargs: Additional context
        """
        if self.config.log_enabled:
            self.log.success(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message.

        Args:
            message: Error message
            **kwargs: Additional context
        """
        if self.config.log_enabled:
            self.log.error(message, **kwargs)

    def clear_cache(self) -> None:
        """Clear all analysis caches.

        Examples:
            Clear caches for fresh analysis::

                debugkit.clear_cache()
                fresh_report = debugkit.analyze_code(my_function)
        """
        self._analysis_cache.clear()
        if self._type_analyzer:
            self._type_analyzer.clear_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get development utilities statistics.

        Returns:
            Dict[str, Any]: Statistics about tool usage and performance
        """
        stats = {
            "config": self.config.to_dict(),
            "analysis_cache_size": len(self._analysis_cache),
            "correlation_id": self._correlation_id,
        }

        if self._type_analyzer:
            stats["type_analyzer"] = self._type_analyzer.get_cache_stats()

        if self._static_orchestrator:
            stats["static_analysis_tools"] = (
                self._static_orchestrator.get_available_tools()
            )

        return stats
