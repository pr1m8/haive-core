"""
Unified development utilities for Python debugging, profiling, and analysis.

This package provides a comprehensive suite of development tools including:
- Enhanced debugging with multiple debugger integrations
- Rich logging with structured output and context management
- Advanced code tracing and execution analysis
- Performance profiling with multiple profiler backends
- Comprehensive static analysis orchestration
- Code complexity and type analysis
- Benchmarking and load testing utilities

All utilities are designed to work together seamlessly and provide
both simple interfaces for quick debugging and advanced features
for comprehensive code analysis.

Examples:
    Quick start with unified interface::

        from haive.core.utils.debugkit import debugkit

        # Enhanced debugging
        debugkit.ice("Debug variable", value=42)

        # Rich logging with context
        with debugkit.context("operation") as ctx:
            ctx.log("Starting process")
            # ... work ...
            ctx.log("Process complete")

        # Complete analysis
        @debugkit.instrument(analyze=True, profile=True)
        def my_function(data: List[str]) -> Dict[str, int]:
            return process_data(data)

    Individual component usage::

        from haive.core.utils.debugkit import debug, log, trace, profile

        # Use specific components
        debug.ice("Variable inspection", data=my_data)
        log.info("Process started", context={"user": "alice"})

        @trace.calls
        @profile.time
        def traced_function():
            return complex_operation()

    Advanced analysis::

        from haive.core.utils.debugkit import debugkit

        # Analyze code quality
        analysis = debugkit.analyze_code(my_function)
        print(f"Type coverage: {analysis.type_analysis.type_coverage:.1%}")
        print(f"Complexity grade: {analysis.complexity_analysis.complexity_grade}")

        # Run static analysis
        results = debugkit.static_analysis.analyze_file(Path("module.py"))
        report = debugkit.static_analysis.generate_report(results)
"""

from typing import TYPE_CHECKING

# Analysis components
from haive.core.utils.debugkit.analysis import (
    get_complexity_analyzer,
    get_static_orchestrator,
    get_type_analyzer,
)
from haive.core.utils.debugkit.benchmarking import benchmark

# Core configuration
from haive.core.utils.debugkit.config import (
    DevConfig,
    Environment,
    LogLevel,
    StorageBackend,
    config,
)

# Core components
from haive.core.utils.debugkit.core import CodeAnalysisReport, DevContext, UnifiedDev

# Individual component interfaces with fallbacks
from haive.core.utils.debugkit.debug import debug
from haive.core.utils.debugkit.logging import log
from haive.core.utils.debugkit.profiling import profile
from haive.core.utils.debugkit.tracing import trace

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


# Create global instance
debugkit = UnifiedDev()

# Export main interface and individual components
__all__ = [
    "debugkit",
    "debug",
    "log",
    "trace",
    "profile",
    "benchmark",
    "config",
    "DevConfig",
    "DevContext",
    "CodeAnalysisReport",
    "UnifiedDev",
    "Environment",
    "LogLevel",
    "StorageBackend",
]
