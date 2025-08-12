"""
Advanced code analysis utilities for type checking and complexity analysis.

This package provides comprehensive code analysis capabilities including:
- Static type analysis with multiple type checkers
- Multi-dimensional complexity analysis
- Code quality scoring and recommendations
- Integration with popular Python analysis tools

The analysis modules work together to provide detailed insights into code
quality, maintainability, and potential issues.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .complexity import ComplexityAnalyzer, ComplexityMetrics, ComplexityReport
    from .static import AnalysisResult, StaticAnalysisOrchestrator
    from .types import FunctionTypeAnalysis, TypeAnalyzer, TypeInfo

# Analysis modules with lazy loading
_type_analyzer: Optional["TypeAnalyzer"] = None
_complexity_analyzer: Optional["ComplexityAnalyzer"] = None
_static_orchestrator: Optional["StaticAnalysisOrchestrator"] = None


def get_type_analyzer() -> "TypeAnalyzer":
    """Get or create the type analyzer instance.

    Returns:
        TypeAnalyzer: The type analyzer instance
    """
    global _type_analyzer
    if _type_analyzer is None:
        from haive.core.utils.debugkit.config import config

        from .types import TypeAnalyzer

        _type_analyzer = TypeAnalyzer(
            use_mypy=config.is_tool_enabled("mypy"),
            cache_enabled=True,
            strict_mode=(
                config.strict_thresholds
                if hasattr(config, "strict_thresholds")
                else False
            ),
        )
    return _type_analyzer


def get_complexity_analyzer() -> "ComplexityAnalyzer":
    """Get or create the complexity analyzer instance.

    Returns:
        ComplexityAnalyzer: The complexity analyzer instance
    """
    global _complexity_analyzer
    if _complexity_analyzer is None:
        from .complexity import ComplexityAnalyzer

        _complexity_analyzer = ComplexityAnalyzer()
    return _complexity_analyzer


def get_static_orchestrator() -> "StaticAnalysisOrchestrator":
    """Get or create the static analysis orchestrator instance.

    Returns:
        StaticAnalysisOrchestrator: The static analysis orchestrator
    """
    global _static_orchestrator
    if _static_orchestrator is None:
        from .static import StaticAnalysisOrchestrator

        _static_orchestrator = StaticAnalysisOrchestrator()
    return _static_orchestrator


__all__ = [
    "AnalysisResult",
    "ComplexityAnalyzer",
    "ComplexityMetrics",
    "ComplexityReport",
    "FunctionTypeAnalysis",
    "StaticAnalysisOrchestrator",
    "TypeAnalyzer",
    "TypeInfo",
    "get_complexity_analyzer",
    "get_static_orchestrator",
    "get_type_analyzer",
]
