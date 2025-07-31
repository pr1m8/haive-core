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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .complexity import ComplexityAnalyzer, ComplexityMetrics, ComplexityReport
    from .static import AnalysisResult, StaticAnalysisOrchestrator
    from .types import FunctionTypeAnalysis, TypeAnalyzer, TypeInfo

__all__ = [
    "TypeAnalyzer",
    "FunctionTypeAnalysis",
    "TypeInfo",
    "ComplexityAnalyzer",
    "ComplexityReport",
    "ComplexityMetrics",
    "StaticAnalysisOrchestrator",
    "AnalysisResult",
]
