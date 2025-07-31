"""
Advanced complexity analysis for Python functions and modules.

This module provides comprehensive code complexity analysis including:
- Cyclomatic complexity (traditional control flow complexity)
- Cognitive complexity (human-perceived complexity)
- Halstead metrics (algorithmic complexity)
- Maintainability index calculation
- Nesting depth analysis
- Custom complexity scoring and grading

The complexity analyzer helps identify code that may be difficult to
maintain, test, or understand, providing actionable recommendations
for improvement.
"""

import ast
import inspect
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import radon.complexity as radon_cc
    import radon.metrics as radon_metrics
    from radon.visitors import HalsteadVisitor

    HAS_RADON = True
except ImportError:
    HAS_RADON = False

try:
    import mccabe

    HAS_MCCABE = True
except ImportError:
    HAS_MCCABE = False


class ComplexityGrade(str, Enum):
    """Complexity grade classifications.

    Attributes:
        A: Excellent - Low complexity, highly maintainable
        B: Good - Moderate complexity, maintainable
        C: Fair - Some complexity, may need refactoring
        D: Poor - High complexity, should be refactored
        F: Critical - Very high complexity, urgent refactoring needed
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class ComplexityType(str, Enum):
    """Types of complexity measurements.

    Attributes:
        CYCLOMATIC: Traditional cyclomatic complexity (control flow)
        COGNITIVE: Cognitive complexity (human perception)
        HALSTEAD: Halstead complexity (algorithmic)
        NESTING: Nesting depth complexity
        PARAMETER: Parameter count complexity
        LENGTH: Code length complexity
    """

    CYCLOMATIC = "cyclomatic"
    COGNITIVE = "cognitive"
    HALSTEAD = "halstead"
    NESTING = "nesting"
    PARAMETER = "parameter"
    LENGTH = "length"


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for a code unit.

    This class contains all complexity measurements for a function,
    providing both traditional and advanced complexity metrics.

    Attributes:
        cyclomatic_complexity: Traditional cyclomatic complexity
        cognitive_complexity: Human-perceived complexity
        halstead_metrics: Dictionary of Halstead metrics
        maintainability_index: Maintainability index (0-100)
        lines_of_code: Total lines including comments and blanks
        logical_lines_of_code: Lines containing actual code
        comment_lines: Number of comment lines
        blank_lines: Number of blank lines
        comment_ratio: Ratio of comments to code lines
        nesting_depth: Maximum nesting depth
        parameter_count: Number of function parameters
        return_count: Number of return statements
        branch_count: Number of conditional branches
        loop_count: Number of loops
        function_calls: Number of function calls
        unique_operators: Number of unique operators (Halstead)
        unique_operands: Number of unique operands (Halstead)
        total_operators: Total number of operators (Halstead)
        total_operands: Total number of operands (Halstead)

    Examples:
        Access specific metrics::

            metrics = complexity_analyzer.analyze_function(my_func).metrics

            print(f"Cyclomatic complexity: {metrics.cyclomatic_complexity}")
            print(f"Cognitive complexity: {metrics.cognitive_complexity}")
            print(f"Maintainability index: {metrics.maintainability_index}")
            print(f"Comment ratio: {metrics.comment_ratio:.1%}")

        Check thresholds::

            if metrics.cyclomatic_complexity > 10:
                print("Function is too complex")

            if metrics.nesting_depth > 4:
                print("Function has excessive nesting")
    """

    # Core complexity metrics
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    halstead_metrics: Dict[str, float] = None
    maintainability_index: float = 0.0

    # Code size metrics
    lines_of_code: int = 0
    logical_lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    comment_ratio: float = 0.0

    # Structure metrics
    nesting_depth: int = 0
    parameter_count: int = 0
    return_count: int = 0
    branch_count: int = 0
    loop_count: int = 0
    function_calls: int = 0

    # Halstead components
    unique_operators: int = 0
    unique_operands: int = 0
    total_operators: int = 0
    total_operands: int = 0

    def __post_init__(self) -> None:
        """Initialize default values for complex fields."""
        if self.halstead_metrics is None:
            self.halstead_metrics = {
                "vocabulary": 0.0,
                "length": 0.0,
                "calculated_length": 0.0,
                "volume": 0.0,
                "difficulty": 0.0,
                "effort": 0.0,
                "time": 0.0,
                "bugs": 0.0,
            }

    def get_overall_complexity_score(self) -> float:
        """Calculate overall complexity score combining all metrics.

        Returns:
            float: Overall complexity score (0-100, higher is more complex)

        Examples:
            Get overall complexity::

                score = metrics.get_overall_complexity_score()
                if score > 75:
                    print("Function is highly complex")
        """
        # Normalize individual scores to 0-25 scale
        cc_score = min(25, (self.cyclomatic_complexity / 20) * 25)
        cognitive_score = min(25, (self.cognitive_complexity / 30) * 25)
        nesting_score = min(25, (self.nesting_depth / 8) * 25)

        # Maintainability index is inverted (higher is better)
        mi_score = max(0, 25 * (1 - self.maintainability_index / 100))

        return cc_score + cognitive_score + nesting_score + mi_score

    def get_complexity_breakdown(self) -> Dict[str, float]:
        """Get detailed breakdown of complexity contributors.

        Returns:
            Dict[str, float]: Breakdown of complexity scores by type
        """
        return {
            "cyclomatic": min(25, (self.cyclomatic_complexity / 20) * 25),
            "cognitive": min(25, (self.cognitive_complexity / 30) * 25),
            "nesting": min(25, (self.nesting_depth / 8) * 25),
            "maintainability": max(0, 25 * (1 - self.maintainability_index / 100)),
            "structure": min(25, (self.parameter_count / 10) * 25),
        }


@dataclass
class ComplexityHotspot:
    """A specific location in code with high complexity.

    Attributes:
        type: Type of complexity hotspot
        line_number: Line number where hotspot occurs
        description: Human-readable description
        severity: Severity level (low/medium/high/critical)
        suggestion: Specific suggestion for improvement
        context: Additional context about the hotspot

    Examples:
        Process hotspots::

            for hotspot in complexity_report.hotspots:
                print(f"Line {hotspot.line_number}: {hotspot.description}")
                print(f"Suggestion: {hotspot.suggestion}")
    """

    type: str
    line_number: int
    description: str
    severity: str = "medium"
    suggestion: str = ""
    context: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Initialize context dictionary."""
        if self.context is None:
            self.context = {}


@dataclass
class ComplexityReport:
    """Complete complexity analysis report for a function.

    This class contains comprehensive complexity analysis results including
    metrics, grading, recommendations, and specific hotspots that need attention.

    Attributes:
        function_name: Name of the analyzed function
        module_name: Module containing the function
        metrics: Detailed complexity metrics
        complexity_grade: Overall complexity grade (A-F)
        grade_breakdown: Grade breakdown by complexity type
        refactoring_suggestions: List of specific refactoring suggestions
        hotspots: List of complexity hotspots in the code
        risk_score: Overall risk score (0-100, higher is riskier)
        maintainability_score: Maintainability score (0-100, higher is better)
        testability_score: How easy the function is to test (0-100)
        analysis_timestamp: When analysis was performed
        thresholds_exceeded: Which complexity thresholds were exceeded

    Examples:
        Review analysis results::

            report = complexity_analyzer.analyze_function(my_function)

            print(f"Overall grade: {report.complexity_grade}")
            print(f"Risk score: {report.risk_score}")

            if report.risk_score > 75:
                print("High risk function - needs refactoring")
                for suggestion in report.refactoring_suggestions:
                    print(f"- {suggestion}")

            # Review specific hotspots
            for hotspot in report.hotspots:
                if hotspot.severity == "critical":
                    print(f"Critical issue at line {hotspot.line_number}")
    """

    function_name: str
    module_name: Optional[str] = None
    metrics: ComplexityMetrics = None
    complexity_grade: ComplexityGrade = ComplexityGrade.C
    grade_breakdown: Dict[str, str] = None
    refactoring_suggestions: List[str] = None
    hotspots: List[ComplexityHotspot] = None
    risk_score: float = 0.0
    maintainability_score: float = 50.0
    testability_score: float = 50.0
    analysis_timestamp: Optional[str] = None
    thresholds_exceeded: List[str] = None

    def __post_init__(self) -> None:
        """Initialize default values for complex fields."""
        if self.metrics is None:
            self.metrics = ComplexityMetrics()
        if self.grade_breakdown is None:
            self.grade_breakdown = {}
        if self.refactoring_suggestions is None:
            self.refactoring_suggestions = []
        if self.hotspots is None:
            self.hotspots = []
        if self.thresholds_exceeded is None:
            self.thresholds_exceeded = []

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of complexity analysis.

        Returns:
            Dict[str, Any]: Summary containing key metrics and assessments
        """
        return {
            "function_name": self.function_name,
            "overall_grade": self.complexity_grade.value,
            "risk_score": self.risk_score,
            "maintainability_score": self.maintainability_score,
            "cyclomatic_complexity": self.metrics.cyclomatic_complexity,
            "cognitive_complexity": self.metrics.cognitive_complexity,
            "lines_of_code": self.metrics.lines_of_code,
            "suggestion_count": len(self.refactoring_suggestions),
            "hotspot_count": len(self.hotspots),
            "critical_hotspots": len(
                [h for h in self.hotspots if h.severity == "critical"]
            ),
            "thresholds_exceeded": self.thresholds_exceeded.copy(),
        }

    def needs_refactoring(self) -> bool:
        """Determine if function needs refactoring based on complexity.

        Returns:
            bool: True if refactoring is recommended
        """
        return (
            self.complexity_grade in [ComplexityGrade.D, ComplexityGrade.F]
            or self.risk_score > 75
            or len([h for h in self.hotspots if h.severity == "critical"]) > 0
        )


class ComplexityAnalyzer:
    """Advanced code complexity analyzer.

    This class provides comprehensive complexity analysis for Python functions
    and modules using multiple complexity metrics and analysis techniques.

    The analyzer integrates with external tools when available (radon, mccabe)
    and provides fallback implementations for core functionality.

    Attributes:
        use_radon: Whether to use radon for enhanced metrics
        use_mccabe: Whether to use mccabe for cyclomatic complexity
        strict_thresholds: Whether to use strict complexity thresholds
        custom_thresholds: Custom complexity thresholds

    Examples:
        Basic complexity analysis::

            analyzer = ComplexityAnalyzer()

            def complex_function(data, config=None):
                if not data:
                    return None

                result = []
                for item in data:
                    if config and 'filter' in config:
                        if item.get('type') == config['filter']:
                            if item.get('status') == 'active':
                                result.append(process_item(item))
                return result

            report = analyzer.analyze_function(complex_function)
            print(f"Complexity grade: {report.complexity_grade}")
            print(f"Suggestions: {len(report.refactoring_suggestions)}")

        Custom thresholds::

            analyzer = ComplexityAnalyzer(
                strict_thresholds=True,
                custom_thresholds={
                    'cyclomatic': {'low': 3, 'medium': 6, 'high': 10},
                    'nesting': {'low': 2, 'medium': 3, 'high': 4}
                }
            )
    """

    def __init__(
        self,
        use_radon: bool = HAS_RADON,
        use_mccabe: bool = HAS_MCCABE,
        strict_thresholds: bool = False,
        custom_thresholds: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        """Initialize the complexity analyzer.

        Args:
            use_radon: Whether to use radon for enhanced complexity metrics
            use_mccabe: Whether to use mccabe for cyclomatic complexity
            strict_thresholds: Whether to use strict complexity thresholds
            custom_thresholds: Custom thresholds for complexity metrics
        """
        self.use_radon = use_radon and HAS_RADON
        self.use_mccabe = use_mccabe and HAS_MCCABE
        self.strict_thresholds = strict_thresholds

        # Default thresholds
        self.thresholds = {
            "cyclomatic": {"low": 5, "medium": 10, "high": 20, "critical": 30},
            "cognitive": {"low": 7, "medium": 15, "high": 25, "critical": 40},
            "nesting": {"low": 3, "medium": 4, "high": 6, "critical": 8},
            "parameters": {"low": 3, "medium": 5, "high": 8, "critical": 12},
            "length": {"low": 25, "medium": 50, "high": 100, "critical": 200},
            "maintainability": {"excellent": 85, "good": 70, "fair": 50, "poor": 25},
        }

        # Apply strict thresholds if requested
        if strict_thresholds:
            self.thresholds["cyclomatic"] = {
                "low": 3,
                "medium": 6,
                "high": 10,
                "critical": 15,
            }
            self.thresholds["cognitive"] = {
                "low": 5,
                "medium": 10,
                "high": 15,
                "critical": 25,
            }
            self.thresholds["nesting"] = {
                "low": 2,
                "medium": 3,
                "high": 4,
                "critical": 6,
            }

        # Apply custom thresholds
        if custom_thresholds:
            for metric, threshold_dict in custom_thresholds.items():
                if metric in self.thresholds:
                    self.thresholds[metric].update(threshold_dict)

    def analyze_function(self, func: Callable[..., Any]) -> ComplexityReport:
        """Perform comprehensive complexity analysis on a function.

        Analyzes all aspects of function complexity including control flow,
        cognitive load, structural complexity, and maintainability metrics.

        Args:
            func: The function to analyze

        Returns:
            ComplexityReport: Complete complexity analysis results

        Raises:
            ValueError: If the function cannot be analyzed
            TypeError: If the input is not a callable

        Examples:
            Analyze function complexity::

                def nested_function(items, config):
                    result = {}
                    for item in items:
                        if item.get('active'):
                            category = item.get('category', 'default')
                            if category not in result:
                                result[category] = []

                            if config.get('validate', True):
                                if validate_item(item):
                                    result[category].append(item)
                            else:
                                result[category].append(item)
                    return result

                report = analyzer.analyze_function(nested_function)

                # Check if refactoring is needed
                if report.needs_refactoring():
                    print("Function should be refactored")
                    for suggestion in report.refactoring_suggestions:
                        print(f"- {suggestion}")
        """
        if not callable(func):
            raise TypeError(f"Expected callable, got {type(func)}")

        func_name = getattr(func, "__name__", str(func))

        try:
            # Get source code
            source = inspect.getsource(func)
        except (OSError, TypeError):
            # Can't get source for built-in functions
            return ComplexityReport(
                function_name=func_name,
                metrics=ComplexityMetrics(),
                complexity_grade=ComplexityGrade.C,
                refactoring_suggestions=[
                    "Cannot analyze built-in or C extension function"
                ],
            )

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return ComplexityReport(
                function_name=func_name,
                complexity_grade=ComplexityGrade.F,
                refactoring_suggestions=[f"Fix syntax error: {str(e)}"],
            )

        # Calculate metrics
        metrics = self._calculate_metrics(tree, source, func)

        # Calculate grades and scores
        complexity_grade = self._calculate_grade(metrics)
        grade_breakdown = self._calculate_grade_breakdown(metrics)
        risk_score = self._calculate_risk_score(metrics)
        maintainability_score = metrics.maintainability_index
        testability_score = self._calculate_testability_score(metrics)

        # Generate suggestions
        suggestions = self._generate_suggestions(metrics)

        # Find hotspots
        hotspots = self._find_hotspots(tree, metrics)

        # Check thresholds
        thresholds_exceeded = self._check_thresholds(metrics)

        return ComplexityReport(
            function_name=func_name,
            module_name=getattr(func, "__module__", None),
            metrics=metrics,
            complexity_grade=complexity_grade,
            grade_breakdown=grade_breakdown,
            refactoring_suggestions=suggestions,
            hotspots=hotspots,
            risk_score=risk_score,
            maintainability_score=maintainability_score,
            testability_score=testability_score,
            thresholds_exceeded=thresholds_exceeded,
        )

    def _calculate_metrics(
        self, tree: ast.AST, source: str, func: Callable[..., Any]
    ) -> ComplexityMetrics:
        """Calculate all complexity metrics for a function."""
        metrics = ComplexityMetrics()

        # Basic code metrics
        lines = source.split("\n")
        metrics.lines_of_code = len(lines)
        metrics.logical_lines_of_code = len(
            [l for l in lines if l.strip() and not l.strip().startswith("#")]
        )
        metrics.comment_lines = len([l for l in lines if l.strip().startswith("#")])
        metrics.blank_lines = len([l for l in lines if not l.strip()])
        metrics.comment_ratio = metrics.comment_lines / max(
            metrics.logical_lines_of_code, 1
        )

        # Use radon if available
        if self.use_radon:
            try:
                metrics = self._calculate_radon_metrics(source, metrics)
            except Exception:
                # Fallback to manual calculation
                metrics = self._calculate_manual_metrics(tree, metrics, func)
        else:
            metrics = self._calculate_manual_metrics(tree, metrics, func)

        return metrics

    def _calculate_radon_metrics(
        self, source: str, metrics: ComplexityMetrics
    ) -> ComplexityMetrics:
        """Calculate metrics using radon library."""
        # Cyclomatic complexity
        cc_visitor = radon_cc.ComplexityVisitor.from_code(source)
        if cc_visitor.complexity:
            # radon returns list of complexities, take the first (main function)
            complexities = [c.complexity for c in cc_visitor.complexity]
            metrics.cyclomatic_complexity = max(complexities) if complexities else 1

        # Halstead metrics
        h_visitor = HalsteadVisitor.from_code(source)
        metrics.halstead_metrics = {
            "vocabulary": h_visitor.vocabulary,
            "length": h_visitor.length,
            "calculated_length": h_visitor.calculated_length,
            "volume": h_visitor.volume,
            "difficulty": h_visitor.difficulty,
            "effort": h_visitor.effort,
            "time": h_visitor.time,
            "bugs": h_visitor.bugs,
        }

        metrics.unique_operators = h_visitor.h1
        metrics.unique_operands = h_visitor.h2
        metrics.total_operators = h_visitor.N1
        metrics.total_operands = h_visitor.N2

        # Maintainability index
        try:
            metrics.maintainability_index = radon_metrics.mi_visit(source, multi=False)
        except Exception:
            metrics.maintainability_index = 50.0  # Default value

        return metrics

    def _calculate_manual_metrics(
        self, tree: ast.AST, metrics: ComplexityMetrics, func: Callable[..., Any]
    ) -> ComplexityMetrics:
        """Calculate metrics manually using AST analysis."""
        analyzer = ManualComplexityAnalyzer()
        analyzer.visit(tree)

        # Copy results to metrics
        metrics.cyclomatic_complexity = analyzer.cyclomatic_complexity
        metrics.cognitive_complexity = analyzer.cognitive_complexity
        metrics.nesting_depth = analyzer.max_nesting_depth
        metrics.parameter_count = analyzer.parameter_count
        metrics.return_count = analyzer.return_count
        metrics.branch_count = analyzer.branch_count
        metrics.loop_count = analyzer.loop_count
        metrics.function_calls = analyzer.function_calls

        # Estimate maintainability index
        metrics.maintainability_index = self._estimate_maintainability_index(metrics)

        return metrics

    def _estimate_maintainability_index(self, metrics: ComplexityMetrics) -> float:
        """Estimate maintainability index without radon."""
        # Simplified MI calculation based on available metrics
        # Real MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * (Cyclomatic Complexity) - 16.2 * ln(Lines of Code)

        # Use approximations
        volume_estimate = metrics.logical_lines_of_code * 2  # Rough estimate

        if volume_estimate > 0 and metrics.lines_of_code > 0:
            mi = (
                171
                - 5.2 * math.log(volume_estimate)
                - 0.23 * metrics.cyclomatic_complexity
                - 16.2 * math.log(metrics.lines_of_code)
            )
        else:
            mi = 50.0  # Default value

        # Normalize to 0-100 range
        return max(0.0, min(100.0, mi))

    def _calculate_grade(self, metrics: ComplexityMetrics) -> ComplexityGrade:
        """Calculate overall complexity grade."""
        score = 0

        # Cyclomatic complexity scoring (25 points)
        if metrics.cyclomatic_complexity <= self.thresholds["cyclomatic"]["low"]:
            score += 25
        elif metrics.cyclomatic_complexity <= self.thresholds["cyclomatic"]["medium"]:
            score += 20
        elif metrics.cyclomatic_complexity <= self.thresholds["cyclomatic"]["high"]:
            score += 10
        else:
            score += 0

        # Cognitive complexity scoring (25 points)
        if metrics.cognitive_complexity <= self.thresholds["cognitive"]["low"]:
            score += 25
        elif metrics.cognitive_complexity <= self.thresholds["cognitive"]["medium"]:
            score += 20
        elif metrics.cognitive_complexity <= self.thresholds["cognitive"]["high"]:
            score += 10
        else:
            score += 0

        # Maintainability index scoring (25 points)
        if (
            metrics.maintainability_index
            >= self.thresholds["maintainability"]["excellent"]
        ):
            score += 25
        elif (
            metrics.maintainability_index >= self.thresholds["maintainability"]["good"]
        ):
            score += 20
        elif (
            metrics.maintainability_index >= self.thresholds["maintainability"]["fair"]
        ):
            score += 10
        else:
            score += 0

        # Nesting depth scoring (25 points)
        if metrics.nesting_depth <= self.thresholds["nesting"]["low"]:
            score += 25
        elif metrics.nesting_depth <= self.thresholds["nesting"]["medium"]:
            score += 20
        elif metrics.nesting_depth <= self.thresholds["nesting"]["high"]:
            score += 10
        else:
            score += 0

        # Convert to grade
        if score >= 90:
            return ComplexityGrade.A
        elif score >= 80:
            return ComplexityGrade.B
        elif score >= 70:
            return ComplexityGrade.C
        elif score >= 60:
            return ComplexityGrade.D
        else:
            return ComplexityGrade.F

    def _calculate_grade_breakdown(self, metrics: ComplexityMetrics) -> Dict[str, str]:
        """Calculate grade breakdown by complexity type."""
        breakdown = {}

        # Cyclomatic complexity grade
        cc = metrics.cyclomatic_complexity
        if cc <= self.thresholds["cyclomatic"]["low"]:
            breakdown["cyclomatic"] = "A"
        elif cc <= self.thresholds["cyclomatic"]["medium"]:
            breakdown["cyclomatic"] = "B"
        elif cc <= self.thresholds["cyclomatic"]["high"]:
            breakdown["cyclomatic"] = "C"
        elif cc <= self.thresholds["cyclomatic"]["critical"]:
            breakdown["cyclomatic"] = "D"
        else:
            breakdown["cyclomatic"] = "F"

        # Cognitive complexity grade
        cog = metrics.cognitive_complexity
        if cog <= self.thresholds["cognitive"]["low"]:
            breakdown["cognitive"] = "A"
        elif cog <= self.thresholds["cognitive"]["medium"]:
            breakdown["cognitive"] = "B"
        elif cog <= self.thresholds["cognitive"]["high"]:
            breakdown["cognitive"] = "C"
        elif cog <= self.thresholds["cognitive"]["critical"]:
            breakdown["cognitive"] = "D"
        else:
            breakdown["cognitive"] = "F"

        # Add other breakdowns...
        return breakdown

    def _calculate_risk_score(self, metrics: ComplexityMetrics) -> float:
        """Calculate overall risk score (0-100)."""
        risk = 0.0

        # Cyclomatic complexity contribution (0-25)
        cc_risk = min(
            25,
            (metrics.cyclomatic_complexity / self.thresholds["cyclomatic"]["critical"])
            * 25,
        )
        risk += cc_risk

        # Cognitive complexity contribution (0-25)
        cog_risk = min(
            25,
            (metrics.cognitive_complexity / self.thresholds["cognitive"]["critical"])
            * 25,
        )
        risk += cog_risk

        # Nesting depth contribution (0-25)
        nest_risk = min(
            25, (metrics.nesting_depth / self.thresholds["nesting"]["critical"]) * 25
        )
        risk += nest_risk

        # Maintainability contribution (0-25, inverted)
        mi_risk = max(0, 25 * (1 - metrics.maintainability_index / 100))
        risk += mi_risk

        return min(100.0, risk)

    def _calculate_testability_score(self, metrics: ComplexityMetrics) -> float:
        """Calculate how easy the function is to test (0-100)."""
        score = 100.0

        # Deduct for high complexity
        score -= min(30, metrics.cyclomatic_complexity * 2)
        score -= min(20, metrics.parameter_count * 3)
        score -= min(20, metrics.nesting_depth * 4)
        score -= min(15, metrics.return_count * 2)

        # Bonus for good practices
        if metrics.comment_ratio > 0.1:
            score += 5

        return max(0.0, min(100.0, score))

    def _generate_suggestions(self, metrics: ComplexityMetrics) -> List[str]:
        """Generate actionable refactoring suggestions."""
        suggestions = []

        if metrics.cyclomatic_complexity > self.thresholds["cyclomatic"]["medium"]:
            suggestions.append(
                "Reduce cyclomatic complexity by extracting methods or simplifying conditions"
            )

        if metrics.cognitive_complexity > self.thresholds["cognitive"]["medium"]:
            suggestions.append(
                "Reduce cognitive complexity by simplifying nested logic and conditions"
            )

        if metrics.nesting_depth > self.thresholds["nesting"]["medium"]:
            suggestions.append(
                "Reduce nesting depth using early returns or guard clauses"
            )

        if metrics.parameter_count > self.thresholds["parameters"]["medium"]:
            suggestions.append(
                "Reduce parameter count by using parameter objects or configuration classes"
            )

        if metrics.lines_of_code > self.thresholds["length"]["medium"]:
            suggestions.append(
                "Function is too long - extract smaller, focused methods"
            )

        if metrics.return_count > 5:
            suggestions.append(
                "Too many return statements - consider using a single exit point"
            )

        if metrics.comment_ratio < 0.05:
            suggestions.append(
                "Add comments to explain complex logic and business rules"
            )

        if metrics.maintainability_index < self.thresholds["maintainability"]["fair"]:
            suggestions.append(
                "Low maintainability - consider major refactoring or redesign"
            )

        return suggestions

    def _find_hotspots(
        self, tree: ast.AST, metrics: ComplexityMetrics
    ) -> List[ComplexityHotspot]:
        """Find specific complexity hotspots in the code."""
        hotspots = []

        class HotspotFinder(ast.NodeVisitor):
            def __init__(self):
                self.hotspots = []
                self.nesting_level = 0

            def visit_If(self, node):
                # Check for complex conditions
                if isinstance(node.test, ast.BoolOp):
                    if len(node.test.values) > 3:
                        self.hotspots.append(
                            ComplexityHotspot(
                                type="complex_condition",
                                line_number=node.lineno,
                                description=f"Complex boolean condition with {len(node.test.values)} parts",
                                severity="medium",
                                suggestion="Break complex condition into smaller, named boolean variables",
                            )
                        )

                self.nesting_level += 1
                if self.nesting_level > 4:
                    self.hotspots.append(
                        ComplexityHotspot(
                            type="deep_nesting",
                            line_number=node.lineno,
                            description=f"Deep nesting level: {self.nesting_level}",
                            severity="high",
                            suggestion="Use early returns or extract methods to reduce nesting",
                        )
                    )

                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_For(self, node):
                # Check for nested loops
                self.nesting_level += 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        self.hotspots.append(
                            ComplexityHotspot(
                                type="nested_loop",
                                line_number=node.lineno,
                                description="Nested loop detected",
                                severity="high",
                                suggestion="Consider using list comprehensions or extracting inner loop",
                            )
                        )
                        break

                self.generic_visit(node)
                self.nesting_level -= 1

        finder = HotspotFinder()
        finder.visit(tree)
        return finder.hotspots

    def _check_thresholds(self, metrics: ComplexityMetrics) -> List[str]:
        """Check which complexity thresholds were exceeded."""
        exceeded = []

        if metrics.cyclomatic_complexity > self.thresholds["cyclomatic"]["high"]:
            exceeded.append(f"cyclomatic_complexity ({metrics.cyclomatic_complexity})")

        if metrics.cognitive_complexity > self.thresholds["cognitive"]["high"]:
            exceeded.append(f"cognitive_complexity ({metrics.cognitive_complexity})")

        if metrics.nesting_depth > self.thresholds["nesting"]["high"]:
            exceeded.append(f"nesting_depth ({metrics.nesting_depth})")

        if metrics.parameter_count > self.thresholds["parameters"]["high"]:
            exceeded.append(f"parameter_count ({metrics.parameter_count})")

        if metrics.lines_of_code > self.thresholds["length"]["high"]:
            exceeded.append(f"lines_of_code ({metrics.lines_of_code})")

        return exceeded


class ManualComplexityAnalyzer(ast.NodeVisitor):
    """Manual complexity analyzer using AST traversal."""

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self.cyclomatic_complexity = 1  # Base complexity
        self.cognitive_complexity = 0
        self.max_nesting_depth = 0
        self.current_nesting_depth = 0
        self.parameter_count = 0
        self.return_count = 0
        self.branch_count = 0
        self.loop_count = 0
        self.function_calls = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition to count parameters."""
        self.parameter_count = len(node.args.args)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Visit if statements for complexity calculation."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting_depth
        self.branch_count += 1

        self.current_nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_nesting_depth)

        self.generic_visit(node)

        self.current_nesting_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        """Visit for loops for complexity calculation."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting_depth
        self.loop_count += 1

        self.current_nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_nesting_depth)

        self.generic_visit(node)

        self.current_nesting_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        """Visit while loops for complexity calculation."""
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting_depth
        self.loop_count += 1

        self.current_nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_nesting_depth)

        self.generic_visit(node)

        self.current_nesting_depth -= 1

    def visit_Try(self, node: ast.Try) -> None:
        """Visit try statements for complexity calculation."""
        self.cyclomatic_complexity += len(node.handlers)
        self.branch_count += len(node.handlers)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Visit boolean operations for cognitive complexity."""
        # Logical operators add cognitive complexity
        self.cognitive_complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Visit return statements."""
        self.return_count += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls."""
        self.function_calls += 1
        self.generic_visit(node)
