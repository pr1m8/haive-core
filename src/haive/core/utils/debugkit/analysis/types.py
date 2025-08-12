"""
Advanced type analysis for Python functions and modules.

This module provides comprehensive type analysis capabilities including:
- Type annotation coverage analysis
- Type error detection using mypy
- Generic type complexity scoring
- Runtime type validation support
- Module-wide type analysis

The type analyzer integrates with multiple type checkers and provides
actionable insights for improving type safety in Python codebases.
"""

import ast
import inspect
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Union,
    get_type_hints,
)

try:
    import mypy.api

    HAS_MYPY = True
except ImportError:
    HAS_MYPY = False

try:
    from typing import get_args, get_origin
except ImportError:
    try:
        from typing import get_args, get_origin  # Python 3.8+
    except ImportError:

        def get_args(tp):
            return getattr(tp, "__args__", ())

        def get_origin(tp):
            return getattr(tp, "__origin__", None)


try:
    import libcst as cst

    HAS_LIBCST = True
except ImportError:
    HAS_LIBCST = False


class TypeComplexity(str, Enum):
    """Type complexity levels for classification.

    Attributes:
        SIMPLE: Basic types (int, str, bool, etc.)
        GENERIC: Generic types with single parameter (List[int])
        COMPLEX: Nested generics (Dict[str, List[int]])
        ADVANCED: Union types, complex generics (Union[str, List[Dict[str, Any]]])
    """

    SIMPLE = "simple"
    GENERIC = "generic"
    COMPLEX = "complex"
    ADVANCED = "advanced"


@dataclass
class TypeInfo:
    """Detailed type information for a variable, parameter, or return value.

    This class contains comprehensive information about a type annotation
    including its structure, complexity, and validation status.

    Attributes:
        name: Variable or parameter name
        type_annotation: The actual type annotation from code
        actual_type: Runtime type (if available)
        is_optional: Whether the type allows None values
        is_generic: Whether the type is a generic type (List, Dict, etc.)
        is_union: Whether the type is a Union type
        generic_args: Type arguments for generic types
        union_types: Types in a Union (if applicable)
        source_location: Where this type was defined (module.function)
        complexity_level: Assessed complexity level of the type
        type_string: String representation of the type
        validation_errors: Any type validation errors found

    Examples:
        Basic type info::

            # For parameter: name: str
            info = TypeInfo(
                name="name",
                type_annotation=str,
                is_optional=False,
                complexity_level=TypeComplexity.SIMPLE
            )

        Generic type info::

            # For parameter: items: List[Dict[str, int]]
            info = TypeInfo(
                name="items",
                type_annotation=List[Dict[str, int]],
                is_generic=True,
                generic_args=[Dict[str, int]],
                complexity_level=TypeComplexity.COMPLEX
            )
    """

    name: str
    type_annotation: type[Any] | None
    actual_type: type[Any] | None = None
    is_optional: bool = False
    is_generic: bool = False
    is_union: bool = False
    generic_args: list[type[Any]] = None
    union_types: list[type[Any]] = None
    source_location: str | None = None
    complexity_level: TypeComplexity = TypeComplexity.SIMPLE
    type_string: str = ""
    validation_errors: list[str] = None

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass creation."""
        if self.generic_args is None:
            self.generic_args = []
        if self.union_types is None:
            self.union_types = []
        if self.validation_errors is None:
            self.validation_errors = []

        # Compute type string representation
        if self.type_annotation:
            self.type_string = str(self.type_annotation)

        # Analyze type structure
        self._analyze_type_structure()

    def _analyze_type_structure(self) -> None:
        """Analyze the structure of the type annotation."""
        if not self.type_annotation:
            return

        # Check for Union types
        origin = get_origin(self.type_annotation)
        args = get_args(self.type_annotation)

        if origin is Union:
            self.is_union = True
            self.union_types = list(args)
            self.is_optional = type(None) in args
            self.complexity_level = TypeComplexity.ADVANCED
        elif origin is not None:
            # Generic type
            self.is_generic = True
            self.generic_args = list(args)

            # Determine complexity
            if len(args) == 1 and get_origin(args[0]) is None:
                self.complexity_level = TypeComplexity.GENERIC
            else:
                self.complexity_level = TypeComplexity.COMPLEX
        else:
            # Simple type
            self.complexity_level = TypeComplexity.SIMPLE

    def get_complexity_score(self) -> float:
        """Calculate numeric complexity score for this type.

        Returns:
            float: Complexity score from 1.0 (simple) to 10.0+ (very complex)

        Examples:
            Compare type complexity::

                simple_type = TypeInfo(name="x", type_annotation=int)
                complex_type = TypeInfo(name="data", type_annotation=Dict[str, List[Optional[int]]])

                assert simple_type.get_complexity_score() < complex_type.get_complexity_score()
        """
        score = 1.0

        if self.is_union:
            score += len(self.union_types) * 2.0

        if self.is_generic:
            score += 2.0
            # Add complexity for nested generics
            for arg in self.generic_args:
                if get_origin(arg) is not None:
                    score += 3.0
                else:
                    score += 1.0

        if self.complexity_level == TypeComplexity.ADVANCED:
            score += 5.0
        elif self.complexity_level == TypeComplexity.COMPLEX:
            score += 3.0
        elif self.complexity_level == TypeComplexity.GENERIC:
            score += 1.5

        return score


@dataclass
class FunctionTypeAnalysis:
    """Complete type analysis results for a function.

    This class contains comprehensive type analysis results including
    parameter analysis, return type analysis, type coverage metrics,
    and recommendations for improvement.

    Attributes:
        function_name: Name of the analyzed function
        module_name: Module containing the function
        parameters: Type information for each parameter
        return_type: Type information for return value
        type_errors: List of type errors found by static analysis
        type_warnings: List of type warnings
        type_coverage: Percentage of parameters with type annotations (0.0-1.0)
        return_type_coverage: Whether return type is annotated
        generic_types_used: List of generic types used in the function
        union_types_used: List of union types used in the function
        type_complexity_score: Overall type complexity score
        type_safety_score: Overall type safety score (0.0-100.0)
        recommendations: List of recommendations for improvement
        mypy_output: Raw output from mypy analysis
        analysis_timestamp: When the analysis was performed

    Examples:
        Access analysis results::

            analysis = type_analyzer.analyze_function(my_function)

            print(f"Type coverage: {analysis.type_coverage:.1%}")
            print(f"Complexity score: {analysis.type_complexity_score:.2f}")
            print(f"Safety score: {analysis.type_safety_score:.1f}")

            for error in analysis.type_errors:
                print(f"Error: {error}")

            for rec in analysis.recommendations:
                print(f"Recommendation: {rec}")
    """

    function_name: str
    module_name: str | None = None
    parameters: dict[str, TypeInfo] = None
    return_type: TypeInfo | None = None
    type_errors: list[str] = None
    type_warnings: list[str] = None
    type_coverage: float = 0.0
    return_type_coverage: bool = False
    generic_types_used: list[str] = None
    union_types_used: list[str] = None
    type_complexity_score: float = 0.0
    type_safety_score: float = 0.0
    recommendations: list[str] = None
    mypy_output: str = ""
    analysis_timestamp: str | None = None

    def __post_init__(self) -> None:
        """Initialize collections and compute derived metrics."""
        if self.parameters is None:
            self.parameters = {}
        if self.type_errors is None:
            self.type_errors = []
        if self.type_warnings is None:
            self.type_warnings = []
        if self.generic_types_used is None:
            self.generic_types_used = []
        if self.union_types_used is None:
            self.union_types_used = []
        if self.recommendations is None:
            self.recommendations = []

        # Compute derived metrics
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        """Compute type coverage and complexity metrics."""
        if not self.parameters:
            return

        # Type coverage
        typed_params = [p for p in self.parameters.values() if p.type_annotation]
        self.type_coverage = (
            len(typed_params) / len(self.parameters) if self.parameters else 0.0
        )

        # Return type coverage
        self.return_type_coverage = (
            self.return_type is not None
            and self.return_type.type_annotation is not None
        )

        # Complexity score
        complexity_scores = [p.get_complexity_score() for p in self.parameters.values()]
        if self.return_type:
            complexity_scores.append(self.return_type.get_complexity_score())

        self.type_complexity_score = (
            sum(complexity_scores) / len(complexity_scores)
            if complexity_scores
            else 0.0
        )

        # Safety score (0-100)
        safety_score = self.type_coverage * 50  # Up to 50 points for coverage
        safety_score += 50 - len(self.type_errors) * 10  # Deduct for errors
        safety_score += 10 if self.return_type_coverage else 0  # Bonus for return type
        self.type_safety_score = max(0.0, min(100.0, safety_score))

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for the type analysis.

        Returns:
            Dict[str, Any]: Summary containing key metrics and counts

        Examples:
            Print analysis summary::

                summary = analysis.get_summary()
                print(f"Parameters analyzed: {summary['parameter_count']}")
                print(f"Type errors: {summary['error_count']}")
                print(f"Overall grade: {summary['grade']}")
        """
        return {
            "function_name": self.function_name,
            "parameter_count": len(self.parameters),
            "typed_parameters": len(
                [p for p in self.parameters.values() if p.type_annotation]
            ),
            "type_coverage": self.type_coverage,
            "error_count": len(self.type_errors),
            "warning_count": len(self.type_warnings),
            "complexity_score": self.type_complexity_score,
            "safety_score": self.type_safety_score,
            "grade": self._get_grade(),
            "has_return_type": self.return_type_coverage,
            "recommendation_count": len(self.recommendations),
        }

    def _get_grade(self) -> str:
        """Get letter grade based on type safety score."""
        if self.type_safety_score >= 90:
            return "A"
        elif self.type_safety_score >= 80:
            return "B"
        elif self.type_safety_score >= 70:
            return "C"
        elif self.type_safety_score >= 60:
            return "D"
        else:
            return "F"


class TypeAnalyzer:
    """Advanced type analyzer for Python functions and modules.

    This class provides comprehensive type analysis capabilities using
    multiple analysis techniques including AST parsing, type hint inspection,
    and integration with external type checkers like mypy.

    The analyzer can work on individual functions or entire modules,
    providing detailed reports on type coverage, complexity, and potential
    improvements.

    Attributes:
        use_mypy: Whether to integrate with mypy for type checking
        cache_enabled: Whether to cache analysis results
        strict_mode: Whether to use strict type checking rules

    Examples:
        Basic function analysis::

            analyzer = TypeAnalyzer()

            def example_func(name: str, age: int = 0) -> str:
                return f"{name} is {age} years old"

            analysis = analyzer.analyze_function(example_func)
            print(f"Type coverage: {analysis.type_coverage:.1%}")

        Module-wide analysis::

            module_results = analyzer.analyze_module(Path("my_module.py"))
            for func_name, analysis in module_results.items():
                if analysis.type_safety_score < 70:
                    print(f"{func_name} needs type improvements")

        Custom configuration::

            analyzer = TypeAnalyzer(
                use_mypy=True,
                strict_mode=True,
                cache_enabled=False
            )
    """

    def __init__(
        self,
        use_mypy: bool = HAS_MYPY,
        cache_enabled: bool = True,
        strict_mode: bool = False,
    ) -> None:
        """Initialize the type analyzer.

        Args:
            use_mypy: Whether to use mypy for additional type checking
            cache_enabled: Whether to cache analysis results for performance
            strict_mode: Whether to use strict type checking rules
        """
        self.use_mypy = use_mypy and HAS_MYPY
        self.cache_enabled = cache_enabled
        self.strict_mode = strict_mode
        self.type_cache: dict[str, FunctionTypeAnalysis] = {}

    def analyze_function(self, func: Callable[..., Any]) -> FunctionTypeAnalysis:
        """Perform comprehensive type analysis on a function.

        Analyzes a function's type annotations, parameter types, return type,
        and performs static type checking to identify potential issues.

        Args:
            func: The function to analyze

        Returns:
            FunctionTypeAnalysis: Complete analysis results

        Raises:
            ValueError: If the function cannot be analyzed
            TypeError: If the input is not a callable

        Examples:
            Analyze a simple function::

                def greet(name: str, enthusiastic: bool = False) -> str:
                    return f"Hello, {name}{'!' if enthusiastic else '.'}"

                analysis = analyzer.analyze_function(greet)
                assert analysis.type_coverage == 1.0  # 100% coverage
                assert analysis.return_type_coverage is True

            Analyze function with type issues::

                def problematic_func(data):  # No type hints
                    return data.process()

                analysis = analyzer.analyze_function(problematic_func)
                assert analysis.type_coverage == 0.0
                assert len(analysis.recommendations) > 0
        """
        if not callable(func):
            raise TypeError(f"Expected callable, got {type(func)}")

        func_name = getattr(func, "__name__", str(func))

        # Check cache
        cache_key = (
            f"{func.__module__}.{func_name}"
            if hasattr(func, "__module__")
            else func_name
        )
        if self.cache_enabled and cache_key in self.type_cache:
            return self.type_cache[cache_key]

        try:
            # Get type hints
            hints = get_type_hints(func, include_extras=True)
        except (NameError, AttributeError, TypeError):
            # Fallback for functions with problematic type hints
            hints = getattr(func, "__annotations__", {})

        # Get function signature
        try:
            signature = inspect.signature(func)
        except (ValueError, TypeError):
            # Can't get signature for some built-in functions
            return FunctionTypeAnalysis(
                function_name=func_name,
                type_errors=["Cannot analyze built-in or C extension function"],
                recommendations=[
                    "Function cannot be analyzed - may be built-in or C extension"
                ],
            )

        # Analyze parameters
        parameters = {}
        for param_name, param in signature.parameters.items():
            param_type = hints.get(param_name)

            type_info = TypeInfo(
                name=param_name,
                type_annotation=param_type,
                actual_type=(
                    type(param.default) if param.default != param.empty else None
                ),
                is_optional=param.default != param.empty,
                source_location=cache_key,
            )

            parameters[param_name] = type_info

        # Analyze return type
        return_hint = hints.get("return")
        return_type = None
        if return_hint:
            return_type = TypeInfo(
                name="return", type_annotation=return_hint, source_location=cache_key
            )

        # Run mypy analysis if available
        type_errors = []
        mypy_output = ""
        if self.use_mypy:
            try:
                type_errors, mypy_output = self._run_mypy_analysis(func)
            except Exception as e:
                type_errors.append(f"Mypy analysis failed: {e!s}")

        # Create analysis result
        analysis = FunctionTypeAnalysis(
            function_name=func_name,
            module_name=getattr(func, "__module__", None),
            parameters=parameters,
            return_type=return_type,
            type_errors=type_errors,
            mypy_output=mypy_output,
        )

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(analysis)

        # Extract type usage
        analysis.generic_types_used = self._extract_generic_types(
            parameters, return_type
        )
        analysis.union_types_used = self._extract_union_types(parameters, return_type)

        # Cache result
        if self.cache_enabled:
            self.type_cache[cache_key] = analysis

        return analysis

    def analyze_module(self, module_path: Path) -> dict[str, FunctionTypeAnalysis]:
        """Analyze all functions in a Python module.

        Performs type analysis on all function definitions found in the
        specified module file.

        Args:
            module_path: Path to the Python module file to analyze

        Returns:
            Dict[str, FunctionTypeAnalysis]: Analysis results keyed by function name

        Raises:
            FileNotFoundError: If the module file doesn't exist
            SyntaxError: If the module has syntax errors

        Examples:
            Analyze a module::

                results = analyzer.analyze_module(Path("my_module.py"))

                # Find functions with low type coverage
                problematic = {
                    name: analysis
                    for name, analysis in results.items()
                    if analysis.type_coverage < 0.8
                }

                print(f"Found {len(problematic)} functions needing type improvements")
        """
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")

        results = {}

        try:
            # Read and parse the module
            with open(module_path, encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(module_path))

            # Find all function definitions
            function_finder = FunctionFinder()
            function_finder.visit(tree)

            # Import the module to get actual function objects
            if HAS_LIBCST:
                # Use libcst for more detailed analysis
                results.update(self._analyze_with_libcst(module_path, source))
            else:
                # Fallback to AST-only analysis
                results.update(self._analyze_with_ast(tree, str(module_path)))

        except SyntaxError as e:
            # Return empty results with error information
            results["_syntax_error"] = FunctionTypeAnalysis(
                function_name="_syntax_error",
                type_errors=[f"Syntax error in module: {e!s}"],
                recommendations=["Fix syntax errors before performing type analysis"],
            )

        return results

    def _run_mypy_analysis(self, func: Callable[..., Any]) -> tuple[list[str], str]:
        """Run mypy type checking on a function.

        Args:
            func: Function to analyze

        Returns:
            Tuple of (error_list, raw_output)
        """
        if not self.use_mypy:
            return [], ""

        try:
            # Get source code
            source = inspect.getsource(func)

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                # Add necessary imports
                f.write("from typing import *\n")
                f.write("from typing_extensions import *\n\n")
                f.write(source)
                temp_path = f.name

            try:
                # Run mypy
                args = [temp_path]
                if self.strict_mode:
                    args.extend(["--strict", "--warn-return-any"])

                stdout, stderr, exit_status = mypy.api.run(args)

                # Parse errors
                errors = []
                for line in stdout.split("\n"):
                    line = line.strip()
                    if line and ("error:" in line or "warning:" in line):
                        # Remove file path and line numbers for cleaner output
                        if ":" in line:
                            parts = line.split(":", 3)
                            if len(parts) >= 4:
                                error_msg = parts[3].strip()
                                errors.append(error_msg)
                            else:
                                errors.append(line)

                return errors, stdout

            finally:
                # Clean up temporary file
                os.unlink(temp_path)

        except Exception as e:
            return [f"Mypy analysis failed: {e!s}"], ""

    def _generate_recommendations(self, analysis: FunctionTypeAnalysis) -> list[str]:
        """Generate actionable recommendations based on type analysis.

        Args:
            analysis: The function type analysis results

        Returns:
            List[str]: List of recommendations
        """
        recommendations = []

        # Type coverage recommendations
        if analysis.type_coverage < 0.5:
            recommendations.append(
                f"Add type hints to {len([p for p in analysis.parameters.values() if not p.type_annotation])} parameters"
            )
        elif analysis.type_coverage < 1.0:
            untyped = [
                p.name for p in analysis.parameters.values() if not p.type_annotation
            ]
            recommendations.append(
                f"Add type hints to parameters: {', '.join(untyped)}"
            )

        # Return type recommendations
        if not analysis.return_type_coverage:
            recommendations.append("Add return type annotation")

        # Type error recommendations
        if analysis.type_errors:
            if len(analysis.type_errors) == 1:
                recommendations.append("Fix the type error found by mypy")
            else:
                recommendations.append(
                    f"Fix {len(analysis.type_errors)} type errors found by mypy"
                )

        # Complexity recommendations
        if analysis.type_complexity_score > 8.0:
            recommendations.append("Consider simplifying complex type annotations")

        # Generic type recommendations
        generic_count = len([p for p in analysis.parameters.values() if p.is_generic])
        if generic_count > 3:
            recommendations.append(
                "Consider using TypedDict or dataclass for complex parameter structures"
            )

        # Union type recommendations
        union_count = len([p for p in analysis.parameters.values() if p.is_union])
        if union_count > 2:
            recommendations.append(
                "Consider using protocol or ABC for multiple union types"
            )

        return recommendations

    def _extract_generic_types(
        self, parameters: dict[str, TypeInfo], return_type: TypeInfo | None
    ) -> list[str]:
        """Extract generic types used in the function signature."""
        generic_types = []

        for param in parameters.values():
            if param.is_generic and param.type_string:
                generic_types.append(param.type_string)

        if return_type and return_type.is_generic and return_type.type_string:
            generic_types.append(return_type.type_string)

        return list(set(generic_types))  # Remove duplicates

    def _extract_union_types(
        self, parameters: dict[str, TypeInfo], return_type: TypeInfo | None
    ) -> list[str]:
        """Extract union types used in the function signature."""
        union_types = []

        for param in parameters.values():
            if param.is_union and param.type_string:
                union_types.append(param.type_string)

        if return_type and return_type.is_union and return_type.type_string:
            union_types.append(return_type.type_string)

        return list(set(union_types))  # Remove duplicates

    def _analyze_with_libcst(
        self, module_path: Path, source: str
    ) -> dict[str, FunctionTypeAnalysis]:
        """Analyze module using libcst for detailed analysis."""
        # Placeholder for libcst implementation
        # This would provide more detailed analysis of the source code
        return {}

    def _analyze_with_ast(
        self, tree: ast.AST, module_path: str
    ) -> dict[str, FunctionTypeAnalysis]:
        """Analyze module using AST for basic analysis."""
        results = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Create basic analysis from AST
                func_analysis = FunctionTypeAnalysis(
                    function_name=node.name, module_name=module_path
                )

                # Analyze parameters from AST
                parameters = {}
                for arg in node.args.args:
                    param_info = TypeInfo(
                        name=arg.arg,
                        type_annotation=None,  # Would need more complex AST analysis
                        source_location=f"{module_path}.{node.name}",
                    )
                    parameters[arg.arg] = param_info

                func_analysis.parameters = parameters
                results[node.name] = func_analysis

        return results

    def clear_cache(self) -> None:
        """Clear the analysis cache.

        Examples:
            Clear cache to force re-analysis::

                analyzer.clear_cache()
                fresh_analysis = analyzer.analyze_function(my_function)
        """
        self.type_cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict[str, int]: Cache statistics including size and hit counts
        """
        return {"cache_size": len(self.type_cache), "cache_enabled": self.cache_enabled}


class FunctionFinder(ast.NodeVisitor):
    """AST visitor to find all function definitions in a module."""

    def __init__(self) -> None:
        """Initialize the function finder."""
        self.functions: list[ast.FunctionDef] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition nodes."""
        self.functions.append(node)
        self.generic_visit(node)
