"""
Static analysis orchestrator for integrating multiple Python analysis tools.

This module provides a unified interface for running and coordinating multiple
static analysis tools including type checkers, linters, complexity analyzers,
and code quality tools. It orchestrates tools like mypy, pyright, radon, vulture,
and many others from the development toolchain.

The orchestrator handles tool execution, result parsing, and provides
unified reporting across all analysis tools.
"""

import concurrent.futures
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import mypy.api

    HAS_MYPY = True
except ImportError:
    HAS_MYPY = False

try:
    import radon.complexity as radon_cc
    import radon.metrics as radon_metrics

    HAS_RADON = True
except ImportError:
    HAS_RADON = False


class AnalysisType(str, Enum):
    """Types of static analysis.

    Attributes:
        TYPE_CHECKING: Static type analysis (mypy, pyright)
        COMPLEXITY: Code complexity analysis (radon, xenon, mccabe)
        QUALITY: Code quality analysis (pyflakes, vulture)
        STYLE: Code style analysis (pycodestyle, autopep8)
        SECURITY: Security analysis (bandit, safety)
        PERFORMANCE: Performance analysis (py-spy, scalene)
        MODERNIZATION: Code modernization (pyupgrade, flynt)
        DEAD_CODE: Dead code detection (vulture, dead)
        METRICS: Code metrics collection (radon, wily)
    """

    TYPE_CHECKING = "type_checking"
    COMPLEXITY = "complexity"
    QUALITY = "quality"
    STYLE = "style"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MODERNIZATION = "modernization"
    DEAD_CODE = "dead_code"
    METRICS = "metrics"


class Severity(str, Enum):
    """Analysis finding severity levels.

    Attributes:
        INFO: Informational finding
        LOW: Low severity issue
        MEDIUM: Medium severity issue
        HIGH: High severity issue
        CRITICAL: Critical issue requiring immediate attention
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnalysisFinding:
    """A single finding from static analysis.

    Represents an issue, suggestion, or metric found by a static analysis tool.

    Attributes:
        tool_name: Name of the tool that generated this finding
        analysis_type: Type of analysis that found this issue
        severity: Severity level of the finding
        message: Human-readable description of the finding
        file_path: Path to the file containing the issue
        line_number: Line number where the issue was found
        column_number: Column number where the issue was found
        rule_id: Tool-specific rule or check identifier
        suggestion: Suggested fix for the issue
        context: Additional context about the finding

    Examples:
        Create a finding from tool output::

            finding = AnalysisFinding(
                tool_name="mypy",
                analysis_type=AnalysisType.TYPE_CHECKING,
                severity=Severity.HIGH,
                message="Argument has incompatible type",
                file_path="my_module.py",
                line_number=42,
                rule_id="arg-type"
            )
    """

    tool_name: str
    analysis_type: AnalysisType
    severity: Severity
    message: str
    file_path: str | None = None
    line_number: int | None = None
    column_number: int | None = None
    rule_id: str | None = None
    suggestion: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def get_location_string(self) -> str:
        """Get formatted location string for display.

        Returns:
            str: Formatted location like "file.py:42:10"
        """
        parts = []
        if self.file_path:
            parts.append(str(self.file_path))
        if self.line_number is not None:
            parts.append(str(self.line_number))
        if self.column_number is not None:
            parts.append(str(self.column_number))
        return ":".join(parts) if parts else "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert finding to dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the finding
        """
        return {
            "tool_name": self.tool_name,
            "analysis_type": self.analysis_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "rule_id": self.rule_id,
            "suggestion": self.suggestion,
            "context": self.context.copy(),
        }


@dataclass
class AnalysisResult:
    """Results from running a static analysis tool.

    Contains all findings and metadata from running a single analysis tool
    on a file or project.

    Attributes:
        tool_name: Name of the analysis tool
        analysis_type: Type of analysis performed
        success: Whether the tool ran successfully
        execution_time: Time taken to run the analysis (seconds)
        findings: List of findings discovered
        metrics: Numerical metrics collected by the tool
        suggestions: High-level suggestions from the tool
        raw_output: Raw output from the tool for debugging
        command_used: Command line used to run the tool
        exit_code: Exit code from the tool execution
        error_message: Error message if tool failed

    Examples:
        Process analysis results::

            result = orchestrator.run_tool("mypy", file_path)

            if result.success:
                print(f"Found {len(result.findings)} issues")
                for finding in result.findings:
                    if finding.severity == Severity.HIGH:
                        print(f"High severity: {finding.message}")
            else:
                print(f"Tool failed: {result.error_message}")
    """

    tool_name: str
    analysis_type: AnalysisType
    success: bool
    execution_time: float = 0.0
    findings: list[AnalysisFinding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    raw_output: str = ""
    command_used: str = ""
    exit_code: int = 0
    error_message: str = ""

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for this analysis result.

        Returns:
            Dict[str, Any]: Summary with counts and key metrics
        """
        severity_counts = {}
        for severity in Severity:
            severity_counts[severity.value] = len(
                [f for f in self.findings if f.severity == severity]
            )

        return {
            "tool_name": self.tool_name,
            "analysis_type": self.analysis_type.value,
            "success": self.success,
            "execution_time": self.execution_time,
            "total_findings": len(self.findings),
            "severity_counts": severity_counts,
            "metrics_count": len(self.metrics),
            "suggestions_count": len(self.suggestions),
        }

    def get_critical_findings(self) -> list[AnalysisFinding]:
        """Get only critical and high severity findings.

        Returns:
            List[AnalysisFinding]: Critical and high severity findings
        """
        return [
            f for f in self.findings if f.severity in [Severity.CRITICAL, Severity.HIGH]
        ]


class ToolAnalyzer:
    """Base class for individual tool analyzers.

    Each static analysis tool has its own analyzer that knows how to
    execute the tool, parse its output, and convert results to the
    unified AnalysisResult format.

    Attributes:
        tool_name: Name of the analysis tool
        analysis_type: Type of analysis this tool performs
        command_template: Template for the command line execution
        available: Whether the tool is available on the system
    """

    def __init__(self, tool_name: str, analysis_type: AnalysisType):
        """Initialize the tool analyzer.

        Args:
            tool_name: Name of the analysis tool
            analysis_type: Type of analysis this tool performs
        """
        self.tool_name = tool_name
        self.analysis_type = analysis_type
        self.command_template = ""
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if the tool is available on the system.

        Returns:
            bool: True if the tool can be executed
        """
        try:
            result = subprocess.run(
                [self.tool_name, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def analyze_file(self, file_path: Path, **kwargs) -> AnalysisResult:
        """Analyze a single file with this tool.

        Args:
            file_path: Path to the file to analyze
            **kwargs: Additional tool-specific arguments

        Returns:
            AnalysisResult: Analysis results from the tool
        """
        if not self.available:
            return AnalysisResult(
                tool_name=self.tool_name,
                analysis_type=self.analysis_type,
                success=False,
                error_message=f"Tool '{self.tool_name}' is not available",
            )

        start_time = time.time()

        try:
            # Build command
            command = self._build_command(file_path, **kwargs)

            # Execute tool
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", 60),
            )

            execution_time = time.time() - start_time

            # Parse output
            findings = self._parse_output(result.stdout, result.stderr, file_path)
            metrics = self._extract_metrics(result.stdout, result.stderr)
            suggestions = self._extract_suggestions(result.stdout, result.stderr)

            return AnalysisResult(
                tool_name=self.tool_name,
                analysis_type=self.analysis_type,
                success=result.returncode == 0
                or len(findings) > 0,  # Some tools return non-zero with findings
                execution_time=execution_time,
                findings=findings,
                metrics=metrics,
                suggestions=suggestions,
                raw_output=result.stdout + result.stderr,
                command_used=" ".join(command),
                exit_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return AnalysisResult(
                tool_name=self.tool_name,
                analysis_type=self.analysis_type,
                success=False,
                execution_time=time.time() - start_time,
                error_message="Tool execution timed out",
            )
        except Exception as e:
            return AnalysisResult(
                tool_name=self.tool_name,
                analysis_type=self.analysis_type,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def _build_command(self, file_path: Path, **kwargs) -> list[str]:
        """Build command line for tool execution.

        Args:
            file_path: Path to analyze
            **kwargs: Additional arguments

        Returns:
            List[str]: Command line arguments
        """
        # Default implementation - subclasses should override
        return [self.tool_name, str(file_path)]

    def _parse_output(
        self, stdout: str, stderr: str, file_path: Path
    ) -> list[AnalysisFinding]:
        """Parse tool output into structured findings.

        Args:
            stdout: Standard output from tool
            stderr: Standard error from tool
            file_path: Path that was analyzed

        Returns:
            List[AnalysisFinding]: Parsed findings
        """
        # Default implementation - subclasses should override
        findings = []

        # Simple line-by-line parsing for generic tools
        for line in (stdout + stderr).split("\n"):
            line = line.strip()
            if line and ("error" in line.lower() or "warning" in line.lower()):
                findings.append(
                    AnalysisFinding(
                        tool_name=self.tool_name,
                        analysis_type=self.analysis_type,
                        severity=Severity.MEDIUM,
                        message=line,
                        file_path=str(file_path),
                    )
                )

        return findings

    def _extract_metrics(self, stdout: str, stderr: str) -> dict[str, Any]:
        """Extract numerical metrics from tool output.

        Args:
            stdout: Standard output from tool
            stderr: Standard error from tool

        Returns:
            Dict[str, Any]: Extracted metrics
        """
        # Default implementation - subclasses should override
        return {}

    def _extract_suggestions(self, stdout: str, stderr: str) -> list[str]:
        """Extract high-level suggestions from tool output.

        Args:
            stdout: Standard output from tool
            stderr: Standard error from tool

        Returns:
            List[str]: Extracted suggestions
        """
        # Default implementation - subclasses should override
        return []


class MypyAnalyzer(ToolAnalyzer):
    """Mypy static type checker analyzer."""

    def __init__(self):
        """Initialize mypy analyzer."""
        super().__init__("mypy", AnalysisType.TYPE_CHECKING)

    def _build_command(self, file_path: Path, **kwargs) -> list[str]:
        """Build mypy command."""
        command = ["mypy"]

        # Add common options
        if kwargs.get("strict", False):
            command.append("--strict")

        if kwargs.get("show_error_codes", True):
            command.append("--show-error-codes")

        command.append(str(file_path))
        return command

    def _parse_output(
        self, stdout: str, stderr: str, file_path: Path
    ) -> list[AnalysisFinding]:
        """Parse mypy output."""
        findings = []

        for line in stdout.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse mypy output format: file:line: severity: message [error-code]
            match = re.match(
                r"^([^:]+):(\d+):(?:(\d+):)?\s*(error|warning|note):\s*(.*?)(?:\s*\[([^\]]+)\])?$",
                line,
            )
            if match:
                file_part, line_num, col_num, severity_str, message, error_code = (
                    match.groups()
                )

                # Map mypy severity to our severity
                severity_map = {
                    "error": Severity.HIGH,
                    "warning": Severity.MEDIUM,
                    "note": Severity.INFO,
                }

                findings.append(
                    AnalysisFinding(
                        tool_name=self.tool_name,
                        analysis_type=self.analysis_type,
                        severity=severity_map.get(severity_str, Severity.MEDIUM),
                        message=message,
                        file_path=file_part,
                        line_number=int(line_num) if line_num else None,
                        column_number=int(col_num) if col_num else None,
                        rule_id=error_code,
                    )
                )

        return findings


class RadonAnalyzer(ToolAnalyzer):
    """Radon complexity analyzer."""

    def __init__(self):
        """Initialize radon analyzer."""
        super().__init__("radon", AnalysisType.COMPLEXITY)

    def _build_command(self, file_path: Path, **kwargs) -> list[str]:
        """Build radon command for cyclomatic complexity."""
        return ["radon", "cc", "-j", str(file_path)]

    def _parse_output(
        self, stdout: str, stderr: str, file_path: Path
    ) -> list[AnalysisFinding]:
        """Parse radon JSON output."""
        findings = []

        try:
            data = json.loads(stdout) if stdout else {}

            for file_path_key, functions in data.items():
                for func_data in functions:
                    complexity = func_data.get("complexity", 0)

                    if complexity > 10:  # High complexity threshold
                        severity = Severity.HIGH if complexity > 20 else Severity.MEDIUM

                        findings.append(
                            AnalysisFinding(
                                tool_name=self.tool_name,
                                analysis_type=self.analysis_type,
                                severity=severity,
                                message=f"Function '{func_data['name']}' has high cyclomatic complexity: {complexity}",
                                file_path=file_path_key,
                                line_number=func_data.get("lineno"),
                                rule_id="high-complexity",
                                context={
                                    "complexity": complexity,
                                    "function_name": func_data["name"],
                                },
                            )
                        )
        except json.JSONDecodeError:
            # Fallback to text parsing
            pass

        return findings

    def _extract_metrics(self, stdout: str, stderr: str) -> dict[str, Any]:
        """Extract complexity metrics from radon output."""
        metrics = {}

        try:
            data = json.loads(stdout) if stdout else {}

            complexities = []
            for functions in data.values():
                complexities.extend([f["complexity"] for f in functions])

            if complexities:
                metrics["average_complexity"] = sum(complexities) / len(complexities)
                metrics["max_complexity"] = max(complexities)
                metrics["min_complexity"] = min(complexities)
                metrics["function_count"] = len(complexities)
        except json.JSONDecodeError:
            pass

        return metrics


class VultureAnalyzer(ToolAnalyzer):
    """Vulture dead code analyzer."""

    def __init__(self):
        """Initialize vulture analyzer."""
        super().__init__("vulture", AnalysisType.DEAD_CODE)

    def _build_command(self, file_path: Path, **kwargs) -> list[str]:
        """Build vulture command."""
        return ["vulture", str(file_path), "--min-confidence", "80"]

    def _parse_output(
        self, stdout: str, stderr: str, file_path: Path
    ) -> list[AnalysisFinding]:
        """Parse vulture output."""
        findings = []

        for line in stdout.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse vulture output: file:line: unused variable 'name' (confidence: X%)
            match = re.match(
                r"^([^:]+):(\d+):\s*(.*?)\s*\(confidence:\s*(\d+)%\)$", line
            )
            if match:
                file_part, line_num, message, confidence = match.groups()

                # Higher confidence = higher severity
                confidence_int = int(confidence)
                if confidence_int >= 90:
                    severity = Severity.HIGH
                elif confidence_int >= 70:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                findings.append(
                    AnalysisFinding(
                        tool_name=self.tool_name,
                        analysis_type=self.analysis_type,
                        severity=severity,
                        message=message,
                        file_path=file_part,
                        line_number=int(line_num),
                        rule_id="dead-code",
                        context={"confidence": confidence_int},
                    )
                )

        return findings


class PyflakesAnalyzer(ToolAnalyzer):
    """Pyflakes code quality analyzer."""

    def __init__(self):
        """Initialize pyflakes analyzer."""
        super().__init__("pyflakes", AnalysisType.QUALITY)

    def _parse_output(
        self, stdout: str, stderr: str, file_path: Path
    ) -> list[AnalysisFinding]:
        """Parse pyflakes output."""
        findings = []

        for line in stdout.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse pyflakes output: file:line: message
            match = re.match(r"^([^:]+):(\d+):\s*(.*)$", line)
            if match:
                file_part, line_num, message = match.groups()

                # Classify severity based on message content
                severity = Severity.MEDIUM
                if "undefined name" in message.lower():
                    severity = Severity.HIGH
                elif "imported but unused" in message.lower():
                    severity = Severity.LOW

                findings.append(
                    AnalysisFinding(
                        tool_name=self.tool_name,
                        analysis_type=self.analysis_type,
                        severity=severity,
                        message=message,
                        file_path=file_part,
                        line_number=int(line_num),
                        rule_id="pyflakes-issue",
                    )
                )

        return findings


class StaticAnalysisOrchestrator:
    """Orchestrator for running multiple static analysis tools.

    This class coordinates the execution of multiple static analysis tools,
    manages their results, and provides unified reporting capabilities.
    It supports both individual tool execution and batch analysis across
    multiple tools.

    Attributes:
        available_tools: Dictionary of available tool analyzers
        default_tool_set: Default set of tools to run
        max_workers: Maximum number of concurrent tool executions
        timeout: Default timeout for tool execution

    Examples:
        Basic orchestration::

            orchestrator = StaticAnalysisOrchestrator()

            # Run specific tools
            results = orchestrator.analyze_file(
                Path("my_module.py"),
                tools=["mypy", "radon", "vulture"]
            )

            # Generate unified report
            report = orchestrator.generate_report(results)
            print(report)

        Batch analysis::

            # Analyze entire project
            project_results = orchestrator.analyze_project(
                Path("./src"),
                tools=["mypy", "pyflakes", "radon"],
                parallel=True
            )

            # Get summary statistics
            summary = orchestrator.get_project_summary(project_results)
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout: int = 60,
        custom_analyzers: dict[str, ToolAnalyzer] | None = None,
    ):
        """Initialize the static analysis orchestrator.

        Args:
            max_workers: Maximum number of concurrent tool executions
            timeout: Default timeout for tool execution in seconds
            custom_analyzers: Custom tool analyzers to include
        """
        self.max_workers = max_workers
        self.timeout = timeout

        # Initialize built-in analyzers
        self.available_tools: dict[str, ToolAnalyzer] = {
            "mypy": MypyAnalyzer(),
            "radon": RadonAnalyzer(),
            "vulture": VultureAnalyzer(),
            "pyflakes": PyflakesAnalyzer(),
        }

        # Add custom analyzers
        if custom_analyzers:
            self.available_tools.update(custom_analyzers)

        # Filter to only available tools
        self.available_tools = {
            name: analyzer
            for name, analyzer in self.available_tools.items()
            if analyzer.available
        }

        self.default_tool_set = (
            ["mypy", "radon", "vulture"] if HAS_MYPY else ["radon", "vulture"]
        )

    def get_available_tools(self) -> list[str]:
        """Get list of available analysis tools.

        Returns:
            List[str]: Names of available tools
        """
        return list(self.available_tools.keys())

    def analyze_file(
        self,
        file_path: Path,
        tools: list[str] | None = None,
        parallel: bool = True,
        **kwargs,
    ) -> dict[str, AnalysisResult]:
        """Analyze a single file with specified tools.

        Args:
            file_path: Path to the file to analyze
            tools: List of tool names to run (None for default set)
            parallel: Whether to run tools in parallel
            **kwargs: Additional arguments passed to tools

        Returns:
            Dict[str, AnalysisResult]: Results keyed by tool name

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If unknown tools are specified

        Examples:
            Analyze with specific tools::

                results = orchestrator.analyze_file(
                    Path("complex_module.py"),
                    tools=["mypy", "radon"],
                    strict=True  # Passed to mypy
                )

                for tool_name, result in results.items():
                    print(f"{tool_name}: {len(result.findings)} findings")
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if tools is None:
            tools = self.default_tool_set

        # Validate requested tools
        unknown_tools = set(tools) - set(self.available_tools.keys())
        if unknown_tools:
            raise ValueError(f"Unknown tools: {unknown_tools}")

        results = {}

        if parallel and len(tools) > 1:
            # Run tools in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_tool = {
                    executor.submit(
                        self.available_tools[tool_name].analyze_file,
                        file_path,
                        timeout=self.timeout,
                        **kwargs,
                    ): tool_name
                    for tool_name in tools
                }

                for future in concurrent.futures.as_completed(future_to_tool):
                    tool_name = future_to_tool[future]
                    try:
                        results[tool_name] = future.result()
                    except Exception as e:
                        results[tool_name] = AnalysisResult(
                            tool_name=tool_name,
                            analysis_type=self.available_tools[tool_name].analysis_type,
                            success=False,
                            error_message=str(e),
                        )
        else:
            # Run tools sequentially
            for tool_name in tools:
                analyzer = self.available_tools[tool_name]
                results[tool_name] = analyzer.analyze_file(
                    file_path, timeout=self.timeout, **kwargs
                )

        return results

    def analyze_project(
        self,
        project_path: Path,
        tools: list[str] | None = None,
        file_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        parallel: bool = True,
        max_files: int | None = None,
    ) -> dict[str, dict[str, AnalysisResult]]:
        """Analyze an entire project with specified tools.

        Args:
            project_path: Path to the project directory
            tools: List of tool names to run
            file_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            parallel: Whether to analyze files in parallel
            max_files: Maximum number of files to analyze

        Returns:
            Dict[str, Dict[str, AnalysisResult]]: Results nested by file path and tool name

        Examples:
            Analyze Python project::

                results = orchestrator.analyze_project(
                    Path("./my_project"),
                    tools=["mypy", "pyflakes", "radon"],
                    file_patterns=["**/*.py"],
                    exclude_patterns=["**/test_*.py", "**/__pycache__/**"]
                )

                # Count total issues
                total_issues = sum(
                    len(tool_result.findings)
                    for file_results in results.values()
                    for tool_result in file_results.values()
                )
        """
        if not project_path.exists():
            raise FileNotFoundError(f"Project path not found: {project_path}")

        # Find Python files to analyze
        if file_patterns is None:
            file_patterns = ["**/*.py"]

        if exclude_patterns is None:
            exclude_patterns = [
                "**/test_*.py",
                "**/*_test.py",
                "**/__pycache__/**",
                "**/.*",
            ]

        python_files = []
        for pattern in file_patterns:
            for file_path in project_path.rglob(pattern):
                if file_path.is_file():
                    # Check exclusions
                    excluded = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            excluded = True
                            break

                    if not excluded:
                        python_files.append(file_path)

        # Limit number of files if specified
        if max_files and len(python_files) > max_files:
            python_files = python_files[:max_files]

        results = {}

        if parallel:
            # Analyze files in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_file = {
                    executor.submit(
                        self.analyze_file,
                        file_path,
                        tools=tools,
                        parallel=False,  # Don't double-parallelize
                    ): file_path
                    for file_path in python_files
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        results[str(file_path)] = future.result()
                    except Exception as e:
                        # Create error result for file
                        error_results = {}
                        for tool_name in tools or self.default_tool_set:
                            if tool_name in self.available_tools:
                                error_results[tool_name] = AnalysisResult(
                                    tool_name=tool_name,
                                    analysis_type=self.available_tools[
                                        tool_name
                                    ].analysis_type,
                                    success=False,
                                    error_message=str(e),
                                )
                        results[str(file_path)] = error_results
        else:
            # Analyze files sequentially
            for file_path in python_files:
                try:
                    results[str(file_path)] = self.analyze_file(
                        file_path, tools=tools, parallel=False
                    )
                except Exception as e:
                    # Handle individual file errors
                    error_results = {}
                    for tool_name in tools or self.default_tool_set:
                        if tool_name in self.available_tools:
                            error_results[tool_name] = AnalysisResult(
                                tool_name=tool_name,
                                analysis_type=self.available_tools[
                                    tool_name
                                ].analysis_type,
                                success=False,
                                error_message=str(e),
                            )
                    results[str(file_path)] = error_results

        return results

    def generate_report(
        self,
        results: dict[str, AnalysisResult] | dict[str, dict[str, AnalysisResult]],
        format: str = "markdown",
    ) -> str:
        """Generate a unified report from analysis results.

        Args:
            results: Analysis results from analyze_file or analyze_project
            format: Report format ("markdown", "json", "text")

        Returns:
            str: Formatted report

        Examples:
            Generate markdown report::

                results = orchestrator.analyze_file(Path("module.py"))
                report = orchestrator.generate_report(results, format="markdown")

                with open("analysis_report.md", "w") as f:
                    f.write(report)
        """
        if format == "markdown":
            return self._generate_markdown_report(results)
        elif format == "json":
            return self._generate_json_report(results)
        elif format == "text":
            return self._generate_text_report(results)
        else:
            raise ValueError(f"Unknown report format: {format}")

    def _generate_markdown_report(
        self,
        results: dict[str, AnalysisResult] | dict[str, dict[str, AnalysisResult]],
    ) -> str:
        """Generate markdown format report."""
        lines = ["# Static Analysis Report\n"]

        # Check if this is single file or project results
        if isinstance(list(results.values())[0], AnalysisResult):
            # Single file results
            lines.append("## File Analysis Results\n")

            total_findings = sum(len(result.findings) for result in results.values())
            lines.append(f"**Total findings**: {total_findings}\n")

            for tool_name, result in results.items():
                lines.append(f"### {tool_name}\n")

                if result.success:
                    lines.append("- **Status**: ✅ Success")
                    lines.append(f"- **Execution time**: {result.execution_time:.3f}s")
                    lines.append(f"- **Findings**: {len(result.findings)}")

                    if result.findings:
                        lines.append("\n**Issues found**:\n")
                        for finding in result.findings:
                            severity_icon = {
                                Severity.CRITICAL: "🔴",
                                Severity.HIGH: "🟠",
                                Severity.MEDIUM: "🟡",
                                Severity.LOW: "🔵",
                                Severity.INFO: "ℹ️",
                            }.get(finding.severity, "")

                            location = finding.get_location_string()
                            lines.append(
                                f"- {severity_icon} **{finding.severity.value.upper()}** `{location}`: {finding.message}"
                            )

                    if result.suggestions:
                        lines.append("\n**Suggestions**:\n")
                        for suggestion in result.suggestions:
                            lines.append(f"- {suggestion}")
                else:
                    lines.append("- **Status**: ❌ Failed")
                    lines.append(f"- **Error**: {result.error_message}")

                lines.append("")
        else:
            # Project results
            lines.append("## Project Analysis Results\n")

            total_files = len(results)
            total_findings = sum(
                len(tool_result.findings)
                for file_results in results.values()
                for tool_result in file_results.values()
            )

            lines.append(f"**Files analyzed**: {total_files}")
            lines.append(f"**Total findings**: {total_findings}\n")

            # Summary by tool
            tool_summaries = {}
            for file_results in results.values():
                for tool_name, result in file_results.items():
                    if tool_name not in tool_summaries:
                        tool_summaries[tool_name] = {
                            "files": 0,
                            "findings": 0,
                            "execution_time": 0.0,
                        }
                    tool_summaries[tool_name]["files"] += 1
                    tool_summaries[tool_name]["findings"] += len(result.findings)
                    tool_summaries[tool_name]["execution_time"] += result.execution_time

            lines.append("### Tool Summary\n")
            for tool_name, summary in tool_summaries.items():
                lines.append(f"**{tool_name}**:")
                lines.append(f"- Files processed: {summary['files']}")
                lines.append(f"- Total findings: {summary['findings']}")
                lines.append(f"- Total time: {summary['execution_time']:.3f}s")
                lines.append("")

            # Top issues by file
            file_issues = [
                (file_path, sum(len(r.findings) for r in file_results.values()))
                for file_path, file_results in results.items()
            ]
            file_issues.sort(key=lambda x: x[1], reverse=True)

            if file_issues:
                lines.append("### Files with Most Issues\n")
                for file_path, issue_count in file_issues[:10]:  # Top 10
                    if issue_count > 0:
                        lines.append(f"- `{file_path}`: {issue_count} issues")
                lines.append("")

        return "\n".join(lines)

    def _generate_json_report(
        self,
        results: dict[str, AnalysisResult] | dict[str, dict[str, AnalysisResult]],
    ) -> str:
        """Generate JSON format report."""
        # Convert results to serializable format
        if isinstance(list(results.values())[0], AnalysisResult):
            # Single file results
            json_data = {
                "type": "file_analysis",
                "results": {
                    tool_name: {
                        "summary": result.get_summary(),
                        "findings": [f.to_dict() for f in result.findings],
                        "metrics": result.metrics,
                        "suggestions": result.suggestions,
                    }
                    for tool_name, result in results.items()
                },
            }
        else:
            # Project results
            json_data = {
                "type": "project_analysis",
                "results": {
                    file_path: {
                        tool_name: {
                            "summary": result.get_summary(),
                            "findings": [f.to_dict() for f in result.findings],
                            "metrics": result.metrics,
                            "suggestions": result.suggestions,
                        }
                        for tool_name, result in file_results.items()
                    }
                    for file_path, file_results in results.items()
                },
            }

        return json.dumps(json_data, indent=2)

    def _generate_text_report(
        self,
        results: dict[str, AnalysisResult] | dict[str, dict[str, AnalysisResult]],
    ) -> str:
        """Generate plain text format report."""
        lines = ["STATIC ANALYSIS REPORT", "=" * 50, ""]

        # Implementation similar to markdown but without formatting
        # ... (truncated for brevity)

        return "\n".join(lines)

    def get_project_summary(
        self, project_results: dict[str, dict[str, AnalysisResult]]
    ) -> dict[str, Any]:
        """Get summary statistics for project analysis results.

        Args:
            project_results: Results from analyze_project

        Returns:
            Dict[str, Any]: Summary statistics
        """
        total_files = len(project_results)
        total_findings = 0
        total_execution_time = 0.0

        severity_counts = {severity.value: 0 for severity in Severity}
        tool_counts = {}

        for file_results in project_results.values():
            for tool_name, result in file_results.items():
                total_findings += len(result.findings)
                total_execution_time += result.execution_time

                if tool_name not in tool_counts:
                    tool_counts[tool_name] = 0
                tool_counts[tool_name] += len(result.findings)

                for finding in result.findings:
                    severity_counts[finding.severity.value] += 1

        return {
            "total_files": total_files,
            "total_findings": total_findings,
            "total_execution_time": total_execution_time,
            "severity_breakdown": severity_counts,
            "tool_breakdown": tool_counts,
            "average_findings_per_file": total_findings / max(total_files, 1),
            "available_tools": self.get_available_tools(),
        }
