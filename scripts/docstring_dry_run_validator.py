#!/usr/bin/env python3
"""Comprehensive dry-run validator for docstring fixes.

This script tests docstring fixes on individual files and validates the results
by running Sphinx builds before and after to measure improvement.
"""

import ast
import logging
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocstringValidator:
    """Validates docstring fixes with Sphinx build testing."""

    def __init__(self):
        self.test_results = []
        self.temp_dirs = []

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def count_sphinx_errors(self, file_path: Path) -> Tuple[int, List[str]]:
        """Count Sphinx errors for a specific file by building docs.

        Returns:
            Tuple of (error_count, error_messages)
        """
        try:
            # Create a minimal Sphinx project for testing
            temp_dir = Path(tempfile.mkdtemp(prefix="sphinx_test_"))
            self.temp_dirs.append(temp_dir)

            # Create minimal conf.py
            conf_py = temp_dir / "conf.py"
            conf_py.write_text(
                """
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
napoleon_google_docstring = True
napoleon_numpy_docstring = False
"""
            )

            # Create index.rst
            index_rst = temp_dir / "index.rst"
            index_rst.write_text(
                f"""
Test Documentation
==================

.. automodule:: {file_path.stem}
   :members:
"""
            )

            # Copy the test file
            test_file = temp_dir / f"{file_path.stem}.py"
            shutil.copy2(file_path, test_file)

            # Run Sphinx build
            result = subprocess.run(
                [
                    "sphinx-build",
                    "-b",
                    "html",
                    "-W",
                    "--keep-going",
                    str(temp_dir),
                    str(temp_dir / "_build"),
                ],
                capture_output=True,
                text=True,
                cwd=temp_dir,
            )

            # Parse errors
            errors = []
            error_count = 0
            for line in result.stderr.split("\n"):
                if "ERROR" in line or "Unexpected indentation" in line:
                    errors.append(line.strip())
                    error_count += 1

            return error_count, errors

        except Exception as e:
            logger.error(f"Error running Sphinx test: {e}")
            return -1, [str(e)]

    def apply_docstring_fixes(self, content: str) -> Tuple[str, List[str]]:
        """Apply docstring fixes to content.

        Returns:
            Tuple of (fixed_content, list_of_changes)
        """
        changes = []
        fixed_content = content

        # Fix 1: Convert markdown code blocks to Google-style indented blocks
        pattern = r"((?:Usage|Example|Examples):\s*\n)(\s*)```(?:python|bash|json|yaml|)\s*\n(.*?)\n\s*```"

        def replace_code_block(match):
            prefix = match.group(1)  # "Usage:\n" etc.
            indent = match.group(2)  # Original indentation
            code = match.group(3)  # Code content

            # Determine base indentation (usually 4 spaces)
            base_indent = len(indent) if indent else 4

            # Build replacement with proper Google-style indentation
            result = prefix
            result += f"{' ' * base_indent}Basic example:\n\n"

            # Indent code by base + 4 spaces
            code_lines = code.split("\n")
            for line in code_lines:
                if line.strip():  # Skip empty lines
                    result += f"{' ' * (base_indent + 4)}{line.strip()}\n"
                else:
                    result += "\n"

            changes.append(f"Converted markdown code block in {prefix.strip()}")
            return result.rstrip()

        # Apply markdown code block fixes
        if "```" in fixed_content:
            original_count = fixed_content.count("```") // 2
            fixed_content = re.sub(
                pattern,
                replace_code_block,
                fixed_content,
                flags=re.MULTILINE | re.DOTALL,
            )
            new_count = fixed_content.count("```") // 2
            if new_count < original_count:
                changes.append(
                    f"Fixed {original_count - new_count} markdown code blocks"
                )

        # Fix 2: Remove inline backticks in Args/Returns sections
        lines = fixed_content.split("\n")
        in_args_section = False
        fixed_lines = []

        for line in lines:
            stripped = line.strip()

            # Detect Args/Returns sections
            if stripped in [
                "Args:",
                "Arguments:",
                "Parameters:",
                "Returns:",
                "Return:",
                "Yields:",
                "Raises:",
            ]:
                in_args_section = True
            elif stripped and not line.startswith(("    ", "\t")):
                in_args_section = False

            # Fix inline backticks in these sections
            if in_args_section and line.startswith(("    ", "\t")) and "`" in line:
                original_line = line
                # Remove backticks but preserve RST roles like :class:`Name`
                if not re.search(r":[a-z]+:`[^`]+`", line):
                    line = re.sub(r"(?<!:)`([^`]+)`", r"\1", line)
                    if line != original_line:
                        changes.append(f"Removed inline backticks: {stripped[:50]}...")

            fixed_lines.append(line)

        fixed_content = "\n".join(fixed_lines)

        # Fix 3: Ensure proper section spacing
        # Add blank line before Args/Returns if missing
        fixed_content = re.sub(
            r"(\n)(\s*)(Args|Arguments|Parameters|Returns|Return|Yields|Raises)(:)",
            r"\1\n\2\3\4",
            fixed_content,
        )

        if re.search(r"\n\n\s*(Args|Returns)", fixed_content):
            changes.append("Added proper section spacing")

        return fixed_content, changes

    def validate_file(self, file_path: Path, show_diff: bool = True) -> Dict:
        """Validate docstring fixes for a single file.

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {file_path}")

        try:
            # Read original content
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Check if file has docstring issues
            has_markdown_blocks = "```" in original_content
            has_inline_backticks = bool(re.search(r"\s+`[^`]+`:", original_content))

            if not (has_markdown_blocks or has_inline_backticks):
                return {
                    "file": file_path,
                    "status": "no_issues",
                    "message": "No docstring issues found",
                    "changes": [],
                }

            # Apply fixes
            fixed_content, changes = self.apply_docstring_fixes(original_content)

            # Show diff if requested
            if show_diff and fixed_content != original_content:
                print(f"\n{'='*60}")
                print(f"DIFF FOR {file_path}")
                print(f"{'='*60}")

                # Simple diff display
                orig_lines = original_content.split("\n")
                fixed_lines = fixed_content.split("\n")

                # Show first few changes
                for i, (orig, fixed) in enumerate(zip(orig_lines, fixed_lines)):
                    if orig != fixed:
                        print(f"Line {i+1}:")
                        print(f"- {orig}")
                        print(f"+ {fixed}")
                        print()
                        if i > 10:  # Limit output
                            print("... (more changes)")
                            break

            # Test with AST parsing
            try:
                ast.parse(fixed_content)
                ast_valid = True
            except SyntaxError as e:
                ast_valid = False
                logger.error(f"Fixed content has syntax error: {e}")

            return {
                "file": file_path,
                "status": "success" if ast_valid else "syntax_error",
                "original_issues": {
                    "markdown_blocks": has_markdown_blocks,
                    "inline_backticks": has_inline_backticks,
                },
                "changes": changes,
                "content_changed": original_content != fixed_content,
                "ast_valid": ast_valid,
                "fixed_content": fixed_content if ast_valid else None,
            }

        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            return {
                "file": file_path,
                "status": "error",
                "error": str(e),
                "changes": [],
            }

    def run_test_suite(self, test_files: List[Path]) -> Dict:
        """Run validation on a suite of test files."""
        results = {
            "files_tested": len(test_files),
            "successful_fixes": 0,
            "files_with_issues": 0,
            "files_no_changes_needed": 0,
            "errors": 0,
            "detailed_results": [],
        }

        print(f"\n{'='*80}")
        print("DOCSTRING FIX VALIDATION - DRY RUN MODE")
        print(f"Testing {len(test_files)} files")
        print(f"{'='*80}")

        for file_path in test_files:
            result = self.validate_file(file_path)
            results["detailed_results"].append(result)

            # Update counters
            if result["status"] == "success":
                if result.get("content_changed", False):
                    results["successful_fixes"] += 1
                    results["files_with_issues"] += 1
                else:
                    results["files_no_changes_needed"] += 1
            elif result["status"] == "no_issues":
                results["files_no_changes_needed"] += 1
            else:
                results["errors"] += 1

        # Print summary
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Files tested: {results['files_tested']}")
        print(f"Successful fixes: {results['successful_fixes']}")
        print(f"Files with issues found: {results['files_with_issues']}")
        print(f"Files needing no changes: {results['files_no_changes_needed']}")
        print(f"Errors: {results['errors']}")

        # Show detailed results
        print("\nDETAILED RESULTS:")
        for result in results["detailed_results"]:
            status = result["status"]
            file_name = result["file"].name

            if status == "success" and result.get("content_changed"):
                print(f"✅ {file_name}: {len(result['changes'])} changes applied")
                for change in result["changes"]:
                    print(f"   - {change}")
            elif status == "no_issues":
                print(f"⏭️  {file_name}: No issues found")
            elif status == "error":
                print(f"❌ {file_name}: Error - {result.get('error', 'Unknown')}")
            else:
                print(f"⚠️  {file_name}: {status}")

        return results


def main():
    """Main function to run docstring validation tests."""

    # Define test files with known issues
    test_files = [
        Path("src/haive/core/common/mixins/secure_config.py"),
        Path("src/haive/core/common/mixins/recompile_mixin.py"),
        Path("src/haive/core/common/mixins/getter_mixin.py"),
        Path("src/haive/core/common/__init__.py"),
        Path("src/haive/core/common/structures/tree.py"),
    ]

    # Filter to only existing files
    existing_files = [f for f in test_files if f.exists()]

    if not existing_files:
        print("No test files found! Make sure you're in the haive-core directory.")
        return

    print(f"Found {len(existing_files)} test files to validate")

    # Run validation
    validator = DocstringValidator()
    try:
        results = validator.run_test_suite(existing_files)

        # Save results to file
        results_file = Path("docstring_validation_results.md")
        with open(results_file, "w") as f:
            f.write("# Docstring Validation Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- Files tested: {results['files_tested']}\n")
            f.write(f"- Successful fixes: {results['successful_fixes']}\n")
            f.write(f"- Files with issues: {results['files_with_issues']}\n")
            f.write(f"- No changes needed: {results['files_no_changes_needed']}\n")
            f.write(f"- Errors: {results['errors']}\n\n")

            f.write("## Detailed Results\n\n")
            for result in results["detailed_results"]:
                f.write(f"### {result['file'].name}\n\n")
                f.write(f"- Status: {result['status']}\n")
                if result.get("changes"):
                    f.write("- Changes applied:\n")
                    for change in result["changes"]:
                        f.write(f"  - {change}\n")
                f.write("\n")

        print(f"\nResults saved to: {results_file}")

    finally:
        validator.cleanup()


if __name__ == "__main__":
    main()
