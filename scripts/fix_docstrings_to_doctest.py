#!/usr/bin/env python3
"""Fix docstring formatting by converting markdown code blocks to doctest format."""

import ast
import logging
import re
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocstringFixer(ast.NodeTransformer):
    """Fix docstring formatting issues by converting to doctest format."""

    def __init__(self):
        self.changes = []

    def visit_Module(self, node):
        """Visit module to fix module-level docstrings."""
        self._fix_docstring(node, is_module=True)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        self._fix_docstring(node)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        self._fix_docstring(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        self._fix_docstring(node)
        self.generic_visit(node)
        return node

    def _fix_docstring(self, node, is_module=False):
        """Fix docstring for a node."""
        node_name = "Module" if is_module else getattr(node, "name", "Unknown")

        if not node.body:
            return

        first_stmt = node.body[0]
        if not isinstance(first_stmt, ast.Expr):
            return

        if not isinstance(first_stmt.value, ast.Constant):
            return

        docstring = first_stmt.value.value
        if not docstring or not isinstance(docstring, str):
            return

        # Fix the docstring
        fixed_docstring, changed = self.fix_docstring_issues(docstring)

        if changed:
            line_no = 1 if is_module else node.lineno
            first_stmt.value.value = fixed_docstring
            self.changes.append(
                {
                    "node": node_name,
                    "line": line_no,
                    "original": docstring,
                    "fixed": fixed_docstring,
                }
            )

    def fix_docstring_issues(self, docstring: str) -> tuple[str, bool]:
        """Convert markdown code blocks to doctest format."""
        original = docstring
        fixed = docstring

        # Pattern to match markdown code blocks in Example sections
        # Captures: prefix (Example/Usage:), description, indent, language, code content
        pattern = r"((?:Example|Examples|Usage):\s*\n)((?:\s*(?!>>>).*?\n)*?)(\s*)```(?:python|)\n(.*?)\n(\s*)```"

        def replacer(match):
            prefix = match.group(1)  # "Example:\n" or "Usage:\n"
            description = match.group(2)  # Any description lines
            indent1 = match.group(3)  # Indentation before ```
            code = match.group(4)  # The code block

            # Determine base indentation (usually 4 or 8 spaces)
            base_indent = len(indent1) if indent1 else 4

            # Build the fixed version with doctest format
            result = prefix

            # Add description if present (but skip for Usage sections - they usually don't need it)
            desc_lines = [line for line in description.split("\n") if line.strip()]
            if desc_lines and not prefix.strip().startswith("Usage"):
                for line in desc_lines:
                    stripped = line.strip()
                    if stripped:
                        result += f"{' ' * base_indent}{stripped}\n"
                result += "\n"  # Blank line before doctest

            # Convert code to doctest format
            code_lines = code.split("\n")

            # Process each line and convert to doctest
            in_class = False
            class_indent = 0

            for i, line in enumerate(code_lines):
                stripped = line.strip()

                if not stripped:
                    # Empty line
                    continue

                # Comments become regular doctest comments
                if stripped.startswith("#"):
                    result += f"{' ' * base_indent}>>> {line.strip()}\n"
                    continue

                # Detect class definitions
                if stripped.startswith("class "):
                    in_class = True
                    class_indent = len(line) - len(line.lstrip())
                    result += f"{' ' * base_indent}>>> {stripped}\n"
                    # Add ... for class body continuation
                    continue

                # Handle class body (methods, attributes)
                if in_class and line and len(line) - len(line.lstrip()) > class_indent:
                    result += f"{' ' * base_indent}... {line[class_indent+4:]}\n"
                    # Check if this ends the class (method def means more content)
                    if stripped.startswith("def "):
                        # Look ahead for method body
                        j = i + 1
                        while j < len(code_lines) and code_lines[j].strip():
                            next_line = code_lines[j]
                            if (
                                next_line.strip()
                                and len(next_line) - len(next_line.lstrip())
                                > class_indent + 4
                            ):
                                # Method body line
                                result += f"{' ' * base_indent}... {next_line[class_indent+8:]}\n"
                            j += 1
                    continue

                # Regular statements - check if it looks like it should have output
                if any(word in stripped for word in ["print(", "config =", "= "]):
                    if "=" in stripped and "print" not in stripped:
                        # Assignment statement
                        result += f"{' ' * base_indent}>>> {stripped}\n"
                    else:
                        # Statement that might produce output
                        result += f"{' ' * base_indent}>>> {stripped}\n"
                else:
                    # Other statements
                    result += f"{' ' * base_indent}>>> {stripped}\n"

                # End class if we're back to base indentation
                if in_class and line and len(line) - len(line.lstrip()) <= class_indent:
                    in_class = False

            return result.rstrip()

        # Apply markdown code block conversion
        if "```" in fixed:
            fixed = re.sub(pattern, replacer, fixed, flags=re.MULTILINE | re.DOTALL)

        # Fix inline backticks in Args/Returns sections (keep this from original script)
        lines = fixed.split("\n")
        in_section = None
        result_lines = []

        for line in lines:
            # Check if we're entering a section
            if line.strip() in [
                "Args:",
                "Arguments:",
                "Parameters:",
                "Returns:",
                "Return:",
                "Yields:",
                "Yield:",
                "Raises:",
                "Raise:",
                "Note:",
                "Notes:",
                "Warning:",
                "Warnings:",
            ]:
                in_section = line.strip()
            elif line.strip() and not line.startswith((" ", "\t")):
                # New section or end of sections
                in_section = None

            # Fix inline backticks in section content
            if in_section and line.strip() and line.startswith((" ", "\t")):
                # Don't remove backticks from RST roles like :class:`Name`
                if not re.search(r":[a-z]+:`[^`]+`", line):
                    # Replace standalone `text` with text
                    line = re.sub(r"(?<!:)`([^`]+)`", r"\1", line)

            result_lines.append(line)

        fixed = "\n".join(result_lines)

        return fixed, fixed != original


def main():
    """Test the docstring fixer on the secure_config.py file."""
    file_path = Path("src/haive/core/common/mixins/secure_config.py")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Testing docstring fixes on {file_path}")

    with open(file_path, encoding="utf-8") as f:
        original_content = f.read()

    try:
        tree = ast.parse(original_content)
        fixer = DocstringFixer()
        fixer.visit(tree)

        if not fixer.changes:
            print("No changes needed!")
            return

        print(f"\nFound {len(fixer.changes)} docstrings to fix:")

        for i, change in enumerate(fixer.changes, 1):
            print(f"\n=== Change {i}: {change['node']} at line {change['line']} ===")
            print("ORIGINAL:")
            print(
                change["original"][:800] + "..."
                if len(change["original"]) > 800
                else change["original"]
            )
            print("\nFIXED:")
            print(
                change["fixed"][:800] + "..."
                if len(change["fixed"]) > 800
                else change["fixed"]
            )

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
