#!/usr/bin/env python3
"""Comprehensive syntax error fixer for Haive codebase."""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple


def fix_common_syntax_errors():
    """Fix the most common syntax errors that prevent documentation build."""

    fixes_applied = []

    # List of files with known syntax errors from the pre-build checker
    files_to_fix = [
        # Critical ones that block documentation build
        "packages/haive-agents/src/haive/agents/supervisor/simple_test.py",
        "packages/haive-games/src/haive/games/chess/example_configurable_players.py",
        "packages/haive-agents/src/haive/agents/rag/db_rag/graph_db/example.py",
        "packages/haive-agents/src/haive/agents/document_modifiers/kg/kg_base/example.py",
        "packages/haive-agents/src/haive/agents/conversation/social_media/example.py",
        "packages/haive-agents/src/haive/agents/conversation/base/examples/basic_state_management.py",
    ]

    for file_path in files_to_fix:
        full_path = Path(file_path)
        if not full_path.exists():
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Fix 1: Unterminated string literals with extra quotes
            content = re.sub(r'(print\([^)]*"[^"]*")!"([^"]*")\)', r"\1\2)", content)

            # Fix 2: Malformed string concatenation
            content = re.sub(r'"([^"]*)"n]"', r'"\1"]', content)

            # Fix 3: Unterminated string literals with pass
            content = re.sub(r'pass"\)', r'pass")', content)

            # Fix 4: Unmatched braces
            content = re.sub(r'pass"\}', r'pass"}', content)

            # Fix 5: Fix imports with hyphens in paths
            content = re.sub(
                r"from haive-([^.]+)\.src\.haive\.", r"from haive.\1.", content
            )

            # Fix 6: Fix global declarations that come after usage
            if "global TQDM_AVAILABLE" in content:
                lines = content.split("\n")
                # Move global declaration to the top of the function
                for i, line in enumerate(lines):
                    if (
                        "def " in line
                        and "TQDM_AVAILABLE" in content[content.find(line) :]
                    ):
                        # Find the global declaration
                        for j in range(i, len(lines)):
                            if "global TQDM_AVAILABLE" in lines[j]:
                                global_line = lines.pop(j)
                                # Insert it right after the function definition
                                lines.insert(i + 1, "    " + global_line.strip())
                                break
                        break
                content = "\n".join(lines)

            # Only write if content changed
            if content != original_content:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
                fixes_applied.append(str(full_path))
                print(f"✅ Fixed syntax errors in {full_path}")

        except Exception as e:
            print(f"❌ Failed to fix {full_path}: {e}")

    return fixes_applied


def main():
    """Main function to run syntax fixes."""
    print("🔧 Fixing critical syntax errors for documentation build...")

    fixes = fix_common_syntax_errors()

    if fixes:
        print(f"\n✅ Applied fixes to {len(fixes)} files:")
        for fix in fixes:
            print(f"  - {fix}")
        print("\n🎉 Ready to try documentation build again!")
    else:
        print("\n✅ No syntax fixes needed or no files found to fix")

    return 0


if __name__ == "__main__":
    sys.exit(main())
