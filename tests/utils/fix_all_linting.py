#!/usr/bin/env python3
"""Comprehensive linting fix script for all Haive packages.

This script runs trunk check and fix across all packages to ensure
consistent code quality across the entire codebase.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def find_packages() -> List[Path]:
    """Find all package directories with trunk configurations."""
    packages_dir = Path("../../..")  # From tests/utils back to packages
    packages = []

    if packages_dir.exists():
        for item in packages_dir.iterdir():
            if item.is_dir() and item.name.startswith("haive-"):
                trunk_config = item / ".trunk" / "trunk.yaml"
                if trunk_config.exists():
                    packages.append(item)

    return sorted(packages)


def run_trunk_command(package_dir: Path, command: List[str]) -> Tuple[int, str, str]:
    """Run a trunk command in the specified package directory."""
    try:
        result = subprocess.run(
            command,
            cwd=package_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out after 5 minutes"
    except Exception as e:
        return 1, "", f"Error running command: {e}"


def fix_package_linting(package_dir: Path) -> Dict[str, any]:
    """Fix linting issues in a single package."""
    package_name = package_dir.name
    print(f"\n🔧 Fixing {package_name}...")

    results = {
        "package": package_name,
        "path": str(package_dir),
        "check_before": None,
        "format_applied": False,
        "check_after": None,
        "issues_fixed": 0,
        "remaining_issues": 0,
        "success": False,
    }

    # Step 1: Check current issues
    print("  📊 Checking current issues..."..")
    returncode, stdout, stderr = run_trunk_command(
        package_dir, ["trunk", "check", "--no-fix", "--ci"]
    )
    results["check_before"] = {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }

    initial_issues = count_issues_from_output(stdout)
    print(f"  📝 Found {initial_issues} initial issues")

    # Step 2: Apply automatic fixes
    print("  🛠️  Applying automatic fixes..."s...")
    returncode, stdout, stderr = run_trunk_command(
        package_dir, ["trunk", "fmt", "--all"]
    )

    if returncode == 0:
        results["format_applied"] = True
        print("  ✅ Formatting applied successfully"y")
    else:
        print(f"  ⚠️  Formatting had issues: {stderr}")

    # Step 3: Run fix for remaining issues
    print("  🔨 Running trunk check with auto-fix..."..")
    returncode, stdout, stderr = run_trunk_command(
        package_dir, ["trunk", "check", "--fix", "--all"]
    )

    # Step 4: Final check
    print("  🏁 Final validation..."..")
    returncode, stdout, stderr = run_trunk_command(
        package_dir, ["trunk", "check", "--no-fix", "--ci"]
    )

    results["check_after"] = {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }

    final_issues = count_issues_from_output(stdout)
    results["issues_fixed"] = max(0, initial_issues - final_issues)
    results["remaining_issues"] = final_issues
    results["success"] = returncode == 0

    print(f"  📈 Issues fixed: {results['issues_fixed']}")
    print(f"  📋 Remaining issues: {results['remaining_issues']}")
    print(
        f"  {'✅' if results['success'] else '❌'} Final status: {'CLEAN' if results['success'] else 'HAS ISSUES'}"
    )

    return results


def count_issues_from_output(output: str) -> int:
    """Count the number of linting issues from trunk output."""
    if not output:
        return 0

    # Look for patterns like "✖ X new lint issues"
    lines = output.split("\n")
    for line in lines:
        if "lint issues" in line and "✖" in line:
            # Extract number
            parts = line.split()
            for i, part in enumerate(parts):
                if part.isdigit() and i < len(parts) - 1:
                    if "lint" in parts[i + 1]:
                        return int(part)

    # If no issues found in summary, assume 0
    if "✔ No issues" in output or "Checked" in output and "✔" in output:
        return 0

    # Count individual issue lines as fallback
    issue_count = 0
    for line in lines:
        if ":high" in line or ":medium" in line or ":low" in line:
            issue_count += 1

    return issue_count


def generate_report(all_results: List[Dict[str, any]]) -> None:
    """Generate a comprehensive report of all linting fixes."""
    print("\n" + "=" * 80)
    print("🎯 LINTING FIX REPORT"RT")
    print("=" * 80)

    total_packages = len(all_results)
    successful_packages = sum(1 for r in all_results if r["success"])
    total_issues_fixed = sum(r["issues_fixed"] for r in all_results)
    total_remaining = sum(r["remaining_issues"] for r in all_results)

    print("📊 SUMMARY:"Y:")
    print(f"  • Total packages processed: {total_packages}")
    print(f"  • Successfully cleaned: {successful_packages}")
    print(f"  • Packages with remaining issues: {total_packages - successful_packages}")
    print(f"  • Total issues fixed: {total_issues_fixed}")
    print(f"  • Total remaining issues: {total_remaining}")

    print("\n📋 PACKAGE DETAILS:"S:")
    for result in all_results:
        status_icon = "✅" if result["success"] else "❌"
        print(
            f"  {status_icon} {result['package']:<20} "
            f"Fixed: {result['issues_fixed']:<3} "
            f"Remaining: {result['remaining_issues']:<3}"
        )

    if total_remaining > 0:
        print("\n⚠️  PACKAGES WITH REMAINING ISSUES:"ES:")
        for result in all_results:
            if not result["success"] and result["remaining_issues"] > 0:
                print(f"  • {result['package']}: {result['remaining_issues']} issues")
                # Show some sample issues
                if result["check_after"] and result["check_after"]["stdout"]:
                    lines = result["check_after"]["stdout"].split("\n")[:5]
                    for line in lines:
                        if ":high" in line or ":medium" in line:
                            print(f"    - {line.strip()}")

    print("\n🎉 NEXT STEPS:"S:")
    if total_remaining == 0:
        print("  ✅ All packages are now lint-free!"!")
    else:
        print("  🔍 Review remaining issues in packages listed above"ve")
        print("  🛠️  Run 'trunk check' in specific packages for details"ails")
        print("  📝 Some issues may require manual fixes"es")


def main():
    """Main script execution."""
    print("🚀 Haive Codebase Linting Fix Tool")
    print("=" * 50)

    # Find all packages
    packages = find_packages()
    if not packages:
        print("❌ No packages with trunk configurations found!")
        sys.exit(1)

    print(f"📦 Found {len(packages)} packages to process:")
    for pkg in packages:
        print(f"  • {pkg.name}")

    # Ask for confirmation
    response = input("\n🤔 Proceed with fixing all packages? (y/N): ": ").strip().lower()
    if response not in ("y", "yes"):
        print("❌ Aborted by user")
        sys.exit(0)

    # Process each package
    all_results = []
    for package_dir in packages:
        result = fix_package_linting(package_dir)
        all_results.append(result)

    # Generate final report
    generate_report(all_results)


if __name__ == "__main__":
    main()
