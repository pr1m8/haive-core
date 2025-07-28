# Haive-Core Package Fixes Guide

**Date**: 2025-07-28  
**Location**: `packages/haive-core/`  
**Purpose**: Fix all trunk issues in haive-core package  
**Status**: Ready for execution

## 🎯 Current Issue Summary

From `trunk check` in haive-core:

- **513 black formatting failures**
- **580 ruff lint issues**
- **91 security issues**
- **Various markdown lint issues**

## 📍 Where to Find Issues

### Run Trunk Check

```bash
# In packages/haive-core/
trunk check                    # See all issues
trunk check --show-existing    # Include existing issues
trunk check src/               # Focus on source code only
```

### Key Problem Files

```bash
# Most problematic areas from trunk output:
src/haive/core/utils/env_utils.py
src/haive/core/utils/haive_discovery/base_analyzer.py
src/haive/core/utils/haive_discovery/component_info.py
src/haive/core/utils/haive_discovery/discovery_engine.py
src/haive/core/utils/haive_discovery/documentation_writer.py
src/haive/core/utils/haive_discovery/retriever_analyzers.py
src/haive/core/utils/haive_discovery/tool_analyzers.py
src/haive/core/utils/haive_discovery/utils.py
src/haive/core/utils/pydantic_utils/ui.py
src/haive/core/utils/tools/tool_schema_generator.py
```

## 🔧 Tools to Use for Fixes

### 1. Trunk (Primary Tool - Already Configured)

```bash
# Auto-fix everything possible
trunk check --fix

# Fix specific tool issues
trunk check --fix --filter=black
trunk check --fix --filter=ruff
trunk check --fix --filter=isort

# Fix specific files
trunk check --fix src/haive/core/utils/env_utils.py
```

### 2. Direct Tool Usage (If Trunk Fails)

```bash
# Black formatting
poetry run black src/

# Ruff linting with auto-fixes
poetry run ruff check src/ --fix
poetry run ruff check src/ --unsafe-fixes

# isort import sorting
poetry run isort src/

# Manual bandit security check
poetry run bandit -r src/
```

### 3. Google-Style Docstring Formatting

```bash
# Better than docformatter - already installed
poetry run pydocstringformatter --write src/ --style google

# Validate Google style
poetry run pydocstyle src/ --convention=google
```

### 4. Custom Docstring Generation Scripts

```bash
# Auto-generate missing docstrings (if scripts exist)
python scripts/add_module_docstrings.py src/
python scripts/add_function_docstrings.py src/
python scripts/fix_docstring_formatting.py src/
```

### 5. Type Hint Generation

```bash
# MonkeyType for runtime type inference
MONKEYTYPE_TRACE=1 poetry run pytest tests/
poetry run monkeytype stub haive.core.utils
poetry run monkeytype apply haive.core.utils
```

## ✅ Step-by-Step Fix Process

### Phase 1: Quick Wins (Auto-fixable)

```bash
# 1. Take baseline measurement
echo "=== BEFORE FIXES ===" > fix_results.log
trunk check --ci >> fix_results.log

# 2. Safety backup
git stash push -m "Before haive-core trunk fixes"

# 3. Run comprehensive auto-fixes
trunk check --fix --all

# 4. Run enhanced formatting
poetry run pydocstringformatter --write src/ --style google
poetry run ruff check src/ --fix --unsafe-fixes

# 5. Measure improvement
echo "=== AFTER FIXES ===" >> fix_results.log
trunk check --ci >> fix_results.log
```

### Phase 2: Validation

```bash
# Test that nothing broke
poetry run pytest tests/ -x

# Check imports still work
poetry run python -c "from haive.core import *; print('✅ Imports OK')"

# Verify specific modules
poetry run python -c "from haive.core.utils.env_utils import *; print('✅ env_utils OK')"
```

### Phase 3: Manual Review (If Needed)

```bash
# Check remaining issues
trunk check --show-existing

# Focus on high-priority issues
trunk check --filter=high

# Review security issues manually
poetry run bandit -r src/ -f json > security_issues.json
```

## 🔍 How to Verify Fixes

### Before/After Comparison

```bash
# Count issues before
trunk check --ci | grep -E "(issues|failures)" > before.txt

# After fixes
trunk check --ci | grep -E "(issues|failures)" > after.txt

# Compare
echo "BEFORE:"; cat before.txt
echo "AFTER:"; cat after.txt
```

### Success Metrics

- **Black failures**: Should go from 513 → 0
- **Ruff issues**: Should reduce by 60-80%
- **Import errors**: Should be 0
- **Test failures**: Should remain 0

### Code Quality Verification

```bash
# Ensure code still works
poetry run python -c "
import haive.core.utils.env_utils
import haive.core.utils.haive_discovery.base_analyzer
print('✅ All imports successful')
"

# Run a few key tests
poetry run pytest tests/test_env_utils.py -v
poetry run pytest tests/test_haive_discovery/ -v
```

## 🗃️ Git Workflow

### Preparation

```bash
# Create fix branch
git checkout -b fix/haive-core-trunk-issues-$(date +%Y%m%d)

# Take safety backup
git stash push -m "Pre-trunk-fixes backup"
```

### Commit Strategy

```bash
# Commit 1: Black formatting fixes
git add src/
git commit -m "fix(haive-core): resolve black formatting issues

- Fixed 513 black formatting violations
- Applied consistent code style across all modules
- No functional changes, formatting only

Fixes: Black failures in haive_discovery, utils, schema modules"

# Commit 2: Ruff lint fixes
git add src/
git commit -m "fix(haive-core): resolve ruff linting issues

- Fixed import ordering and unused imports
- Resolved variable naming and complexity issues
- Applied ruff auto-fixes across codebase

Fixes: 580+ ruff violations"

# Commit 3: Google-style docstrings
git add src/
git commit -m "docs(haive-core): standardize Google-style docstrings

- Applied pydocstringformatter for consistent style
- Converted docstrings to Google convention format
- Improved documentation readability

Enhanced: Documentation consistency across package"
```

### Push and Validation

```bash
# Final validation before push
trunk check --ci
poetry run pytest tests/ -x

# Push changes
git push -u origin fix/haive-core-trunk-issues-$(date +%Y%m%d)

# Create summary
echo "🎉 Haive-Core Fixes Complete!" > FIXES_SUMMARY.md
echo "Branch: fix/haive-core-trunk-issues-$(date +%Y%m%d)" >> FIXES_SUMMARY.md
echo "" >> FIXES_SUMMARY.md
trunk check --ci >> FIXES_SUMMARY.md
```

## 📊 Expected Results

### Issue Reduction Targets

- **Black failures**: 513 → 0 (100% fix rate)
- **Ruff issues**: 580 → 100-150 (75% fix rate)
- **Security issues**: 91 → 20-30 (manual review needed)
- **Overall improvement**: 70-80% issue reduction

### Files Most Improved

1. `src/haive/core/utils/haive_discovery/` - All modules cleaned
2. `src/haive/core/utils/env_utils.py` - Formatting fixed
3. `src/haive/core/utils/pydantic_utils/ui.py` - Style improved
4. `src/haive/core/schema/` modules - Consistency applied

## 🚨 Troubleshooting

### If Trunk Fails

```bash
# Run tools individually
poetry run black src/
poetry run ruff check src/ --fix
poetry run isort src/

# Check for conflicts
git status
git diff
```

### If Tests Break

```bash
# Restore from backup
git stash pop

# Apply fixes more carefully
trunk check --fix --filter=black  # Safe formatting only
```

### If Imports Break

```bash
# Check Python path
poetry run python -c "import sys; print('\\n'.join(sys.path))"

# Verify installation
poetry install
poetry run python -c "import haive.core; print('OK')"
```

## 🎯 Quick Start Commands

```bash
# Complete fix pipeline (copy-paste ready)
git stash push -m "Pre-fixes backup" && \
trunk check --fix --all && \
poetry run pydocstringformatter --write src/ --style google && \
poetry run ruff check src/ --fix --unsafe-fixes && \
poetry run pytest tests/ -x && \
echo "✅ Fixes complete! Check with: trunk check"
```

## 📋 Next Steps After haive-core

Once haive-core is fixed:

1. **haive-agents**: `cd ../haive-agents && trunk check --fix`
2. **haive-tools**: `cd ../haive-tools && trunk check --fix`
3. **haive-games**: `cd ../haive-games && trunk check --fix`
4. **haive-mcp**: `cd ../haive-mcp && trunk check --fix`
5. **haive-prebuilt**: `cd ../haive-prebuilt && trunk check --fix`

Each package will need similar treatment with package-specific commit messages and git workflows.

---

**Ready to execute!** Start with the Quick Start Commands above, then follow the detailed phases for comprehensive fixes.
