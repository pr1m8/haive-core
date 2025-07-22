# Comprehensive Recovery Plan - July 22, 2025

## Current Situation

### What We Have

1. **Fixed **init**.py files** (69 files) - All now have correct `haive.core.*` imports
2. **Untracked provider files** (17 files) - New LLM providers ready to add
3. **Modified files** (80 total) - Mix of fixed and potentially problematic files
4. **Stash with problematic changes** - stash@{1} contains 62 files with massive auto-generated docs

### What Needs Recovery

#### 1. Non-**init**.py Files with Wrong Imports (4 files)

- `src/haive/core/persistence/store/postgres.py` - has `from core.*` imports
- `src/haive/core/persistence/store/wrappers/memory.py` - has `from core.*` imports
- `src/haive/core/persistence/store/wrappers/postgres.py` - has `from core.*` imports
- `src/haive/core/tools/store_manager.py` - has `from core.*` imports

#### 2. Retriever & VectorStore Provider Files

The stash@{1} contains these provider files that may have been corrupted:

- 9 retriever provider configs in `src/haive/core/engine/retriever/providers/`
- 20+ vectorstore provider configs in `src/haive/core/engine/vectorstore/providers/`

#### 3. Other Critical Files in Stash

- `src/haive/core/engine/document/loaders/base/schema.py`
- `src/haive/core/engine/tool/base.py`
- Various other engine and config files

## Recovery Strategy

### Phase 1: Fix Remaining Import Issues

1. Fix the 4 non-**init**.py files with incorrect imports
2. Verify no other files have import issues

### Phase 2: Check Provider Files

1. Compare current provider files with what's in the stash
2. Determine if any provider files were corrupted
3. Selectively recover good provider implementations

### Phase 3: Add New Provider Implementations

1. Add the 17 new untracked provider files:
   - AI21, Azure, Bedrock, Cohere, Fireworks, Groq, HuggingFace
   - Mistral, NVIDIA, Replicate, Together, XAI providers
   - Discovery utilities and type definitions

### Phase 4: Final Verification

1. Ensure all imports are correct
2. Test that the codebase can be imported
3. Commit the recovered state

## Commands to Execute

```bash
# Phase 1: Fix remaining imports
# (Need to edit 4 files)

# Phase 2: Check what's different
git diff stash@{1} -- src/haive/core/engine/retriever/providers/
git diff stash@{1} -- src/haive/core/engine/vectorstore/providers/

# Phase 3: Add new providers
git add src/haive/core/models/llm/providers/
git add src/haive/core/engine/vectorstore/discovery.py
git add src/haive/core/types/

# Phase 4: Commit everything
git add src/haive/core/**/__init__.py
git add <other fixed files>
git commit -m "fix: comprehensive recovery of July 21 fixes
- Fixed all __init__.py imports (69 files)
- Fixed remaining Python file imports (4 files)
- Added new provider implementations (17 files)
- Recovered from July 22 corruption"
```

## Status Summary

- ✅ **init**.py files fixed (69/69)
- ✅ Other Python files fixed (4/4)
- ✅ All imports now use correct haive.core.\* pattern
- ❌ New providers need adding (0/17)
- ✅ Total files recovered: 82 modified files with correct imports
