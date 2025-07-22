# Files That Need Import Fixes

## haive-core Package

### Files with "from core." imports (4 files):

1. **src/haive/core/persistence/store/postgres.py**
   - Line 10: `from core.persistence.postgres_config import PostgresCheckpointerConfig`
   - Should be: `from haive.core.persistence.postgres_config import PostgresCheckpointerConfig`

2. **src/haive/core/persistence/store/wrappers/memory.py**
   - Line 6: `from core.persistence.store.base import SerializableStoreWrapper`
   - Line 7: `from core.persistence.store.embeddings import EmbeddingAdapter`
   - Should use `haive.core.` prefix

3. **src/haive/core/persistence/store/wrappers/postgres.py**
   - Line 6: `from core.persistence.store.base import SerializableStoreWrapper`
   - Line 7: `from core.persistence.store.connection import ConnectionManager`
   - Line 8: `from core.persistence.store.embeddings import EmbeddingAdapter`
   - Line 9: `from core.persistence.store.types import StoreType`
   - Should use `haive.core.` prefix

4. **src/haive/core/tools/store_manager.py**
   - Line 13: `from core.persistence.store.base import SerializableStoreWrapper`
   - Line 14: `from core.persistence.store.factory import create_store`
   - Line 15: `from core.persistence.store.types import StoreType`
   - Should use `haive.core.` prefix

## Summary

Total files to fix: 4
Total import statements to fix: 10

All of these are in the persistence store and tools modules.
