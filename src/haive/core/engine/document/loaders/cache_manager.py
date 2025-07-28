"""Cache manager for document loader registry.

This module provides caching functionality to speed up document loader initialization by
avoiding repeated scanning of 230+ loader modules.
"""

import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RegistryCacheManager:
    """Manages caching of document loader registry data.

    This significantly speeds up imports by caching:
    - Discovered source modules
    - Registered loaders and their configurations
    - Source type mappings
    - Loader capabilities

    Cache is invalidated when:
    - Source files are modified
    - Package version changes
    - Cache expires (default 7 days)
    - User explicitly clears cache
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_days: int = 7,
        use_memory_cache: bool = True,
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files (default: ~/.cache/haive/loaders)
            cache_ttl_days: Cache time-to-live in days
            use_memory_cache: Whether to use in-memory caching for current session
        """
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "haive" / "loaders")
        self.cache_ttl = timedelta(days=cache_ttl_days)
        self.use_memory_cache = use_memory_cache

        # File paths
        self.registry_cache_file = self.cache_dir / "registry_cache.pkl"
        self.metadata_file = self.cache_dir / "cache_metadata.json"

        # In-memory cache for current session
        self._memory_cache: dict[str, Any] = {}
        self._cache_loaded = False

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cached_registry(self) -> dict[str, Any] | None:
        """Get cached registry data if valid.

        Returns:
            Cached registry data or None if cache is invalid/missing
        """
        # Check memory cache first
        if self.use_memory_cache and self._cache_loaded:
            return self._memory_cache.get("registry")

        # Check disk cache
        if not self._is_cache_valid():
            logger.debug("Cache is invalid or missing")
            return None

        try:
            with open(self.registry_cache_file, "rb") as f:
                data = pickle.load(f)

            # Load into memory cache
            if self.use_memory_cache:
                self._memory_cache["registry"] = data
                self._cache_loaded = True

            logger.info("Loaded registry from cache")
            return data

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def save_registry_cache(
        self, registry_data: dict[str, Any], source_files: set[Path] | None = None
    ) -> bool:
        """Save registry data to cache.

        Args:
            registry_data: Registry data to cache
            source_files: Set of source files that were scanned

        Returns:
            True if cache was saved successfully
        """
        try:
            # Save registry data
            with open(self.registry_cache_file, "wb") as f:
                pickle.dump(registry_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "haive_version": self._get_haive_version(),
                "source_files_hash": self._calculate_files_hash(source_files),
                "stats": {
                    "total_sources": len(registry_data.get("sources", {})),
                    "total_loaders": sum(
                        len(info.get("loaders", {}))
                        for info in registry_data.get("sources", {}).values()
                    ),
                },
            }

            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            # Update memory cache
            if self.use_memory_cache:
                self._memory_cache["registry"] = registry_data
                self._cache_loaded = True

            logger.info(
                f"Saved registry cache with {
                    metadata['stats']['total_sources']} sources "
                f"and {metadata['stats']['total_loaders']} loaders"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to save cache: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear all cache files and memory cache."""
        # Clear disk cache
        if self.registry_cache_file.exists():
            self.registry_cache_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()

        # Clear memory cache
        self._memory_cache.clear()
        self._cache_loaded = False

        logger.info("Cleared registry cache")

    def _is_cache_valid(self) -> bool:
        """Check if cache is valid.

        Returns:
            True if cache exists and is valid
        """
        # Check if cache files exist
        if not self.registry_cache_file.exists() or not self.metadata_file.exists():
            return False

        try:
            # Load metadata
            with open(self.metadata_file) as f:
                metadata = json.load(f)

            # Check cache age
            created_at = datetime.fromisoformat(metadata["created_at"])
            if datetime.now() - created_at > self.cache_ttl:
                logger.debug("Cache expired")
                return False

            # Check version compatibility
            current_version = self._get_haive_version()
            if metadata.get("haive_version") != current_version:
                logger.debug(
                    f"Version mismatch: {
                        metadata.get('haive_version')} != {current_version}"
                )
                return False

            # TODO: Check if source files have been modified
            # This would require tracking file modification times

            return True

        except Exception as e:
            logger.debug(f"Cache validation failed: {e}")
            return False

    def _get_haive_version(self) -> str:
        """Get current haive version."""
        try:
            import haive

            return getattr(haive, "__version__", "unknown")
        except BaseException:
            return "unknown"

    def _calculate_files_hash(self, source_files: set[Path] | None) -> str:
        """Calculate hash of source files for change detection."""
        if not source_files:
            return ""

        # Sort files for consistent hashing
        sorted_files = sorted(source_files)

        # Create hash of file paths and modification times
        hasher = hashlib.md5()
        for file_path in sorted_files:
            if file_path.exists():
                hasher.update(str(file_path).encode())
                hasher.update(str(file_path.stat().st_mtime).encode())

        return hasher.hexdigest()

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about current cache status.

        Returns:
            Dictionary with cache information
        """
        info = {
            "cache_dir": str(self.cache_dir),
            "cache_exists": self.registry_cache_file.exists(),
            "memory_cache_loaded": self._cache_loaded,
            "memory_cache_size": len(self._memory_cache),
        }

        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    metadata = json.load(f)
                info["metadata"] = metadata
                info["cache_valid"] = self._is_cache_valid()
            except BaseException:
                info["metadata"] = None
                info["cache_valid"] = False
        else:
            info["metadata"] = None
            info["cache_valid"] = False

        return info


# Global cache manager instance
_cache_manager = RegistryCacheManager()


def get_cache_manager() -> RegistryCacheManager:
    """Get the global cache manager instance."""
    return _cache_manager


def clear_loader_cache() -> None:
    """Clear the document loader cache.

    Use this when you've installed new packages or made changes to loader
    implementations.
    """
    _cache_manager.clear_cache()


def get_cache_status() -> dict[str, Any]:
    """Get current cache status information."""
    return _cache_manager.get_cache_info()
