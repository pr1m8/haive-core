#!/usr/bin/env python3
"""Minimal integration test to identify API issues without heavy imports."""

import asyncio
import sys
import time


def print_status(message: str, success: bool = True):
    """Print test status."""


def test_basic_imports():
    """Test basic imports without heavy dependencies."""
    try:

        print_status("StoreManager import")

        print_status("StoreType import")

        print_status("AugLLMConfig import")

        return True
    except Exception as e:
        print_status(f"Basic imports failed: {e}", False)
        return False


def test_store_manager_api():
    """Test StoreManager API without heavy imports."""
    try:
        from haive.core.persistence.store.types import StoreType
        from haive.core.tools.store_manager import StoreManager

        # Test new API
        store_manager = StoreManager(
            store_config={"type": StoreType.MEMORY},
            default_namespace=("test", "minimal"),
        )
        print_status("StoreManager creation with new API")

        # Test basic operations
        memory_id = store_manager.store_memory(
            content="Test memory", category="test", importance=0.8
        )
        print_status(f"Memory stored with ID: {memory_id}")

        # Test retrieval
        memory = store_manager.retrieve_memory(memory_id)
        if memory and memory.content == "Test memory":
            print_status("Memory retrieval works")
        else:
            print_status("Memory retrieval failed", False)
            return False

        # Test search
        results = store_manager.search_memories("test", limit=5)
        print_status(f"Search found {len(results)} memories")

        return True
    except Exception as e:
        print_status(f"StoreManager API test failed: {e}", False)
        return False


def test_memory_classifier_minimal():
    """Test memory classifier with minimal setup."""
    try:
        # Test import
        from haive.agents.memory.core.classifier import (
            MemoryClassifier,
            MemoryClassifierConfig,
        )

        print_status("MemoryClassifier import")

        # Test config creation
        config = MemoryClassifierConfig(confidence_threshold=0.6)
        print_status("MemoryClassifierConfig creation")

        # Test classifier creation (this might be slow due to LLM)
        MemoryClassifier(config)
        print_status("MemoryClassifier creation")

        return True
    except Exception as e:
        print_status(f"MemoryClassifier test failed: {e}", False)
        return False


def test_memory_store_manager():
    """Test MemoryStoreManager with proper API."""
    try:
        from haive.agents.memory.core.stores import (
            MemoryStoreConfig,
            MemoryStoreManager,
        )
        from haive.core.persistence.store.types import StoreType
        from haive.core.tools.store_manager import StoreManager

        # Create store manager with correct API
        store_manager = StoreManager(
            store_config={"type": StoreType.MEMORY},
            default_namespace=("test", "memory_store"),
        )
        print_status("Base StoreManager created")

        # Create memory store config
        config = MemoryStoreConfig(
            store_manager=store_manager,
            default_namespace=("test", "memory"),
            auto_classify=False,  # Disable to avoid LLM calls
        )
        print_status("MemoryStoreConfig created")

        # Create memory store manager
        MemoryStoreManager(config)
        print_status("MemoryStoreManager created")

        return True
    except Exception as e:
        print_status(f"MemoryStoreManager test failed: {e}", False)
        return False


async def test_memory_store_async():
    """Test async memory store operations."""
    try:
        from haive.agents.memory.core.stores import (
            MemoryStoreConfig,
            MemoryStoreManager,
        )
        from haive.core.persistence.store.types import StoreType
        from haive.core.tools.store_manager import StoreManager

        # Setup
        store_manager = StoreManager(
            store_config={"type": StoreType.MEMORY}, default_namespace=("test", "async")
        )

        config = MemoryStoreConfig(
            store_manager=store_manager,
            default_namespace=("test", "async"),
            auto_classify=False,
        )

        memory_store = MemoryStoreManager(config)

        # Test async store operation
        memory_id = await memory_store.store_memory(
            content="Async test memory", namespace=("test", "async")
        )
        print_status(f"Async memory stored: {memory_id}")

        # Test async retrieval
        memories = await memory_store.retrieve_memories(
            query="async test", namespace=("test", "async"), limit=5
        )
        print_status(f"Async retrieval found {len(memories)} memories")

        return True
    except Exception as e:
        print_status(f"Async memory store test failed: {e}", False)
        return False


def test_config_compatibility():
    """Test configuration compatibility issues."""
    try:
        # Test if UnifiedMemorySystem uses old API
        from haive.agents.memory.unified_memory_api import MemorySystemConfig

        # Test new config format
        MemorySystemConfig(
            store_type="memory",
            collection_name="test_unified",
            default_namespace=("test", "unified"),
            enable_auto_classification=True,
        )
        print_status("MemorySystemConfig created with correct parameters")

        return True
    except Exception as e:
        print_status(f"Config compatibility test failed: {e}", False)
        return False


def main():
    """Run all minimal tests."""
    time.time()

    # Track results
    results = []

    # Test basic imports
    results.append(("Basic Imports", test_basic_imports()))

    # Test StoreManager API
    results.append(("StoreManager API", test_store_manager_api()))

    # Test MemoryClassifier (minimal)
    results.append(("MemoryClassifier", test_memory_classifier_minimal()))

    # Test MemoryStoreManager
    results.append(("MemoryStoreManager", test_memory_store_manager()))

    # Test async operations
    results.append(
        ("Async Operations",
         asyncio.run(
             test_memory_store_async())))

    # Test config compatibility
    results.append(("Config Compatibility", test_config_compatibility()))

    time.time()

    # Print summary

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for _name, _success in results:
        pass

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
