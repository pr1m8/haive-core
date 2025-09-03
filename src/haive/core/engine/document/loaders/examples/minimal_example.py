#!/usr/bin/env python3
"""Minimal working example of the Haive Document Loader System.

This example demonstrates the core functionality of the document loading system
even with the source registration issues. It shows how to use the basic
AutoLoader functionality and path analysis.

Run this example with:
    poetry run python src/haive/core/engine/document/loaders/examples/minimal_example.py
"""

import contextlib
import tempfile
from pathlib import Path
from typing import Any

from haive.core.engine.document.config import LoaderPreference
from haive.core.engine.document.loaders import (
    AutoLoader,
    AutoLoaderConfig,
    get_registration_status,
    list_available_sources,
)


def create_test_files() -> Any:
    """Create temporary test files for demonstration."""
    # Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    # Create various test files
    test_files = {}

    # Text file
    text_file = temp_dir / "test.txt"
    text_file.write_text(
        "This is a test document.\nIt contains multiple lines.\nUsed for testing the document loader system."
    )
    test_files["text"] = text_file

    # JSON file
    json_file = temp_dir / "test.json"
    json_file.write_text(
        '{"title": "Test Document", "content": "This is JSON content", "metadata": {"author": "Test Author"}}'
    )
    test_files["json"] = json_file

    # CSV file
    csv_file = temp_dir / "test.csv"
    csv_file.write_text(
        "name,age,city\nJohn Doe,30,New York\nJane Smith,25,Los Angeles"
    )
    test_files["csv"] = csv_file

    # Markdown file
    md_file = temp_dir / "test.md"
    md_file.write_text(
        """# Test Document

This is a **markdown** document for testing.

## Features

- Supports various file formats
- Auto-detection of source types
- Configurable loading preferences

## Example Code

    Examples:
        >>> loader = AutoLoader()
        >>> docs = loader.load("document.md")
"""
    )
    test_files["markdown"] = md_file

    return temp_dir, test_files


def demonstrate_basic_functionality() -> None:
    """Demonstrate basic AutoLoader functionality."""
    # Create test files
    temp_dir, test_files = create_test_files()

    for _file_type, file_path in test_files.items():
        pass

    # Test 1: Basic AutoLoader initialization

    loader = AutoLoader()

    # Test 2: Custom configuration

    custom_config = AutoLoaderConfig(
        preference=LoaderPreference.QUALITY,
        max_concurrency=5,
        enable_caching=True,
        timeout=120,
    )
    AutoLoader(custom_config)

    # Test 3: Source detection

    for _file_type, file_path in test_files.items():
        with contextlib.suppress(Exception):
            loader.detect_source(str(file_path))

    # Test 4: URL detection

    test_urls = [
        "https://example.com/document.html",
        "https://api.github.com/repos/user/repo",
        "s3://bucket/document.pdf",
        "ftp://files.example.com/data.zip",
    ]

    for url in test_urls:
        with contextlib.suppress(Exception):
            loader.detect_source(url)

    # Test 5: Registry status

    try:
        get_registration_status()

        # Show available sources
        sources = list_available_sources()
        if sources:
            pass
    except Exception:
        pass

    # Test 6: Error handling

    try:
        # Try to detect an invalid path
        loader.detect_source("/invalid/path/that/does/not/exist.xyz")
    except ValueError:
        pass
    except Exception:
        pass

    # Test 7: Loader preferences

    preferences = [
        LoaderPreference.SPEED,
        LoaderPreference.QUALITY,
        LoaderPreference.BALANCED,
    ]

    for pref in preferences:
        with contextlib.suppress(Exception):
            AutoLoader(AutoLoaderConfig(preference=pref))

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    demonstrate_basic_functionality()
