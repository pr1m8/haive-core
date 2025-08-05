"""
📄 File Reader Utilities

This module provides utility functions for reading `.md`, `.yaml`, or `.yml` files into memory.
It supports content integration for system messages and configuration loading.

Features:
- Read Markdown files as plain strings
- Read YAML files into Python dictionaries
- Auto-detect file type based on extension
- Read multiple files at once and return a dictionary
- Designed for use in LLM prompt templates and configuration systems

Author: Your Name
"""

from typing import Any, Union

import yaml


def read_yaml_file(file_path: Union[str, Path]) -> Any:
    """
    Reads a YAML (.yml or .yaml) file and returns the parsed Python object.

    Args:
        file_path (Union[str, Path]): Path to the YAML file.

    Returns:
        Any: Parsed content of the YAML file as a dict, list, or scalar.
    """
    with open(file_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_file_content(file_path: Union[str, Path]) -> Any:
    """
    Reads a Markdown or YAML file and returns its content.

    Automatically detects the file type based on extension.

    Args:
        file_path (Union[str, Path]): Path to the file to read (.md, .yml, or .yaml).

    Returns:
        Any: String content for .md files, parsed Python object for .yaml files,
             or None if the file cannot be read.
    """
    try:
        path = Path(file_path)
        if path.suffix in {".yml", ".yaml"}:
            return read_yaml_file(path)
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def read_multiple_files(file_paths: list[Union[str, Path]]) -> dict[str, Any]:
    """
    Reads multiple Markdown or YAML files and returns a dictionary of filename to content.

    Args:
        file_paths (list[Union[str, Path]]): List of paths to files to read.

    Returns:
        dict[str, Any]: Dictionary mapping file stem (name without extension) to content.
    """
    results = {}
    for path in file_paths:
        path = Path(path)
        content = read_file_content(path)
        if content is not None:
            results[path.stem] = content
    return results
