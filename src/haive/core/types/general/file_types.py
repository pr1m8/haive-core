"""File type definitions and utilities.

This module provides comprehensive file type definitions used throughout
the Haive framework for file processing, validation, and categorization.
"""

from typing import Set


class FileTypes:
    """Comprehensive file type definitions and utilities.

    Provides categorized file extensions and utilities for file type
    detection and validation used in document processing workflows.
    """

    # Document types
    DOCUMENTS: Set[str] = {
        ".txt",
        ".md",
        ".rst",
        ".doc",
        ".docx",
        ".pdf",
        ".rtf",
        ".odt",
    }

    # Code file types
    CODE: Set[str] = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".r",
    }

    # Configuration files
    CONFIG: Set[str] = {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".env",
    }

    # Data files
    DATA: Set[str] = {".csv", ".tsv", ".xlsx", ".parquet", ".jsonl", ".xml", ".sql"}

    # Image files
    IMAGES: Set[str] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".svg",
        ".webp",
        ".ico",
    }

    # Archive files
    ARCHIVES: Set[str] = {".zip", ".tar", ".gz", ".rar", ".7z", ".bz2", ".xz"}

    @classmethod
    def get_category(cls, extension: str) -> str:
        """Get the category for a file extension.

        Args:
            extension: File extension including the dot (e.g., ".py")

        Returns:
            Category name or "unknown" if not found
        """
        extension = extension.lower()

        if extension in cls.DOCUMENTS:
            return "document"
        elif extension in cls.CODE:
            return "code"
        elif extension in cls.CONFIG:
            return "config"
        elif extension in cls.DATA:
            return "data"
        elif extension in cls.IMAGES:
            return "image"
        elif extension in cls.ARCHIVES:
            return "archive"
        else:
            return "unknown"

    @classmethod
    def is_supported(cls, extension: str) -> bool:
        """Check if a file extension is supported.

        Args:
            extension: File extension including the dot

        Returns:
            True if the extension is in any supported category
        """
        return cls.get_category(extension) != "unknown"

    @classmethod
    def get_all_extensions(cls) -> Set[str]:
        """Get all supported file extensions.

        Returns:
            Set of all supported file extensions
        """
        return (
            cls.DOCUMENTS | cls.CODE | cls.CONFIG | cls.DATA | cls.IMAGES | cls.ARCHIVES
        )
