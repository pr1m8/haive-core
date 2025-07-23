"""Programming language definitions and utilities.

This module provides programming language definitions used throughout
the Haive framework for code analysis, syntax highlighting, and processing.
"""

from dataclasses import dataclass


@dataclass
class LanguageInfo:
    """Information about a programming language."""

    name: str
    extensions: set[str]
    type: str  # "compiled", "interpreted", "markup", etc.
    description: str | None = None


class ProgrammingLanguages:
    """Programming language definitions and utilities.

    Provides comprehensive programming language information including
    file extensions, language types, and metadata for code analysis.
    """

    _LANGUAGES: dict[str, LanguageInfo] = {
        "python": LanguageInfo(
            name="Python",
            extensions={".py", ".pyw", ".pyi"},
            type="interpreted",
            description="High-level programming language",
        ),
        "javascript": LanguageInfo(
            name="JavaScript",
            extensions={".js", ".mjs", ".cjs"},
            type="interpreted",
            description="Dynamic programming language for web development",
        ),
        "typescript": LanguageInfo(
            name="TypeScript",
            extensions={".ts", ".tsx"},
            type="compiled",
            description="Typed superset of JavaScript",
        ),
        "java": LanguageInfo(
            name="Java",
            extensions={".java"},
            type="compiled",
            description="Object-oriented programming language",
        ),
        "cpp": LanguageInfo(
            name="C++",
            extensions={".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx"},
            type="compiled",
            description="General-purpose programming language",
        ),
        "c": LanguageInfo(
            name="C",
            extensions={".c", ".h"},
            type="compiled",
            description="General-purpose, procedural programming language",
        ),
        "go": LanguageInfo(
            name="Go",
            extensions={".go"},
            type="compiled",
            description="Statically typed, compiled language",
        ),
        "rust": LanguageInfo(
            name="Rust",
            extensions={".rs"},
            type="compiled",
            description="Systems programming language",
        ),
        "ruby": LanguageInfo(
            name="Ruby",
            extensions={".rb", ".rbw"},
            type="interpreted",
            description="Dynamic, object-oriented language",
        ),
        "php": LanguageInfo(
            name="PHP",
            extensions={".php", ".phtml"},
            type="interpreted",
            description="Server-side scripting language",
        ),
    }

    @classmethod
    def get_by_name(cls, name: str) -> LanguageInfo | None:
        """Get language info by language name.

        Args:
            name: Language name (case-insensitive)

        Returns:
            LanguageInfo if found, None otherwise
        """
        return cls._LANGUAGES.get(name.lower())

    @classmethod
    def get_by_extension(cls, extension: str) -> LanguageInfo | None:
        """Get language info by file extension.

        Args:
            extension: File extension including the dot (e.g., ".py")

        Returns:
            LanguageInfo if found, None otherwise
        """
        extension = extension.lower()
        for lang_info in cls._LANGUAGES.values():
            if extension in lang_info.extensions:
                return lang_info
        return None

    @classmethod
    def get_all_languages(cls) -> list[LanguageInfo]:
        """Get all supported programming languages.

        Returns:
            List of all LanguageInfo objects
        """
        return list(cls._LANGUAGES.values())

    @classmethod
    def get_all_extensions(cls) -> set[str]:
        """Get all supported file extensions.

        Returns:
            Set of all supported file extensions
        """
        extensions = set()
        for lang_info in cls._LANGUAGES.values():
            extensions.update(lang_info.extensions)
        return extensions

    @classmethod
    def is_supported(cls, extension: str) -> bool:
        """Check if a file extension is supported.

        Args:
            extension: File extension including the dot

        Returns:
            True if the extension is supported
        """
        return cls.get_by_extension(extension) is not None
