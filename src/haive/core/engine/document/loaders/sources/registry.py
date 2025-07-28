"""Source registry with decorator-based registration.

This module provides a registry for document sources that maps:
- File extensions to source classes
- URL patterns to source classes
- Schemes to source classes
- Source classes to their associated loaders

The registry enables automatic source detection and loader selection.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from haive.core.engine.document.config import LoaderPreference
from haive.core.engine.document.loaders.path_analyzer import (
    PathAnalysisResult,
    analyze_path,
)
from haive.core.engine.document.loaders.sources.source_base import (
    BaseSource,
    DatabaseSource,
    DirectorySource,
    LocalSource,
    RemoteSource,
)

logger = logging.getLogger(__name__)


@dataclass
class LoaderMapping:
    """Mapping of a loader to a source."""

    name: str  # Loader name in langchain_community
    module: str = "langchain_community.document_loaders"

    # Loader characteristics
    speed: str = "medium"  # fast, medium, slow
    quality: str = "medium"  # low, medium, high

    # Requirements
    requires_packages: list[str] = field(default_factory=list)
    requires_auth: bool = False

    # When to use this loader
    best_for: list[str] = field(default_factory=list)
    conditions: dict[str, Any] = field(
        default_factory=dict
    )  # e.g., {"file_size": "<100MB"}


@dataclass
class SourceRegistration:
    """Complete registration info for a source."""

    name: str
    source_class: type[BaseSource]

    # Pattern matching
    file_extensions: set[str] = field(default_factory=set)
    mime_types: set[str] = field(default_factory=set)
    url_patterns: set[str] = field(default_factory=set)
    schemes: set[str] = field(default_factory=set)
    path_patterns: set[str] = field(default_factory=set)

    # Associated loaders
    loaders: dict[str, LoaderMapping] = field(default_factory=dict)
    default_loader: str | None = None

    # Matching priority (higher = preferred)
    priority: int = 0

    # Custom matcher function
    custom_matcher: Callable[[PathAnalysisResult], bool] | None = None


class SourceRegistry:
    """Registry for document sources and their loaders."""

    def __init__(self) -> None:
        self._sources: dict[str, SourceRegistration] = {}

        # Indexes for fast lookup
        self._extension_index: dict[str, set[str]] = {}  # ext -> source names
        # pattern -> source names
        self._url_pattern_index: dict[str, set[str]] = {}
        self._scheme_index: dict[str, set[str]] = {}  # scheme -> source names
        self._mime_index: dict[str, set[str]] = {}  # mime -> source names

    def register(
        self,
        name: str,
        source_class: type[BaseSource],
        file_extensions: list[str] | None = None,
        mime_types: list[str] | None = None,
        url_patterns: list[str] | None = None,
        schemes: list[str] | None = None,
        path_patterns: list[str] | None = None,
        loaders: dict[str, str | dict[str, Any]] | None = None,
        default_loader: str | None = None,
        priority: int = 0,
        custom_matcher: Callable[[PathAnalysisResult], bool] | None = None,
    ) -> SourceRegistration:
        """Register a source with the registry."""
        # Create registration
        registration = SourceRegistration(
            name=name,
            source_class=source_class,
            file_extensions=set(file_extensions or []),
            mime_types=set(mime_types or []),
            url_patterns=set(url_patterns or []),
            schemes=set(schemes or []),
            path_patterns=set(path_patterns or []),
            default_loader=default_loader,
            priority=priority,
            custom_matcher=custom_matcher,
        )

        # Process loader mappings
        if loaders:
            for loader_name, loader_info in loaders.items():
                if isinstance(loader_info, str):
                    # Simple string mapping
                    registration.loaders[loader_name] = LoaderMapping(name=loader_info)
                elif isinstance(loader_info, dict):
                    # Detailed mapping
                    registration.loaders[loader_name] = LoaderMapping(
                        name=loader_info.get("class", loader_name),
                        module=loader_info.get(
                            "module", "langchain_community.document_loaders"
                        ),
                        speed=loader_info.get("speed", "medium"),
                        quality=loader_info.get("quality", "medium"),
                        requires_packages=loader_info.get("requires_packages", []),
                        requires_auth=loader_info.get("requires_auth", False),
                        best_for=loader_info.get("best_for", []),
                        conditions=loader_info.get("conditions", {}),
                    )

        # Store registration
        self._sources[name] = registration

        # Update indexes
        self._update_indexes(name, registration)

        logger.info(
            f"Registered source '{name}' with {
                len(
                    registration.loaders)} loaders, "
            f"{len(registration.file_extensions)} extensions"
        )

        return registration

    def _update_indexes(self, name: str, registration: SourceRegistration):
        """Update lookup indexes."""
        # File extensions
        for ext in registration.file_extensions:
            if ext not in self._extension_index:
                self._extension_index[ext] = set()
            self._extension_index[ext].add(name)

        # URL patterns
        for pattern in registration.url_patterns:
            if pattern not in self._url_pattern_index:
                self._url_pattern_index[pattern] = set()
            self._url_pattern_index[pattern].add(name)

        # Schemes
        for scheme in registration.schemes:
            if scheme not in self._scheme_index:
                self._scheme_index[scheme] = set()
            self._scheme_index[scheme].add(name)

        # MIME types
        for mime in registration.mime_types:
            if mime not in self._mime_index:
                self._mime_index[mime] = set()
            self._mime_index[mime].add(name)

    def find_source_for_path(
        self, path: str, analysis: PathAnalysisResult | None = None
    ) -> SourceRegistration | None:
        """Find the best source for a given path."""
        # Analyze path if not provided
        if not analysis:
            analysis = analyze_path(path)

        candidates: list[SourceRegistration] = []

        # Check file extension
        if analysis.file_extension:
            for source_name in self._extension_index.get(analysis.file_extension, []):
                candidates.append(self._sources[source_name])

        # Check URL patterns
        if analysis.domain:
            for pattern, source_names in self._url_pattern_index.items():
                if pattern in analysis.domain:
                    for source_name in source_names:
                        candidates.append(self._sources[source_name])

        # Check schemes
        if analysis.url_components and analysis.url_components.get("scheme"):
            scheme = analysis.url_components["scheme"]
            for source_name in self._scheme_index.get(scheme, []):
                candidates.append(self._sources[source_name])

        # Check MIME type
        if analysis.mime_type:
            for source_name in self._mime_index.get(analysis.mime_type, []):
                candidates.append(self._sources[source_name])

        # Check custom matchers
        for registration in self._sources.values():
            if registration.custom_matcher and registration.custom_matcher(analysis):
                candidates.append(registration)

        # Return highest priority match
        if candidates:
            return max(candidates, key=lambda r: r.priority)

        return None

    def create_source(
        self, path: str, source_type: str | None = None, **kwargs
    ) -> BaseSource | None:
        """Create a source instance for a path."""
        # Use specific source if provided
        if source_type and source_type in self._sources:
            registration = self._sources[source_type]
        else:
            # Auto-detect source
            registration = self.find_source_for_path(path)
            if not registration:
                return None

        # Create source instance
        try:
            # Analyze path for additional metadata
            analyze_path(path)

            # Build source kwargs based on source type
            source_kwargs = {
                "source_type": registration.name,
                "source_id": f"{registration.name}:{path}",
            }

            # Add path-specific fields based on base class
            if issubclass(registration.source_class, LocalSource):
                source_kwargs["file_path"] = path
            elif issubclass(registration.source_class, DirectorySource):
                source_kwargs["directory_path"] = path
            elif issubclass(registration.source_class, RemoteSource):
                source_kwargs["url"] = path
            elif issubclass(registration.source_class, DatabaseSource):
                source_kwargs["connection_string"] = path

            # Merge with provided kwargs
            source_kwargs.update(kwargs)

            # Create instance
            return registration.source_class(**source_kwargs)

        except Exception as e:
            logger.exception(f"Failed to create source for {path}: {e}")
            return None

    def get_loader_for_source(
        self,
        source: BaseSource,
        loader_name: str | None = None,
        preference: LoaderPreference = LoaderPreference.BALANCED,
    ) -> LoaderMapping | None:
        """Get the best loader for a source."""
        # Get source registration
        registration = self._sources.get(source.source_type)
        if not registration or not registration.loaders:
            return None

        # Use specific loader if requested
        if loader_name and loader_name in registration.loaders:
            return registration.loaders[loader_name]

        # Use source's preferred loader
        if source.preferred_loader and source.preferred_loader in registration.loaders:
            return registration.loaders[source.preferred_loader]

        # Select based on preference (prioritize over default)
        if preference == LoaderPreference.SPEED:
            # Find fastest loader
            fast_loaders = [
                l for l in registration.loaders.values() if l.speed == "fast"
            ]
            if fast_loaders:
                return fast_loaders[0]

        elif preference == LoaderPreference.QUALITY:
            # Find highest quality loader
            quality_loaders = [
                l for l in registration.loaders.values() if l.quality == "high"
            ]
            if quality_loaders:
                return quality_loaders[0]

        # Use registration's default
        if (
            registration.default_loader
            and registration.default_loader in registration.loaders
        ):
            return registration.loaders[registration.default_loader]

        # Return first available
        return next(iter(registration.loaders.values()))

    def list_sources(self) -> list[str]:
        """List all registered source names."""
        return list(self._sources.keys())

    def get_source_info(self, name: str) -> SourceRegistration | None:
        """Get registration info for a source."""
        return self._sources.get(name)


# Global registry instance
source_registry = SourceRegistry()


# Import to ensure LocalSource is available


def register_source(
    name: str | None = None,
    file_extensions: list[str] | None = None,
    mime_types: list[str] | None = None,
    url_patterns: list[str] | None = None,
    schemes: list[str] | None = None,
    path_patterns: list[str] | None = None,
    loaders: dict[str, str | dict[str, Any]] | None = None,
    default_loader: str | None = None,
    priority: int = 0,
    custom_matcher: Callable[[PathAnalysisResult], bool] | None = None,
) -> Callable[[type[BaseSource]], type[BaseSource]]:
    """Decorator to register a source class.

    Example:
        @register_source(
            name="pdf",
            file_extensions=[".pdf"],
            mime_types=["application/pdf"],
            loaders={
                "fast": "PyPDFLoader",
                "quality": {
                    "class": "UnstructuredPDFLoader",
                    "quality": "high",
                    "requires_packages": ["unstructured", "pdf2image"],
                },
                "ocf": {
                    "class": "PDFPlumberLoader",
                    "speed": "slow",
                    "quality": "high",
                    "best_for": ["tables", "complex_layouts"],
                }
            },
            default_loader="fast",
            priority=10
        )
        class PDFSource(LocalSource):
            '''Source for PDF documents.'''
            pass
    """

    def decorator(source_class: type[BaseSource]) -> type[BaseSource]:
        # Use class name if no name provided
        source_name = name or source_class.__name__.lower().replace("source", "")

        # Register with the global registry
        registration = source_registry.register(
            name=source_name,
            source_class=source_class,
            file_extensions=file_extensions,
            mime_types=mime_types,
            url_patterns=url_patterns,
            schemes=schemes,
            path_patterns=path_patterns,
            loaders=loaders,
            default_loader=default_loader,
            priority=priority,
            custom_matcher=custom_matcher,
        )

        # Attach registration info to class
        source_class._registry_name = source_name
        source_class._registration = registration

        return source_class

    return decorator


__all__ = [
    "LoaderMapping",
    "SourceRegistration",
    "SourceRegistry",
    "register_source",
    "source_registry",
]
