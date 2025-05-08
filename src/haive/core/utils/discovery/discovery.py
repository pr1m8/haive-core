"""
Main component discovery module.

Provides a unified interface for discovering, analyzing, and converting components
from various sources.
"""

import importlib
import inspect
import json
import logging
import os
import pkgutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

# Import analyzers
from .analyzers.loader import DocumentLoaderAnalyzer
from .converters.to_document import DocumentConverter

# Import converters
from .converters.to_tool import ToolConverter
from .extractors.env_var import EnvironmentVariableExtractor
from .extractors.method import MethodExtractor
from .models.component import ComponentCollection, ComponentMetadata, ComponentType

# Set up logging
logger = logging.getLogger(__name__)


class ComponentDiscovery:
    """
    Main component discovery system.

    Discovers and analyzes components from various sources, and provides
    conversion utilities to work with the discovered components.
    """

    def __init__(self):
        """Initialize the component discovery system."""
        self.collection = ComponentCollection()
        self.failed_imports: List[Tuple[str, str]] = []

        # Initialize extractors
        self.method_extractor = MethodExtractor()
        self.env_var_extractor = EnvironmentVariableExtractor()

        # Initialize analyzers
        self.loader_analyzer = DocumentLoaderAnalyzer()

        # Initialize converters
        self.tool_converter = ToolConverter()
        self.document_converter = DocumentConverter()

    def discover_components(
        self,
        base_pkg: Any,
        component_type: ComponentType,
        class_predicate: Callable[[Type], bool],
        analyzer: Any,
    ) -> List[ComponentMetadata]:
        """
        Discover components in a package based on a predicate.

        Args:
            base_pkg: Package to search in
            component_type: Type of component to discover
            class_predicate: Function to determine if a class is a component
            analyzer: Analyzer to use for extracting component metadata

        Returns:
            List of component metadata
        """
        components = []

        # Walk the package to find all modules
        for _, module_name, is_pkg in pkgutil.walk_packages(
            base_pkg.__path__, base_pkg.__name__ + "."
        ):
            try:
                # Skip packages to avoid duplicates
                if is_pkg:
                    continue

                logger.debug(f"Importing module: {module_name}")
                module = importlib.import_module(module_name)
            except Exception as e:
                self.failed_imports.append((module_name, str(e)))
                logger.warning(f"Failed to import {module_name}: {e}")
                continue

            # Find component classes in the module
            for name, cls in inspect.getmembers(module, inspect.isclass):
                try:
                    # Apply the predicate to find components
                    if class_predicate(cls):
                        logger.debug(f"Found component: {name} in {module_name}")

                        # Analyze the component
                        metadata = analyzer.analyze(cls, module_name)

                        # Add to collection and result list
                        self.collection.add(metadata)
                        components.append(metadata)
                except Exception as e:
                    logger.warning(f"Error analyzing {name} in {module_name}: {e}")

        return components

    def discover_document_loaders(self, loaders_package) -> List[ComponentMetadata]:
        """
        Discover all document loader components.

        Args:
            loaders_package: Package containing document loaders

        Returns:
            List of document loader metadata
        """

        def is_document_loader(cls):
            """Check if a class is a document loader."""
            # Check for load method
            has_load = hasattr(cls, "load") and callable(getattr(cls, "load", None))

            # Also check for common class methods
            has_from_file = hasattr(cls, "from_file") and callable(
                getattr(cls, "from_file", None)
            )

            return has_load or has_from_file

        return self.discover_components(
            loaders_package,
            ComponentType.DOCUMENT_LOADER,
            is_document_loader,
            self.loader_analyzer,
        )

    def discover_text_splitters(self, splitters_package) -> List[ComponentMetadata]:
        """
        Discover all text splitter components.

        Args:
            splitters_package: Package containing text splitters

        Returns:
            List of text splitter metadata
        """

        def is_text_splitter(cls):
            """Check if a class is a text splitter."""
            # Check for split_text or split_documents methods
            has_split_text = hasattr(cls, "split_text") and callable(
                getattr(cls, "split_text", None)
            )
            has_split_docs = hasattr(cls, "split_documents") and callable(
                getattr(cls, "split_documents", None)
            )

            return has_split_text or has_split_docs

        # Until we implement a specific analyzer, we'll use the document loader analyzer
        return self.discover_components(
            splitters_package,
            ComponentType.TEXT_SPLITTER,
            is_text_splitter,
            self.loader_analyzer,  # Temporarily use loader analyzer
        )

    def discover_retrievers(self, retrievers_package) -> List[ComponentMetadata]:
        """
        Discover all retriever components.

        Args:
            retrievers_package: Package containing retrievers

        Returns:
            List of retriever metadata
        """

        def is_retriever(cls):
            """Check if a class is a retriever."""
            # Check for get_relevant_documents method
            has_get = hasattr(cls, "get_relevant_documents") and callable(
                getattr(cls, "get_relevant_documents", None)
            )

            # Also check for async variant
            has_aget = hasattr(cls, "aget_relevant_documents") and callable(
                getattr(cls, "aget_relevant_documents", None)
            )

            return has_get or has_aget

        # Until we implement a specific analyzer, we'll use the document loader analyzer
        return self.discover_components(
            retrievers_package,
            ComponentType.RETRIEVER,
            is_retriever,
            self.loader_analyzer,  # Temporarily use loader analyzer
        )

    def discover_document_transformers(
        self, transformers_package
    ) -> List[ComponentMetadata]:
        """
        Discover all document transformer components.

        Args:
            transformers_package: Package containing document transformers

        Returns:
            List of document transformer metadata
        """

        def is_document_transformer(cls):
            """Check if a class is a document transformer."""
            # Check for transform_documents method
            return hasattr(cls, "transform_documents") and callable(
                getattr(cls, "transform_documents", None)
            )

        # Until we implement a specific analyzer, we'll use the document loader analyzer
        return self.discover_components(
            transformers_package,
            ComponentType.DOCUMENT_TRANSFORMER,
            is_document_transformer,
            self.loader_analyzer,  # Temporarily use loader analyzer
        )

    def discover_from_package(
        self, package_name: str
    ) -> Dict[ComponentType, List[ComponentMetadata]]:
        """
        Discover components from a specific package.

        Args:
            package_name: Name of the package to discover components from

        Returns:
            Dictionary mapping component types to lists of component metadata
        """
        result = {}

        try:
            # Try to import the package
            package = importlib.import_module(package_name)

            # Try different component types
            try:
                # Check for document loaders
                if hasattr(package, "document_loaders"):
                    result[ComponentType.DOCUMENT_LOADER] = (
                        self.discover_document_loaders(package.document_loaders)
                    )
                elif "loader" in package_name:
                    result[ComponentType.DOCUMENT_LOADER] = (
                        self.discover_document_loaders(package)
                    )
            except Exception as e:
                logger.warning(f"Error discovering document loaders: {e}")

            try:
                # Check for text splitters
                if hasattr(package, "text_splitters"):
                    result[ComponentType.TEXT_SPLITTER] = self.discover_text_splitters(
                        package.text_splitters
                    )
                elif "splitter" in package_name:
                    result[ComponentType.TEXT_SPLITTER] = self.discover_text_splitters(
                        package
                    )
            except Exception as e:
                logger.warning(f"Error discovering text splitters: {e}")

            try:
                # Check for retrievers
                if hasattr(package, "retrievers"):
                    result[ComponentType.RETRIEVER] = self.discover_retrievers(
                        package.retrievers
                    )
                elif "retriever" in package_name:
                    result[ComponentType.RETRIEVER] = self.discover_retrievers(package)
            except Exception as e:
                logger.warning(f"Error discovering retrievers: {e}")

            try:
                # Check for document transformers
                if hasattr(package, "document_transformers"):
                    result[ComponentType.DOCUMENT_TRANSFORMER] = (
                        self.discover_document_transformers(
                            package.document_transformers
                        )
                    )
                elif "transformer" in package_name:
                    result[ComponentType.DOCUMENT_TRANSFORMER] = (
                        self.discover_document_transformers(package)
                    )
            except Exception as e:
                logger.warning(f"Error discovering document transformers: {e}")

        except ImportError as e:
            logger.warning(f"Could not import package {package_name}: {e}")

        return result

    def discover_all(self) -> ComponentCollection:
        """
        Discover all components from all supported packages.

        Returns:
            Collection of all discovered components
        """
        # Try to discover from common packages
        packages_to_discover = [
            "langchain_community.document_loaders",
            "langchain_text_splitters",
            "langchain_community.text_splitters",
            "langchain_community.retrievers",
            "langchain.retrievers",
            "langchain_community.document_transformers",
        ]

        for package_name in packages_to_discover:
            try:
                logger.info(f"Discovering components from {package_name}")
                results = self.discover_from_package(package_name)

                # Log discovery results
                for component_type, components in results.items():
                    logger.info(
                        f"Discovered {len(components)} {component_type.value} components"
                    )
            except Exception as e:
                logger.warning(f"Error discovering from {package_name}: {e}")

        return self.collection

    def save_to_file(self, filepath: Path) -> None:
        """
        Save discovered components to a JSON file.

        Args:
            filepath: Path to save to
        """
        # Create directories if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for serialization
        data = {
            "discovery_timestamp": self.collection.discovery_timestamp.isoformat(),
            "components": {
                component_id: self._serialize_metadata(metadata)
                for component_id, metadata in self.collection.components.items()
            },
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.collection.components)} components to {filepath}")

    def load_from_file(self, filepath: Path) -> None:
        """
        Load components from a JSON file.

        Args:
            filepath: Path to load from
        """
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Parse timestamp
            timestamp = datetime.fromisoformat(
                data.get("discovery_timestamp", datetime.now().isoformat())
            )

            # Create new collection
            self.collection = ComponentCollection(discovery_timestamp=timestamp)

            # Add components
            for component_id, component_data in data.get("components", {}).items():
                try:
                    # Parse sets
                    for set_field in ["env_vars", "env_vars_found", "env_vars_missing"]:
                        if set_field in component_data:
                            component_data[set_field] = set(component_data[set_field])

                    # Create metadata
                    metadata = ComponentMetadata(**component_data)
                    self.collection.add(metadata)
                except Exception as e:
                    logger.warning(f"Error loading component {component_id}: {e}")

            logger.info(
                f"Loaded {len(self.collection.components)} components from {filepath}"
            )
        except Exception as e:
            logger.error(f"Error loading components from {filepath}: {e}")

    def _serialize_metadata(self, metadata: ComponentMetadata) -> Dict[str, Any]:
        """
        Prepare metadata for JSON serialization.

        Args:
            metadata: Component metadata

        Returns:
            JSON-serializable dict
        """
        # Convert to dict
        data = metadata.model_dump()

        # Convert sets to lists for JSON serialization
        for key, value in data.items():
            if isinstance(value, set):
                data[key] = list(value)

        # Handle datetime objects
        if "discovered_at" in data and data["discovered_at"]:
            if isinstance(data["discovered_at"], datetime):
                data["discovered_at"] = data["discovered_at"].isoformat()

        return data

    def convert_to_tools(self) -> Dict[str, Any]:
        """
        Convert all components to LangChain tools.

        Returns:
            Dictionary of component ID to tool
        """
        tools = {}

        for component_id, metadata in self.collection.components.items():
            tool = self.tool_converter.to_structured_tool(metadata)
            if tool:
                tools[component_id] = tool

        return tools

    def convert_to_documents(self) -> List[Any]:
        """
        Convert all components to Documents.

        Returns:
            List of Documents
        """
        return self.document_converter.collection_to_documents(self.collection)

    def generate_markdown_docs(self, output_dir: Path) -> Dict[str, Path]:
        """
        Generate markdown documentation.

        Args:
            output_dir: Directory to save markdown files to

        Returns:
            Dictionary mapping file names to paths
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group components by type
        components_by_type = {}
        for component_type in ComponentType:
            components = self.collection.list_by_type(component_type)
            if components:
                components_by_type[component_type] = components

        # Generate files
        generated_files = {}

        # Generate index file
        index_path = output_dir / "index.md"
        with open(index_path, "w") as f:
            f.write("# Component Documentation\n\n")
            f.write(
                f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            )
            f.write(
                f"Found {len(self.collection.components)} components across {len(components_by_type)} types.\n\n"
            )

            # Add table of contents
            f.write("## Component Types\n\n")
            for component_type, components in components_by_type.items():
                type_name = component_type.value.replace("_", " ").title()
                f.write(
                    f"- [{type_name}]({component_type.value}.md) ({len(components)} components)\n"
                )

        generated_files["index"] = index_path

        # Generate a file for each component type
        for component_type, components in components_by_type.items():
            file_path = output_dir / f"{component_type.value}.md"
            with open(file_path, "w") as f:
                # Write header
                type_name = component_type.value.replace("_", " ").title()
                f.write(f"# {type_name} Components\n\n")
                f.write(
                    f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
                )
                f.write(f"Found {len(components)} {type_name} components.\n\n")

                # Group by category
                by_category = {}
                for comp in components:
                    if comp.category not in by_category:
                        by_category[comp.category] = []
                    by_category[comp.category].append(comp)

                # Add table of contents
                f.write("## Categories\n\n")
                for category, category_comps in sorted(by_category.items()):
                    category_name = category.replace("_", " ").title()
                    f.write(
                        f"- [{category_name}](#{category.lower().replace(' ', '-')}) ({len(category_comps)} components)\n"
                    )
                f.write("\n")

                # Write each category
                for category, category_comps in sorted(by_category.items()):
                    category_name = category.replace("_", " ").title()
                    f.write(f"## {category_name}\n\n")

                    # Write each component
                    for comp in sorted(category_comps, key=lambda x: x.name):
                        # Convert to document to get markdown content
                        doc = self.document_converter.to_document(comp)
                        if doc:
                            # Extract just the component section (strip the header)
                            content = doc.page_content
                            # Write the content
                            f.write(content)
                            f.write("\n---\n\n")

            generated_files[component_type.value] = file_path

        return generated_files


# Create a function for standalone usage
def discover_components(
    package_names: Optional[List[str]] = None,
    output_file: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    generate_markdown: bool = False,
) -> ComponentCollection:
    """
    Discover components from the specified packages.

    Args:
        package_names: List of packages to discover from (None for all supported)
        output_file: Path to save JSON output (None for no output)
        output_dir: Directory to save markdown docs (None for no docs)
        generate_markdown: Whether to generate markdown documentation

    Returns:
        Collection of discovered components
    """
    # Create discoverer
    discoverer = ComponentDiscovery()

    # Discover components
    if package_names:
        for package_name in package_names:
            discoverer.discover_from_package(package_name)
    else:
        discoverer.discover_all()

    # Save output if requested
    if output_file:
        output_path = Path(output_file)
        discoverer.save_to_file(output_path)

    # Generate markdown if requested
    if generate_markdown and output_dir:
        output_path = Path(output_dir)
        discoverer.generate_markdown_docs(output_path)

    return discoverer.collection
