"""
Document loader engine for loading documents from sources.

This module provides an InvokableEngine implementation for document loaders that can
handle various source types by mapping source objects to appropriate loaders from
langchain_community.
"""

import asyncio
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from pydantic import Field, root_validator

from haive.core.engine.base import InvokableEngine
from haive.core.engine.base.types import EngineType
from haive.core.engine.document.loaders.registry import (
    DocumentLoaderRegistry,
)
from haive.core.engine.document.loaders.sources.base.base import BaseSource
from haive.core.engine.document.loaders.sources.types import SourceType

logger = logging.getLogger(__name__)

# Type variable for document-like objects
D = TypeVar("D", bound=Document)


class DocumentLoaderEngine(
    InvokableEngine[Union[BaseSource, str, Path, Dict[str, Any]], List[Document]]
):
    """
    Engine for loading documents from various sources.

    This engine provides a unified interface for working with document loaders from
    langchain_community, with support for different source types and configurations.
    """

    # Engine type
    engine_type: EngineType = Field(default=EngineType.DOCUMENT_LOADER)

    # Loader configuration
    loader_name: Optional[str] = Field(
        default=None,
        description="Name of the document loader to use (auto-detected if not provided)",
    )

    # Source type filters
    supported_source_types: List[SourceType] = Field(
        default_factory=list,
        description="Source types this loader supports (all if empty)",
    )

    # Document loading options
    recursive: bool = Field(
        default=True, description="Whether to recursively load from directory sources"
    )
    max_documents: Optional[int] = Field(
        default=None,
        description="Maximum number of documents to load (None for unlimited)",
    )
    use_async: bool = Field(
        default=False, description="Whether to use async loading if available"
    )

    # Additional configuration for the loader
    loader_options: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration parameters for the loader"
    )

    model_config = {"arbitrary_types_allowed": True}

    @root_validator(pre=True)
    def ensure_valid_configuration(cls, values):
        """Ensure the engine has a valid configuration."""
        # If loader_name is provided, verify it exists
        if loader_name := values.get("loader_name"):
            registry = DocumentLoaderRegistry.get_instance()
            if not registry.find_by_name(loader_name):
                raise ValueError(
                    f"Document loader '{loader_name}' not found in registry"
                )

        return values

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Get input field definitions for this engine.

        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        return {
            "source": (
                Union[BaseSource, str, Path, Dict[str, Any]],
                Field(description="Source to load documents from"),
            ),
            "recursive": (
                bool,
                Field(
                    default=self.recursive,
                    description="Whether to recursively load from directory sources",
                ),
            ),
            "max_documents": (
                Optional[int],
                Field(
                    default=self.max_documents,
                    description="Maximum number of documents to load (None for unlimited)",
                ),
            ),
            "loader_options": (
                Dict[str, Any],
                Field(
                    default_factory=dict,
                    description="Additional options to pass to the loader",
                ),
            ),
        }

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Get output field definitions for this engine.

        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        return {
            "documents": (
                List[Document],
                Field(description="List of loaded documents"),
            ),
            "metadata": (
                Dict[str, Any],
                Field(
                    default_factory=dict,
                    description="Metadata about the loading process",
                ),
            ),
        }

    def _resolve_source(
        self, source: Union[BaseSource, str, Path, Dict[str, Any]]
    ) -> BaseSource:
        """
        Resolve various source formats to a BaseSource object.

        Args:
            source: Source in various formats

        Returns:
            BaseSource object

        Raises:
            ValueError: If source cannot be resolved
        """
        # Already a BaseSource
        if isinstance(source, BaseSource):
            return source

        # Import source factories here to avoid circular imports
        from haive.core.engine.document.loaders.sources.factory import (
            create_source_from_dict,
            create_source_from_path,
            create_source_from_string,
        )

        # String source
        if isinstance(source, str):
            return create_source_from_string(source)

        # Path source
        if isinstance(source, Path):
            return create_source_from_path(source)

        # Dict source
        if isinstance(source, dict):
            return create_source_from_dict(source)

        raise ValueError(f"Cannot resolve source of type: {type(source)}")

    def _select_loader_name(self, source: BaseSource) -> str:
        """
        Select an appropriate loader based on the source.

        Args:
            source: Source to load from

        Returns:
            Name of the selected loader

        Raises:
            ValueError: If no suitable loader is found
        """
        # Use explicitly specified loader if provided
        if self.loader_name:
            return self.loader_name

        registry = DocumentLoaderRegistry.get_instance()
        source_type = source.source_type

        # Find loaders that support this source type
        loaders_for_type = registry.get_all(source_type)
        if loaders_for_type:
            # Return the first loader
            return next(iter(loaders_for_type.keys()))

        # Try to find a generic loader based on source class
        from haive.core.engine.document.loaders.sources.base.base import SourceClass

        source_class = getattr(source, "source_class", None)
        if source_class:
            if source_class == SourceClass.LOCAL:
                # Try file loader
                if registry.find_by_name("TextLoader"):
                    return "TextLoader"
                elif registry.find_by_name("UnstructuredFileLoader"):
                    return "UnstructuredFileLoader"
            elif source_class == SourceClass.WEB:
                # Try web loader
                if registry.find_by_name("WebBaseLoader"):
                    return "WebBaseLoader"

        # Fall back to a default loader if available
        if registry.find_by_name("UnstructuredLoader"):
            return "UnstructuredLoader"

        raise ValueError(f"No suitable loader found for source type {source_type}")

    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> Any:
        """
        Create a runnable function from this engine configuration.

        Args:
            runnable_config: Optional runtime configuration

        Returns:
            A callable that loads documents
        """
        # Extract relevant parameters from config
        params = self.apply_runnable_config(runnable_config) or {}

        # Update options with runtime overrides
        loader_options = self.loader_options.copy()
        if "loader_options" in params:
            loader_options.update(params["loader_options"])

        # Create a copy of this engine with updated config
        updated_engine = self.model_copy(update={"loader_options": loader_options})

        # Create a function that will be returned as the runnable
        async def load_documents_callable(
            source: Union[BaseSource, str, Path, Dict[str, Any]],
        ) -> List[Document]:
            # Resolve the source
            resolved_source = updated_engine._resolve_source(source)

            # Select loader
            loader_name = updated_engine._select_loader_name(resolved_source)

            # Get loader class
            registry = DocumentLoaderRegistry.get_instance()
            loader_class = registry.find_by_name(loader_name)
            if not loader_class:
                raise ValueError(
                    f"Document loader '{loader_name}' not found in registry"
                )

            # Prepare loader options
            options = loader_options.copy()

            # Add source-specific options
            source_str = str(resolved_source.source)
            if hasattr(loader_class, "__init__"):
                sig = inspect.signature(loader_class.__init__)
                for param_name in sig.parameters:
                    if param_name == "file_path" and os.path.exists(source_str):
                        options.setdefault("file_path", source_str)
                    elif param_name == "path" and os.path.exists(source_str):
                        options.setdefault("path", source_str)
                    elif param_name == "url" and (
                        source_str.startswith("http://")
                        or source_str.startswith("https://")
                    ):
                        options.setdefault("url", source_str)
                    elif param_name == "source":
                        options.setdefault("source", source_str)

            # Create loader instance
            try:
                loader_instance = loader_class(**options)
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate loader '{loader_name}': {str(e)}"
                )

            # Load documents
            try:
                if updated_engine.use_async and hasattr(loader_instance, "aload"):
                    documents = await loader_instance.aload()
                elif hasattr(loader_instance, "load"):
                    if updated_engine.use_async:
                        documents = await asyncio.to_thread(loader_instance.load)
                    else:
                        documents = loader_instance.load()
                else:
                    raise ValueError(
                        f"Loader {loader_name} does not have a load method"
                    )

                # Apply document limit if specified
                if updated_engine.max_documents is not None:
                    documents = documents[: updated_engine.max_documents]

                return documents
            except Exception as e:
                raise ValueError(f"Failed to load documents: {str(e)}")

        # Add reference to the engine for inspection
        load_documents_callable.engine = updated_engine

        return load_documents_callable

    def invoke(
        self,
        source: Union[BaseSource, str, Path, Dict[str, Any]],
        runnable_config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """
        Load documents from the specified source.

        Args:
            source: Source to load documents from
            runnable_config: Optional runtime configuration

        Returns:
            List of loaded documents

        Raises:
            ValueError: If loading fails
        """
        # Create runnable
        runnable = self.create_runnable(runnable_config)

        # Invoke synchronously
        try:
            if asyncio.iscoroutinefunction(runnable):
                # Get or create event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                return loop.run_until_complete(runnable(source))
            else:
                return runnable(source)
        except Exception as e:
            logger.error(f"Error in document loader: {str(e)}")
            raise

    async def ainvoke(
        self,
        source: Union[BaseSource, str, Path, Dict[str, Any]],
        runnable_config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """
        Load documents asynchronously from the specified source.

        Args:
            source: Source to load documents from
            runnable_config: Optional runtime configuration

        Returns:
            List of loaded documents

        Raises:
            ValueError: If loading fails
        """
        # Create runnable
        runnable = self.create_runnable(runnable_config)

        # Invoke asynchronously
        try:
            if asyncio.iscoroutinefunction(runnable):
                return await runnable(source)
            else:
                return await asyncio.to_thread(runnable, source)
        except Exception as e:
            logger.error(f"Error in document loader: {str(e)}")
            raise


# Factory methods for common sources
def create_file_loader_engine(
    file_path: Union[str, Path], loader_name: Optional[str] = None, **loader_options
) -> DocumentLoaderEngine:
    """
    Create a document loader engine for a file.

    Args:
        file_path: Path to the file to load
        loader_name: Optional specific loader to use
        **loader_options: Additional configuration for the loader

    Returns:
        DocumentLoaderEngine configured for the file
    """
    file_path_str = str(file_path)
    file_name = os.path.basename(file_path_str)

    engine_name = f"{loader_name or 'file'}_loader_{file_name}"
    return DocumentLoaderEngine(
        name=engine_name,
        loader_name=loader_name,
        loader_options=loader_options,
    ).register()


def create_url_loader_engine(
    url: str, loader_name: Optional[str] = None, **loader_options
) -> DocumentLoaderEngine:
    """
    Create a document loader engine for a URL.

    Args:
        url: URL to load
        loader_name: Optional specific loader to use
        **loader_options: Additional configuration for the loader

    Returns:
        DocumentLoaderEngine configured for the URL
    """
    from urllib.parse import urlparse

    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    engine_name = f"{loader_name or 'url'}_loader_{domain}"
    return DocumentLoaderEngine(
        name=engine_name,
        loader_name=loader_name,
        loader_options=loader_options,
    ).register()


def create_directory_loader_engine(
    directory_path: Union[str, Path],
    loader_name: Optional[str] = None,
    glob_pattern: Optional[str] = None,
    recursive: bool = True,
    **loader_options,
) -> DocumentLoaderEngine:
    """
    Create a document loader engine for a directory.

    Args:
        directory_path: Path to the directory to load
        loader_name: Optional specific loader to use
        glob_pattern: Optional glob pattern for filtering files
        recursive: Whether to load files recursively
        **loader_options: Additional configuration for the loader

    Returns:
        DocumentLoaderEngine configured for the directory
    """
    directory_str = str(directory_path)
    dir_name = os.path.basename(directory_str)

    options = loader_options.copy()
    if glob_pattern:
        options["glob"] = glob_pattern

    engine_name = f"{loader_name or 'directory'}_loader_{dir_name}"
    return DocumentLoaderEngine(
        name=engine_name,
        loader_name=loader_name,
        recursive=recursive,
        loader_options=options,
    ).register()
