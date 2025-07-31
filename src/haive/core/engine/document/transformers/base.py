"""Document Transformer Engine for Haive Framework.

This module provides an engine for transforming documents using various strategies
such as HTML conversion, document reordering, deduplication, and more.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from pydantic import ConfigDict, Field

from haive.core.engine.base import InvokableEngine
from haive.core.engine.base.types import EngineType
from haive.core.engine.document.transformers.types import DocTransformerType
from haive.core.engine.embeddings import EmbeddingsEngineConfig as EmbeddingsConfig
from haive.core.registry.base import AbstractRegistry
from haive.core.registry.decorators import register_component


@register_component(registry_getter="engine", auto_register=True)
class DocTransformerEngine(InvokableEngine[list[Document], list[Document]]):
    """Engine for transforming documents using various strategies.

    This engine supports multiple document transformation techniques including:
    - HTML to text conversion
    - HTML to markdown conversion
    - HTML content extraction and cleaning
    - Document reordering for long contexts
    - Redundant document filtering
    - Document clustering
    - Text translation
    - Metadata tagging
    """

    engine_type: EngineType = EngineType.DOCUMENT_TRANSFORMER

    # Main configuration
    transformer_type: DocTransformerType = Field(
        default=DocTransformerType.HTML_TO_TEXT,
        description="Type of document transformer to use",
    )

    # HTML transformer options
    ignore_links: bool = Field(
        default=True, description="Whether to ignore links in HTML"
    )
    ignore_images: bool = Field(
        default=True, description="Whether to ignore images in HTML"
    )

    # BeautifulSoup options
    unwanted_tags: list[str] = Field(
        default_factory=lambda: ["script", "style"],
        description="Tags to remove from HTML",
    )
    tags_to_extract: list[str] = Field(
        default_factory=lambda: ["p", "li", "div", "a"],
        description="Tags to extract from HTML",
    )
    unwanted_classnames: list[str] = Field(
        default_factory=list, description="Classnames to remove from HTML"
    )
    remove_comments: bool = Field(
        default=False, description="Whether to remove comments from HTML"
    )

    # Markdown options
    heading_style: str = Field(
        default="ATX", description="Heading style for markdown conversion"
    )
    autolinks: bool = Field(
        default=True, description="Whether to use automatic link style"
    )

    # Embeddings filter options
    similarity_threshold: float = Field(
        default=0.95, description="Threshold for embedding similarity filtering"
    )
    embeddings_model: Any | None = Field(
        default=None, description="Embeddings model for similarity filtering"
    )

    # Translation options
    target_language: str = Field(
        default="en", description="Target language for translation"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_input_fields(self) -> dict[str, tuple[type, Any]]:
        """Define input field requirements."""
        from pydantic import Field

        return {
            "documents": (
                list[Document],
                Field(default_factory=list, description="Documents to transform"),
            )
        }

    def get_output_fields(self) -> dict[str, tuple[type, Any]]:
        """Define output field requirements."""
        from pydantic import Field

        return {
            "documents": (
                list[Document],
                Field(default_factory=list, description="Transformed documents"),
            )
        }

    def create_runnable(self, runnable_config: RunnableConfig | None = None) -> Any:
        """Create a document transformer based on the configuration.

        Args:
            runnable_config: Optional runtime configuration

        Returns:
            Document transformer instance
        """
        # Extract config parameters
        params = self.apply_runnable_config(runnable_config or {})

        # Get parameters with fallbacks to instance values
        transformer_type = params.get("transformer_type", self.transformer_type)

        try:
            # Create the appropriate transformer based on type
            if transformer_type == DocTransformerType.HTML_TO_TEXT:
                from langchain_community.document_transformers import (
                    Html2TextTransformer,
                )

                return Html2TextTransformer(
                    ignore_links=params.get("ignore_links", self.ignore_links),
                    ignore_images=params.get("ignore_images", self.ignore_images),
                )

            if transformer_type == DocTransformerType.HTML_TO_MARKDOWN:
                from langchain_community.document_transformers import (
                    MarkdownifyTransformer,
                )

                return MarkdownifyTransformer(
                    heading_style=params.get("heading_style", self.heading_style),
                    autolinks=params.get("autolinks", self.autolinks),
                )

            if transformer_type == DocTransformerType.BEAUTIFUL_SOUP:
                from langchain_community.document_transformers import (
                    BeautifulSoupTransformer,
                )

                return BeautifulSoupTransformer()

            if transformer_type == DocTransformerType.LONG_CONTEXT_REORDER:
                from langchain_community.document_transformers import LongContextReorder

                return LongContextReorder()

            if transformer_type == DocTransformerType.EMBEDDINGS_REDUNDANT_FILTER:
                from langchain_community.document_transformers import (
                    EmbeddingsRedundantFilter,
                )

                embeddings = params.get("embeddings_model", self.embeddings_model)
                if embeddings is None:
                    raise ValueError(
                        "Embeddings model is required for EmbeddingsRedundantFilter"
                    )
                return EmbeddingsRedundantFilter(
                    embeddings=embeddings,
                    similarity_threshold=params.get(
                        "similarity_threshold", self.similarity_threshold
                    ),
                )

            if transformer_type == DocTransformerType.EMBEDDINGS_CLUSTERING_FILTER:
                from langchain_community.document_transformers import (
                    EmbeddingsClusteringFilter,
                )

                embeddings = params.get("embeddings_model", self.embeddings_model)
                if embeddings is None:
                    raise ValueError(
                        "Embeddings model is required for EmbeddingsClusteringFilter"
                    )
                return EmbeddingsClusteringFilter(embeddings=embeddings)

            if transformer_type == DocTransformerType.GOOGLE_TRANSLATE:
                from langchain_community.document_transformers import (
                    GoogleTranslateTransformer,
                )

                return GoogleTranslateTransformer(
                    target_language=params.get("target_language", self.target_language)
                )

            if transformer_type == DocTransformerType.OPENAI_METADATA_TAGGER:
                from langchain_community.document_transformers import (
                    OpenAIMetadataTagger,
                )

                return OpenAIMetadataTagger()

            # Default to HTML to text transformer
            from langchain_community.document_transformers import Html2TextTransformer

            return Html2TextTransformer()

        except ImportError as e:
            import logging

            logging.exception(f"Error importing document transformer: {e}")
            raise TypeError(f"Required package not installed for {transformer_type}")

        except Exception as e:
            import logging

            logging.exception(f"Error creating document transformer: {e}")
            raise ValueError(f"Failed to create document transformer: {e}")

    def invoke(
        self,
        input_data: list[Document] | dict[str, Any],
        runnable_config: RunnableConfig | None = None,
    ) -> list[Document]:
        """Transform documents using the configured transformer.

        Args:
            input_data: List of documents or dictionary with documents key
            runnable_config: Optional runtime configuration

        Returns:
            List of transformed documents
        """
        # Create the transformer
        transformer = self.create_runnable(runnable_config)

        # Extract documents from input
        if isinstance(input_data, list):
            documents = input_data
        elif isinstance(input_data, dict) and "documents" in input_data:
            documents = input_data["documents"]
        elif hasattr(input_data, "documents"):
            documents = input_data.documents
        else:
            raise ValueError(
                f"Invalid input type: {
                    type(input_data)
                }. Expected list of documents or dict with 'documents' key."
            )

        # Handle empty input
        if not documents:
            return []

        # Transform documents
        try:
            # Get transformer type from parameters or instance
            params = self.apply_runnable_config(runnable_config or {})
            transformer_type = params.get("transformer_type", self.transformer_type)

            # Extract other parameters based on transformer type
            if transformer_type == DocTransformerType.BEAUTIFUL_SOUP:
                return list(
                    transformer.transform_documents(
                        documents,
                        unwanted_tags=params.get("unwanted_tags", self.unwanted_tags),
                        tags_to_extract=params.get(
                            "tags_to_extract", self.tags_to_extract
                        ),
                        unwanted_classnames=params.get(
                            "unwanted_classnames", self.unwanted_classnames
                        ),
                        remove_comments=params.get(
                            "remove_comments", self.remove_comments
                        ),
                    )
                )
            # For other transformers, just pass documents
            return list(transformer.transform_documents(documents))

        except Exception as e:
            import logging

            logging.exception(f"Error transforming documents: {e}")
            # Return original documents if transformation fails
            return documents


# Factory functions for easy creation of document transformers


def create_document_transformer(
    transformer_type: DocTransformerType, name: str | None = None, **kwargs
) -> DocTransformerEngine:
    """Create a document transformer engine with the specified configuration.

    Args:
        transformer_type: Type of document transformer to create
        name: Name for the engine (generated if not provided)
        **kwargs: Additional parameters for specific transformer types

    Returns:
        Configured DocTransformerEngine
    """
    # Generate name if not provided
    if name is None:
        name = f"{transformer_type.value}_transformer"

    # Create the engine
    return DocTransformerEngine(name=name, transformer_type=transformer_type, **kwargs)


def create_html_to_text_transformer(
    name: str = "html_to_text_transformer",
    ignore_links: bool = True,
    ignore_images: bool = True,
) -> DocTransformerEngine:
    """Create an HTML to text transformer.

    Args:
        name: Name for the engine
        ignore_links: Whether to ignore links in HTML
        ignore_images: Whether to ignore images in HTML

    Returns:
        Configured DocTransformerEngine
    """
    return DocTransformerEngine(
        name=name,
        transformer_type=DocTransformerType.HTML_TO_TEXT,
        ignore_links=ignore_links,
        ignore_images=ignore_images,
    )


def create_html_to_markdown_transformer(
    name: str = "html_to_markdown_transformer",
    heading_style: str = "ATX",
    autolinks: bool = True,
) -> DocTransformerEngine:
    """Create an HTML to markdown transformer.

    Args:
        name: Name for the engine
        heading_style: Heading style for markdown conversion
        autolinks: Whether to use automatic link style

    Returns:
        Configured DocTransformerEngine
    """
    return DocTransformerEngine(
        name=name,
        transformer_type=DocTransformerType.HTML_TO_MARKDOWN,
        heading_style=heading_style,
        autolinks=autolinks,
    )


def create_long_context_reorder_transformer(
    name: str = "long_context_reorder_transformer",
) -> DocTransformerEngine:
    """Create a long context reordering transformer.

    This transformer helps address the "lost in the middle" problem where
    performance degrades when models must access relevant information in the
    middle of long contexts.

    Args:
        name: Name for the engine

    Returns:
        Configured DocTransformerEngine
    """
    return DocTransformerEngine(
        name=name, transformer_type=DocTransformerType.LONG_CONTEXT_REORDER
    )


def create_embeddings_filter_transformer(
    embeddings_model: EmbeddingsConfig,
    name: str = "embeddings_filter_transformer",
    similarity_threshold: float = 0.95,
    clustering: bool = False,
) -> DocTransformerEngine:
    """Create an embeddings-based filter transformer.

    This transformer removes redundant documents based on embedding similarity
    or clusters documents based on their embeddings.

    Args:
        name: Name for the engine
        embeddings_model: Embeddings model to use
        similarity_threshold: Threshold for embedding similarity filtering
        clustering: Whether to use clustering filter instead of redundancy filter

    Returns:
        Configured DocTransformerEngine
    """
    transformer_type = (
        DocTransformerType.EMBEDDINGS_CLUSTERING_FILTER
        if clustering
        else DocTransformerType.EMBEDDINGS_REDUNDANT_FILTER
    )

    return DocTransformerEngine(
        name=name,
        transformer_type=transformer_type,
        embeddings_model=embeddings_model,
        similarity_threshold=similarity_threshold,
    )


def create_translate_transformer(
    name: str = "translate_transformer", target_language: str = "en"
) -> DocTransformerEngine:
    """Create a document translation transformer.

    Args:
        name: Name for the engine
        target_language: Target language code

    Returns:
        Configured DocTransformerEngine
    """
    return DocTransformerEngine(
        name=name,
        transformer_type=DocTransformerType.GOOGLE_TRANSLATE,
        target_language=target_language,
    )


# Document Transformer Registry


class DocTransformerRegistry(AbstractRegistry[DocTransformerEngine]):
    """Registry for document transformer engines."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "DocTransformerRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry with empty dictionaries."""
        self.transformers = {}
        self.transformer_ids = {}

    def register(self, item: DocTransformerEngine) -> DocTransformerEngine:
        """Register a document transformer engine."""
        self.transformers[item.name] = item
        self.transformer_ids[item.id] = item
        return item

    def get(self, item_type: Any, name: str) -> DocTransformerEngine | None:
        """Get a document transformer by type and name."""
        return self.transformers.get(name)

    def find_by_id(self, id: str) -> DocTransformerEngine | None:
        """Find a document transformer by its unique ID."""
        return self.transformer_ids.get(id)

    def list(self, item_type: Any) -> list[str]:
        """List all document transformers."""
        return list(self.transformers.keys())

    def get_all(self, item_type: Any) -> dict[str, DocTransformerEngine]:
        """Get all document transformers."""
        return self.transformers

    def clear(self) -> None:
        """Clear the registry."""
        self.transformers = {}
        self.transformer_ids = {}
