"""
Conversion utilities for turning components into Documents.

Converts discovered components into LangChain Documents for vectorization and retrieval.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

try:
    from langchain_core.documents import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from ..models.component import ComponentCollection, ComponentMetadata

logger = logging.getLogger(__name__)


class DocumentConverter:
    """Converts components to LangChain Documents."""

    def to_document(self, metadata: ComponentMetadata) -> Optional[Any]:
        """
        Convert a component metadata to a Document.

        Args:
            metadata: Component metadata to convert

        Returns:
            Document if successful, None if LangChain not available
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, cannot create document")
            return None

        # Build document content as Markdown
        content = self._build_markdown_content(metadata)

        # Create metadata dictionary
        doc_metadata = {
            "component_id": metadata.id,
            "component_name": metadata.name,
            "component_type": metadata.component_type.value,
            "module_path": metadata.module_path,
            "category": metadata.category,
            "tags": metadata.tags,
            "discovered_at": (
                metadata.discovered_at.isoformat() if metadata.discovered_at else None
            ),
        }

        # Add environment variables status
        if metadata.env_vars:
            doc_metadata["env_vars"] = list(metadata.env_vars)
            doc_metadata["env_vars_available"] = len(metadata.env_vars_found) == len(
                metadata.env_vars
            )

        # Create document
        document = Document(page_content=content, metadata=doc_metadata)

        return document

    def collection_to_documents(self, collection: ComponentCollection) -> List[Any]:
        """
        Convert a collection of components to Documents.

        Args:
            collection: Component collection to convert

        Returns:
            List of Documents, empty list if LangChain not available
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, cannot create documents")
            return []

        documents = []

        for metadata in collection.components.values():
            doc = self.to_document(metadata)
            if doc:
                documents.append(doc)

        return documents

    def _build_markdown_content(self, metadata: ComponentMetadata) -> str:
        """
        Build markdown content for the component.

        Args:
            metadata: Component metadata

        Returns:
            Markdown formatted string
        """
        lines = []

        # Title and description
        lines.append(f"# {metadata.name}")
        if metadata.display_name and metadata.display_name != metadata.name:
            lines.append(f"*{metadata.display_name}*")
        lines.append("")

        if metadata.description:
            lines.append(metadata.description)
            lines.append("")

        # Component type and category
        lines.append(
            f"**Type:** {metadata.component_type.value.replace('_', ' ').title()}"
        )
        lines.append(f"**Category:** {metadata.category.replace('_', ' ').title()}")
        if metadata.tags:
            lines.append(f"**Tags:** {', '.join(metadata.tags)}")
        lines.append("")

        # Module path
        lines.append(f"**Module:** `{metadata.module_path}`")
        if metadata.parent_classes:
            lines.append(f"**Parent Classes:** {', '.join(metadata.parent_classes)}")
        lines.append("")

        # Add special capabilities based on component type
        self._add_type_specific_content(metadata, lines)

        # Key parameters
        if metadata.key_parameters:
            lines.append("## Key Parameters")
            lines.append("")
            for param in metadata.key_parameters:
                param_info = self._get_parameter_info(metadata, param)
                if param_info:
                    lines.append(f"- **{param}**: {param_info}")
            lines.append("")

        # Methods
        if metadata.methods:
            lines.append("## Methods")
            lines.append("")

            for name, method in metadata.methods.items():
                # Skip private methods except __init__
                if name.startswith("_") and name != "__init__":
                    continue

                # Method signature
                signature = method.signature_str or f"{name}(...)"
                lines.append(f"### `{name}`")
                lines.append("")
                lines.append(f"```python")
                lines.append(f"def {signature}")
                lines.append(f"```")
                lines.append("")

                # Method description
                if method.docstring:
                    lines.append(method.docstring)
                    lines.append("")

                # Parameters if any
                if method.parameters:
                    lines.append("**Parameters:**")
                    lines.append("")
                    for param_name, param in method.parameters.items():
                        required = "Required" if param.is_required else "Optional"
                        default = (
                            f", default: `{param.default_value}`"
                            if not param.is_required
                            else ""
                        )
                        lines.append(
                            f"- `{param_name}` ({param.type_hint}): {required}{default}"
                        )
                        if param.description:
                            lines.append(f"  {param.description}")
                    lines.append("")

                # Return type
                if method.return_type and method.return_type != "None":
                    lines.append(f"**Returns:** {method.return_type}")
                    lines.append("")

                # Async indicator
                if method.is_async:
                    lines.append("*This method is async.*")
                    lines.append("")

        # Environment variables
        if metadata.env_vars:
            lines.append("## Environment Variables")
            lines.append("")
            for var in sorted(metadata.env_vars):
                status = "✅" if var in metadata.env_vars_found else "❌"
                lines.append(f"- {status} `{var}`")
            lines.append("")

        return "\n".join(lines)

    def _get_parameter_info(self, metadata: ComponentMetadata, param_name: str) -> str:
        """
        Get parameter information from component metadata.

        Args:
            metadata: Component metadata
            param_name: Parameter name to find

        Returns:
            Parameter description or type information
        """
        # Check in __init__ method
        if (
            "__init__" in metadata.methods
            and param_name in metadata.methods["__init__"].parameters
        ):
            param = metadata.methods["__init__"].parameters[param_name]
            desc = param.description or ""
            if param.type_hint and param.type_hint != "Any":
                if desc:
                    return f"{desc} ({param.type_hint})"
                return param.type_hint
            return desc

        # Check in other methods
        for method_name, method in metadata.methods.items():
            if param_name in method.parameters:
                param = method.parameters[param_name]
                desc = param.description or ""
                if param.type_hint and param.type_hint != "Any":
                    if desc:
                        return f"{desc} ({param.type_hint})"
                    return param.type_hint
                return desc

        return ""

    def _add_type_specific_content(self, metadata: ComponentMetadata, lines: List[str]):
        """
        Add component type-specific content to the markdown.

        Args:
            metadata: Component metadata
            lines: List of content lines to append to
        """
        if metadata.component_type.value == "document_loader":
            self._add_loader_content(metadata, lines)
        elif metadata.component_type.value == "text_splitter":
            self._add_splitter_content(metadata, lines)
        elif metadata.component_type.value == "retriever":
            self._add_retriever_content(metadata, lines)

    def _add_loader_content(self, metadata: ComponentMetadata, lines: List[str]):
        """
        Add document loader specific content.

        Args:
            metadata: Component metadata
            lines: List of content lines to append to
        """
        if metadata.loader_methods:
            lines.append("## Loader Capabilities")
            lines.append("")

            # Check each capability
            capabilities = []
            if metadata.loader_methods.get("load"):
                capabilities.append("- ✅ Synchronous loading via `load()`")
            else:
                capabilities.append("- ❌ No synchronous loading")

            if metadata.loader_methods.get("aload"):
                capabilities.append("- ✅ Asynchronous loading via `aload()`")
            else:
                capabilities.append("- ❌ No asynchronous loading")

            if metadata.loader_methods.get("load_and_split"):
                capabilities.append("- ✅ Integrated splitting via `load_and_split()`")
            else:
                capabilities.append("- ❌ No integrated splitting")

            if metadata.loader_methods.get("lazy_load"):
                capabilities.append("- ✅ Lazy loading via `lazy_load()`")

            if metadata.loader_methods.get("from_file"):
                capabilities.append("- ✅ Direct file loading via `from_file()`")

            if metadata.loader_methods.get("from_files"):
                capabilities.append("- ✅ Multi-file loading via `from_files()`")

            lines.extend(capabilities)
            lines.append("")

    def _add_splitter_content(self, metadata: ComponentMetadata, lines: List[str]):
        """
        Add text splitter specific content.

        Args:
            metadata: Component metadata
            lines: List of content lines to append to
        """
        if hasattr(metadata, "splitter_methods") and metadata.splitter_methods:
            lines.append("## Splitter Capabilities")
            lines.append("")

            # Check each capability
            capabilities = []
            if metadata.splitter_methods.get("split_text"):
                capabilities.append("- ✅ Can split raw text via `split_text()`")
            else:
                capabilities.append("- ❌ Cannot split raw text")

            if metadata.splitter_methods.get("split_documents"):
                capabilities.append("- ✅ Can split documents via `split_documents()`")
            else:
                capabilities.append("- ❌ Cannot split documents")

            lines.extend(capabilities)
            lines.append("")

    def _add_retriever_content(self, metadata: ComponentMetadata, lines: List[str]):
        """
        Add retriever specific content.

        Args:
            metadata: Component metadata
            lines: List of content lines to append to
        """
        if hasattr(metadata, "retriever_methods") and metadata.retriever_methods:
            lines.append("## Retriever Capabilities")
            lines.append("")

            # Check each capability
            capabilities = []
            if metadata.retriever_methods.get("get_relevant_documents"):
                capabilities.append(
                    "- ✅ Synchronous retrieval via `get_relevant_documents()`"
                )
            else:
                capabilities.append("- ❌ No synchronous retrieval")

            if metadata.retriever_methods.get("aget_relevant_documents"):
                capabilities.append(
                    "- ✅ Asynchronous retrieval via `aget_relevant_documents()`"
                )
            else:
                capabilities.append("- ❌ No asynchronous retrieval")

            lines.extend(capabilities)
            lines.append("")
