"""Enhanced Document Engine Package.

This package provides comprehensive document processing capabilities including:
- Document loading from various sources (files, URLs, databases, cloud storage)
- Advanced chunking and processing strategies
- Path analysis and source type detection
- Parallel processing and error handling
- Integration with the Haive engine framework

Key Components:
- DocumentEngine: Main engine for document processing
- DocumentEngineConfig: Configuration model for the engine
- Path analysis system for source type detection
- Document loaders for various source types
- Processing strategies for chunking and transformation
"""

from haive.core.engine.document.config import (
    ChunkingStrategy,
    DocumentChunk,
    DocumentEngineConfig,
    DocumentFormat,
    DocumentInput,
    DocumentOutput,
    DocumentSourceType,
    LoaderPreference,
    ProcessedDocument,
    ProcessingStrategy,
)

# Core engine components
from haive.core.engine.document.engine import (
    DocumentEngine,
    create_directory_document_engine,
    create_file_document_engine,
    create_web_document_engine,
)
from haive.core.engine.document.factory import (
    AutoLoaderFactory,
    analyze_source,
    create_document_loader,
)

# Loader components
from haive.core.engine.document.loaders.base.base import (
    BaseDocumentLoader,
    SimpleDocumentLoader,
    TextDocumentLoader,
)
from haive.core.engine.document.loaders.registry import (
    DocumentLoaderRegistry,
    create_loader,
    get_default_registry,
    get_loader,
    register_loader,
)
from haive.core.engine.document.loaders.sources.implementation import (
    CredentialManager,
    EnhancedSource,
)
from haive.core.engine.document.loaders.specific import (  # Database sources
    MongoDBSource,
    PostgreSQLSource,
)
from haive.core.engine.document.loaders.strategy import (
    LoaderCapability,
    LoaderPriority,
    LoaderStrategy,
    strategy_registry,
)

# Path analysis
from haive.core.engine.document.path_analysis import (
    CloudProvider,
    DatabaseType,
    FileCategory,
    PathAnalysisResult,
    PathType,
    analyze_path_comprehensive,
)

# Configuration models
from haive.core.engine.document.processors import (
    ChunkingProcessor,
    ContentNormalizer,
    DocumentProcessor,
    FormatDetector,
    MetadataExtractor,
)


# Factory functions for convenience
def create_document_engine(name: str = "document_engine", **kwargs) -> DocumentEngine:
    """Create a basic document engine with default configuration.

    Args:
        name: Name for the engine instance
        **kwargs: Additional configuration options

    Returns:
        Configured DocumentEngine instance
    """
    config = DocumentEngineConfig(name=name, **kwargs)
    return DocumentEngine(config=config)


def load_documents(
    source,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs,
) -> DocumentOutput:
    """Quick function to load and process documents.

    Args:
        source: Source to load from (path, URL, etc.)
        chunking_strategy: Strategy for chunking documents
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks
        **kwargs: Additional processing options

    Returns:
        DocumentOutput with processed documents
    """
    engine = create_document_engine(
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )

    return engine.invoke(source)


# Agents (conditional import - only if agents package available)
try:
    from haive.core.engine.document.agents import (
        DirectoryDocumentAgent,
        DocumentAgent,
        FileDocumentAgent,
        WebDocumentAgent,
    )
except ImportError:
    # Agents not available - they depend on haive-agents package
    DirectoryDocumentAgent = None
    DocumentAgent = None
    FileDocumentAgent = None
    WebDocumentAgent = None

# Factory and enhanced loaders

# Specific loader implementations - only import what exists

# Processors

# Universal loader system - temporarily commented out until dependencies are fixed
# from .universal_loader import (
#     UniversalDocumentLoader,
#     SmartSourceRegistry,
#     load_document,
#     analyze_document_source,
# )


# Export all public components
__all__ = [
    # Factory and enhanced loaders
    "AutoLoaderFactory",
    # Loaders
    "BaseDocumentLoader",
    "ChunkingProcessor",
    "ChunkingStrategy",
    "CloudProvider",
    "ContentNormalizer",
    "CredentialManager",
    "DatabaseType",
    "DirectoryDocumentAgent",
    # Agents
    "DocumentAgent",
    "DocumentChunk",
    # Core engine
    "DocumentEngine",
    # Configuration
    "DocumentEngineConfig",
    # Enums
    "DocumentFormat",
    "DocumentInput",
    # Registry
    "DocumentLoaderRegistry",
    "DocumentOutput",
    # Processors
    "DocumentProcessor",
    "DocumentSourceType",
    "EnhancedSource",
    # Universal loader system - commented out temporarily
    # "UniversalDocumentLoader",
    # "SmartSourceRegistry",
    # "load_document",
    # "analyze_document_source",
    # Enhanced source system
    "EnhancedSourceType",
    "FileCategory",
    "FileDocumentAgent",
    "FormatDetector",
    "LoaderCapability",
    "LoaderPreference",
    "LoaderPriority",
    # Strategy system
    "LoaderStrategy",
    "MetadataExtractor",
    # Specific loader implementations - only what exists
    # Database sources
    "MongoDBSource",
    "PathAnalysisResult",
    "PathType",
    "PostgreSQLSource",
    "ProcessedDocument",
    "ProcessingStrategy",
    "SimpleDocumentLoader",
    "TextDocumentLoader",
    "WebDocumentAgent",
    # Path analysis
    "analyze_path_comprehensive",
    "analyze_source",
    "create_directory_document_engine",
    "create_document_engine",
    "create_document_loader",
    # Factory functions
    "create_file_document_engine",
    "create_loader",
    "create_web_document_engine",
    "get_default_registry",
    "get_loader",
    "load_documents",
    "register_loader",
    "source_registry",
    "strategy_registry",
]
