"""Engine module for the Haive framework.

This module provides the core engine system that powers all AI agents and workflows
in Haive. It includes LLM configurations, retriever systems, vector stores, and
the base engine infrastructure.

The engine system is designed to be modular and extensible, allowing for easy
integration of new AI models, retrieval systems, and vector storage backends.

Modules:
    aug_llm: Augmented LLM configurations and factories for creating LLM engines
    base: Base engine classes, protocols, and registry system
    retriever: Retriever implementations for RAG (Retrieval-Augmented Generation)
    vectorstore: Vector store integrations for semantic search and storage
    document: Document processing and transformation utilities
    agent: Agent-specific engine components

Key Classes:
    AugLLMConfig: Configuration for augmented LLM engines with enhanced capabilities
    Engine: Base class for all engine implementations
    EngineRegistry: Registry for managing and discovering engine types
    BaseRetrieverConfig: Configuration for retriever engines
    VectorStoreConfig: Configuration for vector store engines

Factory Functions:
    create_retriever: Create a retriever engine from configuration
    create_vectorstore: Create a vector store engine from configuration
    create_retriever_from_documents: Create a retriever with pre-loaded documents

Examples:
    Creating an LLM engine::

        from haive.core.engine import AugLLMConfig

        config = AugLLMConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )

    Creating a vector store::

        from haive.core.engine import create_vectorstore, VectorStoreConfig

        config = VectorStoreConfig(
            type="chroma",
            collection_name="knowledge_base"
        )
        vectorstore = create_vectorstore(config)

    Creating a retriever::

        from haive.core.engine import create_retriever

        retriever = create_retriever(
            vectorstore=vectorstore,
            search_kwargs={"k": 5}
        )
"""

# Agent imports are lazy-loaded to avoid expensive schema_composer initialization (17+ seconds)
# from haive.core.engine.agent import (...)
from haive.core.engine.aug_llm import (
    AugLLMConfig,
    AugLLMFactory,
    MCPAugLLMConfig,
    compose_runnable,
    merge_configs,
)
from haive.core.engine.base import (
    Engine,
    EngineRegistry,
    EngineType,
    InvokableEngine,
    NonInvokableEngine,
)

# Document imports are lazy-loaded to avoid expensive initialization
# from haive.core.engine.document import (...)
# Embedding imports are lazy-loaded to avoid numpy/pandas imports
# from haive.core.engine.embedding import (...)
from haive.core.engine.output_parser import (
    OutputParserEngine,
    OutputParserType,
)

# Prompt template imports are lazy-loaded to avoid circular import with schema_composer
# from haive.core.engine.prompt_template import (...)
# Retriever imports are lazy-loaded to avoid expensive initialization
# from haive.core.engine.retriever import (...)
from haive.core.engine.tool import (
    ToolEngine,
)

# Vectorstore imports are lazy-loaded to avoid pandas imports
# from haive.core.engine.vectorstore import (...)

# ========================================================================
# LAZY LOADING IMPLEMENTATION - Document components loaded on demand
# ========================================================================

# Component names for lazy loading
# Agent components - Heavy due to schema_composer (17+ seconds)
_AGENT_COMPONENTS = {
    "AGENT_REGISTRY",
    "Agent",
    "AgentConfig",
    "AgentProtocol",
    "PatternConfig",
    "PatternManager",
    "PersistentAgentProtocol",
    "StreamingAgentProtocol",
}

_DOCUMENT_COMPONENTS = {
    # Core engine components
    "DocumentEngine",
    "create_document_engine",
    "load_documents",
    # Factory functions
    "create_file_document_engine",
    "create_web_document_engine",
    "create_directory_document_engine",
    # Configuration models
    "DocumentEngineConfig",
    "DocumentInput",
    "DocumentOutput",
    "ProcessedDocument",
    "DocumentChunk",
    # Enums
    "DocumentFormat",
    "DocumentSourceType",
    "LoaderPreference",
    "ProcessingStrategy",
    "ChunkingStrategy",
    # Path analysis
    "analyze_path_comprehensive",
    "PathAnalysisResult",
    "PathType",
    "FileCategory",
    "DatabaseType",
    "CloudProvider",
    # Loaders
    "BaseDocumentLoader",
    "SimpleDocumentLoader",
    "TextDocumentLoader",
    "DocumentLoaderRegistry",
    "get_default_registry",
    "register_loader",
    "get_loader",
    "create_loader",
    # Processors
    "DocumentProcessor",
    "ChunkingProcessor",
    "ContentNormalizer",
    "FormatDetector",
    "MetadataExtractor",
    # Advanced components
    "AutoLoaderFactory",
    "create_document_loader",
    "analyze_source",
    "CredentialManager",
    "EnhancedSource",
    "LoaderStrategy",
    "LoaderCapability",
    "LoaderPriority",
    "MongoDBSource",
    "PostgreSQLSource",
}

_RETRIEVER_COMPONENTS = {
    # Core retriever components
    "BaseRetrieverConfig",
    "RetrieverType",
    "VectorStoreRetrieverConfig",
}

_PROMPT_COMPONENTS = {
    # Prompt template components - lazy to avoid circular import
    "PromptTemplateEngine"
}

_EMBEDDING_COMPONENTS = {
    # Embedding components - lazy to avoid numpy/pandas imports
    "BaseEmbeddingConfig",
    "EmbeddingType",
    "create_embedding_config",
}

_VECTORSTORE_COMPONENTS = {
    # Vectorstore components - lazy to avoid pandas imports
    "VectorStoreConfig",
    "create_retriever",
    "create_retriever_from_documents",
    "create_vectorstore",
    "create_vs_config_from_documents",
    "create_vs_from_documents",
}


def __getattr__(name: str):
    """Lazy loading for agent, document and retriever components to avoid expensive initialization."""
    if name in _AGENT_COMPONENTS:
        # Only import agent module when actually needed (17+ second schema_composer delay)
        from haive.core.engine.agent import (
            AGENT_REGISTRY,
            Agent,
            AgentConfig,
            AgentProtocol,
            PatternConfig,
            PatternManager,
            PersistentAgentProtocol,
            StreamingAgentProtocol,
        )

        # Return the requested component
        return locals()[name]

    if name in _DOCUMENT_COMPONENTS:
        # Only import document module when actually needed
        from haive.core.engine.document import (  # Core engine components; Factory functions; Configuration models; Enums; Path analysis; Loaders; Processors; Advanced components
            AutoLoaderFactory,
            BaseDocumentLoader,
            ChunkingProcessor,
            ChunkingStrategy,
            CloudProvider,
            ContentNormalizer,
            CredentialManager,
            DatabaseType,
            DocumentChunk,
            DocumentEngine,
            DocumentEngineConfig,
            DocumentFormat,
            DocumentInput,
            DocumentLoaderRegistry,
            DocumentOutput,
            DocumentProcessor,
            DocumentSourceType,
            EnhancedSource,
            FileCategory,
            FormatDetector,
            LoaderCapability,
            LoaderPreference,
            LoaderPriority,
            LoaderStrategy,
            MetadataExtractor,
            MongoDBSource,
            PathAnalysisResult,
            PathType,
            PostgreSQLSource,
            ProcessedDocument,
            ProcessingStrategy,
            SimpleDocumentLoader,
            TextDocumentLoader,
            analyze_path_comprehensive,
            analyze_source,
            create_directory_document_engine,
            create_document_engine,
            create_document_loader,
            create_file_document_engine,
            create_loader,
            create_web_document_engine,
            get_default_registry,
            get_loader,
            load_documents,
            register_loader,
        )

        # Return the requested component
        return locals()[name]

    if name in _RETRIEVER_COMPONENTS:
        # Only import retriever module when actually needed
        from haive.core.engine.retriever import (
            BaseRetrieverConfig,
            RetrieverType,
            VectorStoreRetrieverConfig,
        )

        # Return the requested component
        return locals()[name]

    if name in _PROMPT_COMPONENTS:
        # Only import prompt template module when actually needed
        from haive.core.engine.prompt_template import PromptTemplateEngine

        # Return the requested component
        return locals()[name]

    if name in _EMBEDDING_COMPONENTS:
        # Only import embedding module when actually needed
        from haive.core.engine.embedding import (
            BaseEmbeddingConfig,
            EmbeddingType,
            create_embedding_config,
        )

        # Return the requested component
        return locals()[name]

    if name in _VECTORSTORE_COMPONENTS:
        # Only import vectorstore module when actually needed
        from haive.core.engine.vectorstore import (
            VectorStoreConfig,
            create_retriever,
            create_retriever_from_documents,
            create_vectorstore,
            create_vs_config_from_documents,
            create_vs_from_documents,
        )

        # Return the requested component
        return locals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Agent Components - Lazy loaded via __getattr__
    # (All agent components are available but loaded on demand to avoid 17+ second schema_composer delay)
    *_AGENT_COMPONENTS,
    # LLM Components
    "AugLLMConfig",
    "AugLLMFactory",
    "MCPAugLLMConfig",
    "compose_runnable",
    "merge_configs",
    # Base Engine Classes
    "Engine",
    "EngineRegistry",
    "EngineType",
    "InvokableEngine",
    "NonInvokableEngine",
    # Document Components - Lazy loaded via __getattr__
    # (All document components are available but loaded on demand)
    *_DOCUMENT_COMPONENTS,
    # Retriever Components - Lazy loaded via __getattr__
    # (All retriever components are available but loaded on demand)
    *_RETRIEVER_COMPONENTS,
    # Embedding Components - Lazy loaded via __getattr__
    *_EMBEDDING_COMPONENTS,
    # Output Parser Components
    "OutputParserEngine",
    "OutputParserType",
    # Prompt Template Components - Lazy loaded via __getattr__
    *_PROMPT_COMPONENTS,
    # Tool Components
    "ToolEngine",
    # Vector Store Components - Lazy loaded via __getattr__
    *_VECTORSTORE_COMPONENTS,
]
