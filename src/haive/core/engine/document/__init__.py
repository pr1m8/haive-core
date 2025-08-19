"""📚 Document Engine - Intelligent Document Processing Revolution

**THE OMNIPOTENT DOCUMENT CONSCIOUSNESS THAT UNDERSTANDS EVERYTHING**

Welcome to the Document Engine - the revolutionary document intelligence platform 
that transforms static document processing into a living, adaptive understanding 
system. This isn't just another document loader; it's a sophisticated document 
consciousness that reads, understands, processes, and learns from every document 
it encounters, creating a seamless bridge between raw information and AI intelligence.

⚡ REVOLUTIONARY DOCUMENT INTELLIGENCE
------------------------------------

The Document Engine represents a paradigm shift from traditional document processing to 
**intelligent, adaptive document understanding systems** that evolve with content:

**🧠 Universal Document Understanding**: Processes any document type with intelligent format detection
**🔄 Adaptive Processing Strategies**: Dynamic chunking and processing based on content analysis
**⚡ Intelligent Source Detection**: AI-powered identification of optimal loading strategies
**📊 Context-Aware Chunking**: Smart content segmentation that preserves semantic meaning
**🎯 Multi-Source Intelligence**: Seamless processing from files, URLs, databases, and cloud storage

🌟 CORE DOCUMENT INNOVATIONS
---------------------------

**1. Intelligent Document Engine** 🚀
   Revolutionary document processing that thinks and adapts:
   ```python
   from haive.core.engine.document import DocumentEngine, DocumentEngineConfig
   from haive.core.engine.document import ChunkingStrategy, ProcessingStrategy
   
   # Create intelligent document engine with learning capabilities
   engine = DocumentEngine(
       config=DocumentEngineConfig(
           name="intelligent_processor",
           chunking_strategy=ChunkingStrategy.SEMANTIC_AWARE,
           processing_strategy=ProcessingStrategy.ADAPTIVE,
           learning_enabled=True,
           context_preservation=True
       )
   )
   
   # Engine automatically optimizes processing based on content
   engine.enable_content_learning(
       metrics=["chunk_quality", "semantic_coherence", "processing_speed"],
       optimization_target="content_understanding"
   )
   
   # Process documents with intelligent adaptation
   result = engine.invoke([
       "path/to/technical_manual.pdf",
       "https://api.docs.example.com/v1/guide",
       {"database": "mongodb://localhost", "collection": "documents"}
   ])
   
   # Engine learns optimal processing strategies for each content type
   processing_insights = engine.get_processing_insights()
   content_analysis = engine.get_content_analysis_report()
   
   # Apply learned optimizations automatically
   engine.apply_learned_optimizations(
       confidence_threshold=0.85,
       preserve_quality=True
   )
   ```

For complete examples and advanced patterns, see the documentation.
"""

from haive.core.engine.document.engine import (
    create_file_document_engine,
    create_web_document_engine,
    create_directory_document_engine,
)
from haive.core.engine.document.loaders import (
    get_default_registry,
    get_loader,
    load_documents,
    register_loader,
)
from haive.core.engine.document.loaders.adapters import (
    create_loader,
)
from haive.core.engine.document.registry import (
    source_registry,
    strategy_registry,
)

__all__ = [
    "create_directory_document_engine",
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
