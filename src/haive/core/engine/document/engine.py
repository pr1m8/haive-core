"""Enhanced Document Engine Implementation.

This module provides a comprehensive document engine that integrates with the Haive
engine framework for loading, processing, and managing documents from various sources.

The engine supports multiple input types, advanced processing strategies, and
comprehensive error handling while maintaining compatibility with the existing
Haive engine infrastructure.
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Self, Union

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from pydantic import Field, model_validator

from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin
from haive.core.engine.base import InvokableEngine
from haive.core.engine.base.types import EngineType
from haive.core.engine.document.config import (
    ChunkingStrategy,
    DocumentChunk,
    DocumentEngineConfig,
    DocumentFormat,
    DocumentInput,
    DocumentOutput,
    DocumentSourceType,
    ProcessedDocument,
    ProcessingStrategy,
)
from haive.core.engine.document.loaders.base.base import BaseDocumentLoader
from haive.core.engine.document.loaders.registry import DocumentLoaderRegistry
from haive.core.engine.document.path_analysis import (
    PathAnalysisResult,
    analyze_path_comprehensive,
)

logger = logging.getLogger(__name__)

class DocumentEngine(ToolRouteMixin, InvokableEngine[Union[DocumentInput, str, Path, dict[str, Any]], DocumentOutput]):
    """Enhanced Document Engine for comprehensive document processing.

    This engine provides a unified interface for loading documents from various
    sources with advanced processing capabilities including chunking, metadata
    extraction, and parallel processing.

    Features:
    - Multiple input formats (paths, URLs, configs)
    - Auto-detection of source types and formats
    - Advanced chunking strategies
    - Parallel processing capabilities
    - Comprehensive error handling
    - Integration with Haive engine framework
    """
    engine_type: EngineType = Field(default=EngineType.DOCUMENT_LOADER)
    config: DocumentEngineConfig = Field(default_factory=DocumentEngineConfig, description='Engine configuration')
    loader_registry: DocumentLoaderRegistry | None = Field(default=None, description='Document loader registry', exclude=True)
    processing_stats: dict[str, Any] = Field(default_factory=dict, description='Processing statistics', exclude=True)

    def __init__(self, **kwargs) -> None:
        """Initialize the document engine."""
        super().__init__(**kwargs)
        if self.loader_registry is None:
            self.loader_registry = DocumentLoaderRegistry()
        self.processing_stats = {'total_processed': 0, 'successful': 0, 'failed': 0, 'total_time': 0.0}
        logger.debug(f'Initialized DocumentEngine: {self.config.name}')

    @model_validator(mode='after')
    def validate_config(self) -> Self:
        """Validate engine configuration."""
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError('chunk_overlap must be less than chunk_size')
        if self.config.max_workers > 32:
            logger.warning('max_workers > 32 may cause performance issues')
        return self

    def get_input_fields(self) -> dict[str, Any]:
        """Get input field definitions for the engine."""
        return {'source': (Union[str, Path, dict[str, Any]], ...), 'source_type': (Optional[str], None), 'loader_name': (Optional[str], None), 'loader_options': (dict[str, Any], {}), 'chunking_strategy': (Optional[str], None), 'chunk_size': (Optional[int], None), 'chunk_overlap': (Optional[int], None)}

    def get_output_fields(self) -> dict[str, Any]:
        """Get output field definitions for the engine."""
        return {'documents': (list[dict[str, Any]], []), 'total_documents': (int, 0), 'operation_time': (float, 0.0), 'source_type': (str, ''), 'loader_names': (list[str], []), 'original_source': (str, ''), 'processing_strategy': (str, ''), 'errors': (list[dict[str, Any]], []), 'has_errors': (bool, False)}

    def create_runnable(self, runnable_config: dict[str, Any] | None=None) -> 'DocumentEngine':
        """Create a runnable instance from this engine configuration."""
        if runnable_config:
            config_dict = self.config.model_dump()
            config_dict.update(runnable_config)
            new_config = DocumentEngineConfig.model_validate(config_dict)
            return DocumentEngine(config=new_config)
        return self

    def invoke(self, input_data: DocumentInput | str | Path | dict[str, Any], config: RunnableConfig | None=None) -> DocumentOutput:
        """Process documents synchronously.

        Args:
            input_data: Input data (path, URL, or DocumentInput)
            config: Optional runnable configuration

        Returns:
            DocumentOutput with processed documents and metadata
        """
        start_time = time.time()
        try:
            doc_input = self._normalize_input(input_data)
            analysis_result = self._analyze_source(doc_input.source)
            documents = self._load_documents(doc_input, analysis_result)
            processed_docs = self._process_documents(documents, doc_input)
            output = self._create_output(processed_docs, doc_input, analysis_result, time.time() - start_time)
            self._update_stats(output)
            return output
        except Exception as e:
            logger.exception(f'Document processing failed: {e}')
            return DocumentOutput(documents=[], total_documents=0, operation_time=time.time() - start_time, original_source=str(input_data), source_type=DocumentSourceType.UNKNOWN, processing_strategy=self.config.processing_strategy, errors=[{'error': str(e), 'type': type(e).__name__, 'source': str(input_data)}], has_errors=True)

    async def ainvoke(self, input_data: DocumentInput | str | Path | dict[str, Any], config: RunnableConfig | None=None) -> DocumentOutput:
        """Process documents asynchronously.

        Args:
            input_data: Input data (path, URL, or DocumentInput)
            config: Optional runnable configuration

        Returns:
            DocumentOutput with processed documents and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, input_data, config)

    def _normalize_input(self, input_data: DocumentInput | str | Path | dict[str, Any]) -> DocumentInput:
        """Normalize input to DocumentInput object."""
        if isinstance(input_data, DocumentInput):
            return input_data
        if isinstance(input_data, str | Path):
            return DocumentInput(source=str(input_data))
        if isinstance(input_data, dict):
            if 'source' in input_data:
                return DocumentInput(**input_data)
            return DocumentInput(source=input_data)
        raise TypeError(f'Invalid input type: {type(input_data)}')

    def _analyze_source(self, source: str | Path | dict[str, Any]) -> PathAnalysisResult:
        """Analyze the source to determine its properties."""
        if isinstance(source, dict):
            return PathAnalysisResult(original_path=str(source), path_type=DocumentSourceType.TEXT, is_local=False, is_remote=False)
        return analyze_path_comprehensive(source)

    def _load_documents(self, doc_input: DocumentInput, analysis: PathAnalysisResult) -> list[Document]:
        """Load documents from the source."""
        if doc_input.source_type:
            source_type = doc_input.source_type
        else:
            source_type = self._map_path_type_to_source_type(analysis.path_type)
        loader = self._get_loader(doc_input, source_type, analysis)
        try:
            if hasattr(loader, 'load'):
                documents = loader.load()
            elif hasattr(loader, 'load_documents'):
                documents = loader.load_documents()
            else:
                raise TypeError(f'Loader {type(loader)} has no load method')
            logger.info(f'Loaded {len(documents)} documents from {doc_input.source}')
            return documents
        except Exception as e:
            logger.exception(f'Failed to load documents: {e}')
            if self.config.raise_on_error:
                raise
            return []

    def _get_loader(self, doc_input: DocumentInput, source_type: DocumentSourceType, analysis: PathAnalysisResult) -> BaseDocumentLoader:
        """Get the appropriate document loader."""
        if doc_input.loader_name:
            loader = self.loader_registry.get_loader(doc_input.loader_name)
            if loader:
                return loader
        return self._auto_select_loader(source_type, analysis, doc_input)

    def _auto_select_loader(self, source_type: DocumentSourceType, analysis: PathAnalysisResult, doc_input: DocumentInput) -> BaseDocumentLoader:
        """Auto-select the best loader for the source."""
        from haive.core.engine.document.loaders.base.base import SimpleDocumentLoader
        return SimpleDocumentLoader(source=doc_input.source, loader_options=doc_input.loader_options)

    def _process_documents(self, documents: list[Document], doc_input: DocumentInput) -> list[ProcessedDocument]:
        """Process loaded documents according to configuration."""
        if not documents:
            return []
        if self.config.processing_strategy == ProcessingStrategy.PARALLEL and self.config.parallel_processing and (len(documents) > 1):
            return self._process_documents_parallel(documents, doc_input)
        return self._process_documents_sequential(documents, doc_input)

    def _process_documents_sequential(self, documents: list[Document], doc_input: DocumentInput) -> list[ProcessedDocument]:
        """Process documents sequentially."""
        processed = []
        for i, doc in enumerate(documents):
            try:
                start_time = time.time()
                processed_doc = self._process_single_document(doc, doc_input, i)
                processed_doc.processing_time = time.time() - start_time
                processed.append(processed_doc)
            except Exception as e:
                logger.exception(f'Failed to process document {i}: {e}')
                if not self.config.skip_invalid:
                    raise
        return processed

    def _process_documents_parallel(self, documents: list[Document], doc_input: DocumentInput) -> list[ProcessedDocument]:
        """Process documents in parallel."""
        processed = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_doc = {executor.submit(self._process_single_document_safe, doc, doc_input, i): i for i, doc in enumerate(documents)}
            for future in as_completed(future_to_doc):
                doc_index = future_to_doc[future]
                try:
                    processed_doc = future.result()
                    if processed_doc:
                        processed.append(processed_doc)
                except Exception as e:
                    logger.exception(f'Failed to process document {doc_index}: {e}')
                    if not self.config.skip_invalid:
                        raise
        processed.sort(key=lambda x: x.metadata.get('original_index', 0))
        return processed

    def _process_single_document_safe(self, document: Document, doc_input: DocumentInput, index: int) -> ProcessedDocument | None:
        """Safely process a single document with error handling."""
        try:
            start_time = time.time()
            processed_doc = self._process_single_document(document, doc_input, index)
            processed_doc.processing_time = time.time() - start_time
            return processed_doc
        except Exception as e:
            logger.exception(f'Failed to process document {index}: {e}')
            if self.config.skip_invalid:
                return None
            raise

    def _process_single_document(self, document: Document, doc_input: DocumentInput, index: int) -> ProcessedDocument:
        """Process a single document."""
        content = document.page_content
        metadata = dict(document.metadata)
        metadata['original_index'] = index
        if self.config.normalize_content:
            content = self._normalize_content(content)
        doc_format = self._detect_document_format(metadata, content)
        chunks = []
        if self.config.chunking_strategy != ChunkingStrategy.NONE:
            chunks = self._create_chunks(content, doc_input.chunking_strategy or self.config.chunking_strategy, doc_input.chunk_size or self.config.chunk_size, doc_input.chunk_overlap or self.config.chunk_overlap, metadata)
        processed = ProcessedDocument(source=metadata.get('source', str(doc_input.source)), source_type=self._determine_source_type(metadata), format=doc_format, content=content, chunks=chunks, metadata=metadata, loader_name=metadata.get('loader_name', 'unknown'), processing_time=0.0, character_count=len(content), word_count=len(content.split()) if content else 0, chunk_count=len(chunks))
        return processed

    def _normalize_content(self, content: str) -> str:
        """Normalize document content."""
        if not content:
            return content
        content = content.strip()
        import re
        content = re.sub(' ', ' ', content)
        content = re.sub('\\n\\s*\\n', '\n\n', content)
        return content

    def _detect_document_format(self, metadata: dict[str, Any], content: str) -> DocumentFormat:
        """Detect document format from metadata and content."""
        if 'format' in metadata:
            try:
                return DocumentFormat(metadata['format'].lower())
            except ValueError:
                pass
        source = metadata.get('source', '')
        if source:
            ext = Path(source).suffix.lower()
            format_map = {'.pdf': DocumentFormat.PDF, '.docx': DocumentFormat.DOCX, '.txt': DocumentFormat.TXT, '.html': DocumentFormat.HTML, '.md': DocumentFormat.MARKDOWN, '.json': DocumentFormat.JSON, '.csv': DocumentFormat.CSV, '.xml': DocumentFormat.XML}
            if ext in format_map:
                return format_map[ext]
        if content:
            content_lower = content.lower().strip()
            if content_lower.startswith(('<!doctype html', '<html')):
                return DocumentFormat.HTML
            if content_lower.startswith('{') and content_lower.endswith('}'):
                return DocumentFormat.JSON
            if content_lower.startswith('<?xml'):
                return DocumentFormat.XML
        return DocumentFormat.UNKNOWN

    def _determine_source_type(self, metadata: dict[str, Any]) -> DocumentSourceType:
        """Determine source type from metadata."""
        source = metadata.get('source', '')
        if source.startswith(('http://', 'https://')):
            return DocumentSourceType.URL
        if source.startswith(('s3://', 'gs://', 'azure://')):
            return DocumentSourceType.CLOUD
        if '://' in source:
            return DocumentSourceType.DATABASE
        return DocumentSourceType.FILE

    def _create_chunks(self, content: str, strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int, metadata: dict[str, Any]) -> list[DocumentChunk]:
        """Create document chunks based on the specified strategy."""
        if not content or strategy == ChunkingStrategy.NONE:
            return []
        chunks = []
        if strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(content, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_by_paragraph(content, chunk_size)
        elif strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_by_sentence(content, chunk_size)
        elif strategy == ChunkingStrategy.RECURSIVE:
            chunks = self._chunk_recursive(content, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(content, chunk_size)
        else:
            chunks = self._chunk_fixed_size(content, chunk_size, chunk_overlap)
        doc_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({'chunk_index': i, 'chunk_strategy': strategy.value, 'chunk_size': len(chunk_text)})
            doc_chunks.append(DocumentChunk(content=chunk_text, metadata=chunk_metadata, chunk_index=i, chunk_id=f'{metadata.get('source', 'unknown')}_{i}'))
        return doc_chunks

    def _chunk_fixed_size(self, content: str, size: int, overlap: int) -> list[str]:
        """Chunk content into fixed-size pieces."""
        chunks = []
        start = 0
        while start < len(content):
            end = start + size
            chunk = content[start:end]
            chunks.append(chunk)
            if end >= len(content):
                break
            start = end - overlap
        return chunks

    def _chunk_by_paragraph(self, content: str, max_size: int) -> list[str]:
        """Chunk content by paragraphs."""
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ''
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_size:
                if current_chunk:
                    current_chunk += '\n\n'
                current_chunk += paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _chunk_by_sentence(self, content: str, max_size: int) -> list[str]:
        """Chunk content by sentences."""
        import re
        sentences = re.split('[.!?]+', content)
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 1 <= max_size:
                if current_chunk:
                    current_chunk += ' '
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _chunk_recursive(self, content: str, size: int, overlap: int) -> list[str]:
        """Chunk content recursively using multiple separators."""
        separators = ['\n\n', '\n', ' ', '']

        def split_recursive(text: str, separators: list[str]) -> list[str]:
            if len(text) <= size:
                return [text]
            if not separators:
                return self._chunk_fixed_size(text, size, overlap)
            separator = separators[0]
            remaining_separators = separators[1:]
            if separator not in text:
                return split_recursive(text, remaining_separators)
            splits = text.split(separator)
            chunks = []
            current_chunk = ''
            for split in splits:
                test_chunk = current_chunk + (separator if current_chunk else '') + split
                if len(test_chunk) <= size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(split) > size:
                        chunks.extend(split_recursive(split, remaining_separators))
                        current_chunk = ''
                    else:
                        current_chunk = split
            if current_chunk:
                chunks.append(current_chunk)
            return chunks
        return split_recursive(content, separators)

    def _chunk_semantic(self, content: str, size: int) -> list[str]:
        """Chunk content semantically (placeholder implementation)."""
        return self._chunk_by_paragraph(content, size)

    def _map_path_type_to_source_type(self, path_type) -> DocumentSourceType:
        """Map path analysis type to document source type."""
        from haive.core.engine.document.path_analysis import PathType
        mapping = {PathType.LOCAL_FILE: DocumentSourceType.FILE, PathType.LOCAL_DIRECTORY: DocumentSourceType.DIRECTORY, PathType.URL_HTTP: DocumentSourceType.URL, PathType.URL_HTTPS: DocumentSourceType.URL, PathType.DATABASE_URI: DocumentSourceType.DATABASE, PathType.CLOUD_STORAGE: DocumentSourceType.CLOUD}
        return mapping.get(path_type, DocumentSourceType.FILE)

    def _create_output(self, processed_docs: list[ProcessedDocument], doc_input: DocumentInput, analysis: PathAnalysisResult, operation_time: float) -> DocumentOutput:
        """Create final output object."""
        return DocumentOutput(documents=processed_docs, total_documents=len(processed_docs), operation_time=operation_time, original_source=str(doc_input.source), source_type=self._map_path_type_to_source_type(analysis.path_type), processing_strategy=self.config.processing_strategy, loader_names=list({doc.loader_name for doc in processed_docs}))

    def _update_stats(self, output: DocumentOutput):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += output.total_documents
        self.processing_stats['successful'] += output.successful_documents
        self.processing_stats['failed'] += output.failed_documents
        self.processing_stats['total_time'] += output.operation_time

def create_file_document_engine(file_path: str | Path, chunking_strategy: ChunkingStrategy=ChunkingStrategy.RECURSIVE, chunk_size: int=1000, chunk_overlap: int=200, **kwargs) -> DocumentEngine:
    """Create a document engine configured for file processing."""
    config = DocumentEngineConfig(name=f'file_engine_{Path(file_path).name}', source_type=DocumentSourceType.FILE, chunking_strategy=chunking_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    return DocumentEngine(config=config)

def create_web_document_engine(chunking_strategy: ChunkingStrategy=ChunkingStrategy.PARAGRAPH, chunk_size: int=1500, **kwargs) -> DocumentEngine:
    """Create a document engine configured for web content processing."""
    config = DocumentEngineConfig(name='web_document_engine', source_type=DocumentSourceType.URL, chunking_strategy=chunking_strategy, chunk_size=chunk_size, processing_strategy=ProcessingStrategy.ENHANCED, normalize_content=True, detect_language=True, **kwargs)
    return DocumentEngine(config=config)

def create_directory_document_engine(directory_path: str | Path, recursive: bool=True, include_patterns: list[str] | None=None, exclude_patterns: list[str] | None=None, **kwargs) -> DocumentEngine:
    """Create a document engine configured for directory processing."""
    config = DocumentEngineConfig(name=f'directory_engine_{Path(directory_path).name}', source_type=DocumentSourceType.DIRECTORY, recursive=recursive, include_patterns=include_patterns or [], exclude_patterns=exclude_patterns or [], parallel_processing=True, max_workers=4, **kwargs)
    return DocumentEngine(config=config)
__all__ = ['DocumentEngine', 'create_directory_document_engine', 'create_file_document_engine', 'create_web_document_engine']