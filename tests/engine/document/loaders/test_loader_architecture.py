"""Standalone test of the document loader architecture."""

import sys

# Direct imports to avoid cascading import issues
sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")


def test_path_analyzer():
    """Test path analyzer functionality."""
    from haive.core.engine.document.loaders.path_analyzer import (
        FileCategory,
        PathAnalyzer,
        PathType,
    )

    # Test PDF file
    result = PathAnalyzer.analyze("/path/to/document.pdf")
    assert result.path_type == PathType.LOCAL_FILE
    assert result.file_extension == ".pdf"
    assert result.file_category == FileCategory.DOCUMENT

    # Test GitHub URL
    result = PathAnalyzer.analyze("https://github.com/user/repo/blob/main/README.md")
    assert result.service_detected == "github"
    assert result.file_extension == ".md"

    # Test database URI
    result = PathAnalyzer.analyze("postgresql://user:pass@localhost:5432/db")
    assert result.path_type == PathType.DATABASE_URI


def test_source_classes():
    """Test source base classes."""
    from haive.core.engine.document.loaders.sources.source_base import (
        DatabaseSource,
        LocalSource,
        RemoteSource,
    )

    # Test LocalSource
    pdf_source = LocalSource(
        source_type="pdf",
        source_id="test_pdf",
        file_path="/path/to/test.pdf",
        encoding="utf-8",
    )

    kwargs = pdf_source.get_loader_kwargs()
    assert kwargs["file_path"] == "/path/to/test.pdf"
    assert kwargs["encoding"] == "utf-8"

    # Test RemoteSource
    web_source = RemoteSource(
        source_type="web",
        source_id="test_web",
        url="https://example.com/page.html",
        provider="generic",
    )

    kwargs = web_source.get_loader_kwargs()
    assert kwargs["url"] == "https://example.com/page.html"

    # Test DatabaseSource
    db_source = DatabaseSource(
        source_type="database",
        source_id="test_db",
        connection_string="postgresql://user:pass@localhost:5432/db",
        provider="postgresql",
    )

    kwargs = db_source.get_loader_kwargs()
    assert kwargs["connection_string"] == "postgresql://user:pass@localhost:5432/db"


def test_registry():
    """Test source registry and registration."""
    from haive.core.engine.document.config import LoaderPreference
    from haive.core.engine.document.loaders.sources.registry import (
        register_source,
        source_registry,
    )
    from haive.core.engine.document.loaders.sources.source_base import LocalSource

    # Clear the global registry
    source_registry._sources.clear()
    source_registry._extension_index.clear()
    source_registry._url_pattern_index.clear()
    source_registry._scheme_index.clear()
    source_registry._mime_index.clear()

    # Register a PDF source using decorator
    @register_source(
        name="pdf",
        file_extensions=[".pdf"],
        mime_types=["application/pdf"],
        loaders={
            "fast": "PyPDFLoader",
            "quality": {
                "class": "UnstructuredPDFLoader",
                "quality": "high",
                "speed": "slow",
                "requires_packages": ["unstructured"],
            },
        },
        default_loader="fast",
        priority=10,
    )
    class PDFSource(LocalSource):
        """PDF source for testing."""

    # Test finding source by path
    registration = source_registry.find_source_for_path("/path/to/document.pdf")
    assert registration.name == "pdf"
    assert len(registration.loaders) == 2

    # Test creating source
    source = source_registry.create_source("/path/to/document.pdf")
    assert source.source_type == "pdf"
    assert source.file_path == "/path/to/document.pdf"

    # Test loader selection
    fast_loader = source_registry.get_loader_for_source(
        source, preference=LoaderPreference.SPEED
    )
    quality_loader = source_registry.get_loader_for_source(
        source, preference=LoaderPreference.QUALITY
    )

    assert fast_loader.name == "PyPDFLoader"
    assert quality_loader.name == "UnstructuredPDFLoader"


def test_auto_factory():
    """Test the document loader factory."""
    from haive.core.engine.document.loaders.auto_factory import DocumentLoaderFactory

    factory = DocumentLoaderFactory()

    # Test path analysis with sources
    analysis = factory.analyze_path_with_sources("/path/to/document.pdf")

    assert analysis["source"]["name"] == "pdf"
    assert "fast" in analysis["loaders"]
    assert "quality" in analysis["loaders"]

    # Test available loaders lookup
    loaders = factory.find_available_loaders("/path/to/document.pdf")
    assert "fast" in loaders
    assert "quality" in loaders

    # Test source creation via factory
    source = factory.create_source("/path/to/document.pdf")
    assert source.source_type == "pdf"


def test_complete_workflow():
    """Test the complete workflow end-to-end."""
    from haive.core.engine.document.loaders.auto_factory import analyze_document_source

    # Test the convenience function
    result = analyze_document_source("/path/to/document.pdf")

    assert result["source"]["name"] == "pdf"
    assert result["analysis"]["file_extension"] == ".pdf"
    assert len(result["loaders"]) == 2


def main():
    """Run all tests."""

    try:
        test_path_analyzer()
        test_source_classes()
        test_registry()
        test_auto_factory()
        test_complete_workflow()

    except Exception:
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
