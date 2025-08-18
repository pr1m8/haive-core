"""Comprehensive examples for the Haive Document Loaders system.

This module provides detailed examples showing how to use the ultimate auto-loading
system with all 230+ supported document loaders. Examples cover basic usage,
advanced configurations, performance optimization, and real-world scenarios.

Author: Claude (Haive Document Loader System)
Version: 1.0.0
"""

import asyncio
import contextlib

from .auto_loader import (
    AutoLoader,
    AutoLoaderConfig,
    aload_document,
    load_document,
    load_documents_bulk,
)
from .auto_registry import (
    auto_register_all,
    get_registration_status,
    get_sources_by_category,
    list_available_sources,
)
from .sources.enhanced_registry import LoaderPreference
from .sources.source_types import SourceCategory

# =============================================================================
# Basic Usage Examples
# =============================================================================


def example_basic_auto_loading() -> None:
    """Basic auto-loading examples for common use cases."""
    # Initialize the auto-loader
    loader = AutoLoader()

    # Example 1: Load a single PDF document
    try:
        documents = loader.load("/path/to/document.pdf")
        for _i, _doc in enumerate(documents[:2]):  # Show first 2
            pass
    except Exception:
        pass

    # Example 2: Load from a website
    with contextlib.suppress(Exception):
        documents = loader.load("https://python.org/about/")

    # Example 3: Load from cloud storage
    with contextlib.suppress(Exception):
        documents = loader.load(
            "s3://my-bucket/documents/report.pdf",
            aws_access_key_id="your-access-key",
            aws_secret_access_key="your-secret-key",
        )

    # Example 4: Load from database
    with contextlib.suppress(Exception):
        documents = loader.load(
            "postgresql://user:password@localhost:5432/mydb",
            query="SELECT content, title FROM documents WHERE category = 'research'",
        )


def example_convenience_functions() -> None:
    """Examples using convenience functions for quick loading."""
    # Simple one-liner loading
    with contextlib.suppress(Exception):
        load_document("document.pdf")

    # Bulk loading multiple sources
    try:
        sources = ["file1.pdf", "file2.docx", "https://example.com"]
        load_documents_bulk(sources)
    except Exception:
        pass

    # Async loading

    async def async_example():
        """Async Example."""
        with contextlib.suppress(Exception):
            await aload_document("https://example.com")

    asyncio.run(async_example())


# =============================================================================
# Advanced Configuration Examples
# =============================================================================


def example_configuration_options() -> None:
    """Examples of different configuration options."""
    # Speed-optimized configuration
    speed_config = AutoLoaderConfig(
        preference=LoaderPreference.SPEED,
        max_concurrency=20,
        timeout=60,
        retry_attempts=1,
        enable_caching=True,
        cache_ttl=3600,
    )
    AutoLoader(speed_config)

    # Quality-optimized configuration
    quality_config = AutoLoaderConfig(
        preference=LoaderPreference.QUALITY,
        max_concurrency=5,
        timeout=300,
        retry_attempts=5,
        enable_metadata=True,
        default_chunk_size=2000,
    )
    AutoLoader(quality_config)

    # Balanced configuration
    balanced_config = AutoLoaderConfig(
        preference=LoaderPreference.BALANCED,
        max_concurrency=10,
        enable_caching=True,
        enable_metadata=True,
    )
    AutoLoader(balanced_config)


def example_detailed_loading() -> None:
    """Examples of detailed loading with metadata."""
    loader = AutoLoader()

    # Get detailed loading information
    try:
        result = loader.load_detailed("/path/to/document.pdf")

        if result.documents:
            result.documents[0]

    except Exception:
        pass


# =============================================================================
# Bulk and Concurrent Loading Examples
# =============================================================================


def example_bulk_loading() -> None:
    """Examples of bulk loading with different configurations."""
    loader = AutoLoader()

    # Basic bulk loading
    sources = [
        "/documents/report1.pdf",
        "/documents/report2.docx",
        "/documents/data.csv",
        "https://example.com/page1.html",
        "https://example.com/page2.html",
    ]

    try:
        result = loader.load_bulk(sources)

        if result.failed_sources:
            for _source, _error in result.failed_sources:
                pass

    except Exception:
        pass

    # Bulk loading with custom configurations
    mixed_sources = [
        "/local/file.pdf",
        {
            "path": "s3://bucket/file.pdf",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
        },
        {
            "url": "https://api.example.com/data",
            "headers": {"Authorization": "Bearer token"},
            "timeout": 60,
        },
    ]

    with contextlib.suppress(Exception):
        result = loader.load_bulk(mixed_sources, max_workers=5)

    # Progress tracking

    def progress_callback(completed: int, total: int):
        """Progress Callback.

        Args:
            completed: [TODO: Add description]
            total: [TODO: Add description]
        """
        (completed / total) * 100

    try:
        sources = [f"/test{i}.pdf" for i in range(5)]
        result = loader.load_bulk(sources, progress_callback=progress_callback)
    except Exception:
        pass


def example_scrape_all() -> None:
    """Examples of 'scrape all' functionality for bulk processing."""
    loader = AutoLoader()

    # Scrape entire directory
    with contextlib.suppress(Exception):
        loader.load_all("/path/to/documents/")

    # Scrape entire website
    with contextlib.suppress(Exception):
        loader.load_all(
            "https://docs.python.org",
            max_depth=3,
            respect_robots=True,
            include_external_links=False,
        )

    # Scrape all tables from database
    with contextlib.suppress(Exception):
        loader.load_all(
            "postgresql://user:pass@host:5432/db",
            include_system_tables=False,
            table_filter="user_*",
        )


# =============================================================================
# Async and Performance Examples
# =============================================================================


async def example_async_loading():
    """Examples of asynchronous loading for high performance."""
    loader = AutoLoader()

    # Basic async loading
    with contextlib.suppress(Exception):
        await loader.aload("https://example.com/large-document.pdf")

    # Concurrent async loading
    sources = [
        "https://site1.com/doc.pdf",
        "https://site2.com/doc.pdf",
        "https://site3.com/doc.pdf",
    ]

    try:
        tasks = [loader.aload(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_docs = 0
        for _i, result in enumerate(results):
            if isinstance(result, Exception):
                pass
            else:
                total_docs += len(result)

    except Exception:
        pass

    # Async bulk loading
    try:
        sources = [f"https://example.com/doc{i}.pdf" for i in range(5)]
        result = await loader.aload_bulk(sources)

    except Exception:
        pass


def example_performance_optimization() -> None:
    """Examples of performance optimization techniques."""
    # High-concurrency configuration
    perf_config = AutoLoaderConfig(
        preference=LoaderPreference.SPEED,
        max_concurrency=50,
        timeout=30,
        retry_attempts=1,
        enable_caching=True,
        cache_ttl=7200,  # 2 hours
    )
    AutoLoader(perf_config)

    # Caching demonstration
    cached_loader = AutoLoader(AutoLoaderConfig(enable_caching=True))

    import time

    try:
        # First load (slow)
        start = time.time()
        cached_loader.load("/test.pdf")
        time.time() - start

        # Second load (fast from cache)
        start = time.time()
        cached_loader.load("/test.pdf")
        time.time() - start

    except Exception:
        pass


# =============================================================================
# Real-World Use Case Examples
# =============================================================================


def example_enterprise_document_processing() -> None:
    """Example of enterprise-scale document processing."""
    # Enterprise configuration
    enterprise_config = AutoLoaderConfig(
        preference=LoaderPreference.QUALITY,
        max_concurrency=20,
        timeout=600,  # 10 minutes for large files
        retry_attempts=3,
        enable_caching=True,
        enable_metadata=True,
    )
    enterprise_loader = AutoLoader(enterprise_config)

    # Mixed enterprise sources
    enterprise_sources = [
        # Local file shares
        "/mnt/shared/reports/quarterly_report.pdf",
        "/mnt/shared/contracts/contract_2024_001.docx",
        # SharePoint documents
        {
            "path": "https://company.sharepoint.com/sites/docs/contract.pdf",
            "client_id": "sharepoint_client_id",
            "client_secret": "sharepoint_secret",
        },
        # Salesforce attachments
        {
            "path": "salesforce://attachments",
            "username": "sf_user",
            "password": "sf_password",
            "security_token": "sf_token",
        },
        # Google Drive documents
        {
            "path": "https://drive.google.com/folder/xyz",
            "service_account_key": "/path/to/service-account.json",
            "recursive": True,
        },
    ]

    try:
        result = enterprise_loader.load_bulk(enterprise_sources)

        # Analyze results by source type
        source_types = {}
        for loading_result in result.results:
            source_type = loading_result.source_info.source_type
            if source_type not in source_types:
                source_types[source_type] = 0
            source_types[source_type] += len(loading_result.documents)

        for source_type, _count in source_types.items():
            pass

    except Exception:
        pass


def example_research_paper_collection() -> None:
    """Example of collecting research papers from multiple sources."""
    research_loader = AutoLoader(
        AutoLoaderConfig(
            preference=LoaderPreference.QUALITY,
            enable_metadata=True,
        )
    )

    # Academic sources
    research_sources = [
        # ArXiv papers
        "https://arxiv.org/abs/2024.00001",
        "https://arxiv.org/abs/2024.00002",
        # PubMed articles
        {
            "path": "pubmed://search",
            "query": "machine learning healthcare",
            "limit": 50,
        },
        # Google Scholar
        {
            "path": "scholar://search",
            "query": "neural networks 2024",
            "year_range": "2024-2024",
        },
        # Local PDF collection
        "/research/papers/ml_papers/",
        # University repository
        "https://university.edu/research/papers/",
    ]

    try:
        result = research_loader.load_bulk(research_sources)

        # Extract metadata for analysis
        papers_by_year = {}
        papers_by_venue = {}

        for loading_result in result.results:
            for doc in loading_result.documents:
                metadata = doc.metadata

                # Group by year
                year = metadata.get("year", "unknown")
                papers_by_year[year] = papers_by_year.get(year, 0) + 1

                # Group by venue/journal
                venue = metadata.get("venue", metadata.get("journal", "unknown"))
                papers_by_venue[venue] = papers_by_venue.get(venue, 0) + 1

    except Exception:
        pass


def example_legal_document_analysis() -> None:
    """Example of legal document processing and analysis."""
    legal_config = AutoLoaderConfig(
        preference=LoaderPreference.QUALITY,
        enable_metadata=True,
        default_chunk_size=2000,  # Larger chunks for legal text
    )
    legal_loader = AutoLoader(legal_config)

    # Legal document sources
    legal_sources = [
        # Court cases
        {
            "path": "westlaw://cases",
            "jurisdiction": "federal",
            "year_range": "2020-2024",
            "topic": "intellectual property",
        },
        # Legal databases
        {
            "path": "lexis://database",
            "database": "cases",
            "query": "software patents",
        },
        # Government regulations
        "https://federalregister.gov/documents/2024/",
        # Local legal documents
        "/legal/contracts/",
        "/legal/patents/",
        # Law firm documents
        {
            "path": "sharepoint://legal.firm.com/documents",
            "practice_area": "ip",
        },
    ]

    try:

        def legal_progress(completed, total) -> None:
            """Legal Progress.

            Args:
                completed: [TODO: Add description]
                total: [TODO: Add description]

            Returns:
                [TODO: Add return description]
            """
            pass

        result = legal_loader.load_bulk(legal_sources, progress_callback=legal_progress)

        # Analyze legal content
        document_types = {}
        jurisdictions = {}

        for loading_result in result.results:
            for doc in loading_result.documents:
                metadata = doc.metadata

                # Classify document types
                doc_type = metadata.get("document_type", "unknown")
                document_types[doc_type] = document_types.get(doc_type, 0) + 1

                # Track jurisdictions
                jurisdiction = metadata.get("jurisdiction", "unknown")
                jurisdictions[jurisdiction] = jurisdictions.get(jurisdiction, 0) + 1

    except Exception:
        pass


# =============================================================================
# Registry and System Management Examples
# =============================================================================


def example_registry_management() -> None:
    """Examples of managing the document loader registry."""
    # Auto-register all sources
    try:
        stats = auto_register_all()

        if stats.registration_errors:
            for _error in stats.registration_errors[:3]:  # Show first 3
                pass

    except Exception:
        pass

    # Get registration status
    try:
        status = get_registration_status()

        for category, _count in status["category_breakdown"].items():
            pass

        for _reg in status["recent_registrations"][:3]:
            pass

    except Exception:
        pass

    # List available sources
    with contextlib.suppress(Exception):
        sources = list_available_sources()

    # Sources by category
    try:
        for category in SourceCategory:
            sources = get_sources_by_category(category)
            if sources:
                pass  # Show first 5

    except Exception:
        pass


def example_system_capabilities() -> None:
    """Examples of checking system capabilities."""
    loader = AutoLoader()

    # Check supported sources
    try:
        sources_info = loader.get_supported_sources()

        # Show first 5
        for source_type, _info in list(sources_info.items())[:5]:
            pass

    except Exception:
        pass

    # Check source capabilities
    sample_sources = ["pdf", "web", "csv", "json", "database"]

    for source_type in sample_sources:
        with contextlib.suppress(Exception):
            loader.get_capabilities(source_type)

    # Validate credentials
    credential_tests = [
        (
            "aws_s3",
            {"aws_access_key_id": "test_key", "aws_secret_access_key": "test_secret"},
        ),
        (
            "postgresql",
            {"host": "localhost", "username": "test_user", "password": "test_password"},
        ),
        ("google_drive", {"service_account_key": "/path/to/key.json"}),
    ]

    for source_type, credentials in credential_tests:
        with contextlib.suppress(Exception):
            loader.validate_credentials(source_type, **credentials)


# =============================================================================
# Error Handling and Debugging Examples
# =============================================================================


def example_error_handling() -> None:
    """Examples of error handling and debugging."""
    loader = AutoLoader()

    # Handle invalid sources
    invalid_sources = [
        "/nonexistent/file.pdf",
        "https://invalid-url-that-does-not-exist.com",
        "unsupported://protocol/file",
    ]

    for source in invalid_sources:
        try:
            result = loader.load_detailed(source)
            if result.errors:
                for _error in result.errors:
                    pass
            else:
                pass
        except Exception:
            pass

    # Bulk loading with error tolerance
    mixed_sources = [
        "/valid/file.pdf",
        "/invalid/file.pdf",
        "https://example.com",
        "invalid://source",
    ]

    try:
        result = loader.load_bulk(mixed_sources)

        if result.failed_sources:
            for source, _error in result.failed_sources:
                pass

    except Exception:
        pass

    # Retry configuration
    retry_config = AutoLoaderConfig(
        retry_attempts=5,
        timeout=30,
    )
    AutoLoader(retry_config)


# =============================================================================
# Main Examples Runner
# =============================================================================


def run_all_examples() -> None:
    """Run all examples to demonstrate the document loader system."""
    # Basic examples
    example_basic_auto_loading()
    example_convenience_functions()

    # Advanced configuration
    example_configuration_options()
    example_detailed_loading()

    # Bulk and concurrent loading
    example_bulk_loading()
    example_scrape_all()

    # Async examples
    asyncio.run(example_async_loading())

    # Performance examples
    example_performance_optimization()

    # Real-world use cases
    example_enterprise_document_processing()
    example_research_paper_collection()
    example_legal_document_analysis()

    # Registry management
    example_registry_management()
    example_system_capabilities()

    # Error handling
    example_error_handling()


if __name__ == "__main__":
    run_all_examples()
