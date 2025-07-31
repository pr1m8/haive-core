"""Document models for the Haive framework.

This module provides document-related model classes that represent various types
of documents and document sources that can be processed by the Haive framework.
These models define the structure and metadata for different document types,
enabling consistent handling across the system.

The document models support various sources including version control systems,
content management systems, and other document repositories. Each model provides
structured access to document content, metadata, and source-specific information.

Supported Document Sources:
    - GitHub Repositories: Models for GitHub repository structure and content
    - File Systems: Local and remote file system documents
    - Web Content: Web pages and online document sources
    - Database Records: Structured data from various database systems
    - APIs: Documents retrieved from REST APIs and web services

Key Components:
    - Document Models: Structured representations of documents from various sources
    - Settings Classes: Configuration for accessing different document sources
    - Metadata Extraction: Utilities for extracting structured metadata
    - Content Processing: Tools for processing and normalizing document content
    - Source Integration: Seamless integration with external document sources

Typical usage example:
    ```python
    from haive.core.common.models.documents import GithubRepo, GithubSettings

    # Configure GitHub access
    settings = GithubSettings(
        github_token="your_token_here",
        default_branch="main"
    )

    # Create a GitHub repository model
    repo = GithubRepo(
        owner="microsoft",
        name="TypeScript",
        settings=settings
    )

    # Access repository information
    files = repo.get_files()
    readme = repo.get_file_content("README.md")
    metadata = repo.get_metadata()
    ```

Architecture:
    Document models in Haive follow a consistent pattern:

    1. Source Configuration: Settings classes for source-specific parameters
    2. Document Structure: Models that represent document hierarchy and content
    3. Content Access: Methods for retrieving document content and metadata
    4. Processing Pipeline: Integration with content processing and embedding systems

    This architecture ensures consistent document handling regardless of the source
    while providing source-specific optimizations and features.

Advanced Features:
    - Incremental Updates: Track and process only changed documents
    - Batch Processing: Efficient handling of large document collections
    - Content Filtering: Filter documents based on type, size, or content criteria
    - Metadata Enrichment: Automatic extraction and enhancement of document metadata
    - Version Tracking: Handle versioned documents and change detection
    - Access Control: Respect source-level permissions and access controls

Performance Considerations:
    - Lazy Loading: Documents are loaded on-demand to minimize memory usage
    - Caching: Intelligent caching of document content and metadata
    - Parallel Processing: Concurrent document processing for better throughput
    - Streaming: Support for streaming large documents without full memory load
    - Connection Pooling: Reuse connections for better performance with remote sources

Examples:
    Process GitHub repository documents::

        settings = GithubSettings(github_token=os.getenv("GITHUB_TOKEN"))
        repo = GithubRepo(owner="python", name="cpython", settings=settings)

        # Get all Python files
        python_files = repo.get_files(pattern="*.py", max_depth=3)

        # Process files for embedding
        for file_path in python_files:
            content = repo.get_file_content(file_path)
            metadata = repo.get_file_metadata(file_path)
            # Process content...

    Batch process multiple repositories::

        repos = ["microsoft/TypeScript", "python/cpython", "golang/go"]

        for repo_spec in repos:
            owner, name = repo_spec.split("/")
            repo = GithubRepo(owner=owner, name=name, settings=settings)

            # Process repository
            repo.process_documents(
                output_dir=f"./processed/{owner}_{name}",
                filters={"extensions": [".py", ".js", ".go"]}
            )

.. autosummary::
   :toctree: generated/

   GithubRepo
   GithubSettings
"""

from haive.core.common.models.documents.github_repo import GithubRepo, GithubSettings

__all__ = [
    "GithubRepo",
    "GithubSettings",
]
