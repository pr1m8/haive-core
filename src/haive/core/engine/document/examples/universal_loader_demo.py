"""Universal Document Loader Demo.

This script demonstrates how to use the Universal Document Loader to automatically
detect and load documents from any source with intelligent source detection.
"""

import sys
from pathlib import Path

from haive.core.engine.document import (
    CredentialManager,
    UniversalDocumentLoader,
    analyze_document_source,
    load_document,
)

# Add the package root to the path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def demo_universal_loader() -> None:
    """Demonstrate the universal loader capabilities."""
    # Create a universal loader
    credential_manager = CredentialManager()

    # Add some example credentials (you would normally load these from config)
    # credential_manager.add_credential("github", Credential(
    #     credential_type=CredentialType.ACCESS_TOKEN,
    #     value="your_github_token_here"
    # ))

    loader = UniversalDocumentLoader(credential_manager)

    # Test paths for demonstration
    test_paths = [
        # Web sources
        "https://github.com/microsoft/vscode",
        "https://reddit.com/r/programming",
        "https://news.ycombinator.com",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://arxiv.org/abs/2301.00001",
        "https://huggingface.co/datasets/squad",
        # File sources
        "document.pdf",
        "data.csv",
        "code.py",
        "presentation.pptx",
        "README.md",
        "config.yaml",
        "notebook.ipynb",
        # Chat exports
        "whatsapp_chat.txt",
        "discord_export.json",
        # Database sources
        "postgresql://user:pass@localhost/db",
        "mongodb://localhost:27017/mydb",
        # Cloud sources
        "s3://my-bucket/documents/",
        "gs://my-bucket/data.csv",
    ]

    for path in test_paths:
        # Analyze the source
        analysis = loader.analyze_source(path)

        if analysis["recommended"]:
            pass
        if analysis["supports_auth"]:
            pass

        # Show top 3 candidates
        for _i, candidate in enumerate(analysis["candidates"][:3]):
            candidate["confidence"]
            candidate["source_type"]

    # Test actual loader creation for a few examples
    test_loads = [
        ("document.pdf", {"pdf_strategy": "pymupdf", "extract_images": True}),
        ("data.csv", {"csv_args": {"delimiter": ","}}),
        ("https://github.com/microsoft/vscode", {"include_issues": False}),
    ]

    for path, preferences in test_loads:
        try:
            doc_loader = loader.load(path, preferences=preferences)
            if doc_loader:
                pass
            else:
                pass
        except Exception:
            pass

    # Demonstrate convenience functions
    test_convenience = [
        "README.md",
        "https://en.wikipedia.org/wiki/Python",
    ]

    for path in test_convenience:
        try:
            # Use the convenience function
            doc_loader = load_document(path)
            if doc_loader:
                pass
            else:
                pass

            # Analyze with convenience function
            analysis = analyze_document_source(path)

        except Exception:
            pass


def show_supported_sources() -> None:
    """Show all supported source types."""
    loader = UniversalDocumentLoader()
    sources = loader.get_supported_sources()

    # Group by category
    categories = {
        "Web & Social": [
            s
            for s in sources
            if any(
                keyword in s.lower()
                for keyword in [
                    "github",
                    "reddit",
                    "twitter",
                    "web",
                    "wiki",
                    "arxiv",
                    "news",
                    "social",
                ]
            )
        ],
        "File Formats": [
            s
            for s in sources
            if any(
                keyword in s.lower()
                for keyword in [
                    "pdf",
                    "csv",
                    "json",
                    "excel",
                    "word",
                    "text",
                    "image",
                    "code",
                ]
            )
        ],
        "Databases": [
            s
            for s in sources
            if any(keyword in s.lower() for keyword in ["sql", "mongo", "database"])
        ],
        "Cloud Storage": [
            s
            for s in sources
            if any(keyword in s.lower() for keyword in ["s3", "gcs", "azure", "cloud"])
        ],
        "Other": [],
    }

    # Put uncategorized sources in "Other"
    categorized = set()
    for cat_sources in categories.values():
        categorized.update(cat_sources)

    categories["Other"] = [s for s in sources if s not in categorized]

    for _category, cat_sources in categories.items():
        if cat_sources:
            for _source in sorted(cat_sources):
                pass


if __name__ == "__main__":
    demo_universal_loader()
    show_supported_sources()
