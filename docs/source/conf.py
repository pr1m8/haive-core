"""Sphinx configuration for Haive Core Framework documentation."""

import os
import sys

from sphinx.application import Sphinx

# Path setup
sys.path.insert(0, os.path.abspath("../../src"))

# Import shared Haive configuration from pydevelop-docs package
from pydevelop_docs.config import get_haive_config

# Get package-specific configuration
package_name = "haive-core"
package_path = "../../src"

config = get_haive_config(
    package_name=package_name,
    package_path=package_path,
    is_central_hub=False,
    extra_extensions=[
        # Package-specific extensions for haive-core
        "sphinxcontrib.mermaid",  # Graph visualizations
        "sphinx.ext.graphviz",  # Class diagrams
        "sphinx_tabs.tabs",  # Code examples in tabs
    ],
)

# Apply configuration to globals
globals().update(config)

# Package-specific overrides
project = "Haive Core Framework"
html_title = "Haive Core Framework - Core Components and Infrastructure"
html_baseurl = "https://docs.haive.ai/packages/haive-core/"

# Package-specific theme options
html_theme_options.update(
    {
        "announcement": "🚀 Core components, engines, and base infrastructure for Haive AI Agent Framework",
        "source_directory": "packages/haive-core/docs/",
    }
)

# AutoAPI configuration for this package
autoapi_dirs = ["../../src"]


def setup(app: Sphinx):
    """Sphinx setup function for haive-core."""
    # Add package-specific CSS/JS
    app.add_css_file("css/custom.css")
    app.add_js_file("js/api-enhancements.js")

    print("✨ Haive Core Framework documentation loaded!")
    print("📦 Using shared haive-docs configuration")
