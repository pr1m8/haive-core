"""Sphinx configuration for haive-mcp documentation."""

import os
import sys

from sphinx.application import Sphinx

# Path setup
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "haive-core"
copyright = "2025, Haive Team"
author = "Haive Team"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "autoapi.extension",  # Must be first
    "sphinx.ext.autodoc", 
    "sphinx.ext.autosummary",  # Add autosummary for detailed docs
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_codeautolink",  # Automatic GitHub source links
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx.ext.graphviz",
    "myst_parser",  # Parse README.md files
]

# AutoAPI Configuration
autoapi_dirs = ["../../src/haive"]
autoapi_type = "python"
autoapi_add_toctree_entry = True
autoapi_keep_files = True  # Keep generated files like haive-mcp
autoapi_root = "autoapi"
autoapi_include_inheritance_diagram = False
# autoapi_template_dir = "_templates/autoapi"  # Use custom templates
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
]

# CRITICAL: Use module-level pages for hierarchical organization
autoapi_own_page_level = "module"  # Module-level pages like haive-mcp
autoapi_member_order = "groupwise"
autoapi_generate_api_docs = True

# Skip problematic patterns  
autoapi_ignore = ["**/test_*.py", "**/tests/*", "**/*_test.py"]

# Enable both AutoAPI and autosummary to work together
autoapi_python_class_content = "both"  # Include both class and __init__ docstrings
autoapi_python_use_implicit_namespaces = True

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

# Furo theme configuration - Enhanced purple theme with proper contrast
html_theme_options = {
    # Enhanced Navigation Features
    "navigation_with_keys": True,           # Keyboard navigation
    "show_nav_level": 4,                   # Deeper navigation tree  
    "collapse_navigation": False,           # Keep sections expanded
    "sidebar_hide_name": False,            # Keep project name visible
    "navigation_depth": 5,                 # More depth levels
    "show_toc_level": 3,                   # Better table of contents
    "show_prev_next": True,                # Navigation arrows
    
    # Light Mode CSS Variables - Complete contrast specifications
    "light_css_variables": {
        # Brand Colors
        "color-brand-primary": "#8b5cf6",
        "color-brand-content": "#7c3aed", 
        
        # Critical: Foreground/Background Pairs for Contrast
        "color-foreground-primary": "#1f2937",      # Dark gray text
        "color-foreground-secondary": "#6b7280",    # Medium gray text
        "color-foreground-muted": "#9ca3af",        # Light gray text
        "color-background-primary": "#ffffff",      # White background
        "color-background-secondary": "#f9fafb",    # Light gray background
        "color-background-hover": "#f3f4f6",       # Hover background
        "color-background-border": "#e5e7eb",       # Border color
        
        # Sidebar - Light Mode
        "color-sidebar-background": "#faf5ff",      # Light purple background
        "color-sidebar-background-border": "#e9d5ff",
        "color-sidebar-link-text": "#374151",      # Dark text for contrast
        "color-sidebar-link-text--top-level": "#111827",  # Darker top level
        "color-sidebar-item-background--hover": "#f3e8ff",
        "color-sidebar-item-expander-background--hover": "#e9d5ff",
        
        # Content Area
        "color-content-foreground": "#1f2937",     # Dark text on light bg
        "color-content-background": "#ffffff",
        
        # Cards & Components (sphinx-design)
        "color-card-background": "#ffffff",
        "color-card-border": "#e5e7eb",
        "color-card-marginals-background": "#f9fafb",
        
        # Code Blocks
        "color-code-background": "#f8f9fa",
        "color-code-foreground": "#1f2937",
    },
    
    # Dark Mode CSS Variables - Complete contrast specifications  
    "dark_css_variables": {
        # Brand Colors
        "color-brand-primary": "#a78bfa",
        "color-brand-content": "#c084fc",
        
        # Critical: Foreground/Background Pairs for Contrast
        "color-foreground-primary": "#f9fafb",      # Light text
        "color-foreground-secondary": "#d1d5db",    # Medium light text
        "color-foreground-muted": "#9ca3af",        # Muted text
        "color-background-primary": "#0f0a1f",     # Dark purple background
        "color-background-secondary": "#1a0f2e",    # Darker purple
        "color-background-hover": "#2d1b45",        # Purple hover
        "color-background-border": "#4c1d95",       # Purple border
        
        # Sidebar - Dark Mode  
        "color-sidebar-background": "#14001f",      # Very dark purple
        "color-sidebar-background-border": "#4c1d95",
        "color-sidebar-link-text": "#e9d5ff",      # Light purple text
        "color-sidebar-link-text--top-level": "#f3e8ff",  # Lighter top level
        "color-sidebar-item-background--hover": "#2d0059",
        "color-sidebar-item-expander-background--hover": "#4c1d95",
        
        # Content Area
        "color-content-foreground": "#f9fafb",     # Light text on dark bg
        "color-content-background": "#0f0a1f",
        
        # Cards & Components (sphinx-design)
        "color-card-background": "#1e1b3a",
        "color-card-border": "#4c1d95", 
        "color-card-marginals-background": "#2d1b45",
        
        # Code Blocks
        "color-code-background": "#1e0936",        # Dark purple code bg
        "color-code-foreground": "#e9d5ff",
    },
    
    # Footer Icons - GitHub & Discord Integration
    "footer_icons": [
        {
            "name": "GitHub Repository",
            "url": "https://github.com/pr1m8/haive-core",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "Discord Community", 
            "url": "https://discord.gg/haive",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path d="M13.545 2.907a13.227 13.227 0 0 0-3.257-1.011.05.05 0 0 0-.052.025c-.141.25-.297.577-.406.833a12.19 12.19 0 0 0-3.658 0 8.258 8.258 0 0 0-.412-.833.051.051 0 0 0-.052-.025c-1.125.194-2.22.534-3.257 1.011a.041.041 0 0 0-.021.018C.356 6.024-.213 9.047.066 12.032c.001.014.01.028.021.037a13.276 13.276 0 0 0 3.995 2.02.05.05 0 0 0 .056-.019c.308-.42.582-.863.818-1.329a.05.05 0 0 0-.01-.059.051.051 0 0 0-.018-.011 8.875 8.875 0 0 1-1.248-.595.05.05 0 0 1-.02-.066.051.051 0 0 1 .015-.019c.084-.063.168-.129.248-.195a.05.05 0 0 1 .051-.007c2.619 1.196 5.454 1.196 8.041 0a.052.052 0 0 1 .053.007c.08.066.164.132.248.195a.051.051 0 0 1-.004.085 8.254 8.254 0 0 1-1.249.594.05.05 0 0 0-.03.03.052.052 0 0 0 .003.041c.24.465.515.909.817 1.329a.05.05 0 0 0 .056.019 13.235 13.235 0 0 0 4.001-2.02.049.049 0 0 0 .021-.037c.334-3.451-.559-6.449-2.366-9.106a.034.034 0 0 0-.02-.019Zm-8.198 7.307c-.789 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.45.73 1.438 1.613 0 .888-.637 1.612-1.438 1.612Zm5.316 0c-.788 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.451.73 1.438 1.613 0 .888-.631 1.612-1.438 1.612Z"/>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc settings to work with AutoAPI
autodoc_typehints = "description"
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "members": True,
    "member-order": "groupwise", 
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__"
}

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

# -- Purple Theme Configuration ----------------------------------------------
# Syntax highlighting - use purple-friendly themes
pygments_style = "default"  # Better for light mode with our custom CSS
pygments_dark_style = "monokai"  # Good for dark mode

# AutoAPI configuration for prominent API Reference
# Already configured above - removed duplicates
autoapi_toctree_caption = "🔍 API Reference"
autoapi_toctree_first = True  # Put at top!

# Graphviz configuration for beautiful diagrams
graphviz_output_format = "svg"
graphviz_dot_args = [
    "-Kdot",
    "-Tsvg",
    "-Gfontname=Inter",
    "-Nfontname=Inter",
    "-Efontname=Inter",
    "-Gbgcolor=transparent",
    "-Gpad=0.5",
    "-Grankdir=TB",
    "-Gnodesep=0.7",
    "-Granksep=0.8",
    "-Gsplines=true",
]

# CSS files in correct order - purple theme loads last to override
# Simplified - using Furo's built-in theme (no CSS overrides needed)
# The dark_css_variables above already provide the purple theme

# Autosummary settings for detailed API docs
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

# Code autolink configuration for GitHub links
codeautolink_concat_default = True
codeautolink_global_preface = "https://github.com/pr1m8/haive-core"

# MyST parser configuration for README.md files
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Include README.md files in documentation
myst_heading_anchors = 3
myst_title_to_header = True
