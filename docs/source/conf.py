"""Sphinx configuration for the Haive-Core documentation.

This configuration enables:
- AutoAPI for automatic code reference generation
- AutoSummary for summary tables
- autodoc_pydantic for advanced Pydantic v2 integration
- MyST parser for Markdown support
- Furo for clean, responsive theming
- Google-style docstring parsing via Napoleon
- Numerous visual and UX enhancements
"""

import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]  # .../packages/haive-core/
sys.path.insert(0, str(project_root / "src"))

# -----------------------------------------------------------------------------
# Project metadata
# -----------------------------------------------------------------------------
project = "haive-core"
author = "William R. Astley"
release = "0.1.0"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    # Must be first for code API generation
    "autoapi.extension",
    # Core
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google-style + NumPy-style docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    # UX
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_autorun",
    # Typing + Pydantic
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autodoc_pydantic",
]

autosummary_generate = True

# -----------------------------------------------------------------------------
# Source parsers
# -----------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -----------------------------------------------------------------------------
# AutoAPI configuration
# -----------------------------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = [str(project_root / "src" / "haive")]

autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_generate_api_docs = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

# -----------------------------------------------------------------------------
# autodoc_pydantic (Pydantic v2 integration)
# -----------------------------------------------------------------------------
autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_model_summary_list = True
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_field_list_validators = True
autodoc_pydantic_use_schema_description = True

# -----------------------------------------------------------------------------
# MyST configuration
# -----------------------------------------------------------------------------
myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "colon_fence",
    "attrs_block",
    "attrs_inline",
]

# -----------------------------------------------------------------------------
# Napoleon settings
# -----------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {}
napoleon_attr_annotations = True
# -----------------------------------------------------------------------------
# HTML theme settings
# -----------------------------------------------------------------------------
html_theme = "furo"
html_title = "Haive-Core"
html_static_path = ["_static"]
html_css_files = []

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#4c7eff",
        "color-brand-content": "#3456cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4c7eff",
        "color-brand-content": "#94b3ff",
    },
}

# -----------------------------------------------------------------------------
# Intersphinx mapping
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
default_role = "any"
