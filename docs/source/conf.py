# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "haive-core"
copyright = "2025, Haive Team"
author = "Haive Team"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "autoapi.extension",  # Must be first
    "sphinx.ext.autodoc",  # Moved up to be right after autoapi
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",  # NEW: Architecture diagrams
    "sphinxcontrib.autodoc_pydantic",  # NEW: Pydantic model documentation
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_design",
    "sphinx_tabs.tabs",  # NEW: Tabbed content
    "sphinxcontrib.mermaid",
    "sphinx_sitemap",
    "sphinx_last_updated_by_git",  # NEW: Git timestamps
    "sphinx_tippy",  # NEW: Enhanced tooltips
    "notfound.extension",  # NEW: Custom 404 pages
    "sphinxext.opengraph",
    "myst_parser",
    "sphinx_exec_code",  # NEW: Execute example code in docs
    "sphinx_toggleprompt",  # NEW: Toggle shell prompts in code
    "sphinx_issues",  # NEW: Link to GitHub issues
    "sphinx_git",  # NEW: Git integration for changelogs
    "seed_intersphinx_mapping",  # NEW: Auto-populate intersphinx from pyproject.toml
    "sphinx_favicon",  # NEW: Multiple favicon support
    "sphinx_changelog",  # NEW: Structured changelog support
    "sphinx_prompt",  # NEW: Better shell prompts and code blocks
    "sphinxemoji",  # NEW: Emoji support in docs
]

# AutoAPI Configuration
autoapi_dirs = ["../../src/haive"]  # Point directly to haive to remove src prefix
autoapi_type = "python"
autoapi_add_toctree_entry = False  # We'll add manually for better control
autoapi_keep_files = True
autoapi_root = "autoapi"
autoapi_include_inheritance_diagram = False  # Disable for now
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
    "special-members",  # Show __init__ and other special methods
]
autoapi_template_dir = "_templates/autoapi"  # Use custom templates

# CRITICAL: Use module-level pages for hierarchical organization
autoapi_own_page_level = "module"
autoapi_member_order = "groupwise"
autoapi_python_class_content = "both"
autoapi_python_use_implicit_namespaces = True
autoapi_generate_api_docs = True
autoapi_file_patterns = ["*.py"]  # Explicit pattern
autoapi_toctree_depth = 3  # Expand toctree to show more levels by default

# Ignore test files and problematic files
autoapi_ignore = [
    "**/test_*.py",
    "**/tests/*",
    "**/*_test.py",
    "**/conftest.py",
    "**/engine_node_test/**",
    "**/engine_node_test.py",
    "**/graph/state_graph/base.py",  # Metaclass issues
]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

# Furo theme configuration - Enhanced with vibrant colors
html_theme_options = {
    # Light mode color customization - Modern purple/violet theme
    "light_css_variables": {
        "color-brand-primary": "#8b5cf6",  # Vibrant purple
        "color-brand-content": "#7c3aed",  # Deeper purple for content
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#faf5ff",  # Very light purple tint
        "color-background-border": "#e9d5ff",  # Light purple border
        "color-background-hover": "#f3e8ff",  # Hover state
        "color-sidebar-background": "#faf5ff",
        "color-sidebar-background-border": "#e9d5ff",
        "color-sidebar-brand-text": "#7c3aed",
        "color-sidebar-item-background--current": "#ede9fe",
        "color-sidebar-item-background--hover": "#f3e8ff",
        "color-sidebar-search-background": "#ffffff",
        "color-sidebar-search-border": "#e9d5ff",
        "color-sidebar-search-icon": "#8b5cf6",
        "color-admonition-background": "transparent",
        "color-admonition-title-background--note": "#dbeafe",
        "color-admonition-title-background--tip": "#d1fae5",
        "color-admonition-title-background--important": "#fee2e2",
        "color-admonition-title-background--warning": "#fef3c7",
        "color-admonition-title--note": "#1e40af",
        "color-admonition-title--tip": "#047857",
        "color-admonition-title--important": "#b91c1c",
        "color-admonition-title--warning": "#92400e",
        "color-code-background": "#f8f4ff",  # Light purple code bg
        "color-code-foreground": "#1f2937",
        "color-inline-code-background": "#ede9fe",
        "color-highlight-on-target": "#8b5cf640",
        "color-link": "#8b5cf6",
        "color-link--hover": "#7c3aed",
        "color-link-underline": "#8b5cf640",
        "color-link-underline--hover": "#7c3aed80",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif",
        "font-stack--monospace": "JetBrains Mono, Consolas, Monaco, Courier, monospace",
    },
    # Dark mode color customization - Cyberpunk neon theme
    "dark_css_variables": {
        "color-brand-primary": "#a78bfa",  # Bright violet
        "color-brand-content": "#c084fc",  # Bright purple
        "color-background-primary": "#0f0f23",  # Deep dark blue
        "color-background-secondary": "#1a1a2e",  # Slightly lighter
        "color-background-border": "#2e2e4e",  # Purple-tinted border
        "color-background-hover": "#16213e",
        "color-sidebar-background": "#16162d",
        "color-sidebar-background-border": "#2e2e4e",
        "color-sidebar-brand-text": "#c084fc",
        "color-sidebar-item-background--current": "#312e81",
        "color-sidebar-item-background--hover": "#1e1e3f",
        "color-sidebar-search-background": "#1a1a2e",
        "color-sidebar-search-border": "#2e2e4e",
        "color-sidebar-search-icon": "#a78bfa",
        "color-admonition-background": "transparent",
        "color-admonition-title-background--note": "#1e3a8a20",
        "color-admonition-title-background--tip": "#047857​20",
        "color-admonition-title-background--important": "#991b1b20",
        "color-admonition-title-background--warning": "#92400e20",
        "color-admonition-title--note": "#60a5fa",
        "color-admonition-title--tip": "#34d399",
        "color-admonition-title--important": "#f87171",
        "color-admonition-title--warning": "#fbbf24",
        "color-code-background": "#1a1a2e",
        "color-code-foreground": "#e0e7ff",
        "color-inline-code-background": "#312e81",
        "color-highlight-on-target": "#a78bfa40",
        "color-link": "#a78bfa",
        "color-link--hover": "#c084fc",
        "color-link-underline": "#a78bfa40",
        "color-link-underline--hover": "#c084fc80",
        # API doc specific colors
        "color-api-background": "#1a1a2e",
        "color-api-background--hover": "#312e81",
        "color-api-overall": "#e0e7ff",
        "color-api-name": "#c084fc",
        "color-api-pre-name": "#a78bfa",
        "color-api-paren": "#6366f1",
        "color-api-keyword": "#f472b6",
    },
    # Navigation and UI options
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "announcement": "🚀 <b>haive-core</b> - The foundation of the Haive AI Agent Framework",
    "source_repository": "https://github.com/haive-ai/haive",
    "source_branch": "main",
    "source_directory": "packages/haive-core/docs/",
    # Additional Furo features
    "top_of_page_button": "edit",  # Show edit button at top of pages
}

# Additional HTML configuration
html_title = "haive-core"
html_short_title = "haive-core"
html_favicon = "_static/favicon.ico"
html_logo = "_static/logo.png"
html_show_sourcelink = True
html_copy_source = True
html_last_updated_fmt = "%b %d, %Y"  # Add last updated date
html_use_index = True  # Generate index pages

# CSS files
html_css_files = [
    "custom.css",
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap",
]

# JavaScript files for enhanced functionality
html_js_files = [
    ("custom.js", {"defer": "defer"}),
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# TOC Configuration - Enhanced nesting and presentation
toc_object_entries_show_parents = "hide"
toctree_maxdepth = 4  # Maximum depth for nested TOC
toctree_collapse = False  # Don't collapse by default
toctree_titles_only = False  # Show full titles
toctree_includehidden = True  # Include hidden TOC entries

# Custom sidebar configuration for Furo
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_type_aliases = None

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
    "langchain": ("https://api.python.langchain.com/en/latest/", None),
}

# seed_intersphinx_mapping configuration
pkg_requirements_source = "pyproject"  # Read dependencies from pyproject.toml
repository_root = "../.."  # Path to repo root from docs/source

# Autodoc typehints configuration
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented"
typehints_fully_qualified = False
typehints_use_signature = True
typehints_use_signature_return = True
always_use_bars_union = True

# NEW: Pydantic configuration
autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_model_show_validator_summary = True
autodoc_pydantic_model_show_field_summary = True
autodoc_pydantic_model_show_validator_members = True
autodoc_pydantic_field_list_validators = True
autodoc_pydantic_field_show_constraints = True
autodoc_pydantic_model_erdantic_figure = False
autodoc_pydantic_model_erdantic_figure_collapsed = False

# MyST Parser
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "strikethrough",
]
myst_heading_anchors = 3

# Copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

# Togglebutton
togglebutton_hint = "Click to show"
togglebutton_hint_hide = "Click to hide"

# NEW: Sphinx Tabs configuration
sphinx_tabs_disable_tab_closing = True

# Sitemap
html_baseurl = "https://docs.haive.ai/packages/haive-core/"
sitemap_url_scheme = "{link}"
sitemap_filename = "sitemap.xml"
sitemap_locales = [None]
sitemap_excludes = [
    "search",
    "genindex",
]

# OpenGraph
ogp_site_url = "https://docs.haive.ai/packages/haive-core/"
ogp_site_name = "haive-core Documentation"
ogp_social_cards = {
    "enable": True,
}

# Todo extension
todo_include_todos = True

# Coverage extension
coverage_show_missing_items = True

# Mermaid
mermaid_output_format = "svg"
mermaid_params = [
    "--theme",
    "default",
    "--width",
    "100%",
    "--backgroundColor",
    "transparent",
]

# NEW: Graphviz configuration
graphviz_output_format = "svg"
graphviz_dot_args = ["-Kdot", "-Tsvg"]

# Viewcode extension settings
viewcode_enable_epub = False
viewcode_follow_imported_members = True

# Add source link prefix for GitHub
html_context = {
    "display_github": True,
    "github_user": "haive-ai",
    "github_repo": "haive",
    "github_version": "main",
    "conf_py_path": "/packages/haive-core/docs/source/",
}

# NEW: Git last updated configuration
git_last_updated_show_commit_hash = True
git_last_updated_format = "%Y-%m-%d %H:%M"

# NEW: Sphinx Tippy configuration - Enhanced hover tooltips
# CRITICAL for Furo theme - use correct anchor selector
tippy_anchor_parent_selector = "div.content"  # Required for Furo theme

# Tooltip appearance and behavior
tippy_props = {
    "placement": "auto-start",  # Smart positioning
    "maxWidth": 600,  # Maximum tooltip width
    "theme": "light-border",  # Visual theme
    "delay": [200, 100],  # Show/hide delays
    "duration": [200, 100],  # Animation duration
    "interactive": True,  # Allow interaction with tooltip
    "arrow": True,  # Show arrow
    "animation": "shift-away",  # Animation style
}

# Enable various tooltip types
tippy_enable_mathjax = True  # Math equation tooltips (requires mathjax)
tippy_enable_doitips = True  # DOI link tooltips
tippy_enable_wikitips = True  # Wikipedia link tooltips
tippy_enable_footnotes = True  # Footnote tooltips
tippy_rtd_urls = ["https://docs.haive.ai", "https://haive-core.readthedocs.io"]

# Skip certain classes (avoid tooltips on these)
tippy_skip_anchor_classes = ["headerlink", "sd-stretched-link", "reference-external"]

# Add CSS class to elements with tooltips
tippy_add_class = "has-tooltip"

# Custom tooltips for specific URLs or patterns
tippy_custom_tips = {
    "haive.core.engine": "The core engine module provides the foundation for all agent operations.",
    "AugLLMConfig": "Enhanced LLM configuration with support for multiple providers and advanced features.",
    "ReactAgent": "An agent that uses ReAct (Reasoning and Acting) pattern for complex tasks.",
}

# NEW: Not found page configuration
notfound_context = {
    "title": "Page Not Found",
    "body": """
<h1>🚀 Oops! Page Not Found</h1>
<p>The page you're looking for seems to have wandered off into the documentation cosmos.</p>

<div class="admonition tip">
<p class="admonition-title">Try these options:</p>
<ul>
<li><strong>Search:</strong> Use the search box above to find what you need</li>
<li><strong>API Reference:</strong> Check our <a href="/autoapi/">complete API documentation</a></li> 
<li><strong>Home:</strong> Return to the <a href="/">main documentation</a></li>
</ul>
</div>

<p>Still can't find what you're looking for? <a href="https://github.com/haive-ai/haive/issues">Report an issue</a> and we'll help you out!</p>
    """,
}
notfound_template = "page.html"
notfound_no_urls_prefix = True

# NEW: Sphinx-exec-code configuration
exec_code_working_dir = "../../"
exec_code_source_file_link = True

# NEW: Toggle prompt configuration
toggleprompt_offset_right = 30
toggleprompt_default_hidden = "true"

# NEW: Issues configuration
issues_github_path = "haive-ai/haive"
issues_uri = "https://github.com/haive-ai/haive/issues/{issue}"

# NEW: Sphinx-git configuration
sphinx_git_changelog = True
sphinx_git_changelog_title = "📝 Documentation Changes"
sphinx_git_show_tags = True
sphinx_git_show_branch = True
sphinx_git_tracked_files = ["docs/source/"]
sphinx_git_untracked = False

# NEW: Sphinx-favicon configuration - Multiple favicon support
favicons = [
    {
        "rel": "icon",
        "sizes": "32x32",
        "href": "favicon-32x32.png",
        "type": "image/png",
    },
    {
        "rel": "icon",
        "sizes": "16x16",
        "href": "favicon-16x16.png",
        "type": "image/png",
    },
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "href": "apple-touch-icon.png",
        "type": "image/png",
    },
    {
        "rel": "shortcut icon",
        "href": "favicon.ico",
        "type": "image/x-icon",
    },
]

# NEW: Sphinx-changelog configuration - Structured changelog support
changelog_sections_past = 10  # How many past releases to show
changelog_inner_tag_sort = [
    "breaking",
    "feature",
    "bugfix",
    "improvement",
    "docs",
    "other",
]
changelog_hide_empty_releases = True
changelog_only_releases_branch = "main"
changelog_hide_filtered_commits = True

# NEW: Sphinx-prompt configuration - Enhanced shell prompts
prompt_style = "bash"
prompt_types = {
    "bash": "$",
    "zsh": "$",
    "cmd": ">",
    "powershell": "PS>",
    "python": ">>>",
}

# NEW: Sphinx-hoverxref configuration - Hover cross-references
hoverxref_auto_ref = True
hoverxref_domains = ["py"]
hoverxref_roles = ["doc", "ref", "class", "func", "meth", "mod"]
hoverxref_role_types = {
    "doc": "tooltip",
    "ref": "tooltip",
    "class": "modal",
    "func": "modal",
    "meth": "modal",
}

# NEW: Sphinx-paramlinks configuration - Parameter linking
paramlinks_hyperlink_param = "name"

# NEW: Sphinx-math-dollar configuration - LaTeX math with dollar signs
mathjax_config = {
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
    }
}

# NEW: Sphinx-needs configuration - Requirements tracking
needs_types = [
    {
        "directive": "req",
        "title": "Requirement",
        "prefix": "REQ_",
        "color": "#BFD8D2",
        "style": "node",
    },
    {
        "directive": "spec",
        "title": "Specification",
        "prefix": "SPEC_",
        "color": "#FEDCD2",
        "style": "node",
    },
    {
        "directive": "test",
        "title": "Test Case",
        "prefix": "TEST_",
        "color": "#DF744A",
        "style": "node",
    },
]

needs_id_regex = "^[A-Z0-9_]{5,}"
needs_show_link_type = True
needs_show_link_title = True
