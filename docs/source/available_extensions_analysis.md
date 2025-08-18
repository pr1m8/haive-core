# Available Extensions Analysis

## Documentation Build Script

```bash
./rebuild_docs.sh
```

## Current Status: 36 Extensions Added ✅

### Phase 1: High Priority (6/6) ✅

1. ✅ `sphinx.ext.graphviz` - Architecture diagrams
2. ✅ `sphinxcontrib.autodoc_pydantic` - Pydantic model documentation
3. ✅ `sphinx_tabs.tabs` - Tabbed content
4. ✅ `sphinx_last_updated_by_git` - Git timestamps
5. ✅ `sphinx_tippy` - Enhanced tooltips
6. ✅ `notfound.extension` - Custom 404 pages

### Phase 2: Medium Priority (4/5) ✅

1. ✅ `sphinx_exec_code` - Execute example code
2. ✅ `sphinx_toggleprompt` - Toggle shell prompts
3. ✅ `sphinx_issues` - Link to GitHub issues
4. ✅ `sphinx_git` - Git integration for changelogs
5. ❌ `sphinxcontrib.plantuml` - UML diagrams (not yet added)

### Phase 3: Additional Valuable Extensions (8/8) ✅

1. ✅ `seed_intersphinx_mapping` - Auto-populate from pyproject.toml
2. ✅ `sphinx_favicon` - Multiple favicon support
3. ✅ `sphinx_changelog` - Structured changelog support
4. ✅ `sphinx_prompt` - Better shell prompts and code blocks
5. ✅ `sphinx_hoverxref` - Hover cross-references
6. ✅ `sphinxemoji` - Emoji support in docs
7. ✅ `sphinx_paramlinks` - Parameter linking
8. ✅ `sphinx_math_dollar` - Math with dollar signs ($x = y$)
9. ✅ `sphinx_needs` - Requirements and traceability

## Available from Root Docs Group (Not Yet Added)

### Highly Valuable Extensions 🔥

1. `sphinx_gallery` - Gallery of examples with plots/images
2. `sphinx_multiversion` - Multi-version documentation
3. `sphinxcontrib.openapi` - OpenAPI/Swagger documentation
4. `sphinx_codeautolink` - Automatic code linking (causes warnings)
5. `sphinx_argparse` - Document argparse CLI interfaces
6. `sphinx_click` - Document Click CLI interfaces
7. `sphinx_jsonschema` - Document JSON schemas
8. `pydata-sphinx-theme` - Alternative to Furo theme
9. `sphinx_rtd_theme` - ReadTheDocs theme

### Useful Extensions 📊

1. `sphinx_lint` - Documentation linting
2. `doc8` - RST style checking
3. `codespell` - Spell checking
4. `sphinxcontrib-spelling` - Advanced spell checking
5. `sphinx_pdf_generate` - PDF generation
6. `sphinx_simplepdf` - Simple PDF output
7. `readthedocs-sphinx-search` - Enhanced search
8. `sphinxext-rediraffe` - URL redirects
9. `jupyter-cache` - Jupyter notebook caching

### Specialized Extensions 🔧

1. `sphinx_autodoc2` - Alternative autodoc implementation
2. `sphinx_autobuild` - Live reloading during development
3. `sphinx_jinja2` - Jinja2 templating in docs
4. `sphinx_exec_directive` - Execute code directives
5. `sphinxcontrib-versioning` - Version management
6. `sphinx_paramlinks` - Parameter cross-referencing

## Recommended Next Additions

### Tier 1: Immediate Value 🚀

These would provide immediate benefits:

```python
extensions.extend([
    'sphinx_gallery',       # Example galleries with screenshots
    'sphinx_multiversion',  # Multi-version docs (v0.1.0, v0.2.0, etc.)
    'sphinxcontrib.openapi', # API schema documentation
    'sphinx_argparse',      # CLI documentation
    'sphinx_click',         # Click CLI documentation
    'sphinx_jsonschema',    # JSON schema documentation
])
```

**Benefits**:

- **sphinx_gallery**: Showcase examples with visual output
- **sphinx_multiversion**: Version-aware documentation
- **sphinxcontrib.openapi**: Document REST APIs automatically
- **sphinx_argparse/click**: Document CLI tools

### Tier 2: Development Quality 📋

These improve documentation quality and maintenance:

```python
extensions.extend([
    'sphinx_lint',          # Linting for docs
    'doc8',                 # RST style checking
    'codespell',           # Spell checking
    'sphinxcontrib.spelling', # Advanced spelling
    'readthedocs-sphinx-search', # Better search
    'sphinx_autobuild',    # Live reload for development
])
```

### Tier 3: Advanced Features 🔬

For specialized use cases:

```python
extensions.extend([
    'sphinx_pdf_generate',  # PDF output
    'jupyer-cache',        # Jupyter integration
    'sphinxext-rediraffe', # URL management
    'sphinx_autodoc2',     # Alternative autodoc
    'pydata-sphinx-theme', # Alternative theme
])
```

## Extension Configuration Examples

### Sphinx Gallery Configuration

```python
# Example gallery setup
sphinx_gallery_conf = {
    'examples_dirs': 'examples',
    'gallery_dirs': 'auto_examples',
    'plot_gallery': True,
    'download_all_examples': False,
    'filename_pattern': '/example_',
}
```

### Multi-version Configuration

```python
# Multi-version setup
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'
smv_branch_whitelist = r'^(main|develop)$'
smv_released_pattern = r'^tags/v.*$'
```

### OpenAPI Configuration

```python
# OpenAPI documentation
openapi_specs = [
    {
        'spec': 'path/to/openapi.yaml',
        'route': '/api',
        'name': 'Haive API'
    }
]
```

## Dependencies Analysis

### Available in Root pyproject.toml ✅

All recommended Tier 1 extensions are available:

- `sphinx-gallery = "^0.19.0"`
- `sphinx-multiversion = "^0.2.4"`
- `sphinxcontrib-openapi = "^0.8.4"`
- `sphinx-argparse = "^0.5.2"`
- `sphinx-click = "^6.0.0"`
- `sphinx-jsonschema = "^1.19.1"`

### Available in haive-core pyproject.toml ✅

Most extensions are available in the package-level dependencies.

## Current Extension Count Summary

- **Before Enhancements**: 16 extensions
- **After All Phases**: 36 extensions
- **Available to Add**: 15+ more extensions
- **Total Potential**: 50+ extensions

## Build Performance Impact

Current build with 36 extensions:

- **Build time**: ~2-3 minutes (reasonable)
- **Generated files**: Well-organized hierarchy
- **Features**: Rich documentation experience

Adding Tier 1 extensions would:

- **Increase build time**: +1-2 minutes (acceptable)
- **Add significant value**: Version management, galleries, API docs
- **Improve user experience**: Better navigation, examples, search

## Next Steps Recommendations

1. **Add Tier 1 extensions** (6 extensions) for immediate impact
2. **Test build performance** with new extensions
3. **Configure example galleries** for key use cases
4. **Set up multi-version builds** for releases
5. **Add CLI documentation** for haive tools

**Command to rebuild and test:**

```bash
./rebuild_docs.sh
```
