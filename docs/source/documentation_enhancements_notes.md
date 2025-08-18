# haive-core Documentation Enhancement Notes

## Current Issues Identified

### 1. Template Integration

- AutoAPI is generating everything but our templates aren't leveraging specialized extensions
- We have `sphinxcontrib.autodoc_pydantic` installed but not being used in templates
- Enum extensions are available but not integrated
- Special method documentation could be enhanced

### 2. Docstring Formatting Issues

- Some docstrings are using markdown-style code blocks (```python) instead of RST format
- This causes raw code blocks to appear in the documentation
- Need to use proper RST directives like:
  - `.. code-block:: python` for code examples
  - `.. note::` for notes
  - `.. warning::` for warnings
  - `.. admonition::` for custom callouts

### 3. Cross-Linking Problems

- Lack of automatic cross-references between modules
- Missing intersphinx mappings for external libraries
- AutoAPI references not being properly linked

### 4. Navigation Structure

- API reference needs to be more prominent in TOC
- haive-core should be top-level with dropdowns
- Need better autosummary integration for quick navigation

### 5. Extension Utilization

Currently installed but underutilized extensions:

- `sphinx_autodoc_typehints` - Could enhance type documentation
- `sphinx_design` - Grid layouts, cards, tabs, etc.
- `sphinxcontrib.mermaid` - Diagrams in docstrings
- `sphinx_exec_code` - Execute examples in documentation
- `sphinx_issues` - Link to GitHub issues
- `sphinx_git` - Git integration for changelogs
- `sphinx_tippy` - Better tooltips for terms
- `sphinx_toggleprompt` - Toggle shell prompts

### 6. AutoAPI Template Enhancements Needed

- Custom module template with better organization
- Class template with Pydantic field documentation
- Function template with better parameter tables
- Package template with autosummary integration

## Proposed Enhancements

### 1. Fix Docstring Format

- Create a script to convert markdown code blocks to RST
- Add proper RST directives for examples, notes, warnings
- Use sphinx-specific cross-reference syntax

### 2. Enhanced Templates

- Create custom templates that leverage:
  - Pydantic autodoc for model fields
  - Enum documentation for choices
  - Type hint integration
  - Autosummary for quick navigation

### 3. Better Configuration

- Add intersphinx mappings for common libraries
- Configure autodoc_typehints for better type documentation
- Set up proper cross-referencing

### 4. Navigation Improvements

- Restructure TOC to have API at top level
- Use sphinx-design for better layout
- Add quick navigation cards

### 5. Rich Content Integration

- Use admonitions for important information
- Add diagrams with mermaid where helpful
- Include executable examples
- Add copy buttons to code blocks

## Quick Wins

1. Fix the docstring format issues (RST vs markdown)
2. Update templates to use autosummary
3. Add intersphinx mappings
4. Configure pydantic autodoc properly
5. Enhance the main index.rst with better navigation

## Template Files to Create/Modify

1. `_templates/autoapi/python/module.rst` - Enhanced with autosummary
2. `_templates/autoapi/python/class.rst` - Pydantic field documentation
3. `_templates/autoapi/python/function.rst` - Better parameter tables
4. `_templates/autoapi/python/package.rst` - Overview with cards
5. `_templates/autosummary/module.rst` - Module summary template
6. `_templates/autosummary/class.rst` - Class summary template
