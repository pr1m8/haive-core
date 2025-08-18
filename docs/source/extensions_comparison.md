# Extensions Comparison: PyDevelop-Docs vs haive-core

## Current haive-core Extensions (16 total)

1. `autoapi.extension` - Automatic API documentation
2. `sphinx.ext.autodoc` - Extract docstrings
3. `sphinx.ext.napoleon` - Google/NumPy docstring styles
4. `sphinx.ext.viewcode` - View source code links
5. `sphinx.ext.intersphinx` - Link to other projects
6. `sphinx.ext.todo` - Todo notes
7. `sphinx.ext.coverage` - Documentation coverage
8. `sphinx.ext.mathjax` - Math rendering
9. `sphinx_autodoc_typehints` - Type hints in docs
10. `sphinx_copybutton` - Copy button for code blocks
11. `sphinx_togglebutton` - Collapsible sections
12. `sphinx_design` - Cards, tabs, grids
13. `sphinxcontrib.mermaid` - Mermaid diagrams
14. `sphinx_sitemap` - Generate sitemap.xml
15. `sphinxext.opengraph` - Social media previews
16. `myst_parser` - Markdown support

## PyDevelop-Docs Extensions (40+ total)

### Core Extensions (we have these)

- ✅ sphinx.ext.autodoc
- ✅ sphinx.ext.napoleon
- ✅ sphinx.ext.viewcode
- ✅ sphinx.ext.intersphinx
- ✅ autoapi.extension

### Enhanced API (partially have)

- ✅ sphinx_autodoc_typehints
- ❌ sphinxcontrib.autodoc_pydantic - Pydantic model documentation
- ❌ seed_intersphinx_mapping - Auto-populate from pyproject.toml

### Content & Design (partially have)

- ✅ myst_parser
- ✅ sphinx_design
- ✅ sphinx_togglebutton
- ✅ sphinx_copybutton
- ❌ sphinx_tabs.tabs - Tabbed content

### Execution Extensions (we don't have)

- ❌ sphinxcontrib.programoutput - Show command output
- ❌ sphinx_exec_code - Execute code blocks

### Diagram Extensions (partially have)

- ❌ sphinx.ext.graphviz - Graphviz diagrams
- ✅ sphinxcontrib.mermaid
- ❌ sphinxcontrib.plantuml - PlantUML diagrams
- ❌ sphinxcontrib.blockdiag - Block diagrams
- ❌ sphinxcontrib.seqdiag - Sequence diagrams
- ❌ sphinxcontrib.nwdiag - Network diagrams
- ❌ sphinxcontrib.actdiag - Activity diagrams

### Utilities (partially have)

- ✅ sphinx_sitemap
- ❌ sphinx_codeautolink - Auto-link code references (disabled due to Python 3.10 issues)

### TOC Enhancements (we don't have)

- ❌ sphinx_treeview - Tree view navigation

### Enhanced Features (we don't have)

- ❌ sphinx_toggleprompt - Toggle shell prompts
- ❌ sphinx_prompt - Better prompt display
- ❌ sphinx_last_updated_by_git - Git-based timestamps
- ❌ sphinx_library - Library documentation
- ❌ sphinx_icontract - Contract documentation
- ❌ sphinx_tippy - Advanced tooltips

### Documentation Tools (we don't have)

- ❌ sphinx_comments - Comments system
- ❌ sphinx_contributors - Show contributors
- ❌ sphinx_issues - GitHub issue links
- ❌ sphinx_needs - Requirements tracking
- ❌ sphinxarg.ext - Argparse documentation
- ❌ notfound.extension - Custom 404 pages
- ❌ sphinx_reredirects - URL redirects
- ❌ sphinxext.rediraffe - Smart redirects
- ❌ sphinx_git - Git integration
- ❌ sphinx_debuginfo - Debug information
- ❌ sphinx_tags - Tag system
- ❌ sphinx_favicon - Multiple favicons
- ❌ sphinx_combine - Combine documentation

## Key Configuration Differences

### PyDevelop-Docs Has:

1. **Tippy tooltips configuration** - Advanced hover tooltips
2. **Git integration** - Last updated dates, changelog
3. **404 page customization** - Better error pages
4. **Requirements tracking** - sphinx_needs for tracking requirements
5. **Pydantic model documentation** - Full Pydantic support
6. **Multiple diagram formats** - PlantUML, Graphviz, various diagrams
7. **Code execution** - Run code examples in docs
8. **Advanced TOC** - Treeview navigation
9. **Social features** - Comments, contributors
10. **Debug info** - Performance metrics

### Recommended Additions for haive-core:

#### High Priority (would significantly improve docs):

1. `sphinxcontrib.autodoc_pydantic` - Since Haive uses Pydantic extensively
2. `sphinx_last_updated_by_git` - Show when docs were last updated
3. `sphinx.ext.graphviz` - For architecture diagrams
4. `sphinx_tabs.tabs` - For multi-language code examples
5. `notfound.extension` - Better 404 pages
6. `sphinx_tippy` - Enhanced tooltips

#### Medium Priority (nice to have):

1. `sphinx_exec_code` - Execute example code
2. `sphinx_toggleprompt` - Clean command examples
3. `sphinx_issues` - Link to GitHub issues
4. `sphinxcontrib.plantuml` - UML diagrams
5. `sphinx_git` - Git integration

#### Low Priority (optional):

1. `sphinx_contributors` - Show contributors
2. `sphinx_tags` - Tag system
3. `sphinx_needs` - Requirements tracking
4. Various diagram extensions (blockdiag, seqdiag, etc.)

## Configuration Enhancements to Add:

1. **Tippy tooltips** - Better hover information
2. **Git timestamps** - Show last updated
3. **404 page** - Custom not found page
4. **Pydantic config** - Full model documentation
5. **Multiple favicons** - Better browser support
6. **Breadcrumbs** - Already added via custom template
7. **Code execution** - Run examples in docs
