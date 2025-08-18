# haive-core Documentation Enhancement Plan

## 🎯 Goals

1. Fix docstring formatting (markdown → RST)
2. Enable breadcrumb navigation with Furo
3. Improve cross-linking and references
4. Leverage all installed extensions
5. Create better AutoAPI templates
6. Enhance navigation and discoverability

## 🔍 Specific Issues Found

### Example: haive.core.models.vectorstore Documentation

1. **Markdown formatting instead of RST**:
   - Bullet lists using `-` instead of `*` or proper RST lists
   - No section underlines (====, ----, ~~~~)
   - Code blocks without `.. code-block::` directive

2. **Poor structure**:
   - Enum values listed as separate attributes (confusing)
   - No autosummary tables
   - Missing proper class/method organization

3. **No cross-references**:
   - Plain text instead of `:class:`, `:func:`, `:mod:` references
   - No intersphinx links to external libraries

## 📋 Phase 1: Foundation Fixes (Immediate)

### 1.1 Fix Docstring Formatting

- **Problem**: Docstrings using ```python instead of RST format
- **Solution**: Create a script to convert all markdown-style code blocks to RST
- **Actions**:
  ````
  - Convert ```python to .. code-block:: python
  - Convert markdown lists to RST lists
  - Add proper indentation for code blocks
  - Convert inline `code` to :code:`code`
  ````

### 1.2 Enable Breadcrumbs

- **Furo Feature**: Built-in breadcrumb support
- **Configuration**:
  ```python
  html_theme_options = {
      "navigation_depth": 4,  # Enable deeper navigation
      "show_nav_level": 2,    # Show navigation levels
      "show_toc_level": 3,    # Table of contents depth
  }
  ```
- **CSS Enhancement**: Custom breadcrumb styling

### 1.3 Configure Intersphinx

- **Purpose**: Enable cross-references to external docs
- **Mappings to add**:
  ```python
  intersphinx_mapping = {
      "python": ("https://docs.python.org/3/", None),
      "langchain": ("https://python.langchain.com/", None),
      "pydantic": ("https://docs.pydantic.dev/", None),
      "numpy": ("https://numpy.org/doc/stable/", None),
  }
  ```

## 📋 Phase 2: Template Enhancements

### 2.1 AutoAPI Module Template

- **Features**:
  - Autosummary tables for classes/functions
  - Admonitions for important info
  - Better organization with sections
  - Breadcrumb navigation

### 2.2 Pydantic Integration

- **Use sphinxcontrib.autodoc_pydantic**:
  ```rst
  .. autopydantic_model:: MyModel
     :model-show-json: True
     :model-show-field-summary: True
     :model-show-validator-members: True
  ```

### 2.3 Enhanced Class Template

- **Features**:
  - Field tables for Pydantic models
  - Method grouping (public, private, special)
  - Inheritance diagrams
  - Examples in tabs

## 📋 Phase 3: Navigation & Discovery

### 3.1 Enhanced Index Structure

```rst
.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Quick Start
      :link: quickstart
      :link-type: doc

      Get started with haive-core in minutes

   .. grid-item-card:: 📚 API Reference
      :link: autoapi/haive/core/index
      :link-type: doc

      Complete API documentation
```

### 3.2 API Reference Reorganization

- Move API to top-level TOC
- Create category pages with overviews
- Add "see also" sections
- Include usage examples

### 3.3 Search Enhancements

- Configure search_language
- Add search suggestions
- Include code examples in search

## 📋 Phase 4: Rich Content Integration

### 4.1 Admonitions & Callouts

```rst
.. admonition:: Best Practice
   :class: tip

   Always use AugLLMConfig for agent configuration

.. warning::
   This API is experimental and may change
```

### 4.2 Code Examples

- Use sphinx_exec_code for runnable examples
- Add copy buttons (already configured)
- Show/hide output with toggles
- Include error handling examples

### 4.3 Diagrams

```rst
.. mermaid::

   graph LR
       A[User] --> B[Agent]
       B --> C[Engine]
       C --> D[LLM]
```

## 📋 Phase 5: Extensions Configuration

### 5.1 Autodoc Type Hints

```python
autodoc_typehints = "both"  # Show in signature and description
autodoc_typehints_format = "short"  # Shorter type representations
autodoc_type_aliases = {
    "StateSchema": "haive.core.schema.StateSchema",
}
```

### 5.2 Pydantic Configuration

```python
autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_field_summary = True
autodoc_pydantic_field_list_validators = True
autodoc_pydantic_field_show_constraints = True
```

### 5.3 Git Integration

```python
# For changelog generation
sphinx_git_changelog_titles = {
    "feature": "New Features",
    "bugfix": "Bug Fixes",
    "doc": "Documentation",
}
```

## 🚀 Implementation Order

1. **Day 1**: Fix docstrings, enable breadcrumbs, configure intersphinx
2. **Day 2**: Create enhanced AutoAPI templates
3. **Day 3**: Reorganize navigation, add rich content
4. **Day 4**: Configure all extensions properly
5. **Day 5**: Polish and optimize

## 📝 Custom CSS for Breadcrumbs

```css
/* _static/custom.css */
.breadcrumb-nav {
  padding: 0.5rem 0;
  font-size: 0.875rem;
  border-bottom: 1px solid var(--color-background-border);
}

.breadcrumb-item {
  display: inline;
  color: var(--color-foreground-secondary);
}

.breadcrumb-item::after {
  content: " › ";
  margin: 0 0.25rem;
}

.breadcrumb-item:last-child::after {
  content: "";
}
```

## 🎨 Furo-Specific Features to Enable

1. **Announcement Bar**: Already configured ✓
2. **Edit Button**: Already configured ✓
3. **Navigation with Keys**: Already configured ✓
4. **Collapsible Sidebar**: Add configuration
5. **Version Switcher**: Add if needed
6. **Custom Footer**: Add copyright/links

## 📊 Success Metrics

- [ ] All docstrings in RST format
- [ ] Breadcrumbs visible on all pages
- [ ] Cross-references working (internal & external)
- [ ] Pydantic models properly documented
- [ ] Autosummary tables generated
- [ ] Rich content (admonitions, tabs, grids)
- [ ] Improved search functionality
- [ ] Clear navigation hierarchy

## 🔧 Tools Needed

1. **Docstring converter script** (Python → RST)
2. **Template generator** for AutoAPI
3. **CSS customization** for breadcrumbs
4. **Build validation script** to check for issues

This plan will transform the haive-core documentation into a professional, navigable, and feature-rich resource!
