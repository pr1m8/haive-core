{# Enhanced AutoAPI module template #}
{% if not obj.display %}
:orphan:

{% endif %}
:py:mod:`{{ obj.name }}`
=========={{ "=" * obj.name|length }}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|indent(3) }}

{% endif %}

{% block subpackages %}
{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% if visible_subpackages %}
{{ _('Subpackages') }}
{{ "-" * _('Subpackages')|length }}

.. toctree::
   :titlesonly:
   :maxdepth: 1

{% for subpackage in visible_subpackages %}
   {{ subpackage.short_name }}/index.rst
{%- endfor %}
{% endif %}
{% endblock %}

{% block submodules %}
{% set visible_submodules = obj.submodules|selectattr("display")|list %}
{% if visible_submodules %}
{{ _('Submodules') }}
{{ "-" * _('Submodules')|length }}

.. toctree::
   :titlesonly:
   :maxdepth: 1

{% for submodule in visible_submodules %}
   {{ submodule.short_name }}.rst
{%- endfor %}

{% endif %}
{% endblock %}

{% block summary %}
{% if obj.all is not none %}
{% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% elif obj.type is equalto("package") %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% else %}
{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}
{% if visible_children %}

{{ _('Package Contents') }}
{{ "-" * _('Package Contents')|length }}

.. raw:: html

   <div class="autoapi-summary">

{% block classes %}
{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
{% if visible_classes %}

Classes
~~~~~~~

.. autoapisummary::

{% for klass in visible_classes %}
   {{ obj.name }}.{{ klass.short_name }}
{%- endfor %}

{% endif %}
{% endblock %}

{% block functions %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
{% if visible_functions %}

Functions
~~~~~~~~~

.. autoapisummary::

{% for function in visible_functions %}
   {{ obj.name }}.{{ function.short_name }}
{%- endfor %}

{% endif %}
{% endblock %}

{% block exceptions %}
{% set visible_exceptions = visible_children|selectattr("type", "equalto", "exception")|list %}
{% if visible_exceptions %}

Exceptions
~~~~~~~~~~

.. autoapisummary::

{% for exception in visible_exceptions %}
   {{ obj.name }}.{{ exception.short_name }}
{%- endfor %}

{% endif %}
{% endblock %}

{% block attributes %}
{% set visible_attributes = visible_children|selectattr("type", "equalto", "data")|list %}
{% if visible_attributes %}

Module Attributes
~~~~~~~~~~~~~~~~~

.. autoapisummary::

{% for attribute in visible_attributes %}
   {{ obj.name }}.{{ attribute.short_name }}
{%- endfor %}

{% endif %}
{% endblock %}

.. raw:: html

   </div>

{% endif %}
{% endblock %}

{% block content %}
{% if obj.all is not none %}
{% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% elif obj.type is equalto("package") %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% else %}
{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}

{% if visible_children %}

{{ _('API Reference') }}
{{ "-" * _('API Reference')|length }}

{% for obj_item in visible_children %}
{{ obj_item.rendered|indent(0) }}
{% endfor %}

{% endif %}
{% endblock %}