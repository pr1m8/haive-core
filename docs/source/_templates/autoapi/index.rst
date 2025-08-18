{# Enhanced AutoAPI index template #}
API Reference
=============

This section contains the complete API reference for **{{ project }}**.

.. contents:: API Contents
   :local:
   :depth: 2

Overview
--------

The **{{ project }}** API is organized into the following main components:

Core Components
~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: autoapi
   :recursive:

   haive.core.engine
   haive.core.schema
   haive.core.graph
   haive.core.memory
   haive.core.tools

Common Utilities
~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: autoapi
   :recursive:

   haive.core.common
   haive.core.utils
   haive.core.config

Full API
--------

.. autoapitoctree:: {{ autoapi_root }}
   :maxdepth: 3

Indices
-------

* :ref:`genindex`
* :ref:`modindex`