Haive Development Utilities Documentation
==========================================

**Unified debugging, logging, tracing, and profiling for Python development**

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code Style

Welcome to the **Haive Development Utilities** documentation! This comprehensive toolkit
provides powerful, easy-to-use debugging, logging, tracing, profiling, and benchmarking
capabilities for Python development.

🚀 Quick Start
--------------

Get started immediately with a single import:

.. code-block:: python

   from haive.core.utils.dev import debug, log, trace, profile, benchmark
   
   # Enhanced debugging (icecream replacement)
   debug.ice("Hello, World!", variable=42)
   
   # Rich logging with context
   with log.context("my_operation"):
       log.info("Starting process...")
       log.success("Complete!")
   
   # Performance profiling
   @profile.time
   def my_function():
       return sum(i**2 for i in range(10000))

✨ Why Choose Haive Dev Utilities?
----------------------------------

**Traditional Problems:**

- 🐛 Basic ``print()`` statements clutter output
- 📋 Standard logging lacks visual appeal  
- ⚡ Separate tools for profiling and benchmarking
- 🔧 Complex setup for performance analysis
- 🔄 Inconsistent APIs across different tools

**Our Solutions:**

- 🎨 **Rich Output**: Beautiful terminal formatting with colors and structure
- 🔧 **Unified API**: Single import for all development utilities
- 🛡️ **Smart Fallbacks**: Works even when optional dependencies are missing
- 📈 **Progressive Enhancement**: Simple for quick debugging, powerful for deep analysis
- ⚙️ **Zero Configuration**: Works out of the box with sensible defaults

🎯 Core Features
----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🐛 Enhanced Debugging
      :link: debugging
      :link-type: doc

      icecream replacement, web debugging, visual debugging, auto-breakpoints

   .. grid-item-card:: 📋 Rich Logging
      :link: logging
      :link-type: doc

      Structured logging, beautiful output, context managers, data visualization

   .. grid-item-card:: 🔍 Code Tracing
      :link: tracing
      :link-type: doc

      Function call tracking, variable monitoring, advanced tracing, reports

   .. grid-item-card:: ⚡ Performance Profiling
      :link: profiling
      :link-type: doc

      Function timing, memory profiling, CPU analysis, comprehensive profiling

   .. grid-item-card:: 🧪 Benchmarking
      :link: benchmarking
      :link-type: doc

      Precise timing, load testing, spike testing, async benchmarking

   .. grid-item-card:: 🔗 Integration
      :link: integration
      :link-type: doc

      Unified workflows, production usage, best practices

📚 Documentation
----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/debugging
   guides/logging
   guides/tracing
   guides/profiling
   guides/benchmarking
   guides/integration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/debugging
   api/logging
   api/tracing
   api/profiling
   api/benchmarking

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_usage
   examples/advanced_workflows
   examples/production_patterns
   examples/performance_analysis

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   contributing
   license

🏃‍♂️ Quick Examples
-------------------

**Simple Debugging:**

.. code-block:: python

   def calculate_tax(income, rate):
       debug.ice(income, rate)  # See inputs beautifully
       
       tax = income * rate
       debug.ice(tax)  # See result
       
       return tax

**Rich Logging Workflow:**

.. code-block:: python

   def process_orders(orders):
       log.info("Processing orders", count=len(orders))
       
       with log.context("validation"):
           log.progress("Validating orders...")
           # validation code
           log.success("Orders validated!")
       
       with log.context("processing"):
           for i, order in enumerate(orders):
               if i % 10 == 0:
                   log.progress(f"Processed {i}/{len(orders)}")
           log.success("All orders processed!")

**Performance Analysis:**

.. code-block:: python

   @profile.comprehensive  # CPU, memory, and timing
   @trace.calls           # Track all function calls
   def data_processor(dataset):
       with log.context("data_processing"):
           debug.ice("Processing dataset", size=len(dataset))
           
           result = heavy_computation(dataset)
           
           log.success("Processing complete!")
           return result

🛡️ Production Ready
-------------------

Designed for production use with zero overhead when disabled:

.. code-block:: python

   import os
   from haive.core.utils.dev import debug, trace
   
   # Environment-based control
   DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
   
   if not DEBUG_MODE:
       debug.disable()
       trace.call_tracker.disable()
       trace.var_tracker.disable()

When disabled, all operations become no-ops with minimal performance impact.

🔗 Dependencies
---------------

**Core (always available):**

- Python 3.8+
- Rich (with fallback to standard output)

**Optional (graceful fallbacks):**

- icecream, pdb++, web-pdb, pudb, birdseye
- pysnooper, hunter, line_profiler, memory_profiler  
- pyinstrument, scalene, matplotlib, psutil

The module provides smart fallbacks when optional dependencies are missing.

📊 Features Overview
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Module
     - Key Features
     - Use Cases
   * - **debug**
     - icecream replacement, web debugging, visual debugging
     - Variable inspection, interactive debugging, remote debugging
   * - **log**
     - Rich output, context managers, data visualization
     - Structured logging, progress tracking, data display
   * - **trace**
     - Call tracking, variable monitoring, detailed tracing
     - Function analysis, execution flow, debugging complex logic
   * - **profile**
     - Timing, memory, CPU profiling, benchmarking
     - Performance optimization, bottleneck identification
   * - **benchmark**
     - Load testing, spike testing, statistical analysis
     - Performance comparison, stress testing, capacity planning

💡 Need Help?
-------------

- 📖 **Documentation**: You're reading it! Explore the sections above
- 🐛 **Issues**: Report bugs and request features on `GitHub <https://github.com/haive-ai/haive/issues>`_
- 💬 **Discussions**: Join the community on `GitHub Discussions <https://github.com/haive-ai/haive/discussions>`_
- 📧 **Email**: Contact us at dev@haive.ai

🤝 Contributing
---------------

We welcome contributions! See our :doc:`contributing` guide for details.

📄 License
----------

This project is licensed under the MIT License - see the :doc:`license` file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`