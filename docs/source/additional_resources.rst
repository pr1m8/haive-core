Additional Resources
====================

.. raw:: html

   <style>
   .resource-section {
       margin: 1.5rem 0;
       padding: 1.5rem;
       background: var(--color-background-secondary, #f9fafb);
       border-radius: 8px;
       border: 1px solid var(--color-brand-primary, #8b5cf6);
   }
   
   .resource-grid {
       display: grid;
       grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
       gap: 1rem;
       margin-top: 1rem;
   }
   
   .resource-card {
       padding: 1rem;
       background: var(--color-background-primary, #ffffff);
       border-radius: 6px;
       border: 1px solid var(--color-background-border, #e5e7eb);
       transition: all 0.2s;
   }
   
   .resource-card:hover {
       border-color: var(--color-brand-primary, #8b5cf6);
       box-shadow: 0 2px 8px rgba(139, 92, 246, 0.1);
   }
   </style>

External Links
--------------

.. raw:: html

   <div class="resource-section">
      <div class="resource-grid">
         <div class="resource-card">
            <h3>🐙 GitHub Repository</h3>
            <p>Source code, issues, and contributions</p>
            <a href="https://github.com/pr1m8/haive-core" target="_blank">
               github.com/pr1m8/haive-core →
            </a>
         </div>
         
         <div class="resource-card">
            <h3>💬 Discord Community</h3>
            <p>Chat with developers and get help</p>
            <a href="https://discord.gg/haive" target="_blank">
               Join Discord →
            </a>
         </div>
         
         <div class="resource-card">
            <h3>📚 Haive Central Docs</h3>
            <p>Complete Haive ecosystem documentation</p>
            <a href="https://docs.haive.io" target="_blank">
               docs.haive.io →
            </a>
         </div>
         
         <div class="resource-card">
            <h3>📦 PyPI Package</h3>
            <p>Install via pip or poetry</p>
            <a href="https://pypi.org/project/haive-core/" target="_blank">
               View on PyPI →
            </a>
         </div>
      </div>
   </div>

Developer Resources
-------------------

.. dropdown:: Code Examples
   :animate: fade-in-slide-down
   :open:
   
   * `Example Agents <https://github.com/pr1m8/haive-core/tree/main/examples>`_
   * `Test Suite <https://github.com/pr1m8/haive-core/tree/main/tests>`_
   * `Notebooks <https://github.com/pr1m8/haive-core/tree/main/notebooks>`_

.. dropdown:: Contributing
   :animate: fade-in-slide-down
   
   * `Contributing Guide <https://github.com/pr1m8/haive-core/blob/main/CONTRIBUTING.md>`_
   * `Code of Conduct <https://github.com/pr1m8/haive-core/blob/main/CODE_OF_CONDUCT.md>`_
   * `Development Setup <https://github.com/pr1m8/haive-core/blob/main/docs/DEVELOPMENT.md>`_

.. dropdown:: API Documentation
   :animate: fade-in-slide-down
   
   * :doc:`autoapi/haive/core/index`
   * :doc:`common_module_overview`
   * :doc:`api_reference`

Related Projects
----------------

.. dropdown:: Haive Ecosystem
   :animate: fade-in-slide-down
   
   * `haive-agents <https://github.com/pr1m8/haive-agents>`_ - Pre-built agent implementations
   * `haive-tools <https://github.com/pr1m8/haive-tools>`_ - Tool integrations
   * `haive-games <https://github.com/pr1m8/haive-games>`_ - Game environments
   * `haive-mcp <https://github.com/pr1m8/haive-mcp>`_ - Model Context Protocol
   * `haive-dataflow <https://github.com/pr1m8/haive-dataflow>`_ - Stream processing
   * `haive-prebuilt <https://github.com/pr1m8/haive-prebuilt>`_ - Ready-to-use configurations

Support & Help
--------------

.. raw:: html

   <div class="resource-section" style="background: #fef3c7; border-color: #f59e0b;">
      <h3 style="margin-top: 0;">🆘 Need Help?</h3>
      <ul>
         <li>📖 Check the <a href="getting_started.html">Getting Started Guide</a></li>
         <li>💬 Ask in <a href="https://discord.gg/haive" target="_blank">Discord</a></li>
         <li>🐛 Report bugs on <a href="https://github.com/pr1m8/haive-core/issues" target="_blank">GitHub Issues</a></li>
         <li>💡 Request features via <a href="https://github.com/pr1m8/haive-core/discussions" target="_blank">GitHub Discussions</a></li>
      </ul>
   </div>

License & Citation
------------------

.. dropdown:: License Information
   :animate: fade-in-slide-down
   
   haive-core is released under the MIT License.
   
   See the full `LICENSE <https://github.com/pr1m8/haive-core/blob/main/LICENSE>`_ file.

.. dropdown:: How to Cite
   :animate: fade-in-slide-down
   
   If you use haive-core in your research, please cite:
   
   .. code-block:: bibtex
   
      @software{haive-core,
        author = {Haive Team},
        title = {haive-core: Foundation for Intelligent AI Agents},
        year = {2025},
        url = {https://github.com/pr1m8/haive-core}
      }