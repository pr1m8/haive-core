"""Haive Core Tools Package.

This package provides tools that agents can use, including store management
tools for memory operations similar to LangMem.
"""

import lazy_loader as lazy

# Define submodules to lazy load
submodules = ["interrupt_tool_wrapper", "store_manager", "store_tools"]

# Define specific attributes from submodules to expose
# TODO: Customize this based on actual exports from each submodule
submod_attrs = {
    "interrupt_tool_wrapper": [],  # TODO: Add specific exports from interrupt_tool_wrapper
    "store_manager": [],  # TODO: Add specific exports from store_manager
    "store_tools": [],  # TODO: Add specific exports from store_tools
}

# Attach lazy loading - this creates __getattr__, __dir__, and __all__
__getattr__, __dir__, __all__ = lazy.attach(
    __name__, submodules=submodules, submod_attrs=submod_attrs
)

# Add any eager imports here (lightweight utilities, etc.)
# Example: from .metadata import SomeUtility
# __all__ += ['SomeUtility']
