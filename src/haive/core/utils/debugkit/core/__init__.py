"""
Core debugkit components.

This submodule contains the main classes and utilities that form
the foundation of the debugkit system.
"""

from haive.core.utils.debugkit.core.context import DevContext

# UnifiedDev and CodeAnalysisReport will be imported from unified.py once created
try:
    from haive.core.utils.debugkit.core.unified import CodeAnalysisReport, UnifiedDev
except ImportError:
    # Temporary - these will be moved here later
    UnifiedDev = None
    CodeAnalysisReport = None

__all__ = [
    "DevContext",
    "UnifiedDev",
    "CodeAnalysisReport",
]
