"""
Debug utilities submodule.

This submodule provides enhanced debugging capabilities including
variable inspection, interactive debugging, and decorators.
"""

try:
    from haive.core.utils.debugkit.debug.enhanced import EnhancedDebug
except ImportError:
    from haive.core.utils.debugkit.fallbacks import FallbackDebug as EnhancedDebug

try:
    from haive.core.utils.debugkit.debug.decorators import debug_decorators
except ImportError:
    debug_decorators = None

try:
    from haive.core.utils.debugkit.debug.inspection import variable_inspector
except ImportError:
    variable_inspector = None

try:
    from haive.core.utils.debugkit.debug.interactive import interactive_debugger
except ImportError:
    interactive_debugger = None

# Create default debug instance
debug = EnhancedDebug()

__all__ = [
    "debug",
    "EnhancedDebug",
    "debug_decorators",
    "variable_inspector",
    "interactive_debugger",
]
