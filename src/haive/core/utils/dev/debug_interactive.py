"""
Interactive Debugging Utilities

Provides enhanced interactive debugging with pdb variants, web debugging, and visual interfaces.
"""

import inspect
import pdb

# Try to import enhanced debuggers
try:
    import pdbpp  # pdb++  # noqa: F401

    HAS_PDBPP = True
except ImportError:
    HAS_PDBPP = False

try:
    import ipdb

    HAS_IPDB = True
except ImportError:
    HAS_IPDB = False

try:
    import web_pdb

    HAS_WEB_PDB = True
except ImportError:
    HAS_WEB_PDB = False

try:
    import pudb

    HAS_PUDB = True
except ImportError:
    HAS_PUDB = False


class InteractiveDebugger:
    """Interactive debugging with enhanced debuggers."""

    def __init__(self):
        self.debug_enabled = True

    def pdb(self, condition: bool = True) -> None:
        """Start interactive debugging session."""
        if not self.debug_enabled or not condition:
            return

        if HAS_PDBPP:
            pdb.set_trace()
        elif HAS_IPDB:
            ipdb.set_trace()
        else:
            pdb.set_trace()

    def web(self, port: int = 5555, condition: bool = True) -> None:
        """Start web-based debugging session."""
        if not self.debug_enabled or not condition:
            return

        if not HAS_WEB_PDB:
            # Fallback to regular pdb
            self.pdb(condition)
            return

        web_pdb.set_trace(port=port)

    def visual(self, condition: bool = True) -> None:
        """Start visual debugging with pudb."""
        if not self.debug_enabled or not condition:
            return

        if not HAS_PUDB:
            # Fallback to enhanced pdb
            self.pdb(condition)
            return

        inspect.currentframe().f_back
        pudb.set_trace()

    def status(self) -> dict[str, bool]:
        """Get status of available debugging tools."""
        return {
            "debug_enabled": self.debug_enabled,
            "pdb": True,  # Always available
            "pdb++": HAS_PDBPP,
            "ipdb": HAS_IPDB,
            "web_pdb": HAS_WEB_PDB,
            "pudb": HAS_PUDB,
        }

    def enable(self) -> None:
        """Enable interactive debugging."""
        self.debug_enabled = True

    def disable(self) -> None:
        """Disable interactive debugging."""
        self.debug_enabled = False


# Global instance for easy import
interactive_debugger = InteractiveDebugger()
