"""
Haive Core Package

Provides core functionality for the haive framework including:
- Logging management with automatic source tracking
- Engine components
- Graph builders
- Model abstractions
"""

# Set version
__version__ = "0.1.0"
"""
# Monkey-patch rich print if requested before ANY imports
import os
if os.getenv("HAIVE_NO_RICH"):
    try:
        import rich
        rich.print = lambda *args, **kwargs: None
        # Also create a no-op rprint
        import sys
        sys.modules['rprint'] = type(sys)('rprint')
        sys.modules['rprint'].rprint = lambda *args, **kwargs: None
    except ImportError:
        pass

# Auto-configure logging to reduce clutter AND show sources
try:
    from haive.core.logging.auto_config import auto_configure_logging
    
    # Apply default configuration WITH source tracking
    # This happens silently - no messages printed
    auto_configure_logging(preset="default", use_source_formatter=True)
    
    # Only enable print tracking if explicitly requested
    if os.getenv("HAIVE_TRACK_PRINTS"):
        from haive.core.logging.quick_setup import intercept_prints_silent
        intercept_prints_silent()
    
    # Redirect rich print to logging if quiet mode requested
    if os.getenv("HAIVE_QUIET_RICH"):
        from haive.core.logging.quick_setup import redirect_rich_print_to_logging
        redirect_rich_print_to_logging()
    
except ImportError:
    # Logging not available yet - that's okay
    pass

# Export version
__all__ = ["__version__"]
"""
