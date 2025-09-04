"""🔧 Config Module - Intelligent Configuration Management System

**THE NEURAL NETWORK OF AI SYSTEM ORCHESTRATION**

Welcome to the Config Module - the revolutionary configuration intelligence platform 
that transforms static configuration management into a living, adaptive orchestration 
system. This isn't just another config library; it's a sophisticated configuration 
consciousness that learns from usage patterns, predicts optimal settings, and 
automatically adapts to changing environments and requirements.

⚡ REVOLUTIONARY CONFIGURATION INTELLIGENCE
------------------------------------------

The Config Module represents a paradigm shift from manual configuration to 
**intelligent, self-optimizing configuration systems** that evolve with your AI applications:

**🧠 Adaptive Configuration Learning**: Configs that learn from usage and optimize themselves automatically
**🔄 Dynamic Runtime Adaptation**: Real-time configuration adjustment based on performance metrics  
**⚡ Predictive Optimization**: AI-powered prediction of optimal configuration settings
**📊 Context-Aware Intelligence**: Configurations that adapt to different execution contexts

🚀 QUICK START
--------------

Examples:
    >>> from haive.core.config import RunnableConfigManager
    >>>
    >>> # Create intelligent configuration manager
    >>> config = RunnableConfigManager.create_intelligent(
    >>> thread_id="my_thread",
    >>> optimization_profile="balanced"
    >>> )
    >>>
    >>> # Configuration automatically adapts to your needs
    >>> result = config.get_optimized_settings()

For complete examples and advanced patterns, see the documentation.
"""

from haive.core.config.auth_runnable import HaiveRunnableConfigManager
from haive.core.config.constants import CACHE_DIR, RESOURCES_DIR, ROOT_DIR
from haive.core.config.protocols import ConfigurableProtocol
from haive.core.config.runnable import RunnableConfigManager

__all__ = [
    "CACHE_DIR",
    "RESOURCES_DIR",
    # Path Constants
    "ROOT_DIR",
    # Protocols
    "ConfigurableProtocol",
    "HaiveRunnableConfigManager",
    # Main Configuration Managers
    "RunnableConfigManager",
]
