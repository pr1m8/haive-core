"""Haive Component Discovery System.

A comprehensive tool for discovering, analyzing, and documenting components
in the Haive AI framework.
"""

from haive.core.utils.haive_discovery.base_analyzer import ComponentAnalyzer
from haive.core.utils.haive_discovery.component_info import ComponentInfo
from haive.core.utils.haive_discovery.discovery_engine import EnhancedComponentDiscovery
from haive.core.utils.haive_discovery.documentation_writer import DocumentationWriter
from haive.core.utils.haive_discovery.engine_analyzer import EngineAnalyzer
from haive.core.utils.haive_discovery.haive_discovery import HaiveComponentDiscovery
from haive.core.utils.haive_discovery.retriever_analyzers import (
    RetrieverAnalyzer,
    VectorStoreAnalyzer,
)
from haive.core.utils.haive_discovery.tool_analyzers import (
    DocumentLoaderAnalyzer,
    ToolAnalyzer,
)

# Import utility functions
from haive.core.utils.haive_discovery.utils import (  # Quick discovery functions; Tool extraction functions; Engine functions; Documentation functions; Analysis functions; Factory functions
    analyze_failed_imports,
    create_custom_analyzer,
    create_discovery,
    create_tool_from_component,
    discover_all,
    discover_engines,
    discover_loaders,
    discover_retrievers,
    discover_tools,
    discover_vector_stores,
    find_components_by_name,
    find_components_with_env_vars,
    generate_json_catalog,
    generate_markdown_report,
    get_all_engine_configs,
    get_all_tools,
    get_discovery_stats,
    get_engines_by_type,
    get_tools_by_type,
    quick_discover,
    save_discovery_report,
)

__all__ = [
    "ComponentAnalyzer",
    # Core classes
    "ComponentInfo",
    "DocumentLoaderAnalyzer",
    "DocumentationWriter",
    "EngineAnalyzer",
    "EnhancedComponentDiscovery",
    "HaiveComponentDiscovery",
    "RetrieverAnalyzer",
    "ToolAnalyzer",
    "VectorStoreAnalyzer",
    # Analysis functions
    "analyze_failed_imports",
    "create_custom_analyzer",
    # Factory functions
    "create_discovery",
    "create_tool_from_component",
    "discover_all",
    "discover_engines",
    "discover_loaders",
    "discover_retrievers",
    "discover_tools",
    "discover_vector_stores",
    "find_components_by_name",
    "find_components_with_env_vars",
    "generate_json_catalog",
    "generate_markdown_report",
    # Engine functions
    "get_all_engine_configs",
    # Tool functions
    "get_all_tools",
    "get_discovery_stats",
    "get_engines_by_type",
    "get_tools_by_type",
    # Quick discovery functions
    "quick_discover",
    # Documentation functions
    "save_discovery_report",
]

__version__ = "0.1.0"
