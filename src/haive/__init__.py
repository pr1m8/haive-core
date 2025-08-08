"""Haive - AI Agent Framework

This is the top-level namespace package for the Haive framework.

The framework is organized into focused packages:

- haive.core: Core framework infrastructure
- haive.agents: Pre-built agent implementations
- haive.tools: Tool integrations and toolkits
- haive.games: Game environments and agents
- haive.dataflow: Streaming and data processing
- haive.mcp: Model Context Protocol integration
- haive.prebuilt: Ready-to-use agent configurations
"""

# This is a namespace package - no code needed here
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
