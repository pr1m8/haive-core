"""Tool_Utils utility module.

This module provides tool utils functionality for the Haive framework.

Functions:
"""

from langchain_core.tools import BaseTool


def _format_tool_descriptions(tools: list[BaseTool]) -> str:
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(tool_descriptions)
