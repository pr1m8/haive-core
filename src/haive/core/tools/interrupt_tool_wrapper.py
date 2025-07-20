"""from typing import Any
Human-in-the-Loop Tool Wrapper for LangGraph Agents.

This module defines a utility function `add_human_in_the_loop` that allows
LangChain tools to be wrapped with interrupt-based human review via LangGraph.
This enables human approval, editing, or feedback substitution before a tool is executed.

Typical usage:
    from this_module import add_human_in_the_loop

    @tool
    def search_docs(query: str) -> str:
        return f"Results for: {query}"

    safe_tool = add_human_in_the_loop(search_docs)
    result = safe_tool.invoke({"query": "pydantic base models"})
"""

from collections.abc import Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig
from langgraph.types import interrupt


def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    """Wrap a LangChain tool with human-in-the-loop interrupt logic for approval, editing, or feedback.

    This function wraps an existing LangChain tool (or plain callable) with LangGraph's interrupt system,
    allowing a human to review each call before it is executed.

    Args:
        tool (Callable | BaseTool):
            The LangChain tool (or callable function) to wrap with human-in-the-loop support.
            If a plain callable is passed, it will be converted into a LangChain `BaseTool`.

        interrupt_config (HumanInterruptConfig, optional):
            Configuration dict defining which types of human input are allowed. If not provided,
            the default enables all three options:
            ```
            {
                "allow_accept": True,
                "allow_edit": True,
                "allow_respond": True,
            }
            ```

    Returns:
        BaseTool:
            A LangChain-compatible tool that prompts a human to approve, edit, or respond to each call
            before invoking the original tool logic.

    Raises:
        ValueError: If the human interrupt returns an unsupported response type.

    Example:
        >>> @tool
        ... def get_user_profile(user_id: str) -> str:
        ...     return f"Profile for {user_id}"

        >>> reviewed_tool = add_human_in_the_loop(get_user_profile)
        >>> reviewed_tool.invoke({"user_id": "123"})
        # Will prompt human before invoking
    """
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    @create_tool(tool.name, description=tool.description, args_schema=tool.args_schema)
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request: HumanInterrupt = {
            "action_request": {"action": tool.name, "args": tool_input},
            "config": interrupt_config,
            "description": "Please review the tool call",
        }
        response = interrupt([request])[0]

        if response["type"] == "accept":
            tool_response = tool.invoke(tool_input, config)
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            tool_response = tool.invoke(tool_input, config)
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(
                f"Unsupported interrupt response type: {
                    response['type']}"
            )

        return tool_response

    return call_tool_with_interrupt
