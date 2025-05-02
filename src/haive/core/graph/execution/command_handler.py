# execution/command_handler.py
from typing import Any, Optional

from langgraph.types import Command, Send


def process_engine_result(result: Any, command_goto: Optional[Any] = None) -> Any:
    """Process engine result according to Command pattern."""
    # If result is already a Command or Send, pass it through
    if isinstance(result, Command):
        return result

    if isinstance(result, Send) or (
        isinstance(result, list) and all(isinstance(x, Send) for x in result)
    ):
        return result

    # Otherwise, wrap in Command if goto specified
    if command_goto is not None:
        return Command(update=result, goto=command_goto)

    # Just return result directly
    return result
