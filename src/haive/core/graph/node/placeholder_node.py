from langgraph.types import Command
from pydantic import BaseModel

from haive.core.schema.state_schema import StateSchema


def placeholder_node(state: StateSchema | BaseModel) -> Command[None]:
    """Placeholder Node.

    Args:
        state: [TODO: Add description]

    Returns:
        [TODO: Add return description]
    """
    return Command(update={})
