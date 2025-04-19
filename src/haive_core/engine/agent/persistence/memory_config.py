from typing import Literal

from langgraph.checkpoint.memory import MemorySaver

from haive_core.engine.agent.persistence.base import CheckpointerConfig
from haive_core.engine.agent.persistence.types import CheckpointerType


class MemoryCheckpointerConfig(CheckpointerConfig):
    type: Literal[CheckpointerType.memory] = CheckpointerType.memory

    def create_checkpointer(self):
        return MemorySaver()

    """
    def build(self) -> MemorySaver:
        return MemorySaver()
    """
