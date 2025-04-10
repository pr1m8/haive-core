from typing import Literal
from src.haive.core.engine.agent.persistence.base import CheckpointerConfig
from src.haive.core.engine.agent.persistence.types import CheckpointerType
from langgraph.checkpoint.memory import MemorySaver
class MemoryCheckpointerConfig(CheckpointerConfig):
    type: Literal[CheckpointerType.memory] = CheckpointerType.memory

    def create_checkpointer(self):
        return MemorySaver()
    
    """
    def build(self) -> MemorySaver:
        return MemorySaver()
    """
