from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel
from src.haive.core.engine.agent.persistence.types import CheckpointerType

class CheckpointerConfig(BaseModel, ABC):
    type: CheckpointerType

    @abstractmethod
    def create_checkpointer(self) -> Any:
        ...

    async def create_checkpointer(self) -> Any:
        raise NotImplementedError("Async not supported for this checkpointer.")
