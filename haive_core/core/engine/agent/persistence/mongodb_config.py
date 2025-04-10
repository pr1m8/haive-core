from typing import Literal
from pydantic import Field
from src.haive.core.engine.agent.persistence.base import CheckpointerConfig
from src.haive.core.engine.agent.persistence.types import CheckpointerType

try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


class MongoDBCheckpointerConfig(CheckpointerConfig):
    """MongoDB checkpointer configuration."""
    type: Literal[CheckpointerType.mongodb] = CheckpointerType.mongodb
    uri: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URI")
    use_async: bool = Field(default=False, description="Use async MongoDB client")

    def _validate_installed(self) -> None:
        if not MONGODB_AVAILABLE:
            raise ImportError("langgraph-checkpoint-mongodb is not installed.")
        

    def create_checkpointer(self):
        self._validate_installed()
        return MongoDBSaver.from_conn_string(self.uri)
    
    """
    def build(self):
        self._validate_installed()
        return MongoDBSaver.from_conn_string(self.uri)
    """


    """
    async def abuild(self):
        self._validate_installed()
        return await AsyncMongoDBSaver.from_conn_string(self.uri)
    """