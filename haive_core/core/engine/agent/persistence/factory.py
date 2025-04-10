from typing import Dict, Type
from src.haive.core.engine.agent.persistence.types import CheckpointerType
from src.haive.core.engine.agent.persistence.base import CheckpointerConfig
from src.haive.core.engine.agent.persistence.postgres_config import PostgresCheckpointerConfig
from src.haive.core.engine.agent.persistence.mongodb_config import MongoDBCheckpointerConfig
from src.haive.core.engine.agent.persistence.memory_config import MemoryCheckpointerConfig

CHECKPOINTER_CONFIG_MAP: Dict[str, Type[CheckpointerConfig]] = {
    CheckpointerType.memory: MemoryCheckpointerConfig,
    CheckpointerType.postgres: PostgresCheckpointerConfig,
    CheckpointerType.mongodb: MongoDBCheckpointerConfig,
}



def load_checkpointer_config(data: dict) -> CheckpointerConfig:
    """Instantiate the correct CheckpointerConfig subclass from src.config dict."""
    type_str = data.get("type", CheckpointerType.memory)
    config_cls = CHECKPOINTER_CONFIG_MAP.get(type_str)

    if not config_cls:
        raise ValueError(f"Unsupported checkpointer type: {type_str}")

    return config_cls(**data)