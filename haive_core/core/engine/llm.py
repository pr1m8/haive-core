from src.haive.core.models.llm.base import LLMConfig

class LLMEngine:
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config

    def 