# src/haive/core/engine.py

from typing import Any, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.haive.core.models.llm.base import LLMConfig, AzureLLMConfig
from src.haive.core.engine.aug_llm import AugLLMConfig, compose_runnable


class Engine(ABC):
    """
    Base class for all engines.
    
    An engine is responsible for processing inputs and producing outputs.
    Different engine types (LLM, Runnable, Agent) have different implementations.
    """
    
    @abstractmethod
    def invoke(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """
        Invoke the engine on the given inputs.
        
        Args:
            inputs: The inputs to the engine
            **kwargs: Additional parameters for the invocation
            
        Returns:
            The result of invoking the engine
        """
        pass
    
    @abstractmethod
    async def ainvoke(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """
        Asynchronously invoke the engine on the given inputs.
        
        Args:
            inputs: The inputs to the engine
            **kwargs: Additional parameters for the invocation
            
        Returns:
            The result of invoking the engine
        """
        pass

        
class LLMEngine(Engine):
    """
    Engine that uses a language model directly.
    """
    
    def __init__(self, llm: Any):
        """
        Initialize with a language model.
        
        Args:
            llm: The language model to use
        """
        self.llm = llm
    
    def invoke(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Invoke the language model."""
        return self.llm.invoke(inputs, **kwargs)
    
    async def ainvoke(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Asynchronously invoke the language model."""
        return await self.llm.ainvoke(inputs, **kwargs)
    
    @classmethod
    def from_config(cls, config: LLMConfig) -> 'LLMEngine':
        """Create from an LLMConfig."""
        llm = config.instantiate_llm()
        return cls(llm)


class RunnableEngine(Engine):
    """
    Engine that uses a Runnable pipeline.
    """
    
    def __init__(self, runnable: Runnable):
        """
        Initialize with a Runnable.
        
        Args:
            runnable: The Runnable to use
        """
        self.runnable = runnable
    
    def invoke(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Invoke the Runnable."""
        print(f"inputs: {inputs}")
        #print(f"kwargs: {kwargs}")
        #print(type(inputs))
        #print(type(kwargs))
        #print(self.runnable)
        return self.runnable.invoke(inputs, **kwargs)
    
    async def ainvoke(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Asynchronously invoke the Runnable."""
        return await self.runnable.ainvoke(inputs, **kwargs)
    
    @classmethod
    def from_config(cls, config: AugLLMConfig) -> 'RunnableEngine':
        """Create from an AugLLMConfig."""
        #print(f"config: {config}")
        runnable = compose_runnable(config)
        #print(f"runnable: {runnable}")
        return cls(runnable).runnable


class AgentEngine(Engine):
    """
    Engine that uses another agent.
    """
    
    def __init__(self, agent: Any):
        """
        Initialize with an agent.
        
        Args:
            agent: The agent to use
        """
        self.agent = agent
    
    def invoke(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Invoke the agent."""
        return self.agent.app.invoke(inputs, **kwargs)
    
    async def ainvoke(self, inputs: Dict[str, Any], **kwargs) -> Any:
        """Asynchronously invoke the agent."""
        return await self.agent.app.ainvoke(inputs, **kwargs)
    
    #@classmethod
    #def from_config(cls, config: AgentConfig) -> 'AgentEngine':
        #"""Create from an AgentConfig."""
        #return cls(config.build_agent())

# To avoid circular imports, create_engine function should be in a different module
# We'll define it in engine_factory.py