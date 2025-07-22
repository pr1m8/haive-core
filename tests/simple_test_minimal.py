"""Minimal test to reproduce the engine validation error."""

import asyncio
import contextlib

from haive.agents.simple.agent_v2 import SimpleAgentV2
from langchain_core.prompts import ChatPromptTemplate

from haive.core.engine.aug_llm import AugLLMConfig


async def test_simple_agent_v2():
    """Test SimpleAgent v2 with minimal setup."""
    # Create engine with prompt template
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("human", "{query}")]
    )

    engine = AugLLMConfig(
        name="test_engine", prompt_template=prompt, model="gpt-4o-mini"
    )

    # Create agent
    agent = SimpleAgentV2(name="test_agent", engine=engine)

    # This should work
    agent.input_schema(query="hello")

    # This is where the error occurs
    with contextlib.suppress(Exception):
        await agent.arun("hello")


if __name__ == "__main__":
    asyncio.run(test_simple_agent_v2())
