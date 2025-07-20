"""Simple debug test for self-discover agent."""

import asyncio
import logging
import sys

from langchain_core.messages import HumanMessage

from haive.agents.reasoning_and_critique.self_discover.v2.agent import (
    DEFAULT_REASONING_MODULES,
    self_discovery,
)

# Suppress all logging except errors
logging.getLogger().setLevel(logging.ERROR)

# Add direct paths to avoid import issues
sys.path.insert(
    0, "/home/will/Projects/haive/backend/haive/packages/haive-agents/src")
sys.path.insert(
    0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")


async def simple_test():
    """Simple test with minimal output."""
    test_input = {
        "messages": [HumanMessage(content="What is 2+2?")],
        "reasoning_modules": DEFAULT_REASONING_MODULES[:3],
        "task_description": "What is 2+2?",
    }

    try:
        result = await self_discovery.ainvoke(test_input)
        return result
    except Exception:
        return None


if __name__ == "__main__":
    result = asyncio.run(simple_test())
