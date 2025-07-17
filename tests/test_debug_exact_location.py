#!/usr/bin/env python3
"""Debug script to find EXACT location of remaining serialization issues."""

import logging
import os
import sys
import traceback

# Add the packages to Python path
sys.path.insert(
    0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")
sys.path.insert(
    0, "/home/will/Projects/haive/backend/haive/packages/haive-agents/src")

# Enable all debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run_react_agent_with_debug():
    """Run ReactAgent test with full debug tracing."""

    try:
        # Import and run the exact same test
        from haive.agents.react.agent import ReactAgent
        from langchain_core.tools import tool

        from haive.core.engine.aug_llm import AugLLMConfig

        # Create the tool
        @tool
        def calc_add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # Create engine config
        engine_config = AugLLMConfig(tools=[calc_add])

        # Create ReactAgent
        agent = ReactAgent(name="debug_agent", engine=engine_config)

        # Try to run the agent - this is where the error should occur

        # Patch the agent execution to catch exact error location
        original_run = agent.run
        original_arun = agent.arun

        def debug_run(*args, **kwargs):
            try:
                return original_run(*args, **kwargs)
            except Exception as e:
                traceback.print_exc()
                raise

        def debug_arun(*args, **kwargs):
            try:
                return original_arun(*args, **kwargs)
            except Exception as e:
                traceback.print_exc()
                raise

        agent.run = debug_run
        agent.arun = debug_arun

        # Now try to run
        result = agent.run("What is 2 + 2?")

    except Exception as e:
        pass

        # Get the full traceback
        tb = traceback.format_exc()

        # Try to pinpoint the exact line

        exc_type, exc_value, exc_traceback = sys.exc_info()

        for frame_info in traceback.extract_tb(exc_traceback):


if __name__ == "__main__":
    run_react_agent_with_debug()
