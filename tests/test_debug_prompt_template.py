#!/usr/bin/env python3
"""Debug script to test prompt template variable handling."""

import asyncio

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Create the same prompt template from the test
RAG_QUERY_REFINEMENT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert query optimization specialist."),
        (
            "human",
            """Analyze and refine the following user query.

**Original Query:** {query}
**Context (if provided):** {context}

Provide analysis and suggestions.""",
        ),
    ]
).partial(context="")


class QueryRefinementResponse(BaseModel):
    """Query refinement analysis."""

    original_query: str = Field(description="The original user query")
    refined_query: str = Field(description="The improved query")


def test_prompt_template_directly():
    """Test the prompt template directly to see what it expects."""

    # Check template properties

    # Test with proper variables
    try:
        formatted = RAG_QUERY_REFINEMENT.format(
            query="what is the tallest building in france"
        )
    except Exception as e:
        pass")


def test_augllm_config():
    """Test AugLLMConfig with the prompt template."""
    from haive.core.engine.aug_llm import AugLLMConfig


    config = AugLLMConfig(
        prompt_template=RAG_QUERY_REFINEMENT,
        structured_output_model=QueryRefinementResponse,
        structured_output_version="v2",
    )


    # Test what happens when we provide input data as dict
    input_data = {"query": "what is the tallest building in france"}

    # Check if the engine can handle this


async def test_simple_agent():
    """Test SimpleAgent with debug output."""
    from haive.agents.simple.agent_v2 import SimpleAgentV2
    from haive.core.engine.aug_llm import AugLLMConfig


    try:
        agent = SimpleAgentV2(
            engine=AugLLMConfig(
                prompt_template=RAG_QUERY_REFINEMENT,
                structured_output_model=QueryRefinementResponse,
                structured_output_version="v2",
            ),
            persistence=None,  # Disable persistence to avoid DB issues
        )

        # Check the input schema
        input_schema = agent.input_schema
        if hasattr(input_schema, "model_fields"):
            pass

        # This is where it will likely fail
        await agent.arun(
            {"query": "what is the tallest building in france"}, debug=True
        )

    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_prompt_template_directly()
    test_augllm_config()
    asyncio.run(test_simple_agent())
