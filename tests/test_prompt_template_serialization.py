#!/usr/bin/env python3
"""Test script to reproduce BasePromptTemplate serialization issue."""

import contextlib
import sys

from langchain_core.prompts import ChatPromptTemplate

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.persistence.serializers import SecureSecretStrSerializer

# Add the packages to Python path
sys.path.insert(
    0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")
sys.path.insert(
    0, "/home/will/Projects/haive/backend/haive/packages/haive-agents/src")


def test_prompt_template_serialization():
    """Test serializing AugLLMConfig with prompt templates."""
    # Create a simple chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("human", "{input}")]
    )

    # Create AugLLMConfig with prompt template
    config = AugLLMConfig(name="test_config", prompt_template=chat_prompt)

    # Test model_dump
    with contextlib.suppress(Exception):
        config.model_dump()

    # Test serializer
    serializer = SecureSecretStrSerializer()

    try:
        # Test serializing the config object
        serialized = serializer.dumps(config)

        # Test deserializing
        serializer.loads(serialized)

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_prompt_template_serialization()
