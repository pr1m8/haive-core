#!/usr/bin/env python3
"""Debug script to trace BasePromptTemplate serialization issue step by step."""

import os
import sys
import traceback

from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.persistence.serializers import SecureSecretStrSerializer

# Add the packages to Python path
sys.path.insert(
    0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")
sys.path.insert(
    0, "/home/will/Projects/haive/backend/haive/packages/haive-agents/src")


def step_1_create_config():
    """Step 1: Create AugLLMConfig with prompt template."""

    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("human", "{input}")]
    )

    config = AugLLMConfig(name="test_config", prompt_template=chat_prompt)

    return config


def step_2_test_model_dump(config):
    """Step 2: Test model_dump serialization."""

    try:
        dumped = config.model_dump()
        prompt_template_data = dumped.get("prompt_template")
        if isinstance(prompt_template_data, dict):
            pass
        else:
            pass
        return dumped
    except Exception as e:
        traceback.print_exc()
        return None


def step_3_test_serialization(config):
    """Step 3: Test SecureSecretStrSerializer."""

    serializer = SecureSecretStrSerializer()

    try:
        # Serialize
        serialized = serializer.dumps(config)

        # Deserialize
        deserialized = serializer.loads(serialized)

        if hasattr(deserialized, "prompt_template") and deserialized.prompt_template:
            pass

        return deserialized

    except Exception as e:
        traceback.print_exc()
        return None


def step_4_test_recreate_from_dict(dumped_data):
    """Step 4: Test recreating AugLLMConfig from dumped dict."""

    if dumped_data is None:
        return None

    try:
        # Try to create AugLLMConfig from the dumped dict
        recreated = AugLLMConfig(**dumped_data)

        if hasattr(recreated, "prompt_template") and recreated.prompt_template:
            pass

        return recreated

    except Exception as e:
        traceback.print_exc()
        return None


def step_5_test_isinstance_checks(config):
    """Step 5: Test isinstance checks that might be failing."""

    if config is None or not hasattr(config, "prompt_template"):
        return

    prompt_template = config.prompt_template

    # Test some of the actual isinstance checks from the code
    if isinstance(prompt_template, ChatPromptTemplate):
        pass")
    else:
        pass")

    if hasattr(prompt_template, "__dict__"):
        pass


def step_6_test_model_validator(config):
    """Step 6: Test if model validators are triggered."""

    if config is None:
        return

    try:
        # Try to trigger model validation by accessing properties

        # Try to trigger any lazy initialization
        if hasattr(config, "_validate_and_setup"):
            config._validate_and_setup()


    except Exception as e:
        traceback.print_exc()


def main():
    """Main debug workflow."""

    # Step 1: Create config
    original_config = step_1_create_config()

    # Step 2: Test model_dump
    dumped_data = step_2_test_model_dump(original_config)

    # Step 3: Test serialization/deserialization
    deserialized_config = step_3_test_serialization(original_config)

    # Step 4: Test recreation from dict
    recreated_config = step_4_test_recreate_from_dict(dumped_data)

    # Step 5: Test isinstance checks on different configs
    step_5_test_isinstance_checks(original_config)

    step_5_test_isinstance_checks(deserialized_config)

    step_5_test_isinstance_checks(recreated_config)

    # Step 6: Test model validators
    step_6_test_model_validator(deserialized_config)



if __name__ == "__main__":
    main()
