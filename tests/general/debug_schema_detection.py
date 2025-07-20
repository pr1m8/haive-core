#!/usr/bin/env python3
"""Debug the base class detection logic in detail."""

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.schema.composer import SchemaComposer

# Create AugLLMConfig
config = AugLLMConfig(temperature=0.7)


# Create composer
composer = SchemaComposer(name="TestSchema")

# Check initial state

# Manually trace through the detection logic
if hasattr(config, "engine_type"):
    engine_type_value = getattr(
        config.engine_type,
        "value",
        config.engine_type)
    engine_type_str = str(engine_type_value).lower()

# Call the detection method
composer._detect_base_class_requirements([config])


# Check if it's detecting has_llm_engine correctly
has_llm_engine = False
if hasattr(config, "engine_type"):
    engine_type_value = getattr(
        config.engine_type,
        "value",
        config.engine_type)
    engine_type_str = str(engine_type_value).lower()
    if engine_type_str == "llm":
        has_llm_engine = True

# Import the base classes to check what should be used


# Check the MRO of each base class

# Build the schema to see the full result
schema = composer.build()
