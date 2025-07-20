#!/usr/bin/env python3
"""Debug engine tracking in SchemaComposer."""

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.schema.composer import SchemaComposer

# Create AugLLMConfig
config = AugLLMConfig(temperature=0.7, name="test_llm")


# Create composer
composer = SchemaComposer(name="TestSchema")


# Add fields from the engine
composer.add_fields_from_components([config])


# Check if LLM engines are detected in the second call

# Test the detection logic directly
has_llm_engine = False
for _engine_name in composer.engines_by_type.get("llm", []):
    has_llm_engine = True
    break

# Now call build
schema = composer.build()
