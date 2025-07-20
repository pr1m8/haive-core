#!/usr/bin/env python3
"""Debug what happens during build method."""

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.schema.composer import SchemaComposer

# Create AugLLMConfig
config = AugLLMConfig(temperature=0.7)


# Create composer
composer = SchemaComposer(name="TestSchema")

# Add fields from components
composer.add_fields_from_components([config])


# Now build
schema = composer.build()


# Check if the detected_base_class was None before build
composer2 = SchemaComposer(name="TestSchema2")

# Add fields but don't detect base class
composer2.add_fields_from_components([config])
# Manually set it to None to simulate the issue
composer2.detected_base_class = None

# Now build and see what happens
schema2 = composer2.build()
