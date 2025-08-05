#!/usr/bin/env python3
"""Debug with logging to see what's happening."""

import contextlib
import logging

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.schema.composer import SchemaComposer

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")


# Create AugLLMConfig
config = AugLLMConfig(temperature=0.7)


# Use from_components to see the full flow
schema_class = SchemaComposer.from_components([config], name="TestSchema")


# Check if it has the required fields
for field in ["messages", "token_usage", "engine", "engines"]:
    has_field = field in schema_class.model_fields

# Try to create instance
with contextlib.suppress(Exception):
    instance = schema_class()
