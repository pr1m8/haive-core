"""JSON Schema utilities for handling Pydantic models with callable fields.

This module provides utilities to create JSON schemas that are compatible with
external systems like CopilotKit while preserving callable fields for runtime use.
"""

import logging
from typing import Any, Dict, Type, get_type_hints
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic.json_schema import JsonSchemaMode
from pydantic._internal._typing_extra import eval_type_lenient

logger = logging.getLogger(__name__)


class CallableAwareJsonSchemaGenerator(GenerateJsonSchema):
    """Custom JSON schema generator that handles callable fields gracefully.
    
    This generator extends Pydantic's default JSON schema generation to:
    1. Skip callable fields that can't be represented in JSON schema
    2. Log skipped fields for debugging
    3. Preserve all other fields including complex types that JsonPlusSerializer can handle
    """

    def handle_invalid_for_json_schema(self, schema: Any, error_message: str) -> JsonSchemaValue:
        """Handle fields that are invalid for JSON schema generation.
        
        This method is called when Pydantic encounters a field that cannot be
        represented in JSON schema (like Callable types).
        """
        # Check if this is a callable schema error
        if "CallableSchema" in error_message:
            logger.debug(f"Skipping callable field from JSON schema: {error_message}")
            # Return None to exclude this field from the schema
            return None
        
        # For other errors, fall back to default behavior
        return super().handle_invalid_for_json_schema(schema, error_message)


def create_copilotkit_compatible_schema(model: Type[BaseModel], **kwargs) -> Dict[str, Any]:
    """Create a JSON schema that's compatible with CopilotKit and other external systems.
    
    Args:
        model: Pydantic model class to generate schema for
        **kwargs: Additional arguments passed to model_json_schema()
        
    Returns:
        JSON schema dict with callable fields excluded but all other fields preserved
    """
    try:
        # First try with custom generator
        schema = model.model_json_schema(
            schema_generator=CallableAwareJsonSchemaGenerator,
            **kwargs
        )
        logger.debug(f"Generated JSON schema for {model.__name__} with {len(schema.get('properties', {}))} properties")
        return schema
    except Exception as e:
        logger.warning(f"Failed to generate schema with custom generator for {model.__name__}: {e}")
        
        # Fallback: try with default generator
        try:
            return model.model_json_schema(**kwargs)
        except Exception as fallback_error:
            logger.error(f"Complete failure generating schema for {model.__name__}: {fallback_error}")
            
            # Ultimate fallback: minimal schema
            return {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Chat message history"
                    }
                },
                "title": model.__name__
            }


def patch_model_for_json_schema_compatibility(model: Type[BaseModel]) -> Type[BaseModel]:
    """Patch a Pydantic model to use CallableAwareJsonSchemaGenerator by default.
    
    This modifies the model's model_json_schema method to automatically use
    the callable-aware generator.
    
    Args:
        model: Pydantic model class to patch
        
    Returns:
        The same model class with patched JSON schema generation
    """
    original_model_json_schema = model.model_json_schema
    
    @classmethod
    def patched_model_json_schema(cls, **kwargs):
        """Patched version that uses CallableAwareJsonSchemaGenerator by default."""
        if 'schema_generator' not in kwargs:
            kwargs['schema_generator'] = CallableAwareJsonSchemaGenerator
        return original_model_json_schema(**kwargs)
    
    model.model_json_schema = patched_model_json_schema
    logger.debug(f"Patched {model.__name__} for JSON schema compatibility")
    return model


class JsonSchemaCompatibilityMixin:
    """Mixin that can be added to Pydantic models to automatically handle callable fields.
    
    Usage:
        class MyState(JsonSchemaCompatibilityMixin, BaseModel):
            messages: List[BaseMessage] = Field(default_factory=list)
            my_callable: Callable = Field(default=None)
    
    The mixin will automatically exclude callable fields from JSON schema generation
    while preserving them for serialization and runtime use.
    """
    
    @classmethod
    def model_json_schema(cls, **kwargs) -> Dict[str, Any]:
        """Override to use CallableAwareJsonSchemaGenerator by default."""
        if 'schema_generator' not in kwargs:
            kwargs['schema_generator'] = CallableAwareJsonSchemaGenerator
        return super().model_json_schema(**kwargs)


# Convenience function for common use cases
def make_copilotkit_compatible(model: Type[BaseModel]) -> Type[BaseModel]:
    """Make a Pydantic model compatible with CopilotKit JSON schema requirements.
    
    This is a convenience function that applies the necessary patches to handle
    callable fields gracefully.
    
    Args:
        model: Pydantic model class
        
    Returns:
        The patched model class
    """
    return patch_model_for_json_schema_compatibility(model)