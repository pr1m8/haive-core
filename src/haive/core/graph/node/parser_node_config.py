# src/haive/core/graph/node/parser.py

import json
import logging
from typing import Any, Dict, List, Optional, Type, Union

from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from haive.core.graph.node.base_config import NodeConfig
from haive.core.graph.node.types import NodeType

logger = logging.getLogger(__name__)


class ParserNodeConfig(NodeConfig):
    """
    Configuration for a parser node that converts tool outputs to structured Pydantic models.
    """

    # Override node_type for this specific node type
    node_type: NodeType = Field(
        default=NodeType.PARSER, description="Type of node (always PARSER)"
    )

    # Parser configuration
    model_registry_key: str = Field(
        default="output_schemas",
        description="Key in state to find output model registry",
    )

    parsed_models_key: str = Field(
        default="parsed_models", description="Key in state to store parsed models"
    )

    fallback_node: Optional[str] = Field(
        default="agent", description="Node to route to if parsing fails"
    )

    def __call__(
        self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Command:
        """
        Parse input using the appropriate output schema.

        Args:
            state: The current state, either complete state or a tool call
            config: Optional runtime configuration

        Returns:
            Command with parsed model and routing
        """
        # Determine if state is a full state dict or just a tool call
        if isinstance(state, dict) and "id" in state and "name" in state:
            # This is a single tool call from Send
            tool_call = state
            # Create an empty updates dict
            updates = {}

            # Try to parse the tool call result
            parsed_model = self._parse_tool_call(tool_call, state)
            if parsed_model is not None:
                # Get model class name
                model_name = parsed_model.__class__.__name__
                updates[model_name] = parsed_model

                # Return Command with model update
                return Command(update=updates, goto=self.command_goto)
            else:
                # Parsing failed, route to fallback
                return Command(goto=self.fallback_node)
        else:
            # This is a complete state dict
            tool_calls = state.get("completed_tool_calls", [])
            updates = {}

            # Parse each completed tool call
            for tool_call in tool_calls:
                parsed_model = self._parse_tool_call(tool_call, state)
                if parsed_model is not None:
                    # Get model class name
                    model_name = parsed_model.__class__.__name__
                    updates[model_name] = parsed_model

            # Return Command with all parsed models
            return Command(update=updates, goto=self.command_goto)

    def _parse_tool_call(
        self, tool_call: Dict[str, Any], full_state: Dict[str, Any]
    ) -> Optional[BaseModel]:
        """
        Parse a single tool call using the appropriate output schema.

        Args:
            tool_call: The tool call to parse
            full_state: Full state containing schemas

        Returns:
            Parsed Pydantic model or None if parsing failed
        """
        try:
            # Get tool name and result
            tool_name = tool_call.get("name")
            result = tool_call.get("result")
            output_schema_name = tool_call.get("output_schema")

            if not result:
                logger.warning(f"No result found in tool call {tool_call.get('id')}")
                return None

            # Get output schema - try multiple approaches
            output_schema = None

            # 1. Try from output_schema attribute on tool call
            if output_schema_name:
                # Get from state's registry
                schema_registry = {}
                if hasattr(full_state, self.model_registry_key):
                    schema_registry = getattr(full_state, self.model_registry_key)
                elif self.model_registry_key in full_state:
                    schema_registry = full_state[self.model_registry_key]

                # Get schema from registry
                output_schema = schema_registry.get(output_schema_name)

            # 2. Try direct from model type in tool call
            if not output_schema and "model_type" in tool_call:
                model_type = tool_call["model_type"]
                # Try to find in schema registry
                schema_registry = {}
                if hasattr(full_state, self.model_registry_key):
                    schema_registry = getattr(full_state, self.model_registry_key)
                elif self.model_registry_key in full_state:
                    schema_registry = full_state[self.model_registry_key]

                # Look for the model type in registry
                output_schema = schema_registry.get(model_type)

            # 3. Try by tool name
            if not output_schema and tool_name:
                # Try to find in registry by tool name
                schema_registry = {}
                if hasattr(full_state, self.model_registry_key):
                    schema_registry = getattr(full_state, self.model_registry_key)
                elif self.model_registry_key in full_state:
                    schema_registry = full_state[self.model_registry_key]

                # Look through registry for schemas with matching tool_name attribute
                for schema_name, schema in schema_registry.items():
                    if hasattr(schema, "tool_name") and schema.tool_name == tool_name:
                        output_schema = schema
                        break

            # If no schema found, cannot parse
            if not output_schema:
                logger.warning(f"No output schema found for tool {tool_name}")
                return None

            # Parse result with schema using Pydantic
            if isinstance(result, str):
                # Try to parse as JSON first
                try:
                    result_dict = json.loads(result)
                except json.JSONDecodeError:
                    # Not valid JSON, use as raw text
                    result_dict = {"text": result}
            elif isinstance(result, dict):
                result_dict = result
            else:
                # Other types, convert to string and use as text
                result_dict = {"text": str(result)}

            # Use Pydantic parsing (respecting v1 vs v2)
            if hasattr(output_schema, "model_validate"):
                # Pydantic v2
                parsed_model = output_schema.model_validate(result_dict)
            else:
                # Pydantic v1
                parsed_model = output_schema.parse_obj(result_dict)

            return parsed_model

        except ValidationError as e:
            logger.warning(f"Validation error parsing result: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing tool result: {e}")
            return None
