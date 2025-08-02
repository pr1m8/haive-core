import logging
import uuid
from collections.abc import Callable, Sequence
from typing import Any, Self

from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field, model_validator

from haive.core.engine.base import Engine
from haive.core.graph.node.types import CommandGoto, NodeType

logger = logging.getLogger(__name__)


class NodeConfig(BaseModel):
    """Configuration for a node in a graph.

    A NodeConfig defines all aspects of a node's behavior, including:
    - Core identification (id, name)
    - Engine/callable to execute
    - State schema integration
    - Input/output field mappings
    - Control flow behavior
    - Node type-specific options
    """

    id: str = Field(
        default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this node",
    )
    name: str = Field(description="Name of the node in the graph")
    node_type: NodeType | None = Field(
        default=None,
        description="Type of node (determined automatically if not specified)",
    )
    schemas: Sequence[BaseTool | type[BaseModel] | Callable] = Field(
        default_factory=list, description="The schemas to use for the node"
    )
    engine: Engine | None = Field(
        default=None, description="Engine instance to use for this node"
    )
    engine_name: str | None = Field(
        default=None, description="Name of engine to look up in registry"
    )
    callable_func: Callable | None = Field(
        default=None, description="Callable function to use for this node", exclude=True
    )
    callable_ref: str | None = Field(
        default=None, description="Reference to callable function (module.function)"
    )
    state_schema: type[BaseModel] | None = Field(
        default=None, description="State schema class for this node"
    )
    input_schema: type[BaseModel] | None = Field(
        default=None, description="Input schema for this node"
    )
    output_schema: type[BaseModel] | None = Field(
        default=None, description="Output schema for this node"
    )
    input_fields: list[str] | dict[str, str] | None = Field(
        default=None,
        description="List of input fields or mapping from state keys to node input keys",
    )
    output_fields: list[str] | dict[str, str] | None = Field(
        default=None,
        description="List of output fields or mapping from node output keys to state keys",
    )
    command_goto: CommandGoto | None = Field(
        default=None, description="Next node to go to after this node (or END)"
    )
    retry_policy: RetryPolicy | None = Field(
        default=None, description="Retry policy for node execution"
    )
    tools: list[Any] | None = Field(default=None, description="Tools for tool nodes")
    messages_field: str | None = Field(
        default="messages",
        description="Field containing messages for tool/validation nodes",
    )
    handle_tool_errors: bool | str | Callable = Field(
        default=True, description="How to handle tool errors"
    )
    validation_schemas: list[type[BaseModel] | Callable] | None = Field(
        default=None, description="Validation schemas for validation nodes"
    )
    condition: Callable | None = Field(
        default=None, description="Condition function for branch nodes", exclude=True
    )
    condition_ref: str | None = Field(
        default=None, description="Reference to condition function (module.function)"
    )
    routes: dict[Any, str] | None = Field(
        default=None, description="Routes mapping condition results to node names"
    )
    send_targets: list[str] | None = Field(
        default=None, description="Target nodes for send operations"
    )
    send_field: str | None = Field(
        default=None, description="Field containing items to distribute to targets"
    )
    config_overrides: dict[str, Any] = Field(
        default_factory=dict, description="Engine configuration overrides for this node"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for this node"
    )
    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    def to_dict(self) -> dict[str, Any]:
        """Convert this node config to a dictionary representation.

        Returns:
            Dictionary representation of this node config
        """
        data = self.model_dump(exclude={"engine", "callable_func", "condition"})
        if self.engine:
            if hasattr(self.engine, "to_dict"):
                data["engine"] = self.engine.to_dict()
            elif hasattr(self.engine, "model_dump"):
                data["engine"] = self.engine.model_dump()
            else:
                data["engine"] = {
                    "name": getattr(self.engine, "name", "unknown"),
                    "type": getattr(self.engine, "engine_type", "unknown"),
                }
        if self.state_schema:
            data["state_schema"] = (
                f"{self.state_schema.__module__}.{self.state_schema.__name__}"
            )
        if self.input_schema:
            data["input_schema"] = (
                f"{self.input_schema.__module__}.{self.input_schema.__name__}"
            )
        if self.output_schema:
            data["output_schema"] = (
                f"{self.output_schema.__module__}.{self.output_schema.__name__}"
            )
        if self.command_goto == END:
            data["command_goto"] = "END"
        return data

    @model_validator(mode="after")
    def validate_and_determine_node_type(self) -> Self:
        """Validate the configuration and determine the node type automatically if not specified."""
        if self.command_goto == "END":
            self.command_goto = END
        if (
            self.engine is None
            and self.engine_name is None
            and (self.callable_func is None)
            and (self.tools is None)
            and (self.schemas is None)
        ):
            raise ValueError(
                "At least one of engine, engine_name, tools,schemas or callable_func must be set"
            )
        if isinstance(self.input_fields, list):
            self.input_fields = {field: field for field in self.input_fields}
        if isinstance(self.output_fields, list):
            self.output_fields = {field: field for field in self.output_fields}
        if self.node_type is None:
            if self.tools is not None:
                self.node_type = NodeType.TOOL
            elif self.validation_schemas is not None:
                self.node_type = NodeType.VALIDATION
            elif self.condition is not None and self.routes is not None:
                self.node_type = NodeType.BRANCH
            elif self.send_targets is not None:
                self.node_type = NodeType.SEND
            elif self.engine is not None or self.engine_name is not None:
                self.node_type = NodeType.ENGINE
            elif self.callable_func is not None:
                self.node_type = NodeType.CALLABLE
            else:
                self.node_type = NodeType.CUSTOM
        if self.engine and (not self.state_schema):
            try:
                from haive.core.schema.schema_composer import SchemaComposer

                schema = SchemaComposer.from_components([self.engine])
                self.state_schema = schema.build()
                if not self.input_schema:
                    self.input_schema = schema.create_input_schema()
                if not self.output_schema:
                    self.output_schema = schema.create_output_schema()
            except Exception as e:
                logger.warning(f"Could not auto-generate schema from engine: {e}")
        if self.engine and (not self.input_fields) and (not self.output_fields):
            try:
                engine_name = getattr(self.engine, "name", "default")
                if hasattr(self.state_schema, "__engine_io_mappings__"):
                    io_mappings = getattr(
                        self.state_schema, "__engine_io_mappings__", {}
                    )
                    if engine_name in io_mappings:
                        mapping = io_mappings[engine_name]
                        if "inputs" in mapping and (not self.input_fields):
                            input_fields = mapping["inputs"]
                            self.input_fields = {field: field for field in input_fields}
                        if "outputs" in mapping and (not self.output_fields):
                            output_fields = mapping["outputs"]
                            self.output_fields = {
                                field: field for field in output_fields
                            }
            except Exception as e:
                logger.warning(f"Could not extract I/O mappings from schema: {e}")
        return self

    def get_engine(self) -> tuple[Engine | None, str | None]:
        """Get the engine for this node, resolving from registry if needed.

        Returns:
            Tuple of (engine_instance, engine_id)
        """
        if self.engine is None and self.engine_name is None:
            return (self.callable_func, None)
        if self.engine is not None:
            engine_id = getattr(self.engine, "id", None)
            return (self.engine, engine_id)
        try:
            from haive.core.engine.base.registry import EngineRegistry

            registry = EngineRegistry.get_instance()
            engine = registry.find(self.engine_name)
            if engine:
                engine_id = getattr(engine, "id", None)
                return (engine, engine_id)
        except ImportError:
            logger.warning(
                f"Could not import EngineRegistry to resolve engine: {self.engine_name}"
            )
        return (None, None)

    def get_input_mapping(self) -> dict[str, str]:
        """Get the input mapping for this node.

        Returns:
            Dictionary mapping state keys to node input keys
        """
        if self.input_fields:
            return dict(self.input_fields)
        return {}

    def get_output_mapping(self) -> dict[str, str]:
        """Get the output mapping for this node.

        Returns:
            Dictionary mapping node output keys to state keys
        """
        if self.output_fields:
            return dict(self.output_fields)
        return {}
