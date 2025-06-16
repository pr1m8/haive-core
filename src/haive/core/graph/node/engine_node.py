# src/haive/core/graph/node/engine_node.py

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langgraph.types import Command, RetryPolicy, Send
from pydantic import BaseModel, Field

from haive.core.engine.base import Engine, EngineType
from haive.core.graph.common.types import ConfigLike, StateLike
from haive.core.graph.node.base_config import NodeConfig
from haive.core.graph.node.types import NodeType
from haive.core.logging.rich_logger import LogLevel, get_logger

# Get module logger
logger = get_logger(__name__)
logger.set_level(LogLevel.WARNING)


class EngineNodeConfig(NodeConfig):
    """Elegant engine-based node with intelligent I/O handling."""

    # Core identity
    node_type: NodeType = Field(default=NodeType.ENGINE)
    engine: Optional[Engine] = Field(default=None)
    engine_name: Optional[str] = Field(default=None)

    # Field mappings (auto-normalized)
    input_fields: Optional[Union[List[str], Dict[str, str]]] = Field(default=None)
    output_fields: Optional[Union[List[str], Dict[str, str]]] = Field(default=None)

    # Options
    retry_policy: Optional[RetryPolicy] = Field(default=None)
    use_send: bool = Field(default=False)
    debug: bool = Field(default=False)

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Union[Command, Send]:
        """Execute engine node with intelligent I/O handling."""
        with logger.track_time(f"Executing node {self.name}"):
            try:
                # Get engine and validate
                engine = self._get_engine()
                if not engine:
                    raise ValueError(f"No engine available for node '{self.name}'")

                # Extract input intelligently
                input_data = self._extract_smart_input(state, engine)

                # Execute with merged config
                result = self._execute_with_config(engine, input_data, config)

                # Wrap result intelligently
                return self._wrap_smart_result(result, state, engine)

            except Exception as e:
                self._log_error(e)
                raise

    def _get_engine(self) -> Optional[Engine]:
        """Get engine from direct reference or registry."""
        if self.engine:
            return self.engine

        if self.engine_name:
            engine = self._lookup_engine(self.engine_name)
            if engine:
                self.engine = engine  # Cache it
                return engine

        return None

    def _extract_smart_input(self, state: StateLike, engine: Engine) -> Any:
        """Extract input using the most appropriate strategy."""
        # Strategy 1: Explicit mapping
        if self.input_fields:
            return self._extract_mapped_input(
                state, self._normalize_mapping(self.input_fields)
            )

        # Strategy 2: Schema-defined inputs
        schema_inputs = self._get_schema_inputs(state, engine.name)
        if schema_inputs:
            return self._extract_typed_input(state, schema_inputs, engine.engine_type)

        # Strategy 3: Engine-defined inputs
        engine_inputs = self._get_engine_inputs(engine)
        if engine_inputs:
            return self._extract_typed_input(state, engine_inputs, engine.engine_type)

        # Strategy 4: Type-based defaults
        return self._extract_default_input(state, engine.engine_type)

    def _extract_typed_input(
        self, state: StateLike, fields: List[str], engine_type: EngineType
    ) -> Dict[str, Any]:
        """Extract fields with type-specific intelligence."""
        extractors = {
            EngineType.RETRIEVER: self._extract_retriever_fields,
            EngineType.LLM: self._extract_llm_fields,
            EngineType.VECTOR_STORE: self._extract_vectorstore_fields,
            EngineType.EMBEDDINGS: self._extract_embeddings_fields,
        }

        extractor = extractors.get(engine_type, self._extract_generic_fields)
        return extractor(state, fields)

    def _extract_retriever_fields(
        self, state: StateLike, fields: List[str]
    ) -> Dict[str, Any]:
        """Retriever-specific extraction: always include query, filter None values."""
        input_data = {}

        for field in fields:
            value = self._get_state_value(state, field)

            if field == "query":
                # Always include query, even if empty
                input_data[field] = value or ""
                logger.debug(f"Retriever query: '{value or ''}'")
            elif value is not None:
                # Only include other fields if they have values
                input_data[field] = value
                logger.debug(f"Retriever {field}: {value}")
            else:
                logger.debug(f"Skipping None value for {field}")

        return input_data

    def _extract_llm_fields(
        self, state: StateLike, fields: List[str]
    ) -> Dict[str, Any]:
        """LLM-specific extraction: include all fields."""
        return {field: self._get_state_value(state, field) for field in fields}

    def _extract_vectorstore_fields(
        self, state: StateLike, fields: List[str]
    ) -> Dict[str, Any]:
        """Vector store extraction: filter None values except query."""
        input_data = {}
        for field in fields:
            value = self._get_state_value(state, field)
            if field == "query" or value is not None:
                input_data[field] = value
        return input_data

    def _extract_embeddings_fields(self, state: StateLike, fields: List[str]) -> Any:
        """Embeddings extraction: often just needs text."""
        # Try to get text/query field first
        for field in ["query", "text", "content"]:
            if field in fields:
                value = self._get_state_value(state, field)
                if value:
                    return value

        # Fall back to all fields as dict
        return {field: self._get_state_value(state, field) for field in fields}

    def _extract_generic_fields(
        self, state: StateLike, fields: List[str]
    ) -> Dict[str, Any]:
        """Generic extraction: include non-None values."""
        return {
            field: value
            for field in fields
            if (value := self._get_state_value(state, field)) is not None
        }

    def _extract_default_input(self, state: StateLike, engine_type: EngineType) -> Any:
        """Default extraction when no fields are specified."""
        defaults = {
            EngineType.RETRIEVER: lambda: self._extract_retriever_fields(
                state, ["query", "k", "filter", "search_type", "score_threshold"]
            ),
            EngineType.LLM: lambda: {
                "messages": self._get_state_value(state, "messages", [])
            },
            EngineType.VECTOR_STORE: lambda: self._extract_vectorstore_fields(
                state, ["query", "k", "filter"]
            ),
            EngineType.EMBEDDINGS: lambda: self._get_state_value(state, "query", ""),
        }

        extractor = defaults.get(engine_type, lambda: self._state_as_dict(state))
        return extractor()

    def _wrap_smart_result(
        self, result: Any, state: StateLike, engine: Engine
    ) -> Union[Command, Send]:
        """Intelligently wrap result based on type and configuration."""
        # Already wrapped? Return as-is
        if isinstance(result, (Command, Send)):
            return result

        # Generate update dictionary
        update = self._create_update_dict(result, state, engine)

        # Return appropriate wrapper
        if self.use_send and self.command_goto:
            return Send(node=self.command_goto, arg=update)
        else:
            return Command(update=update, goto=self.command_goto)

    def _create_update_dict(
        self, result: Any, state: StateLike, engine: Engine
    ) -> Dict[str, Any]:
        """Create state update dictionary from result."""
        # Strategy 1: Explicit output mapping
        if self.output_fields:
            return self._apply_output_mapping(result)

        # Strategy 2: Schema-defined outputs
        schema_outputs = self._get_schema_outputs(state, engine.name)
        if schema_outputs:
            return self._map_to_outputs(result, schema_outputs)

        # Strategy 3: Smart type-based mapping
        return self._smart_result_mapping(result, state, engine.engine_type)

    def _smart_result_mapping(
        self, result: Any, state: StateLike, engine_type: EngineType
    ) -> Dict[str, Any]:
        """Smart result mapping based on result type and engine type."""
        # Message results go to messages
        if self._is_message_like(result):
            return self._update_messages(result, state)

        # Dictionary results
        if isinstance(result, dict):
            return result

        # Engine-specific single value mapping
        field_map = {
            EngineType.RETRIEVER: "documents",
            EngineType.LLM: "response",
            EngineType.EMBEDDINGS: "embeddings",
            EngineType.VECTOR_STORE: "documents",
        }

        field = field_map.get(engine_type, "result")
        return {field: result}

    def _update_messages(self, result: Any, state: StateLike) -> Dict[str, Any]:
        """Update messages list with new message(s)."""
        existing = self._get_state_value(state, "messages", [])
        messages = list(existing) if existing else []

        if isinstance(result, list):
            messages.extend(result)
        else:
            messages.append(result)

        return {"messages": messages}

    # Utility methods
    def _get_state_value(self, state: StateLike, key: str, default: Any = None) -> Any:
        """Get value from state with fallback."""
        if hasattr(state, key):
            return getattr(state, key)
        elif isinstance(state, dict):
            return state.get(key, default)
        return default

    def _get_schema_inputs(
        self, state: StateLike, engine_name: str
    ) -> Optional[List[str]]:
        """Get engine inputs from schema."""
        if not hasattr(state, "__engine_io_mappings__") or not engine_name:
            return None
        return (
            getattr(state, "__engine_io_mappings__", {})
            .get(engine_name, {})
            .get("inputs")
        )

    def _get_schema_outputs(
        self, state: StateLike, engine_name: str
    ) -> Optional[List[str]]:
        """Get engine outputs from schema."""
        if not hasattr(state, "__engine_io_mappings__") or not engine_name:
            return None
        return (
            getattr(state, "__engine_io_mappings__", {})
            .get(engine_name, {})
            .get("outputs")
        )

    def _get_engine_inputs(self, engine: Engine) -> Optional[List[str]]:
        """Get input fields from engine definition."""
        if hasattr(engine, "get_input_fields"):
            return list(engine.get_input_fields().keys())
        return None

    def _extract_mapped_input(
        self, state: StateLike, mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract using explicit state->input mapping."""
        return {
            input_key: self._get_state_value(state, state_key)
            for state_key, input_key in mapping.items()
            if self._get_state_value(state, state_key) is not None
        }

    def _apply_output_mapping(self, result: Any) -> Dict[str, Any]:
        """Apply explicit output mapping."""
        mapping = self._normalize_mapping(self.output_fields)

        if isinstance(result, dict):
            return {
                state_key: result.get(result_key)
                for result_key, state_key in mapping.items()
                if result_key in result
            }
        else:
            # Single value to first mapped field
            first_state_key = next(iter(mapping.values()))
            return {first_state_key: result}

    def _map_to_outputs(self, result: Any, output_fields: List[str]) -> Dict[str, Any]:
        """Map result to schema output fields."""
        if isinstance(result, dict):
            return {
                field: result.get(field) for field in output_fields if field in result
            }
        else:
            return {output_fields[0]: result} if output_fields else {"result": result}

    def _execute_with_config(
        self, engine: Engine, input_data: Any, config: Optional[ConfigLike]
    ) -> Any:
        """Execute engine with merged configuration."""
        merged_config = self._build_merged_config(config, engine)

        # Log execution details in debug mode
        if self.debug or logger.is_debug_mode():
            logger.debug_table(
                "_execute_with_config",
                {
                    "Engine": engine.name,
                    "Engine Type": getattr(engine, "engine_type", "unknown"),
                    "Input Type": type(input_data).__name__,
                    "Input Data": input_data,
                    "Config": merged_config,
                },
            )

        # Special handling for retrievers - they need string queries
        if hasattr(engine, "engine_type") and engine.engine_type.value == "retriever":
            logger.debug("RETRIEVER DETECTED - Special handling")

            if isinstance(input_data, dict):
                if "query" in input_data:
                    query_str = str(input_data["query"])
                    logger.debug(f"Extracting query string: '{query_str}'")
                    logger.debug(
                        f"Other params: {[k for k in input_data.keys() if k != 'query']}"
                    )
                    return engine.invoke(query_str, merged_config)
                else:
                    logger.debug("No 'query' key in dict, using whole dict as string")
                    return engine.invoke(str(input_data), merged_config)
            else:
                logger.debug(
                    f"Input is not dict, converting to string: '{str(input_data)}'"
                )
                return engine.invoke(str(input_data), merged_config)

        logger.debug("Standard engine invoke")
        return engine.invoke(input_data, merged_config)

    def _build_merged_config(
        self, runtime_config: Optional[ConfigLike], engine: Engine
    ) -> Optional[Dict[str, Any]]:
        """Build merged configuration."""
        if not runtime_config and not self.config_overrides:
            return None

        config = dict(runtime_config or {})

        # Ensure configurable section
        config.setdefault("configurable", {})

        # Apply node-level overrides
        config["configurable"].update(self.config_overrides)

        # Apply engine-specific overrides
        engine_id = getattr(engine, "id", None)
        if engine_id and self.config_overrides:
            config["configurable"].setdefault("engine_configs", {})
            config["configurable"]["engine_configs"].setdefault(engine_id, {})
            config["configurable"]["engine_configs"][engine_id].update(
                self.config_overrides
            )

        return config

    def _lookup_engine(self, name: str) -> Optional[Engine]:
        """Lookup engine in registry."""
        try:
            from haive.core.engine.base import EngineRegistry

            return EngineRegistry.get_instance().find(name)
        except ImportError:
            logger.warning(f"Could not import EngineRegistry for {name}")
            return None

    def _normalize_mapping(
        self, fields: Optional[Union[List[str], Dict[str, str]]]
    ) -> Dict[str, str]:
        """Normalize field mapping to dict."""
        if isinstance(fields, list):
            return {field: field for field in fields}
        return fields or {}

    def _state_as_dict(self, state: StateLike) -> Dict[str, Any]:
        """Convert state to dictionary."""
        if isinstance(state, dict):
            return state
        elif hasattr(state, "model_dump"):
            return state.model_dump()
        elif hasattr(state, "__dict__"):
            return state.__dict__
        else:
            return {"value": state}

    def _is_message_like(self, obj: Any) -> bool:
        """Check if object is message-like."""
        try:
            from langchain_core.messages import BaseMessage

            return isinstance(obj, BaseMessage)
        except ImportError:
            return hasattr(obj, "content") and hasattr(obj, "type")

    def _log_error(self, error: Exception) -> None:
        """Log error with full context."""
        logger.log_exception(error, f"Engine node '{self.name}' failed")

    def __repr__(self) -> str:
        """Clean string representation."""
        engine_ref = self.engine.name if self.engine else self.engine_name or "None"
        return f"EngineNode(name='{self.name}', engine='{engine_ref}')"
