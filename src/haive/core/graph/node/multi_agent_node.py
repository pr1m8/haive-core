"""Multi-agent node with hierarchical state projection.

This module provides node configurations for multi-agent systems that properly
handle state projection between the container state and individual agent states.
"""

import logging
from typing import TYPE_CHECKING, Any, Self, TypeVar, Union

if TYPE_CHECKING:
    from haive.agents.base.agent import Agent
else:
    # Placeholder for runtime
    Agent = "Agent"

from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from haive.core.graph.common.types import ConfigLike, NodeType, StateLike
from haive.core.graph.node.base_node_config import BaseNodeConfig
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState

logger = logging.getLogger(__name__)
TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class MultiAgentNode(BaseNodeConfig[MultiAgentState, MultiAgentState]):
    """Node for executing agents within a multi-agent state container.

    This node handles:
    - State projection from MultiAgentState to agent-specific schema
    - Agent execution with its expected state type
    - State updates back to the container
    - Recompilation tracking

    The key innovation is that each agent receives its exact expected
    state schema, not a flattened global state.
    """

    node_type: NodeType = Field(
        default=NodeType.AGENT, description="Node type for multi-agent execution"
    )
    agent_name: str = Field(
        description="Name of the agent to execute (key in agents dict)"
    )
    agent: Union["Agent", None] = Field(
        default=None,
        description="Agent instance (extracted from state if not provided)",
    )
    project_state: bool = Field(
        default=True, description="Whether to project state to agent's expected schema"
    )
    share_messages: bool = Field(
        default=True, description="Whether to share messages with agent state"
    )
    update_mode: str = Field(
        default="merge",
        description="How to update container state: 'merge' or 'replace'",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Validate node configuration."""
        if not self.agent_name:
            raise ValueError("agent_name is required")
        return self

    def __call__(
        self, state: MultiAgentState, config: ConfigLike | None = None
    ) -> Command:
        """Execute agent with state projection."""
        logger.info(f"{'=' * 60}")
        logger.info(f"MULTI-AGENT NODE: {self.name}")
        logger.info(f"Agent: {self.agent_name}")
        logger.info(f"{'=' * 60}")
        try:
            agent = self._get_agent(state)
            if not agent:
                raise ValueError(f"Agent '{self.agent_name}' not found in state")
            state.set_active_agent(self.agent_name)
            agent_input = self._project_state_for_agent(state, agent)
            logger.info(f"Projected state with {len(agent_input)} fields")
            logger.debug(f"Fields: {list(agent_input.keys())}")
            if hasattr(agent, "_app") and agent._app:
                result = agent._app.invoke(agent_input, config)
            else:
                result = agent.invoke(agent_input, config)
            state_update = self._update_container_state(state, result)
            if self._needs_recompilation(agent):
                state.mark_agent_for_recompile(self.agent_name, "Agent graph changed")
            logger.info("✅ Agent completed successfully")
            return Command(update=state_update, goto=self._get_goto_node())
        except Exception as e:
            logger.exception(f"❌ Agent execution failed: {e}")
            state_update = {
                "agent_outputs": {
                    **state.agent_outputs,
                    self.agent_name: {"error": str(e)},
                }
            }
            return Command(update=state_update, goto=self._get_goto_node())

    def _get_agent(self, state: MultiAgentState) -> Agent | None:
        """Get agent from state or use provided agent."""
        if self.agent:
            return self.agent
        return state.get_agent(self.agent_name)

    def _project_state_for_agent(
        self, state: MultiAgentState, agent: "Agent"
    ) -> dict[str, Any]:
        """Project container state to agent's expected schema.

        This is the key method that gives each agent its exact
        expected state type instead of a flattened global state.
        """
        agent_state = state.get_agent_state(self.agent_name)
        projected = agent_state.copy()
        if not self.project_state:
            return projected
        if hasattr(agent, "state_schema") and agent.state_schema:
            expected_schema = agent.state_schema
            for field_name, field_info in expected_schema.model_fields.items():
                if field_name not in projected:
                    if field_name == "messages" and self.share_messages:
                        projected["messages"] = state.messages
                    elif hasattr(state, field_name):
                        shared_fields = getattr(
                            expected_schema, "__shared_fields__", set()
                        )
                        if field_name in shared_fields:
                            projected[field_name] = getattr(state, field_name)
                    elif field_info.default is not None:
                        projected[field_name] = field_info.default
                    elif field_info.default_factory:
                        projected[field_name] = field_info.default_factory()
            try:
                validated = expected_schema(**projected)
                return validated.model_dump()
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")
                return projected
        else:
            if self.share_messages and "messages" not in projected:
                projected["messages"] = state.messages
            return projected

    def _update_container_state(
        self, state: MultiAgentState, agent_result: Any
    ) -> dict[str, Any]:
        """Update container state with agent results."""
        state_update = {}
        if isinstance(agent_result, BaseModel):
            result_dict = agent_result.model_dump()
        elif isinstance(agent_result, dict):
            result_dict = agent_result
        else:
            result_dict = {"result": agent_result}
        if self.update_mode == "merge":
            current_agent_state = state.get_agent_state(self.agent_name)
            updated_agent_state = {**current_agent_state, **result_dict}
        else:
            updated_agent_state = result_dict
        state_update["agent_states"] = {
            **state.agent_states,
            self.agent_name: updated_agent_state,
        }
        state.record_agent_output(self.agent_name, result_dict)
        state_update["agent_outputs"] = state.agent_outputs
        if "messages" in result_dict and self.share_messages:
            state_update["messages"] = result_dict["messages"]
        if hasattr(self._get_agent(state), "state_schema"):
            schema = self._get_agent(state).state_schema
            shared_fields = getattr(schema, "__shared_fields__", set())
            for field in shared_fields:
                if field in result_dict and field != "messages":
                    state_update[field] = result_dict[field]
        return state_update

    def _needs_recompilation(self, agent: "Agent") -> bool:
        """Check if agent needs recompilation."""
        if hasattr(agent, "graph") and hasattr(agent.graph, "needs_recompile"):
            return agent.graph.needs_recompile()
        return False

    def _get_goto_node(self) -> str | None:
        """Get next node to execute."""
        return self.command_goto


class StateProjectionNode(BaseNodeConfig[TInput, TOutput]):
    """Generic state projection node for any schema transformation.

    This node can project from any input schema to any output schema,
    useful for bridging between different state representations.
    """

    node_type: NodeType = Field(
        default=NodeType.TRANSFORM, description="Transform node type"
    )
    input_schema: type[BaseModel] = Field(description="Expected input schema")
    output_schema: type[BaseModel] = Field(description="Output schema to produce")
    field_mappings: dict[str, str] = Field(
        default_factory=dict, description="Map input fields to output fields"
    )
    defaults: dict[str, Any] = Field(
        default_factory=dict, description="Default values for output fields"
    )

    def __call__(self, state: StateLike, config: ConfigLike | None = None) -> Command:
        """Project state from input to output schema."""
        logger.info(
            f"Projecting state: {self.input_schema.__name__} → {self.output_schema.__name__}"
        )
        if isinstance(state, self.input_schema):
            input_data = state.model_dump()
        elif isinstance(state, dict):
            input_data = state
        else:
            input_data = {}
        output_data = {}
        for out_field, in_field in self.field_mappings.items():
            if in_field in input_data:
                output_data[out_field] = input_data[in_field]
        for field, default in self.defaults.items():
            if field not in output_data:
                output_data[field] = default
        for field_name in self.output_schema.model_fields:
            if field_name not in output_data and field_name in input_data:
                output_data[field_name] = input_data[field_name]
        try:
            validated = self.output_schema(**output_data)
            result = validated.model_dump()
        except Exception as e:
            logger.exception(f"Schema validation failed: {e}")
            result = output_data
        return Command(update=result, goto=self._get_goto_node())


def create_multi_agent_node(
    agent_name: str, name: str | None = None, **kwargs
) -> MultiAgentNode:
    """Create a multi-agent node for executing an agent from MultiAgentState."""
    if not name:
        name = f"execute_{agent_name}"
    return MultiAgentNode(name=name, agent_name=agent_name, **kwargs)


def create_projection_node(
    input_schema: type[BaseModel],
    output_schema: type[BaseModel],
    name: str | None = None,
    **kwargs,
) -> StateProjectionNode:
    """Create a state projection node."""
    if not name:
        name = f"project_{input_schema.__name__}_to_{output_schema.__name__}"
    return StateProjectionNode(
        name=name, input_schema=input_schema, output_schema=output_schema, **kwargs
    )
