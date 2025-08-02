"""Intelligent Multi-Agent Node with sequence inference and branching.

This module provides an enhanced multi-agent node that can automatically:
- Infer execution sequence from agent naming patterns and dependencies
- Handle conditional branching between agents
- Dynamically route execution based on conditions
- Manage complex multi-agent workflows
"""
import logging
from typing import Any, Self

from langgraph.graph import END
from langgraph.types import Command
from pydantic import Field, model_validator

from haive.core.graph.common.types import ConfigLike, NodeType
from haive.core.graph.node.base_node_config import BaseNodeConfig
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState

logger = logging.getLogger(__name__)

class IntelligentMultiAgentNode(BaseNodeConfig[MultiAgentState, MultiAgentState]):
    """Intelligent multi-agent node with sequence inference and branching.

    This node provides advanced multi-agent coordination with:
    - Automatic sequence inference from agent patterns
    - Conditional branching and routing
    - Dynamic agent execution planning
    - Smart fallback strategies
    """
    node_type: NodeType = Field(default=NodeType.AGENT, description='Node type for intelligent multi-agent execution')
    execution_mode: str = Field(default='infer', description='Execution mode: infer, sequential, parallel, branch, conditional')
    infer_sequence: bool = Field(default=True, description='Whether to automatically infer execution sequence')
    branches: dict[str, dict[str, Any]] = Field(default_factory=dict, description='Branch configurations for conditional routing')
    current_sequence: list[str] | None = Field(default=None, description='Current inferred or manually set sequence')
    execution_index: int = Field(default=0, description='Current position in execution sequence')

    @model_validator(mode='after')
    def validate_config(self) -> Self:
        """Validate node configuration."""
        if self.execution_mode not in ['infer', 'sequential', 'parallel', 'branch', 'conditional']:
            raise ValueError(f'Invalid execution_mode: {self.execution_mode}')
        return self

    def __call__(self, state: MultiAgentState, config: ConfigLike | None=None) -> Command:
        """Execute intelligent multi-agent coordination."""
        logger.info(f'{'=' * 60}')
        logger.info(f'INTELLIGENT MULTI-AGENT NODE: {self.name}')
        logger.info(f'Mode: {self.execution_mode}')
        logger.info(f'{'=' * 60}')
        try:
            if self.execution_mode == 'infer' and self.infer_sequence:
                sequence = self._infer_execution_sequence(state)
                logger.info(f'Inferred sequence: {sequence}')
            else:
                sequence = self._get_execution_sequence(state)
                logger.info(f'Using sequence: {sequence}')
            self.current_sequence = sequence
            if self.execution_mode in ['infer', 'sequential']:
                return self._execute_sequential(state, sequence)
            if self.execution_mode == 'parallel':
                return self._execute_parallel(state, sequence)
            if self.execution_mode == 'branch':
                return self._execute_branch(state, sequence)
            if self.execution_mode == 'conditional':
                return self._execute_conditional(state, sequence)
            raise ValueError(f'Unknown execution mode: {self.execution_mode}')
        except Exception as e:
            logger.exception(f'Error in intelligent multi-agent execution: {e}')
            return Command(goto=END)

    def _infer_execution_sequence(self, state: MultiAgentState) -> list[str]:
        """Infer the optimal execution sequence from agent characteristics."""
        agents = state.agents
        if not agents:
            return []
        agent_names = list(agents.keys())
        if len(agent_names) <= 1:
            return agent_names
        sequence = self._infer_from_naming_patterns(agent_names)
        if sequence:
            logger.debug(f'Sequence from naming patterns: {sequence}')
            return sequence
        sequence = self._infer_from_agent_types(agent_names, agents)
        if sequence:
            logger.debug(f'Sequence from agent types: {sequence}')
            return sequence
        sequence = self._infer_from_prompt_dependencies(agent_names, agents)
        if sequence:
            logger.debug(f'Sequence from prompt dependencies: {sequence}')
            return sequence
        logger.debug(f'Fallback sequence: {agent_names}')
        return agent_names

    def _infer_from_naming_patterns(self, agent_names: list[str]) -> list[str]:
        """Infer sequence from common naming patterns."""
        patterns = ['planner', 'plan', 'planning', 'analyzer', 'analysis', 'analyze', 'researcher', 'research', 'search', 'executor', 'execute', 'execution', 'worker', 'validator', 'validate', 'validation', 'reviewer', 'review', 'critique', 'replanner', 'replan', 'replanning', 'formatter', 'format', 'output', 'summary', 'summarize', 'summarizer']
        agent_scores = {}
        for agent_name in agent_names:
            score = len(patterns)
            for i, pattern in enumerate(patterns):
                if pattern in agent_name.lower():
                    score = i
                    break
            agent_scores[agent_name] = score
        sorted_agents = sorted(agent_names, key=lambda x: agent_scores[x])
        if len(set(agent_scores.values())) > 1:
            return sorted_agents
        return []

    def _infer_from_agent_types(self, agent_names: list[str], agents: dict[str, Any]) -> list[str]:
        """Infer sequence from agent types/classes."""
        type_priority = {'ReactAgent': 1, 'SimpleAgent': 2, 'RAGAgent': 3, 'ToolAgent': 4}
        agent_scores = {}
        for agent_name in agent_names:
            agent = agents[agent_name]
            agent_type = type(agent).__name__
            agent_scores[agent_name] = type_priority.get(agent_type, 5)
        sorted_agents = sorted(agent_names, key=lambda x: agent_scores[x])
        if len(set(agent_scores.values())) > 1:
            return sorted_agents
        return []

    def _infer_from_prompt_dependencies(self, agent_names: list[str], agents: dict[str, Any]) -> list[str]:
        """Infer sequence from prompt template dependencies."""
        dependencies = {}
        for agent_name in agent_names:
            agent = agents[agent_name]
            dependencies[agent_name] = set()
            if hasattr(agent, 'engine') and hasattr(agent.engine, 'prompt_template'):
                prompt = str(agent.engine.prompt_template)
                for other_agent in agent_names:
                    if other_agent != agent_name:
                        if any((field in prompt.lower() for field in [f'{other_agent}_result', f'{other_agent}_output', f'result_from_{other_agent}', f'output_from_{other_agent}'])):
                            dependencies[agent_name].add(other_agent)
        sequence = []
        remaining = set(agent_names)
        while remaining:
            ready = []
            for agent_name in remaining:
                if not dependencies[agent_name] & remaining:
                    ready.append(agent_name)
            if not ready:
                ready = [next(iter(remaining))]
            for agent_name in ready:
                sequence.append(agent_name)
                remaining.remove(agent_name)
        return sequence if len(sequence) > 1 else []

    def _get_execution_sequence(self, state: MultiAgentState) -> list[str]:
        """Get execution sequence (manual or from state)."""
        if self.current_sequence:
            return self.current_sequence
        return list(state.agents.keys())

    def _execute_sequential(self, state: MultiAgentState, sequence: list[str]) -> Command:
        """Execute agents in sequence."""
        if not sequence:
            return Command(goto=END)
        if self.execution_index >= len(sequence):
            return Command(goto=END)
        current_agent = sequence[self.execution_index]
        self.execution_index += 1
        state.set_active_agent(current_agent)
        if self.execution_index < len(sequence):
            next_agent = sequence[self.execution_index]
            return Command(goto=f'agent_{next_agent}')
        return Command(goto=END)

    def _execute_parallel(self, state: MultiAgentState, sequence: list[str]) -> Command:
        """Execute agents in parallel."""
        if not sequence:
            return Command(goto=END)
        if sequence:
            state.set_active_agent(sequence[0])
            return Command(goto=f'agent_{sequence[0]}')
        return Command(goto=END)

    def _execute_branch(self, state: MultiAgentState, sequence: list[str]) -> Command:
        """Execute agents with conditional branching."""
        if not sequence:
            return Command(goto=END)
        current_agent = sequence[0] if sequence else None
        if not current_agent:
            return Command(goto=END)
        if current_agent in self.branches:
            branch_config = self.branches[current_agent]
            next_agent = self._evaluate_branch_condition(state, branch_config)
            if next_agent:
                state.set_active_agent(next_agent)
                return Command(goto=f'agent_{next_agent}')
        state.set_active_agent(current_agent)
        return Command(goto=f'agent_{current_agent}')

    def _execute_conditional(self, state: MultiAgentState, sequence: list[str]) -> Command:
        """Execute agents with conditional flow."""
        if not sequence:
            return Command(goto=END)
        current_agent = sequence[self.execution_index] if self.execution_index < len(sequence) else None
        if not current_agent:
            return Command(goto=END)
        state.set_active_agent(current_agent)
        if self.execution_index + 1 < len(sequence):
            next_agent = sequence[self.execution_index + 1]
            if self._should_continue_to_next_agent(state, current_agent, next_agent):
                self.execution_index += 1
                return Command(goto=f'agent_{next_agent}')
        return Command(goto=END)

    def _evaluate_branch_condition(self, state: MultiAgentState, branch_config: dict[str, Any]) -> str | None:
        """Evaluate branch condition and return next agent."""
        condition = branch_config.get('condition', 'default')
        targets = branch_config.get('targets', [])
        if condition == 'default' and targets:
            return targets[0]
        return targets[0] if targets else None

    def _should_continue_to_next_agent(self, state: MultiAgentState, current_agent: str, next_agent: str) -> bool:
        """Determine if execution should continue to next agent."""
        return True

    def add_branch_condition(self, source_agent: str, condition: str, target_agents: list[str]):
        """Add a branch condition for an agent."""
        self.branches[source_agent] = {'condition': condition, 'targets': target_agents}

    def set_execution_sequence(self, sequence: list[str]):
        """Manually set the execution sequence."""
        self.current_sequence = sequence
        self.execution_index = 0
        self.execution_mode = 'sequential'
        self.infer_sequence = False

def create_intelligent_multi_agent_node(name: str, execution_mode: str='infer', branches: dict[str, dict[str, Any]] | None=None, **kwargs) -> IntelligentMultiAgentNode:
    """Factory function to create an intelligent multi-agent node.

    Args:
        name: Name of the node
        execution_mode: Execution mode (infer, sequential, parallel, branch, conditional)
        branches: Branch configurations
        **kwargs: Additional configuration parameters

    Returns:
        Configured IntelligentMultiAgentNode
    """
    return IntelligentMultiAgentNode(name=name, execution_mode=execution_mode, branches=branches or {}, **kwargs)