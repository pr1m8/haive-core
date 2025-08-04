"""Enhanced Multi-Agent State with V3/V4 support and extended capabilities.

This module provides EnhancedMultiAgentState, which extends MultiAgentState with additional
features for advanced multi-agent workflows, particularly for V3/V4 agent implementations.

The EnhancedMultiAgentState provides backward compatibility while adding enhanced features
for complex multi-agent orchestration patterns.

Key Features:
    - **Full MultiAgentState compatibility**: All features from base MultiAgentState
    - **Enhanced V3/V4 support**: Additional fields and methods for V3/V4 agents
    - **Performance tracking**: Agent performance metrics and adaptive routing
    - **Extended debugging**: Enhanced debug capabilities and rich display
    - **Backward compatibility**: Drop-in replacement for MultiAgentState

Example:
    Basic usage (identical to MultiAgentState)::

        from haive.core.schema.prebuilt.enhanced_multi_agent_state import EnhancedMultiAgentState
        from haive.agents.simple import SimpleAgent

        # Create agents
        planner = SimpleAgent(name="planner")
        executor = SimpleAgent(name="executor")

        # Initialize enhanced state
        state = EnhancedMultiAgentState(agents=[planner, executor])

    With enhanced features::

        # Enable performance tracking
        state = EnhancedMultiAgentState(
            agents=[planner, executor],
            performance_tracking=True
        )

        # Access performance metrics
        metrics = state.get_performance_metrics("planner")
        print(f"Planner performance: {metrics}")

See Also:
    - :class:`haive.core.schema.prebuilt.multi_agent_state.MultiAgentState`: Base state class
    - :class:`haive.agents.multi.enhanced.multi_agent_v3.EnhancedMultiAgent`: V3 implementation
    - :class:`haive.agents.multi.enhanced_multi_agent_v4.EnhancedMultiAgentV4`: V4 implementation
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


class EnhancedMultiAgentState(MultiAgentState):
    """Enhanced multi-agent state with V3/V4 support and extended capabilities.

    This class extends MultiAgentState with additional features for advanced multi-agent
    workflows, particularly for V3/V4 agent implementations. It provides full backward
    compatibility while adding enhanced features for complex orchestration patterns.

    The EnhancedMultiAgentState is designed to be a drop-in replacement for MultiAgentState
    with additional capabilities for performance tracking, enhanced debugging, and
    advanced coordination patterns.

    Key Features:
        - **Full MultiAgentState compatibility**: All base features preserved
        - **Performance tracking**: Agent execution metrics and adaptive routing
        - **Enhanced debugging**: Extended debug capabilities and rich display
        - **V3/V4 support**: Additional fields and methods for V3/V4 agents
        - **Extended coordination**: Advanced routing and execution patterns

    Additional Attributes:
        performance_tracking (bool): Enable performance metrics collection.
        performance_metrics (Dict[str, Dict[str, Any]]): Performance data per agent.
        execution_metadata (Dict[str, Any]): Additional execution metadata.
        debug_mode (bool): Enable enhanced debug capabilities.

    Examples:
        Basic usage (identical to MultiAgentState)::

            state = EnhancedMultiAgentState(agents=[agent1, agent2])

        With enhanced features::

            state = EnhancedMultiAgentState(
                agents=[agent1, agent2],
                performance_tracking=True,
                debug_mode=True
            )

            # Access performance metrics
            metrics = state.get_performance_metrics("agent1")

        Performance-based routing::

            if state.should_route_to_best_performer():
                best_agent = state.get_best_performing_agent()
                state.set_active_agent(best_agent)

    See Also:
        - :class:`haive.core.schema.prebuilt.multi_agent_state.MultiAgentState`: Base state
        - :class:`haive.agents.multi.enhanced.multi_agent_v3.EnhancedMultiAgent`: V3 usage
        - :class:`haive.agents.multi.enhanced_multi_agent_v4.EnhancedMultiAgentV4`: V4 usage
    """

    # Enhanced fields for V3/V4 support
    performance_tracking: bool = Field(
        default=False,
        description="Enable performance metrics collection for adaptive routing",
    )

    performance_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Performance metrics per agent (execution time, success rate, etc.)",
    )

    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata for advanced coordination patterns",
    )

    debug_mode: bool = Field(
        default=False,
        description="Enable enhanced debug capabilities and detailed logging",
    )

    # Enhanced methods for V3/V4 workflows

    def get_performance_metrics(self, agent_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary containing performance metrics, or empty dict if none available

        Example:
            >>> metrics = state.get_performance_metrics("planner")
            >>> print(f"Average execution time: {metrics.get('avg_time', 0)}")
        """
        return self.performance_metrics.get(agent_name, {})

    def record_agent_performance(
        self,
        agent_name: str,
        execution_time: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record performance metrics for an agent execution.

        Args:
            agent_name: Name of the agent
            execution_time: Time taken for execution in seconds
            success: Whether the execution was successful
            metadata: Additional performance metadata

        Example:
            >>> state.record_agent_performance("planner", 2.5, True, {"tokens": 150})
        """
        if not self.performance_tracking:
            return

        if agent_name not in self.performance_metrics:
            self.performance_metrics[agent_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "success_rate": 0.0,
                "last_execution": None,
            }

        metrics = self.performance_metrics[agent_name]
        metrics["total_executions"] += 1
        metrics["total_time"] += execution_time
        metrics["avg_time"] = metrics["total_time"] / metrics["total_executions"]

        if success:
            metrics["successful_executions"] += 1

        metrics["success_rate"] = (
            metrics["successful_executions"] / metrics["total_executions"]
        )
        metrics["last_execution"] = {
            "time": execution_time,
            "success": success,
            "metadata": metadata or {},
        }

    def get_best_performing_agent(self, metric: str = "success_rate") -> Optional[str]:
        """Get the name of the best performing agent based on a metric.

        Args:
            metric: Metric to optimize for ("success_rate", "avg_time", etc.)

        Returns:
            Name of best performing agent, or None if no metrics available

        Example:
            >>> best = state.get_best_performing_agent("success_rate")
            >>> if best:
            ...     state.set_active_agent(best)
        """
        if not self.performance_metrics:
            return None

        best_agent = None
        best_value = None

        for agent_name, metrics in self.performance_metrics.items():
            if metric not in metrics:
                continue

            value = metrics[metric]

            # For time-based metrics, lower is better
            if metric.endswith("_time"):
                if best_value is None or value < best_value:
                    best_value = value
                    best_agent = agent_name
            else:
                # For other metrics, higher is better
                if best_value is None or value > best_value:
                    best_value = value
                    best_agent = agent_name

        return best_agent

    def should_route_to_best_performer(self, threshold: float = 0.1) -> bool:
        """Determine if routing should be done based on performance.

        Args:
            threshold: Minimum performance difference to trigger routing

        Returns:
            True if performance-based routing is recommended

        Example:
            >>> if state.should_route_to_best_performer():
            ...     best = state.get_best_performing_agent()
            ...     state.set_active_agent(best)
        """
        if not self.performance_tracking or len(self.performance_metrics) < 2:
            return False

        success_rates = [
            metrics.get("success_rate", 0.0)
            for metrics in self.performance_metrics.values()
        ]

        if not success_rates:
            return False

        max_rate = max(success_rates)
        min_rate = min(success_rates)

        return (max_rate - min_rate) > threshold

    def set_execution_metadata(self, key: str, value: Any) -> None:
        """Set execution metadata for advanced coordination.

        Args:
            key: Metadata key
            value: Metadata value

        Example:
            >>> state.set_execution_metadata("workflow_type", "sequential")
            >>> state.set_execution_metadata("priority", "high")
        """
        self.execution_metadata[key] = value

    def get_execution_metadata(self, key: str, default: Any = None) -> Any:
        """Get execution metadata.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default

        Example:
            >>> workflow_type = state.get_execution_metadata("workflow_type", "parallel")
        """
        return self.execution_metadata.get(key, default)

    def display_enhanced_debug_info(
        self, title: str = "Enhanced MultiAgent State"
    ) -> None:
        """Display enhanced debug information with performance metrics.

        Args:
            title: Title for the debug display

        Example:
            >>> state.display_enhanced_debug_info("My Workflow Debug")
        """
        # Call parent debug display
        self.display_debug_info(title)

        # Add enhanced information if available
        if self.performance_tracking and self.performance_metrics:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            # Performance metrics table
            perf_table = Table(title="🚀 Agent Performance Metrics")
            perf_table.add_column("Agent", style="cyan")
            perf_table.add_column("Executions", style="green")
            perf_table.add_column("Success Rate", style="yellow")
            perf_table.add_column("Avg Time", style="magenta")
            perf_table.add_column("Status", style="blue")

            for agent_name, metrics in self.performance_metrics.items():
                executions = str(metrics.get("total_executions", 0))
                success_rate = f"{metrics.get('success_rate', 0.0):.2%}"
                avg_time = f"{metrics.get('avg_time', 0.0):.2f}s"

                # Determine status
                if metrics.get("success_rate", 0.0) > 0.9:
                    status = "🟢 Excellent"
                elif metrics.get("success_rate", 0.0) > 0.7:
                    status = "🟡 Good"
                else:
                    status = "🔴 Needs Attention"

                perf_table.add_row(
                    agent_name, executions, success_rate, avg_time, status
                )

            console.print(perf_table)
            console.print()


# For backward compatibility, also export under common alias
MultiAgentStateEnhanced = EnhancedMultiAgentState

# Export both names
__all__ = ["EnhancedMultiAgentState", "MultiAgentStateEnhanced"]
