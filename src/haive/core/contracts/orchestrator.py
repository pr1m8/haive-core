"""Orchestrator for contract enforcement.

This module provides the central orchestrator that ensures all contracts
are respected during execution, providing the control layer over the
dynamic runtime system.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from haive.core.contracts.boundaries import AccessPermissions, BoundedState, StateView
from haive.core.contracts.engine_contracts import EngineContract, EngineInterface
from haive.core.contracts.node_contracts import ContractualNode, ContractViolation, NodeContract

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates execution with contract enforcement.
    
    This is the central coordinator that ensures all contracts
    are respected during execution, providing the control layer
    over the dynamic runtime system.
    
    Attributes:
        components: Registered components (engines and nodes)
        contracts: Contracts for each component
        access_rules: Derived access permissions
        execution_log: Log of all executions
        _dependency_graph: Component dependency graph
        
    Examples:
        >>> # Create orchestrator
        >>> orchestrator = Orchestrator()
        >>> 
        >>> # Register components
        >>> orchestrator.register_engine("llm", llm_engine)
        >>> orchestrator.register_node(validation_node)
        >>> 
        >>> # Execute with contract enforcement
        >>> state = BoundedState(initial_data)
        >>> result = orchestrator.execute("llm", state)
    """
    
    def __init__(self):
        """Initialize orchestrator."""
        self.components: Dict[str, Any] = {}
        self.contracts: Dict[str, Union[EngineContract, NodeContract]] = {}
        self.access_rules: Dict[str, AccessPermissions] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self._dependency_graph: Dict[str, List[str]] = {}
        self._execution_order: List[str] = []
    
    def register_engine(
        self, 
        name: str, 
        engine: EngineInterface,
        auto_register_state: bool = True
    ) -> None:
        """Register engine with its contract.
        
        Args:
            name: Engine identifier
            engine: Engine implementing EngineInterface
            auto_register_state: Whether to auto-register with state
            
        Examples:
            >>> orchestrator.register_engine("llm", ContractualLLMEngine())
            >>> orchestrator.register_engine("retriever", retriever_engine)
        """
        self.components[name] = engine
        self.contracts[name] = engine.get_contract()
        
        # Derive access rules from contract
        permissions = self._derive_permissions_from_contract(engine.get_contract())
        self.access_rules[name] = permissions
        
        # Update dependency graph
        self._update_dependency_graph(name, [])
        
        logger.info(
            f"Registered engine '{name}' with contract: "
            f"inputs={len(engine.get_contract().inputs)}, "
            f"outputs={len(engine.get_contract().outputs)}"
        )
    
    def register_node(
        self, 
        node: ContractualNode,
        auto_register_state: bool = True
    ) -> None:
        """Register node with its contract.
        
        Args:
            node: Node with contract
            auto_register_state: Whether to auto-register with state
            
        Examples:
            >>> validation_node = ContractualNode("validator", contract, validate_fn)
            >>> orchestrator.register_node(validation_node)
        """
        self.components[node.name] = node
        self.contracts[node.name] = node.contract
        
        # Derive access rules from node contract
        permissions = self._derive_permissions_from_node_contract(node.contract)
        self.access_rules[node.name] = permissions
        
        # Update dependency graph
        self._update_dependency_graph(node.name, node.contract.dependencies)
        
        logger.info(
            f"Registered node '{node.name}' with contract: "
            f"inputs={node.contract.inputs}, "
            f"outputs={node.contract.outputs}, "
            f"dependencies={node.contract.dependencies}"
        )
    
    def execute(
        self, 
        component_name: str, 
        state: BoundedState,
        validate_only: bool = False
    ) -> Any:
        """Execute component with full contract enforcement.
        
        Args:
            component_name: Component to execute
            state: Bounded state container
            validate_only: Only validate, don't execute
            
        Returns:
            Execution result
            
        Raises:
            ContractViolation: If any contract violated
            ValueError: If component not found
            
        Examples:
            >>> # Execute with enforcement
            >>> result = orchestrator.execute("llm", state)
            >>> 
            >>> # Validate only
            >>> orchestrator.execute("llm", state, validate_only=True)
        """
        if component_name not in self.components:
            available = list(self.components.keys())
            raise ValueError(
                f"Component '{component_name}' not registered. "
                f"Available: {available}"
            )
        
        component = self.components[component_name]
        contract = self.contracts[component_name]
        
        # Register component with state if not already
        if component_name not in state._access_rules:
            state.register_component(component_name, self.access_rules[component_name])
        
        # Get bounded view for component
        state_view = state.get_view_for(component_name)
        
        # Pre-execution validation
        self._validate_pre_execution(component, component_name, state)
        
        if validate_only:
            logger.info(f"Validation passed for component '{component_name}'")
            return {"validation": "passed"}
        
        # Execute with monitoring
        start_time = datetime.now()
        try:
            if isinstance(component, ContractualNode):
                result = component(state_view)
            elif isinstance(component, EngineInterface):
                # Convert state view to dict for engine
                state_dict = self._state_view_to_dict(state_view, component.get_contract())
                result = component.execute(state_dict)
            else:
                raise TypeError(f"Unknown component type: {type(component)}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Post-execution validation
            self._validate_post_execution(component, component_name, result, state)
            
            # Log successful execution
            self._log_execution(component_name, execution_time, "success")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._log_execution(component_name, execution_time, "failed", str(e))
            raise
    
    def execute_chain(
        self, 
        component_names: List[str], 
        state: BoundedState,
        stop_on_error: bool = True
    ) -> Dict[str, Any]:
        """Execute chain of components in sequence.
        
        Args:
            component_names: Components to execute in order
            state: Bounded state container
            stop_on_error: Whether to stop on first error
            
        Returns:
            Combined results from all components
            
        Examples:
            >>> # Execute pipeline
            >>> results = orchestrator.execute_chain(
            ...     ["retriever", "llm", "validator"],
            ...     state
            ... )
        """
        results = {}
        executed = []
        
        for name in component_names:
            try:
                # Check dependencies
                if name in self._dependency_graph:
                    missing = [
                        dep for dep in self._dependency_graph[name]
                        if dep not in executed
                    ]
                    if missing:
                        raise ContractViolation({
                            "component": name,
                            "type": "dependency",
                            "details": f"Missing dependencies: {missing}"
                        })
                
                # Execute component
                result = self.execute(name, state)
                results[name] = result
                executed.append(name)
                
                # Update state with results if dict
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key in self.access_rules[name].writable:
                            state._data[key] = value
                
            except Exception as e:
                logger.error(f"Component '{name}' failed: {e}")
                if stop_on_error:
                    raise
                results[name] = {"error": str(e)}
        
        return results
    
    def validate_composition(
        self, 
        components: List[str]
    ) -> List[str]:
        """Validate that components can be composed.
        
        Args:
            components: List of component names in execution order
            
        Returns:
            List of compatibility issues
            
        Examples:
            >>> issues = orchestrator.validate_composition(["llm", "validator"])
            >>> if issues:
            ...     print(f"Issues found: {issues}")
        """
        issues = []
        available_fields = set()
        
        for i, current in enumerate(components):
            if current not in self.contracts:
                issues.append(f"Component '{current}' not registered")
                continue
            
            contract = self.contracts[current]
            
            # Check inputs are available
            if isinstance(contract, EngineContract):
                required = contract.get_required_inputs()
            elif isinstance(contract, NodeContract):
                required = contract.inputs
            else:
                required = []
            
            if i > 0:  # Not first component
                missing = set(required) - available_fields
                if missing:
                    issues.append(
                        f"Component '{current}' requires {missing} "
                        f"but previous components don't provide them"
                    )
            
            # Add outputs to available
            if isinstance(contract, EngineContract):
                outputs = contract.get_guaranteed_outputs()
            elif isinstance(contract, NodeContract):
                outputs = contract.outputs
            else:
                outputs = []
            
            available_fields.update(outputs)
            
            # Add transform targets if node
            if isinstance(contract, NodeContract):
                available_fields.update(contract.transforms.values())
        
        return issues
    
    def get_execution_plan(
        self, 
        components: List[str]
    ) -> List[Dict[str, Any]]:
        """Get execution plan for components.
        
        Args:
            components: Component names
            
        Returns:
            Execution plan with details
        """
        plan = []
        
        for i, name in enumerate(components):
            if name not in self.contracts:
                plan.append({
                    "step": i + 1,
                    "component": name,
                    "status": "not_registered"
                })
                continue
            
            contract = self.contracts[name]
            
            if isinstance(contract, EngineContract):
                inputs = contract.get_required_inputs()
                outputs = contract.get_guaranteed_outputs()
                dependencies = []
            else:
                inputs = contract.inputs
                outputs = contract.outputs
                dependencies = contract.dependencies
            
            plan.append({
                "step": i + 1,
                "component": name,
                "type": "engine" if isinstance(contract, EngineContract) else "node",
                "inputs": inputs,
                "outputs": outputs,
                "dependencies": dependencies
            })
        
        return plan
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions.
        
        Returns:
            Execution statistics
        """
        total = len(self.execution_log)
        successful = sum(1 for e in self.execution_log if e["status"] == "success")
        failed = total - successful
        
        avg_duration = 0
        if successful > 0:
            durations = [
                e["duration"] for e in self.execution_log 
                if e["status"] == "success"
            ]
            avg_duration = sum(durations) / len(durations)
        
        # Component statistics
        component_stats = {}
        for log in self.execution_log:
            comp = log["component"]
            if comp not in component_stats:
                component_stats[comp] = {"success": 0, "failed": 0}
            
            if log["status"] == "success":
                component_stats[comp]["success"] += 1
            else:
                component_stats[comp]["failed"] += 1
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "average_duration": avg_duration,
            "registered_components": len(self.components),
            "component_stats": component_stats,
            "recent_failures": [
                e for e in self.execution_log[-10:] 
                if e["status"] == "failed"
            ]
        }
    
    def _validate_pre_execution(
        self, 
        component: Any, 
        name: str, 
        state: BoundedState
    ) -> None:
        """Validate pre-execution conditions.
        
        Args:
            component: Component to validate
            name: Component name
            state: Current state
            
        Raises:
            ContractViolation: If validation fails
        """
        state_snapshot = state.snapshot()
        
        if isinstance(component, EngineInterface):
            # Validate inputs
            if not component.validate_input(state_snapshot):
                raise ContractViolation({
                    "component": name,
                    "phase": "pre-execution",
                    "details": "Input validation failed"
                })
            
            # Check preconditions
            unmet = component.check_preconditions(state_snapshot)
            if unmet:
                raise ContractViolation({
                    "component": name,
                    "phase": "preconditions",
                    "details": f"Unmet preconditions: {unmet}"
                })
    
    def _validate_post_execution(
        self, 
        component: Any, 
        name: str, 
        result: Any, 
        state: BoundedState
    ) -> None:
        """Validate post-execution conditions.
        
        Args:
            component: Component that executed
            name: Component name
            result: Execution result
            state: Current state
            
        Raises:
            ContractViolation: If validation fails
        """
        if isinstance(component, EngineInterface):
            # Validate output
            if not component.validate_output(result):
                raise ContractViolation({
                    "component": name,
                    "phase": "post-execution",
                    "details": "Output validation failed"
                })
            
            # Check postconditions
            state_snapshot = state.snapshot()
            unmet = component.check_postconditions(state_snapshot)
            if unmet:
                raise ContractViolation({
                    "component": name,
                    "phase": "postconditions",
                    "details": f"Unmet postconditions: {unmet}"
                })
    
    def _derive_permissions_from_contract(
        self, 
        contract: EngineContract
    ) -> AccessPermissions:
        """Derive access permissions from engine contract.
        
        Args:
            contract: Engine contract
            
        Returns:
            Access permissions
        """
        permissions = AccessPermissions()
        
        # Inputs are readable
        for field in contract.inputs:
            permissions.readable.add(field.name)
        
        # Outputs are writable
        for field in contract.outputs:
            permissions.writable.add(field.name)
        
        # Side effects need write access
        for field in contract.side_effects:
            permissions.writable.add(field)
        
        return permissions
    
    def _derive_permissions_from_node_contract(
        self, 
        contract: NodeContract
    ) -> AccessPermissions:
        """Derive access permissions from node contract.
        
        Args:
            contract: Node contract
            
        Returns:
            Access permissions
        """
        permissions = AccessPermissions()
        
        # Inputs are readable
        permissions.readable.update(contract.inputs)
        
        # Outputs are writable
        permissions.writable.update(contract.outputs)
        
        # Transforms need both read and write
        for source, target in contract.transforms.items():
            permissions.readable.add(source)
            permissions.writable.add(target)
        
        return permissions
    
    def _state_view_to_dict(
        self, 
        state_view: StateView, 
        contract: EngineContract
    ) -> Dict[str, Any]:
        """Convert state view to dict for engine.
        
        Args:
            state_view: State view
            contract: Engine contract
            
        Returns:
            Dictionary with required fields
        """
        result = {}
        
        # Get required inputs
        for field in contract.inputs:
            if field.required:
                try:
                    result[field.name] = state_view.get(field.name)
                except:
                    if field.default is not None:
                        result[field.name] = field.default
            elif state_view.has_permission("read", field.name):
                result[field.name] = state_view.get(field.name, field.default)
        
        return result
    
    def _update_dependency_graph(
        self, 
        component: str, 
        dependencies: List[str]
    ) -> None:
        """Update component dependency graph.
        
        Args:
            component: Component name
            dependencies: Component dependencies
        """
        self._dependency_graph[component] = dependencies
        
        # Update execution order
        self._update_execution_order()
    
    def _update_execution_order(self) -> None:
        """Update topological execution order based on dependencies."""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(node: str):
            if node in visited:
                return
            visited.add(node)
            
            if node in self._dependency_graph:
                for dep in self._dependency_graph[node]:
                    if dep in self._dependency_graph:
                        visit(dep)
            
            order.append(node)
        
        for component in self._dependency_graph:
            visit(component)
        
        self._execution_order = order
    
    def _log_execution(
        self, 
        component: str, 
        duration: float, 
        status: str, 
        error: Optional[str] = None
    ) -> None:
        """Log execution details.
        
        Args:
            component: Component name
            duration: Execution duration
            status: Execution status
            error: Error message if failed
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "duration": duration,
            "status": status
        }
        
        if error:
            log_entry["error"] = error
        
        self.execution_log.append(log_entry)
        
        if status == "success":
            logger.info(f"Component '{component}' executed successfully in {duration:.3f}s")
        else:
            logger.error(f"Component '{component}' failed after {duration:.3f}s: {error}")