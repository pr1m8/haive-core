"""Node contracts for graph execution.

This module defines contracts for graph nodes, ensuring explicit
declaration of behavior and dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from haive.core.contracts.boundaries import StateView

logger = logging.getLogger(__name__)


class NodeContract(BaseModel):
    """Contract for a graph node.
    
    Defines what a node requires, produces, and guarantees during
    execution in a graph workflow.
    
    Attributes:
        inputs: Required input fields from state
        outputs: Produced output fields to state
        transforms: How fields are transformed (source -> target)
        dependencies: Other nodes this depends on
        guarantees: What this node guarantees after execution
        error_handling: How errors are handled
        can_retry: Whether node can be retried on failure
        
    Examples:
        >>> # Validation node contract
        >>> validation_contract = NodeContract(
        ...     inputs=["response", "criteria"],
        ...     outputs=["validated_response", "validation_score"],
        ...     transforms={"response": "validated_response"},
        ...     guarantees=["validation_score between 0 and 1"]
        ... )
        >>> 
        >>> # Tool node contract
        >>> tool_contract = NodeContract(
        ...     inputs=["tool_calls", "tools"],
        ...     outputs=["tool_results"],
        ...     dependencies=["llm_node"],
        ...     can_retry=True
        ... )
    """
    
    # Required fields
    inputs: List[str] = Field(
        default_factory=list,
        description="Required input fields"
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="Produced output fields"
    )
    
    # Transformations
    transforms: Dict[str, str] = Field(
        default_factory=dict,
        description="Field transformations (source -> target)"
    )
    
    # Dependencies and guarantees
    dependencies: List[str] = Field(
        default_factory=list,
        description="Other nodes this depends on"
    )
    guarantees: List[str] = Field(
        default_factory=list,
        description="Post-execution guarantees"
    )
    
    # Error handling
    error_handling: str = Field(
        default="raise",
        description="How to handle errors (raise, log, default)"
    )
    can_retry: bool = Field(
        default=False,
        description="Whether node can be retried"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts if can_retry"
    )
    
    def validate_dependencies(self, executed_nodes: List[str]) -> List[str]:
        """Check if dependencies have been executed.
        
        Args:
            executed_nodes: List of already executed node names
            
        Returns:
            List of missing dependencies
        """
        return [dep for dep in self.dependencies if dep not in executed_nodes]
    
    def get_all_fields(self) -> Dict[str, str]:
        """Get all fields this node interacts with.
        
        Returns:
            Dictionary mapping field to interaction type
        """
        fields = {}
        
        for field in self.inputs:
            fields[field] = "input"
        
        for field in self.outputs:
            fields[field] = "output"
        
        for source, target in self.transforms.items():
            fields[source] = "transform_source"
            fields[target] = "transform_target"
        
        return fields


class ContractualNode:
    """Node that declares and enforces its contract.
    
    This ensures nodes explicitly declare their behavior,
    making graph composition predictable and debuggable.
    
    Attributes:
        name: Node identifier
        contract: Node's contract specification
        execute_fn: Function to execute
        _execution_count: Number of times executed
        _contract_violations: List of contract violations
        _execution_history: History of executions
        
    Examples:
        >>> # Create validation node
        >>> def validate(state_view):
        ...     response = state_view.get("response")
        ...     # Validation logic
        ...     return {"validated_response": response, "score": 0.95}
        >>> 
        >>> node = ContractualNode(
        ...     name="validator",
        ...     contract=validation_contract,
        ...     execute_fn=validate
        ... )
        >>> 
        >>> # Execute with state view
        >>> result = node(state_view)
    """
    
    def __init__(
        self, 
        name: str, 
        contract: NodeContract, 
        execute_fn: Callable[[StateView], Dict[str, Any]]
    ):
        """Initialize contractual node.
        
        Args:
            name: Node identifier
            contract: Node's contract
            execute_fn: Function to execute with StateView
        """
        self.name = name
        self.contract = contract
        self.execute_fn = execute_fn
        self._execution_count = 0
        self._contract_violations: List[Dict[str, Any]] = []
        self._execution_history: List[Dict[str, Any]] = []
        self._retry_count = 0
    
    def __call__(self, state_view: StateView) -> Dict[str, Any]:
        """Execute node with contract enforcement.
        
        Args:
            state_view: Bounded view of state
            
        Returns:
            Execution results
            
        Raises:
            ContractViolation: If contract is violated
        """
        from datetime import datetime
        
        start_time = datetime.now()
        
        # Validate inputs
        if not self._validate_inputs(state_view):
            violation = {
                "node": self.name,
                "type": "input",
                "details": f"Missing required inputs: {self.contract.inputs}",
                "timestamp": datetime.now().isoformat()
            }
            self._contract_violations.append(violation)
            raise ContractViolation(violation)
        
        # Execute with retry logic
        result = None
        last_error = None
        
        for attempt in range(self.contract.max_retries if self.contract.can_retry else 1):
            try:
                result = self.execute_fn(state_view)
                self._execution_count += 1
                break
            except Exception as e:
                last_error = e
                self._retry_count += 1
                
                if not self.contract.can_retry or attempt == self.contract.max_retries - 1:
                    # Handle error based on policy
                    if self.contract.error_handling == "raise":
                        violation = {
                            "node": self.name,
                            "type": "execution",
                            "details": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        self._contract_violations.append(violation)
                        raise ContractViolation(violation)
                    elif self.contract.error_handling == "log":
                        logger.error(f"Node '{self.name}' execution failed: {e}")
                        result = {}
                    elif self.contract.error_handling == "default":
                        result = {field: None for field in self.contract.outputs}
                else:
                    logger.warning(
                        f"Node '{self.name}' attempt {attempt + 1} failed, retrying: {e}"
                    )
        
        if result is None:
            result = {}
        
        # Validate outputs
        if not self._validate_outputs(result):
            violation = {
                "node": self.name,
                "type": "output",
                "details": f"Missing required outputs: {self.contract.outputs}",
                "timestamp": datetime.now().isoformat()
            }
            self._contract_violations.append(violation)
            raise ContractViolation(violation)
        
        # Apply transformations to state
        self._apply_transforms(state_view, result)
        
        # Record execution
        execution_time = (datetime.now() - start_time).total_seconds()
        self._execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "duration": execution_time,
            "status": "success",
            "retry_count": self._retry_count
        })
        
        return result
    
    def _validate_inputs(self, state_view: StateView) -> bool:
        """Validate all required inputs are available.
        
        Args:
            state_view: State view to validate
            
        Returns:
            True if all inputs available
        """
        for field in self.contract.inputs:
            if not state_view.has_permission("read", field):
                logger.error(f"Node '{self.name}' cannot read required field '{field}'")
                return False
            
            try:
                # Try to get the field to ensure it exists
                state_view.get(field)
            except (PermissionError, KeyError):
                return False
        
        return True
    
    def _validate_outputs(self, result: Dict[str, Any]) -> bool:
        """Validate all required outputs are produced.
        
        Args:
            result: Execution result
            
        Returns:
            True if all outputs produced
        """
        if not isinstance(result, dict):
            return False
        
        for field in self.contract.outputs:
            if field not in result:
                logger.error(f"Node '{self.name}' missing required output '{field}'")
                return False
        
        return True
    
    def _apply_transforms(self, state_view: StateView, result: Dict[str, Any]) -> None:
        """Apply field transformations to state.
        
        Args:
            state_view: State view to update
            result: Execution result
        """
        for source, target in self.contract.transforms.items():
            if source in result:
                # Transform and write to target field
                if state_view.has_permission("write", target):
                    state_view.set(target, result[source])
                else:
                    logger.warning(
                        f"Node '{self.name}' cannot write transform target '{target}'"
                    )
    
    def get_contract_summary(self) -> Dict[str, Any]:
        """Get human-readable contract summary.
        
        Returns:
            Contract details and execution statistics
        """
        return {
            "name": self.name,
            "contract": {
                "inputs": self.contract.inputs,
                "outputs": self.contract.outputs,
                "transforms": self.contract.transforms,
                "dependencies": self.contract.dependencies,
                "guarantees": self.contract.guarantees,
                "can_retry": self.contract.can_retry
            },
            "statistics": {
                "executions": self._execution_count,
                "violations": len(self._contract_violations),
                "retries": self._retry_count,
                "success_rate": (
                    (self._execution_count - len(self._contract_violations)) / 
                    self._execution_count if self._execution_count > 0 else 0
                )
            },
            "recent_violations": self._contract_violations[-3:]
        }
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self._execution_count = 0
        self._contract_violations = []
        self._execution_history = []
        self._retry_count = 0


class ContractViolation(Exception):
    """Exception raised when contract is violated.
    
    Attributes:
        violation: Dictionary containing violation details
    """
    
    def __init__(self, violation: Dict[str, Any]):
        """Initialize with violation details.
        
        Args:
            violation: Violation information
        """
        self.violation = violation
        super().__init__(f"Contract violation: {violation}")


class NodeChain:
    """Chain of contractual nodes with dependency validation.
    
    Ensures nodes are executed in valid order with all dependencies met.
    
    Attributes:
        nodes: Dictionary of nodes by name
        execution_order: Order to execute nodes
        _executed: Set of executed node names
        
    Examples:
        >>> # Create node chain
        >>> chain = NodeChain()
        >>> chain.add_node(llm_node)
        >>> chain.add_node(validation_node)
        >>> chain.add_node(output_node)
        >>> 
        >>> # Validate and execute
        >>> issues = chain.validate_chain()
        >>> if not issues:
        ...     result = chain.execute(state_view)
    """
    
    def __init__(self):
        """Initialize empty node chain."""
        self.nodes: Dict[str, ContractualNode] = {}
        self.execution_order: List[str] = []
        self._executed: set[str] = set()
    
    def add_node(self, node: ContractualNode) -> None:
        """Add node to chain.
        
        Args:
            node: Node to add
        """
        self.nodes[node.name] = node
        self.execution_order.append(node.name)
    
    def validate_chain(self) -> List[str]:
        """Validate the node chain for consistency.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check dependencies are met
        for name in self.execution_order:
            node = self.nodes[name]
            missing = node.contract.validate_dependencies(
                self.execution_order[:self.execution_order.index(name)]
            )
            if missing:
                issues.append(
                    f"Node '{name}' has unmet dependencies: {missing}"
                )
        
        # Check output-input compatibility
        available_outputs = set()
        for name in self.execution_order:
            node = self.nodes[name]
            
            # Check inputs are available
            missing_inputs = set(node.contract.inputs) - available_outputs
            if missing_inputs and name != self.execution_order[0]:
                issues.append(
                    f"Node '{name}' requires {missing_inputs} but they're not produced"
                )
            
            # Add outputs to available
            available_outputs.update(node.contract.outputs)
            
            # Add transform targets
            available_outputs.update(node.contract.transforms.values())
        
        return issues
    
    def execute(self, state_view: StateView) -> Dict[str, Any]:
        """Execute the node chain.
        
        Args:
            state_view: State view for execution
            
        Returns:
            Combined results from all nodes
            
        Raises:
            ContractViolation: If any node violates contract
        """
        combined_result = {}
        
        for name in self.execution_order:
            node = self.nodes[name]
            
            # Check dependencies
            missing = node.contract.validate_dependencies(list(self._executed))
            if missing:
                raise ContractViolation({
                    "node": name,
                    "type": "dependency",
                    "details": f"Missing dependencies: {missing}"
                })
            
            # Execute node
            result = node(state_view)
            combined_result.update(result)
            
            # Mark as executed
            self._executed.add(name)
        
        return combined_result
    
    def get_execution_plan(self) -> List[Dict[str, Any]]:
        """Get execution plan showing order and dependencies.
        
        Returns:
            List of execution steps
        """
        plan = []
        
        for i, name in enumerate(self.execution_order):
            node = self.nodes[name]
            plan.append({
                "step": i + 1,
                "node": name,
                "inputs": node.contract.inputs,
                "outputs": node.contract.outputs,
                "dependencies": node.contract.dependencies
            })
        
        return plan