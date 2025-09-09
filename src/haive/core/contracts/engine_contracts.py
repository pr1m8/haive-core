"""Engine contracts for explicit behavior specification.

This module defines contracts that engines must implement to make their
behavior explicit and verifiable at runtime.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FieldContract(BaseModel):
    """Contract for a single field.
    
    Defines the contract for an individual field including its type,
    requirements, validation, and documentation.
    
    Attributes:
        name: Field identifier
        field_type: Expected type for the field
        required: Whether field must be present
        default: Default value if not required
        description: Human-readable field purpose
        validator: Optional validation function
        examples: Example values for documentation
        
    Examples:
        >>> # Define a messages field contract
        >>> messages_field = FieldContract(
        ...     name="messages",
        ...     field_type=list,
        ...     required=True,
        ...     description="List of conversation messages",
        ...     validator=lambda x: len(x) > 0
        ... )
        >>> 
        >>> # Define optional temperature field
        >>> temp_field = FieldContract(
        ...     name="temperature",
        ...     field_type=float,
        ...     required=False,
        ...     default=0.7,
        ...     validator=lambda x: 0 <= x <= 2
        ... )
    """
    
    name: str = Field(description="Field identifier")
    field_type: Type = Field(description="Expected type")
    required: bool = Field(default=True, description="Whether field is required")
    default: Any = Field(default=None, description="Default value if not required")
    description: str = Field(default="", description="Field purpose")
    validator: Optional[Callable[[Any], bool]] = Field(
        default=None, 
        description="Validation function",
        exclude=True  # Exclude from serialization
    )
    examples: List[Any] = Field(
        default_factory=list,
        description="Example values"
    )
    
    def validate_value(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this field contract.
        
        Args:
            value: Value to validate
            
        Returns:
            (is_valid, error_message)
        """
        # Type check
        if not isinstance(value, self.field_type):
            return False, f"Expected {self.field_type.__name__}, got {type(value).__name__}"
        
        # Custom validation
        if self.validator:
            try:
                if not self.validator(value):
                    return False, "Custom validation failed"
            except Exception as e:
                return False, f"Validation error: {e}"
        
        return True, None


class EngineContract(BaseModel):
    """Complete contract for an engine.
    
    Defines the full contract for an engine including inputs, outputs,
    side effects, and conditions that must hold before and after execution.
    
    Attributes:
        inputs: Input field contracts
        outputs: Output field contracts  
        side_effects: Fields modified as side effects
        preconditions: Conditions that must be true before execution
        postconditions: Conditions guaranteed after execution
        error_handling: How errors are handled
        performance: Performance characteristics
        
    Examples:
        >>> # LLM engine contract
        >>> llm_contract = EngineContract(
        ...     inputs=[
        ...         FieldContract(name="messages", field_type=list, required=True),
        ...         FieldContract(name="temperature", field_type=float, required=False)
        ...     ],
        ...     outputs=[
        ...         FieldContract(name="response", field_type=str, required=True)
        ...     ],
        ...     side_effects=["conversation_history"],
        ...     preconditions=["len(messages) > 0"],
        ...     postconditions=["response is not empty"]
        ... )
    """
    
    # Field contracts
    inputs: List[FieldContract] = Field(
        default_factory=list,
        description="Input field contracts"
    )
    outputs: List[FieldContract] = Field(
        default_factory=list,
        description="Output field contracts"
    )
    
    # Side effects and conditions
    side_effects: List[str] = Field(
        default_factory=list,
        description="Fields modified as side effects"
    )
    preconditions: List[str] = Field(
        default_factory=list,
        description="Conditions required before execution"
    )
    postconditions: List[str] = Field(
        default_factory=list,
        description="Conditions guaranteed after execution"
    )
    
    # Error handling
    error_handling: Dict[str, str] = Field(
        default_factory=dict,
        description="Error types and how they're handled"
    )
    
    # Performance characteristics
    performance: Dict[str, Any] = Field(
        default_factory=lambda: {
            "timeout": None,
            "max_retries": 0,
            "is_async": False,
            "is_streaming": False
        },
        description="Performance characteristics"
    )
    
    def get_required_inputs(self) -> List[str]:
        """Get list of required input field names.
        
        Returns:
            List of required field names
        """
        return [f.name for f in self.inputs if f.required]
    
    def get_optional_inputs(self) -> Dict[str, Any]:
        """Get optional inputs with defaults.
        
        Returns:
            Dictionary of optional field names to defaults
        """
        return {
            f.name: f.default 
            for f in self.inputs 
            if not f.required
        }
    
    def get_guaranteed_outputs(self) -> List[str]:
        """Get list of guaranteed output field names.
        
        Returns:
            List of guaranteed output field names
        """
        return [f.name for f in self.outputs if f.required]
    
    def validate_inputs(self, state: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate state against input contracts.
        
        Args:
            state: State to validate
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        for field_contract in self.inputs:
            if field_contract.required and field_contract.name not in state:
                issues.append(f"Missing required field: {field_contract.name}")
            elif field_contract.name in state:
                value = state[field_contract.name]
                valid, error = field_contract.validate_value(value)
                if not valid:
                    issues.append(f"Field '{field_contract.name}': {error}")
        
        return len(issues) == 0, issues
    
    def validate_outputs(self, result: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate result against output contracts.
        
        Args:
            result: Result to validate
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        for field_contract in self.outputs:
            if field_contract.required and field_contract.name not in result:
                issues.append(f"Missing guaranteed output: {field_contract.name}")
            elif field_contract.name in result:
                value = result[field_contract.name]
                valid, error = field_contract.validate_value(value)
                if not valid:
                    issues.append(f"Output '{field_contract.name}': {error}")
        
        return len(issues) == 0, issues


class EngineInterface(ABC):
    """Interface all engines must implement for contracts.
    
    This ensures every engine explicitly declares its contract,
    making dependencies and effects clear at runtime.
    
    Methods:
        get_contract: Return the engine's contract
        validate_input: Check if state is valid for execution
        validate_output: Check if output meets contract
        execute: Execute the engine with contract enforcement
    """
    
    @abstractmethod
    def get_contract(self) -> EngineContract:
        """Get engine's contract.
        
        Returns:
            Complete contract specification
            
        Examples:
            >>> contract = engine.get_contract()
            >>> print(f"Required inputs: {contract.get_required_inputs()}")
            >>> print(f"Guaranteed outputs: {contract.get_guaranteed_outputs()}")
        """
        pass
    
    @abstractmethod
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate state meets input requirements.
        
        Args:
            state: Current state
            
        Returns:
            True if state is valid for execution
        """
        pass
    
    @abstractmethod
    def validate_output(self, result: Any) -> bool:
        """Validate output meets contract.
        
        Args:
            result: Execution result
            
        Returns:
            True if output is valid
        """
        pass
    
    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute engine with state.
        
        Args:
            state: Input state
            
        Returns:
            Execution result
            
        Raises:
            ContractViolation: If contract is violated
        """
        pass
    
    def check_preconditions(self, state: Dict[str, Any]) -> List[str]:
        """Check which preconditions are not met.
        
        Args:
            state: Current state
            
        Returns:
            List of unmet preconditions
        """
        contract = self.get_contract()
        unmet = []
        
        for condition in contract.preconditions:
            if not self._evaluate_condition(condition, state):
                unmet.append(condition)
        
        return unmet
    
    def check_postconditions(self, state: Dict[str, Any]) -> List[str]:
        """Check which postconditions are not met.
        
        Args:
            state: State after execution
            
        Returns:
            List of unmet postconditions
        """
        contract = self.get_contract()
        unmet = []
        
        for condition in contract.postconditions:
            if not self._evaluate_condition(condition, state):
                unmet.append(condition)
        
        return unmet
    
    def _evaluate_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        """Evaluate a condition against state.
        
        This is a simplified implementation. A real implementation would
        use a safe expression evaluator or a DSL for conditions.
        
        Args:
            condition: Condition expression
            state: Current state
            
        Returns:
            True if condition is met
        """
        # Simple conditions for now
        if condition == "len(messages) > 0":
            return "messages" in state and len(state.get("messages", [])) > 0
        elif condition == "response is not empty":
            return "response" in state and bool(state.get("response"))
        elif condition == "response is not None":
            return "response" in state and state.get("response") is not None
        elif condition == "tools are callable":
            tools = state.get("tools", [])
            return all(callable(t) for t in tools)
        else:
            # Unknown condition - log and assume true
            logger.warning(f"Unknown condition: {condition}")
            return True
    
    def get_contract_summary(self) -> Dict[str, Any]:
        """Get human-readable contract summary.
        
        Returns:
            Contract summary with key information
        """
        contract = self.get_contract()
        return {
            "required_inputs": contract.get_required_inputs(),
            "optional_inputs": list(contract.get_optional_inputs().keys()),
            "guaranteed_outputs": contract.get_guaranteed_outputs(),
            "side_effects": contract.side_effects,
            "preconditions": contract.preconditions,
            "postconditions": contract.postconditions,
            "performance": contract.performance
        }


class ContractAdapter(EngineInterface):
    """Base adapter for adding contracts to existing engines.
    
    This provides a base implementation for adapting existing engines
    to support contracts without modifying the original implementation.
    
    Attributes:
        engine: The wrapped engine
        contract: The engine's contract
        
    Examples:
        >>> # Adapt existing engine
        >>> class MyEngineAdapter(ContractAdapter):
        ...     def build_contract(self):
        ...         return EngineContract(...)
        ...     
        ...     def execute(self, state):
        ...         # Adapt state and call engine
        ...         return self.engine.invoke(state)
    """
    
    def __init__(self, engine: Any):
        """Initialize adapter with existing engine.
        
        Args:
            engine: Engine to wrap with contracts
        """
        self.engine = engine
        self.contract = self.build_contract()
        self._execution_count = 0
        self._contract_violations: List[Dict[str, Any]] = []
    
    @abstractmethod
    def build_contract(self) -> EngineContract:
        """Build contract for the wrapped engine.
        
        Returns:
            Engine contract specification
        """
        pass
    
    def get_contract(self) -> EngineContract:
        """Get engine's contract.
        
        Returns:
            Engine contract
        """
        return self.contract
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate input against contract.
        
        Args:
            state: Input state
            
        Returns:
            True if valid
        """
        valid, issues = self.contract.validate_inputs(state)
        if not valid:
            logger.warning(f"Input validation issues: {issues}")
        return valid
    
    def validate_output(self, result: Any) -> bool:
        """Validate output against contract.
        
        Args:
            result: Execution result
            
        Returns:
            True if valid
        """
        if not isinstance(result, dict):
            return False
        
        valid, issues = self.contract.validate_outputs(result)
        if not valid:
            logger.warning(f"Output validation issues: {issues}")
        return valid
    
    def log_violation(self, phase: str, details: str) -> None:
        """Log a contract violation.
        
        Args:
            phase: Phase where violation occurred
            details: Violation details
        """
        violation = {
            "phase": phase,
            "details": details,
            "execution": self._execution_count
        }
        self._contract_violations.append(violation)
        logger.error(f"Contract violation in {phase}: {details}")