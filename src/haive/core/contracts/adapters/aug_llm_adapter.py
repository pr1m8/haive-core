"""Contract adapter for AugLLMConfig.

This module provides a contract adapter for AugLLMConfig, adding explicit
contracts without modifying the original implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from haive.core.contracts.engine_contracts import (
    ContractAdapter,
    EngineContract,
    FieldContract,
)
from haive.core.contracts.node_contracts import ContractViolation
from haive.core.engine.aug_llm import AugLLMConfig

logger = logging.getLogger(__name__)


class AugLLMContract(BaseModel):
    """Contract specification for AugLLMConfig.
    
    This defines what the LLM engine needs and guarantees,
    making its behavior explicit and verifiable.
    
    Attributes:
        required_inputs: Fields that must be present
        optional_inputs: Optional fields with defaults
        guaranteed_outputs: Fields guaranteed to be produced
        possible_outputs: Conditional outputs based on configuration
        side_effects: State modifications the engine makes
        preconditions: Conditions before execution
        postconditions: Conditions after execution
    """
    
    # Required inputs
    required_inputs: set[str] = Field(
        default={"messages"},
        description="Fields that must be present"
    )
    
    # Optional inputs with defaults
    optional_inputs: Dict[str, Any] = Field(
        default={
            "temperature": 0.7,
            "max_tokens": None,
            "tools": [],
            "tool_choice": "auto",
            "stream": False,
        },
        description="Optional fields with defaults"
    )
    
    # Guaranteed outputs
    guaranteed_outputs: set[str] = Field(
        default={"response", "message_added"},
        description="Fields guaranteed to be produced"
    )
    
    # Possible outputs (conditional)
    possible_outputs: Dict[str, str] = Field(
        default={
            "tool_calls": "if tools provided",
            "structured_output": "if structured_output_model set",
            "token_usage": "if tracking enabled",
            "stream_chunks": "if streaming enabled",
        },
        description="Conditional outputs"
    )
    
    # Side effects
    side_effects: List[str] = Field(
        default=[
            "appends to messages",
            "may call tools",
            "updates conversation_history if present",
            "may update token_count",
        ],
        description="State modifications"
    )
    
    # Preconditions
    preconditions: List[str] = Field(
        default=[
            "messages must be list of BaseMessage or dicts",
            "if tools provided, they must be callable",
            "temperature must be between 0 and 2",
            "max_tokens must be positive if set",
        ],
        description="Conditions that must be true before execution"
    )
    
    # Postconditions
    postconditions: List[str] = Field(
        default=[
            "response will be non-empty string",
            "messages will have new assistant message",
            "if tool_calls made, tool_messages will be added",
            "if structured_output_model, output will validate",
        ],
        description="Conditions guaranteed after execution"
    )


class ContractualAugLLMConfig(ContractAdapter):
    """AugLLMConfig with explicit contracts.
    
    This adapter wraps AugLLMConfig to add contract enforcement
    without modifying the original implementation.
    
    Attributes:
        config: The wrapped AugLLMConfig
        contract: The engine's contract
        _execution_log: Log of executions
        _contract_violations: List of violations
        _tool_contracts: Contracts for registered tools
        
    Examples:
        >>> # Create with new config
        >>> contractual = ContractualAugLLMConfig(
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
        >>> 
        >>> # Wrap existing config
        >>> config = AugLLMConfig()
        >>> contractual = ContractualAugLLMConfig(config=config)
        >>> 
        >>> # Execute with contract enforcement
        >>> state = {"messages": [{"role": "user", "content": "Hello"}]}
        >>> result = contractual.execute(state)
    """
    
    def __init__(self, config: Optional[AugLLMConfig] = None, **kwargs):
        """Initialize with config or create new one.
        
        Args:
            config: Existing AugLLMConfig to wrap
            **kwargs: Arguments to create new AugLLMConfig
        """
        # Create or use provided config
        if config:
            self.config = config
        else:
            self.config = AugLLMConfig(**kwargs)
        
        # Initialize adapter
        super().__init__(self.config)
        
        # Additional tracking
        self._tool_contracts: Dict[str, Dict[str, Any]] = {}
        self._streaming_chunks: List[Any] = []
    
    def build_contract(self) -> EngineContract:
        """Build contract based on configuration.
        
        Returns:
            Contract specification for this engine
        """
        # Build base contract
        base_contract = AugLLMContract()
        
        # Create field contracts
        inputs = []
        
        # Messages field (required)
        inputs.append(FieldContract(
            name="messages",
            field_type=list,
            required=True,
            description="List of conversation messages",
            validator=lambda x: isinstance(x, list) and len(x) > 0
        ))
        
        # Temperature field (optional)
        inputs.append(FieldContract(
            name="temperature",
            field_type=float,
            required=False,
            default=0.7,
            description="Sampling temperature",
            validator=lambda x: 0 <= x <= 2
        ))
        
        # Max tokens field (optional)
        inputs.append(FieldContract(
            name="max_tokens",
            field_type=int,
            required=False,
            default=None,
            description="Maximum response tokens",
            validator=lambda x: x is None or x > 0
        ))
        
        # Tools field (optional)
        if self.config.tools:
            inputs.append(FieldContract(
                name="tools",
                field_type=list,
                required=False,
                default=self.config.tools,
                description="Available tools"
            ))
            base_contract.guaranteed_outputs.add("tool_calls")
        
        # Structured output field
        if self.config.structured_output_model:
            base_contract.guaranteed_outputs.add("structured_output")
            base_contract.postconditions.append(
                f"structured_output will match {self.config.structured_output_model}"
            )
        
        # Streaming
        if self.config.streaming:
            base_contract.possible_outputs["stream_chunks"] = "if streaming enabled"
        
        # Create output contracts
        outputs = []
        
        # Response field (guaranteed)
        outputs.append(FieldContract(
            name="response",
            field_type=str,
            required=True,
            description="LLM response text"
        ))
        
        # Message added flag
        outputs.append(FieldContract(
            name="message_added",
            field_type=bool,
            required=True,
            description="Whether message was added to history"
        ))
        
        # Tool calls (conditional)
        if self.config.tools:
            outputs.append(FieldContract(
                name="tool_calls",
                field_type=list,
                required=False,
                description="Tool invocations made"
            ))
        
        # Structured output (conditional)
        if self.config.structured_output_model:
            outputs.append(FieldContract(
                name="structured_output",
                field_type=dict,
                required=True,
                description="Structured output matching model"
            ))
        
        # Build engine contract
        return EngineContract(
            inputs=inputs,
            outputs=outputs,
            side_effects=list(base_contract.side_effects),
            preconditions=list(base_contract.preconditions),
            postconditions=list(base_contract.postconditions),
            performance={
                "timeout": getattr(self.config, "request_timeout", None),
                "max_retries": getattr(self.config, "max_retries", 0),
                "is_async": True,
                "is_streaming": self.config.streaming,
            }
        )
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate state meets input requirements.
        
        Args:
            state: Current state
            
        Returns:
            True if state is valid for execution
        """
        # Check messages
        if "messages" not in state:
            logger.error("Missing required field: messages")
            return False
        
        messages = state["messages"]
        if not isinstance(messages, list):
            logger.error("messages must be a list")
            return False
        
        if len(messages) == 0:
            logger.error("messages cannot be empty")
            return False
        
        # Validate message format
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.error(f"messages[{i}] must be a dict")
                return False
            if "role" not in msg or "content" not in msg:
                logger.error(f"messages[{i}] missing role or content")
                return False
        
        # Check temperature if provided
        if "temperature" in state:
            temp = state["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                logger.error("temperature must be between 0 and 2")
                return False
        
        # Check max_tokens if provided
        if "max_tokens" in state:
            tokens = state["max_tokens"]
            if tokens is not None and (not isinstance(tokens, int) or tokens <= 0):
                logger.error("max_tokens must be positive integer or None")
                return False
        
        return True
    
    def validate_output(self, result: Any) -> bool:
        """Validate output meets contract.
        
        Args:
            result: Execution result
            
        Returns:
            True if output is valid
        """
        if not isinstance(result, dict):
            logger.error("Result must be a dictionary")
            return False
        
        # Check guaranteed outputs
        if "response" not in result:
            logger.error("Missing guaranteed output: response")
            return False
        
        if not isinstance(result["response"], str) or not result["response"]:
            logger.error("response must be non-empty string")
            return False
        
        # Check structured output if expected
        if self.config.structured_output_model:
            if "structured_output" not in result:
                logger.error("Missing structured output")
                return False
            
            try:
                # Validate against model
                self.config.structured_output_model.model_validate(result["structured_output"])
            except Exception as e:
                logger.error(f"Structured output validation failed: {e}")
                return False
        
        return True
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with contract enforcement.
        
        Args:
            state: Input state
            
        Returns:
            Execution result with contract validation
            
        Raises:
            ContractViolation: If contract is violated
        """
        start_time = datetime.now()
        
        # Pre-execution validation
        if not self.validate_input(state):
            self.log_violation("input_validation", "Input validation failed")
            raise ContractViolation({
                "phase": "input_validation",
                "details": "Input validation failed"
            })
        
        # Check preconditions
        unmet = self.check_preconditions(state)
        if unmet:
            self.log_violation("preconditions", f"Unmet: {unmet}")
            raise ContractViolation({
                "phase": "preconditions",
                "details": f"Preconditions not met: {unmet}"
            })
        
        # Execute
        try:
            # Create runnable and execute
            runnable = self.config.create_runnable()
            
            # Prepare input
            invoke_input = {
                "messages": state["messages"]
            }
            
            # Add optional fields
            if "temperature" in state:
                invoke_input["temperature"] = state["temperature"]
            if "max_tokens" in state:
                invoke_input["max_tokens"] = state["max_tokens"]
            
            # Execute
            raw_result = runnable.invoke(invoke_input)
            
            # Transform to expected format
            result = self._transform_result(raw_result, state)
            
        except Exception as e:
            self.log_violation("execution", str(e))
            raise ContractViolation({
                "phase": "execution",
                "details": f"Execution failed: {e}"
            })
        
        # Post-execution validation
        if not self.validate_output(result):
            self.log_violation("output_validation", "Output validation failed")
            raise ContractViolation({
                "phase": "output_validation",
                "details": "Output validation failed"
            })
        
        # Check postconditions
        unmet_post = self.check_postconditions(state)
        if unmet_post:
            self.log_violation("postconditions", f"Unmet: {unmet_post}")
            raise ContractViolation({
                "phase": "postconditions",
                "details": f"Postconditions not met: {unmet_post}"
            })
        
        # Log successful execution
        self._execution_count += 1
        self._execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "duration": (datetime.now() - start_time).total_seconds(),
            "status": "success",
            "input_size": len(str(state)),
            "output_size": len(str(result))
        })
        
        return result
    
    def _transform_result(self, raw_result: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw result to contract format.
        
        Args:
            raw_result: Raw execution result
            state: Input state
            
        Returns:
            Transformed result matching contract
        """
        result = {}
        
        # Extract response
        if isinstance(raw_result, str):
            result["response"] = raw_result
        elif isinstance(raw_result, dict):
            result["response"] = raw_result.get("content", str(raw_result))
        elif hasattr(raw_result, "content"):
            result["response"] = raw_result.content
        else:
            result["response"] = str(raw_result)
        
        # Track message addition
        result["message_added"] = True
        
        # Add tool calls if present
        if hasattr(raw_result, "tool_calls") and raw_result.tool_calls:
            result["tool_calls"] = raw_result.tool_calls
        
        # Add structured output if present
        if self.config.structured_output_model:
            if hasattr(raw_result, "structured_output"):
                result["structured_output"] = raw_result.structured_output
            elif isinstance(raw_result, dict) and "structured_output" in raw_result:
                result["structured_output"] = raw_result["structured_output"]
        
        # Add token usage if available
        if hasattr(raw_result, "usage"):
            result["token_usage"] = {
                "total": raw_result.usage.total_tokens,
                "prompt": raw_result.usage.prompt_tokens,
                "completion": raw_result.usage.completion_tokens
            }
        elif hasattr(raw_result, "response_metadata"):
            metadata = raw_result.response_metadata
            if "token_usage" in metadata:
                result["token_usage"] = metadata["token_usage"]
        
        return result
    
    def add_tool_with_contract(self, tool: Any, contract: Dict[str, Any]) -> None:
        """Add tool with its contract.
        
        Args:
            tool: Tool to add
            contract: Tool's contract specification
        """
        # Add to config
        self.config.add_tool(tool)
        
        # Update our contract
        tool_name = getattr(tool, "__name__", str(tool))
        self.contract.possible_outputs[f"tool_{tool_name}_result"] = "if tool is called"
        self.contract.side_effects.append(f"may call {tool_name}")
        
        # Store tool contract for validation
        self._tool_contracts[tool_name] = contract
        
        # Rebuild contract to include tool
        self.contract = self.build_contract()
        
        logger.info(f"Added tool '{tool_name}' with contract")
    
    def get_contract_summary(self) -> Dict[str, Any]:
        """Get human-readable contract summary.
        
        Returns:
            Contract details and statistics
        """
        total_executions = len(self._execution_log)
        successful = sum(1 for e in self._execution_log if e["status"] == "success")
        violations = len(self._contract_violations)
        
        # Get average execution time
        if successful > 0:
            avg_duration = sum(
                e["duration"] for e in self._execution_log 
                if e["status"] == "success"
            ) / successful
        else:
            avg_duration = 0
        
        return {
            "contract": {
                "required_inputs": list(self.contract.get_required_inputs()),
                "optional_inputs": list(self.contract.get_optional_inputs().keys()),
                "guaranteed_outputs": list(self.contract.get_guaranteed_outputs()),
                "side_effects": self.contract.side_effects,
                "preconditions": self.contract.preconditions,
                "postconditions": self.contract.postconditions,
                "performance": self.contract.performance
            },
            "statistics": {
                "total_executions": total_executions,
                "successful": successful,
                "violations": violations,
                "success_rate": successful / total_executions if total_executions > 0 else 0,
                "average_duration": avg_duration
            },
            "tools": list(self._tool_contracts.keys()),
            "recent_violations": self._contract_violations[-5:] if self._contract_violations else []
        }