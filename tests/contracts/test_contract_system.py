"""Test the runtime contract system.

This test demonstrates the contract system working with bounded state,
contract enforcement, and orchestration.
"""

import pytest
from typing import Dict, Any

from haive.core.contracts import (
    AccessPermissions,
    BoundedState,
    StateView,
    FieldContract,
    EngineContract,
    EngineInterface,
    NodeContract,
    ContractualNode,
    ContractViolation,
    Orchestrator,
)


class TestBoundedState:
    """Test bounded state with access control."""
    
    def test_state_creation_and_access(self):
        """Test creating bounded state and controlled access."""
        # Create state with initial data
        state = BoundedState(initial_data={
            "messages": [],
            "context": {"session": "test"},
            "private_data": "secret"
        })
        
        # Register component with permissions
        permissions = AccessPermissions(
            readable={"messages", "context"},
            writable={"messages"},
            append_only={"history"}
        )
        state.register_component("llm", permissions)
        
        # Get view for component
        view = state.get_view_for("llm")
        
        # Test read access
        messages = view.get("messages")
        assert messages == []
        
        context = view.get("context")
        assert context == {"session": "test"}
        
        # Test write access
        view.set("messages", [{"role": "user", "content": "Hello"}])
        assert state._data["messages"] == [{"role": "user", "content": "Hello"}]
        
        # Test permission violation
        with pytest.raises(PermissionError):
            view.get("private_data")  # Not readable
        
        with pytest.raises(PermissionError):
            view.set("context", {})  # Not writable
    
    def test_state_checkpointing(self):
        """Test state checkpointing and rollback."""
        state = BoundedState(initial_data={"counter": 0})
        
        # Create checkpoint (already has initial checkpoint at v1)
        v2 = state.checkpoint("Second checkpoint")
        assert v2 == 2
        
        # Modify state
        state._data["counter"] = 5
        v3 = state.checkpoint("After increment")
        
        # Modify again
        state._data["counter"] = 10
        
        # Rollback to v1 (initial)
        state.rollback(1)
        assert state._data["counter"] == 0
        
        # Rollback to v3
        state.rollback(v3)
        assert state._data["counter"] == 5
    
    def test_access_logging(self):
        """Test access logging and summary."""
        state = BoundedState()
        
        permissions = AccessPermissions(
            readable={"field1", "field2"},
            writable={"field1"}
        )
        state.register_component("test", permissions)
        
        view = state.get_view_for("test")
        
        # Perform operations
        view.get("field1", "default")
        view.set("field1", "value")
        
        try:
            view.get("field3")  # Should fail
        except PermissionError:
            pass
        
        # Check access log
        logs = view.get_access_log()
        assert len(logs) == 3
        assert logs[0]["operation"] == "read"
        assert logs[1]["operation"] == "write"
        assert logs[2]["status"] == "denied"


class MockEngine(EngineInterface):
    """Mock engine for testing contracts."""
    
    def get_contract(self) -> EngineContract:
        return EngineContract(
            inputs=[
                FieldContract(name="input1", field_type=str, required=True),
                FieldContract(name="input2", field_type=int, required=False, default=10)
            ],
            outputs=[
                FieldContract(name="result", field_type=str, required=True)
            ],
            preconditions=["len(input1) > 0"],
            postconditions=["result is not None"]
        )
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        return "input1" in state and isinstance(state["input1"], str)
    
    def validate_output(self, result: Any) -> bool:
        return isinstance(result, dict) and "result" in result
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        input1 = state.get("input1", "")
        input2 = state.get("input2", 10)
        return {"result": f"{input1}-{input2}"}


class TestContractualNode:
    """Test contractual nodes with enforcement."""
    
    def test_node_creation_and_execution(self):
        """Test creating and executing a contractual node."""
        # Create node contract
        contract = NodeContract(
            inputs=["messages"],
            outputs=["processed"],
            transforms={"messages": "processed_messages"}
        )
        
        # Create node function
        def process(view: StateView) -> Dict[str, Any]:
            messages = view.get("messages")
            return {
                "processed": f"Processed {len(messages)} messages",
                "messages": messages  # For transform
            }
        
        # Create contractual node
        node = ContractualNode("processor", contract, process)
        
        # Create state and view
        state = BoundedState({"messages": ["msg1", "msg2"]})
        permissions = AccessPermissions(
            readable={"messages"},
            writable={"processed", "processed_messages"}
        )
        state.register_component("processor", permissions)
        view = state.get_view_for("processor")
        
        # Execute node
        result = node(view)
        assert result["processed"] == "Processed 2 messages"
    
    def test_node_contract_violation(self):
        """Test node contract violations."""
        contract = NodeContract(
            inputs=["required_field"],
            outputs=["output_field"]
        )
        
        def failing_process(view: StateView) -> Dict[str, Any]:
            return {}  # Missing required output
        
        node = ContractualNode("failing", contract, failing_process)
        
        # Create state without required field
        state = BoundedState({})
        permissions = AccessPermissions(readable=set(), writable=set())
        state.register_component("failing", permissions)
        view = state.get_view_for("failing")
        
        # Should raise contract violation
        with pytest.raises(ContractViolation) as exc_info:
            node(view)
        
        assert exc_info.value.violation["type"] == "input"


class TestOrchestrator:
    """Test orchestrator with contract enforcement."""
    
    def test_orchestrator_registration(self):
        """Test registering components with orchestrator."""
        orchestrator = Orchestrator()
        
        # Register engine
        engine = MockEngine()di
        orchestrator.register_engine("test_engine", engine)
        
        # Register node
        contract = NodeContract(inputs=["data"], outputs=["result"])
        node = ContractualNode(
            "test_node",
            contract,
            lambda v: {"result": "processed"}
        )
        orchestrator.register_node(node)
        
        # Check registration
        assert "test_engine" in orchestrator.components
        assert "test_node" in orchestrator.components
        assert "test_engine" in orchestrator.contracts
        assert "test_node" in orchestrator.contracts
    
    def test_orchestrator_execution(self):
        """Test orchestrator execution with contract enforcement."""
        orchestrator = Orchestrator()
        
        # Register engine
        engine = MockEngine()
        orchestrator.register_engine("engine1", engine)
        
        # Create state
        state = BoundedState({"input1": "test"})
        
        # Execute engine
        result = orchestrator.execute("engine1", state)
        assert result["result"] == "test-10"
        
        # Check execution log
        summary = orchestrator.get_execution_summary()
        assert summary["successful"] == 1
        assert summary["failed"] == 0
    
    def test_orchestrator_chain_execution(self):
        """Test executing chain of components."""
        orchestrator = Orchestrator()
        
        # Register first node
        contract1 = NodeContract(
            inputs=["input"],
            outputs=["intermediate"]
        )
        node1 = ContractualNode(
            "node1",
            contract1,
            lambda v: {"intermediate": v.get("input") + "-processed"}
        )
        orchestrator.register_node(node1)
        
        # Register second node
        contract2 = NodeContract(
            inputs=["intermediate"],
            outputs=["final"]
        )
        node2 = ContractualNode(
            "node2",
            contract2,
            lambda v: {"final": v.get("intermediate") + "-final"}
        )
        orchestrator.register_node(node2)
        
        # Create state
        state = BoundedState({"input": "data"})
        
        # Execute chain
        results = orchestrator.execute_chain(["node1", "node2"], state)
        
        assert "node1" in results
        assert "node2" in results
        assert state._data["final"] == "data-processed-final"
    
    def test_composition_validation(self):
        """Test validating component composition."""
        orchestrator = Orchestrator()
        
        # Register components with incompatible contracts
        contract1 = NodeContract(outputs=["field_a"])
        node1 = ContractualNode("node1", contract1, lambda v: {"field_a": "a"})
        orchestrator.register_node(node1)
        
        contract2 = NodeContract(inputs=["field_b"])  # Needs field_b, not field_a
        node2 = ContractualNode("node2", contract2, lambda v: {})
        orchestrator.register_node(node2)
        
        # Validate composition
        issues = orchestrator.validate_composition(["node1", "node2"])
        assert len(issues) > 0
        assert "field_b" in issues[0]


class TestContractViolations:
    """Test contract violation handling."""
    
    def test_input_validation_failure(self):
        """Test input validation failures."""
        orchestrator = Orchestrator()
        engine = MockEngine()
        orchestrator.register_engine("engine", engine)
        
        # State missing required field
        state = BoundedState({})  # Missing "input1"
        
        with pytest.raises(ContractViolation) as exc_info:
            orchestrator.execute("engine", state)
        
        assert "pre-execution" in str(exc_info.value)
    
    def test_retry_logic(self):
        """Test node retry on failure."""
        call_count = 0
        
        def flaky_function(view: StateView) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"output": "success"}
        
        contract = NodeContract(
            outputs=["output"],
            can_retry=True,
            max_retries=3
        )
        
        node = ContractualNode("flaky", contract, flaky_function)
        
        # Create state and view
        state = BoundedState({})
        permissions = AccessPermissions(writable={"output"})
        state.register_component("flaky", permissions)
        view = state.get_view_for("flaky")
        
        # Should succeed after retries
        result = node(view)
        assert result["output"] == "success"
        assert call_count == 3


def test_end_to_end_workflow():
    """Test complete workflow with all components."""
    # Create bounded state
    state = BoundedState({
        "messages": [{"role": "user", "content": "Hello"}],
        "context": {}
    })
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Register processing node
    process_contract = NodeContract(
        inputs=["messages"],
        outputs=["processed_messages", "message_count"]
    )
    
    def process_messages(view: StateView) -> Dict[str, Any]:
        messages = view.get("messages")
        return {
            "processed_messages": messages,
            "message_count": len(messages)
        }
    
    process_node = ContractualNode("processor", process_contract, process_messages)
    orchestrator.register_node(process_node)
    
    # Register validation node
    validate_contract = NodeContract(
        inputs=["message_count"],
        outputs=["is_valid"],
        dependencies=["processor"]
    )
    
    def validate(view: StateView) -> Dict[str, Any]:
        count = view.get("message_count")
        return {"is_valid": count > 0}
    
    validate_node = ContractualNode("validator", validate_contract, validate)
    orchestrator.register_node(validate_node)
    
    # Execute workflow
    results = orchestrator.execute_chain(["processor", "validator"], state)
    
    # Verify results
    assert results["processor"]["message_count"] == 1
    assert results["validator"]["is_valid"] is True
    
    # Check state was updated
    assert "message_count" in state._data
    assert "is_valid" in state._data
    
    # Get execution summary
    summary = orchestrator.get_execution_summary()
    assert summary["successful"] == 2
    assert summary["success_rate"] == 1.0


if __name__ == "__main__":
    # Run basic tests
    test_end_to_end_workflow()
    print("✅ End-to-end workflow test passed!")
    
    # Test bounded state
    test_state = TestBoundedState()
    test_state.test_state_creation_and_access()
    print("✅ Bounded state access control test passed!")
    
    test_state.test_state_checkpointing()
    print("✅ State checkpointing test passed!")
    
    # Test nodes
    test_node = TestContractualNode()
    test_node.test_node_creation_and_execution()
    print("✅ Contractual node execution test passed!")
    
    # Test orchestrator
    test_orch = TestOrchestrator()
    test_orch.test_orchestrator_execution()
    print("✅ Orchestrator execution test passed!")
    
    print("\n🎉 All contract system tests passed!")