"""Full integration test for all contract components.

Tests the complete contract system including boundaries, engines, nodes,
tools, prompts, and orchestration working together.
"""

import pytest
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Import all our contract components
from haive.core.contracts.boundaries import (
    BoundedState,
    AccessPermissions,
    StateView
)
from haive.core.contracts.engine_contracts import (
    EngineInterface,
    EngineContract,
    FieldContract
)
from haive.core.contracts.node_contracts import (
    ContractualNode,
    NodeContract
)
from haive.core.contracts.orchestrator import Orchestrator
from haive.core.contracts.tool_config import (
    ToolConfig,
    ToolContract,
    ToolCapability
)
from haive.core.contracts.enhanced_prompt_config import (
    EnhancedPromptConfig,
    FewShotConfig,
    MessagesConfig,
    FormatInstructionsConfig
)
from haive.core.contracts.tool_registry import ToolRegistry
from haive.core.contracts.prompt_library import PromptLibrary


# Test models
class AnalysisResult(BaseModel):
    """Structured output for analysis."""
    summary: str = Field(..., description="Summary of analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    recommendations: list[str] = Field(default_factory=list)


# Test tools
@tool
def analyzer(text: str) -> str:
    """Analyze text and return insights."""
    return f"Analysis of {len(text)} characters: Contains {len(text.split())} words"


@tool
def summarizer(text: str, max_length: int = 100) -> str:
    """Summarize text to specified length."""
    words = text.split()[:max_length]
    return " ".join(words)


class TestFullIntegration:
    """Test full integration of all contract components."""
    
    def test_complete_workflow(self):
        """Test a complete workflow using all components."""
        
        # 1. Create bounded state with permissions
        permissions = AccessPermissions(
            readable={"input", "output", "context"},
            writable={"output"},
            append_only={"history"},
            compute_only={"temp"}
        )
        
        state = BoundedState(
            initial_data={
                "input": "Analyze this important document about AI safety",
                "context": {"domain": "AI", "importance": "high"},
                "history": [],
                "temp": {}
            }
        )
        
        # Register component with permissions
        state.register_component("test_component", permissions)
        
        # 2. Setup tool configuration with contracts
        tool_config = ToolConfig()
        
        analyzer_contract = ToolContract(
            name="analyzer",
            description="Text analyzer",
            capabilities=ToolCapability(
                can_read_state=True,
                can_write_state=False,
                computational_cost="low"
            ),
            side_effects=[]
        )
        
        summarizer_contract = ToolContract(
            name="summarizer",
            description="Text summarizer",
            capabilities=ToolCapability(
                can_read_state=True,
                can_write_state=True,
                computational_cost="medium"
            ),
            side_effects=["modifies_output"]
        )
        
        tool_config.add_tool(analyzer, analyzer_contract)
        tool_config.add_tool(summarizer, summarizer_contract)
        
        # 3. Setup enhanced prompt configuration
        prompt_config = EnhancedPromptConfig(
            system_message="You are an AI safety analyst",
            messages=MessagesConfig(
                add_messages_placeholder=True,
                messages_placeholder_name="history"
            ),
            format_instructions=FormatInstructionsConfig(
                include_format_instructions=True,
                auto_generate=True
            )
        )
        
        # Create template
        template = prompt_config.create_template()
        assert template is not None
        
        # Add format instructions for structured output
        prompt_config.add_format_instructions(AnalysisResult)
        
        # 4. Create tool registry and register tools
        registry = ToolRegistry()
        registry.register("analyzer", analyzer, analyzer_contract)
        registry.register("summarizer", summarizer, summarizer_contract)
        
        # 5. Create prompt library and store templates
        library = PromptLibrary()
        from haive.core.contracts.prompt_config import PromptContract
        
        analysis_contract = PromptContract(
            name="analysis",
            description="Analysis prompt",
            output_format="structured"
        )
        
        library.add_template(
            "analysis",
            template,
            analysis_contract,
            tags={"safety", "analysis"}
        )
        
        # 6. Create engine with contracts
        engine_contract = EngineContract(
            inputs=[
                FieldContract(name="input", field_type=str, required=True),
                FieldContract(name="context", field_type=dict, required=False)
            ],
            outputs=[
                FieldContract(name="output", field_type=str),
                FieldContract(name="analysis", field_type=AnalysisResult)
            ],
            side_effects=["updates_history"],
            preconditions=["input_not_empty"],
            postconditions=["output_generated"]
        )
        
        # Note: We'd normally create ContractualAugLLMConfig here
        # but for testing we'll simulate the interface
        class MockEngine(EngineInterface):
            def get_contract(self) -> EngineContract:
                return engine_contract
            
            def validate_input(self, data: Dict[str, Any]) -> bool:
                return "input" in data and bool(data["input"])
            
            def validate_output(self, result: Any) -> bool:
                """Validate that output meets contract requirements."""
                return isinstance(result, dict) and "output" in result and "analysis" in result
            
            def execute(self, state: StateView) -> Dict[str, Any]:
                # Simulate engine execution
                input_text = state.get("input")
                
                # Use tools
                analysis = analyzer.invoke({"text": input_text})
                summary = summarizer.invoke({"text": input_text, "max_length": 10})
                
                return {
                    "output": summary,
                    "analysis": {
                        "summary": analysis,
                        "confidence": 0.85,
                        "recommendations": ["Review safety implications"]
                    }
                }
        
        engine = MockEngine()
        
        # 7. Create contractual node
        node_contract = NodeContract(
            name="analysis_node",
            inputs=["input", "context"],
            outputs=["output", "analysis"],
            required_state_fields=["input"],
            state_modifications=["output", "history"]
        )
        
        # Create execution function that uses the engine
        def execute_with_engine(state_view: StateView) -> Dict[str, Any]:
            return engine.execute(state_view)
        
        node = ContractualNode(
            name="analysis_node",
            contract=node_contract,
            execute_fn=execute_with_engine
        )
        
        # 8. Create orchestrator and register components
        orchestrator = Orchestrator()
        
        # Register node
        orchestrator.register_node(node)
        
        # Note: In real usage we'd register the engine
        # orchestrator.register_engine("main_engine", engine)
        
        # 9. Execute the workflow
        # Get state view for component
        state_view = state.get_view_for("test_component")
        
        # Validate we can read input
        assert state_view.get("input") == "Analyze this important document about AI safety"
        
        # Execute through node
        result = node(state_view)
        
        # Verify results
        assert "output" in result
        assert "analysis" in result
        assert isinstance(result["analysis"], dict)
        assert "summary" in result["analysis"]
        assert result["analysis"]["confidence"] == 0.85
        
        # 10. Verify tool usage tracking
        registry.track_usage("analyzer", execution_time=0.1)
        registry.track_usage("summarizer", execution_time=0.2)
        
        stats = registry.get_usage_stats()
        assert stats["analyzer"]["usage_count"] == 1
        assert stats["summarizer"]["usage_count"] == 1
        
        # 11. Verify prompt library usage
        retrieved_template = library.get_template("analysis")
        assert retrieved_template is not None
        assert library.templates["analysis:1.0.0"].usage_count == 1
        
        # 12. Test capability-based tool discovery
        safe_tools = registry.find_by_capability("can_write_state", False)
        assert analyzer in safe_tools
        assert summarizer not in safe_tools
        
        # 13. Test permission validation
        available_permissions = {"read_state"}
        valid, missing = registry.validate_permissions(
            "summarizer",
            available_permissions
        )
        # Summarizer has no required permissions in our setup
        assert valid is True
        
        # 14. Verify state checkpointing
        checkpoint_id = state.checkpoint("before_modification")
        
        # Modify state
        state_view.set("output", "Modified output")
        
        # Restore checkpoint
        state.rollback(checkpoint_id)
        restored_view = state.get_view_for("test_component")
        assert restored_view.get("output") != "Modified output"
    
    def test_contract_violations(self):
        """Test that contract violations are properly caught."""
        
        # Create state with strict permissions
        permissions = AccessPermissions(
            readable={"input"},
            writable=set(),  # Nothing writable
            append_only=set(),
            compute_only=set()
        )
        
        state = BoundedState(
            initial_data={"input": "test", "protected": "secret"}
        )
        
        # Register component with permissions
        state.register_component("test", permissions)
        
        view = state.get_view_for("test")
        
        # Test read permission violation
        with pytest.raises(PermissionError):
            view.get("protected")  # Not in readable
        
        # Test write permission violation
        with pytest.raises(PermissionError):
            view.set("input", "modified")  # Not in writable
    
    def test_tool_routing_with_contracts(self):
        """Test tool routing based on capabilities."""
        
        config = ToolConfig(routing_strategy="capability")
        
        # Add tools with different capabilities
        safe_tool_contract = ToolContract(
            name="safe",
            description="Safe tool",
            capabilities=ToolCapability(
                can_write_state=False,
                can_call_external=False,
                computational_cost="low"
            )
        )
        
        unsafe_tool_contract = ToolContract(
            name="unsafe",
            description="Unsafe tool",
            capabilities=ToolCapability(
                can_write_state=True,
                can_call_external=True,
                computational_cost="high"
            ),
            required_permissions={"admin", "write_external"}
        )
        
        config.add_tool(analyzer, safe_tool_contract)
        config.add_tool(summarizer, unsafe_tool_contract)
        
        # Get tools by capability
        low_cost_tools = config.get_tools_by_capability("computational_cost", "low")
        assert len(low_cost_tools) == 1  # safe tool has computational_cost='low'
        
        safe_tools = config.get_tools_by_capability("can_write_state", False)
        assert len(safe_tools) == 1
        assert analyzer in safe_tools
        
        # Validate permissions
        available_perms = {"read"}
        assert config.validate_permissions(analyzer, available_perms) is True
        assert config.validate_permissions(summarizer, available_perms) is False
    
    def test_prompt_composition_workflow(self):
        """Test composing prompts from library."""
        
        library = PromptLibrary()
        from haive.core.contracts.prompt_config import PromptContract
        
        # Add base templates
        base_template = ChatPromptTemplate.from_template("Base: {input}")
        specific_template = ChatPromptTemplate.from_template("Specific: {detail}")
        
        base_contract = PromptContract(name="base", description="Base template")
        specific_contract = PromptContract(name="specific", description="Specific template")
        
        library.add_template("base", base_template, base_contract, tags={"foundation"})
        library.add_template("specific", specific_template, specific_contract, tags={"detail"})
        
        # Compose templates
        composed = library.compose_templates(
            ["base", "specific"],
            "composed",
            mode="sequential"
        )
        
        assert composed is not None
        assert "composed:1.0.0" in library.templates
        assert library.composition_rules["composed"] == ["base", "specific"]
        
        # Verify composition
        messages = composed.messages
        assert len(messages) >= 2  # Should have both templates
    
    def test_orchestrator_coordination(self):
        """Test orchestrator coordinating multiple components."""
        
        orchestrator = Orchestrator()
        
        # Create multiple nodes with contracts
        node1_contract = NodeContract(
            inputs=["input"],
            outputs=["intermediate"]
        )
        
        node2_contract = NodeContract(
            inputs=["intermediate"],
            outputs=["output"]
        )
        
        # Create mock engines
        class MockEngine1(EngineInterface):
            def get_contract(self) -> EngineContract:
                return EngineContract(
                    inputs=[FieldContract(name="input", field_type=str, required=True)],
                    outputs=[FieldContract(name="intermediate", field_type=str)],
                    side_effects=[],
                    preconditions=[],
                    postconditions=[]
                )
            
            def validate_input(self, data: Dict[str, Any]) -> bool:
                return "input" in data
            
            def validate_output(self, result: Any) -> bool:
                """Validate that output meets contract requirements."""
                return isinstance(result, dict) and "intermediate" in result
            
            def execute(self, state: StateView) -> Dict[str, Any]:
                return {"intermediate": f"Processed: {state.get('input')}"}
        
        class MockEngine2(EngineInterface):
            def get_contract(self) -> EngineContract:
                return EngineContract(
                    inputs=[FieldContract(name="intermediate", field_type=str, required=True)],
                    outputs=[FieldContract(name="output", field_type=str)],
                    side_effects=[],
                    preconditions=[],
                    postconditions=[]
                )
            
            def validate_input(self, data: Dict[str, Any]) -> bool:
                return "intermediate" in data
            
            def validate_output(self, result: Any) -> bool:
                """Validate that output meets contract requirements."""
                return isinstance(result, dict) and "output" in result
            
            def execute(self, state: StateView) -> Dict[str, Any]:
                return {"output": f"Final: {state.get('intermediate')}"}
        
        # Create engines
        engine1 = MockEngine1()
        engine2 = MockEngine2()
        
        # Create execution functions
        def execute1(state_view: StateView) -> Dict[str, Any]:
            return engine1.execute(state_view)
        
        def execute2(state_view: StateView) -> Dict[str, Any]:
            return engine2.execute(state_view)
        
        node1 = ContractualNode(name="processor", contract=node1_contract, execute_fn=execute1)
        node2 = ContractualNode(name="finalizer", contract=node2_contract, execute_fn=execute2)
        
        # Register nodes
        orchestrator.register_node(node1)
        orchestrator.register_node(node2)
        
        # Execute would normally be done through orchestrator
        # but we're testing the structure here
        assert "processor" in orchestrator.components
        assert "finalizer" in orchestrator.components
        
        # Verify node contracts are enforced
        assert orchestrator.components["processor"].contract.inputs == ["input"]
        assert orchestrator.components["finalizer"].contract.outputs == ["output"]