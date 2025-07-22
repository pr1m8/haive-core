"""Tests for Dynamic Activation Pattern - Core Components.

This module tests the core components of the Dynamic Activation Pattern
using real components (no mocks) following the Haive testing philosophy.

Based on:
- @project_docs/active/patterns/dynamic_activation_pattern.md
- @project_docs/active/standards/testing/philosophy.md (no mocks)
- Real Azure OpenAI integration for testing
"""

import pytest
from langchain_core.tools import tool

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.registry import DynamicRegistry, RegistryItem
from haive.core.schema.prebuilt.dynamic_activation_state import DynamicActivationState
from haive.core.schema.prebuilt.meta_state import MetaStateSchema


# Real tools for testing (no mocks)
@tool
def calculator_tool(expression: str) -> float:
    """Calculate mathematical expression."""
    try:
        return eval(expression)
    except Exception:
        return 0.0


@tool
def text_processor_tool(text: str) -> str:
    """Process text by converting to uppercase."""
    return text.upper()


@tool
def word_counter_tool(text: str) -> int:
    """Count words in text."""
    return len(text.split())


class TestDynamicRegistry:
    """Test suite for DynamicRegistry with real components."""

    def test_registry_creation_with_type_safety(self):
        """Test creating registry with generic type safety."""
        # Test with MockTool type
        tool_registry = DynamicRegistry[MockTool]()

        assert tool_registry.items == {}
        assert tool_registry.active_items == set()
        assert tool_registry.max_active is None

        # Test with MockComponent type
        component_registry = DynamicRegistry[MockComponent]()

        assert component_registry.items == {}
        assert component_registry.active_items == set()
        assert component_registry.max_active is None

    def test_registry_item_registration(self):
        """Test registering items in registry."""
        registry = DynamicRegistry[MockTool]()

        # Create test tool
        tool = MockTool(name="calculator", description="Mathematical calculations")

        # Create registry item
        item = RegistryItem(
            id="calc_001",
            name="Calculator",
            description="Mathematical calculations",
            component=tool,
        )

        # Register item
        registry.register(item)

        # Verify registration
        assert "calc_001" in registry.items
        assert registry.items["calc_001"].component.name == "calculator"
        assert (
            registry.items["calc_001"].component.description
            == "Mathematical calculations"
        )

        # Verify not active yet
        assert "calc_001" not in registry.active_items

    def test_component_activation(self):
        """Test activating components in registry."""
        registry = DynamicRegistry[TestTool]()

        # Create and register multiple tools
        tools = [
            MockTool(name="calculator", description="Math calculations"),
            MockTool(name="search", description="Web search"),
            MockTool(name="file_reader", description="File operations"),
        ]

        for i, tool in enumerate(tools):
            item = RegistryItem(
                id=f"tool_{i:03d}",
                name=tool.name.title(),
                description=tool.description,
                component=tool,
            )
            registry.register(item)

        # Activate first tool
        success = registry.activate("tool_000")
        assert success is True
        assert "tool_000" in registry.active_items

        # Activate second tool
        success = registry.activate("tool_001")
        assert success is True
        assert "tool_001" in registry.active_items

        # Verify both are active
        assert len(registry.active_items) == 2
        active_items = registry.get_active_items()
        assert len(active_items) == 2

        # Verify correct components are active
        active_names = {item.component.name for item in active_items}
        assert active_names == {"calculator", "search"}

    def test_component_deactivation(self):
        """Test deactivating components."""
        registry = DynamicRegistry[TestTool]()

        # Create and register tool
        tool = TestTool(name="calculator", description="Math calculations")
        item = RegistryItem(
            id="calc_001",
            name="Calculator",
            description="Mathematical calculations",
            component=tool,
        )
        registry.register(item)

        # Activate tool
        registry.activate("calc_001")
        assert "calc_001" in registry.active_items

        # Deactivate tool
        success = registry.deactivate("calc_001")
        assert success is True
        assert "calc_001" not in registry.active_items

        # Verify deactivation
        active_items = registry.get_active_items()
        assert len(active_items) == 0

    def test_max_active_constraint(self):
        """Test maximum active items constraint."""
        registry = DynamicRegistry[TestTool](max_active=2)

        # Create and register three tools
        tools = [
            TestTool(name="calculator", description="Math calculations"),
            TestTool(name="search", description="Web search"),
            TestTool(name="file_reader", description="File operations"),
        ]

        for i, tool in enumerate(tools):
            item = RegistryItem(
                id=f"tool_{i:03d}",
                name=tool.name.title(),
                description=tool.description,
                component=tool,
            )
            registry.register(item)

        # Activate first two tools (should succeed)
        assert registry.activate("tool_000") is True
        assert registry.activate("tool_001") is True
        assert len(registry.active_items) == 2

        # Try to activate third tool (should fail due to max_active=2)
        assert registry.activate("tool_002") is False
        assert len(registry.active_items) == 2

        # Deactivate one tool and try again (should succeed)
        registry.deactivate("tool_000")
        assert registry.activate("tool_002") is True
        assert len(registry.active_items) == 2

    def test_registry_stats(self):
        """Test registry statistics."""
        registry = DynamicRegistry[TestTool]()

        # Create and register tools
        tools = [
            TestTool(name="calculator", description="Math calculations"),
            TestTool(name="search", description="Web search"),
            TestTool(name="file_reader", description="File operations"),
        ]

        for i, tool in enumerate(tools):
            item = RegistryItem(
                id=f"tool_{i:03d}",
                name=tool.name.title(),
                description=tool.description,
                component=tool,
            )
            registry.register(item)

        # Activate some tools
        registry.activate("tool_000")
        registry.activate("tool_001")

        # Get stats
        stats = registry.get_stats()

        assert stats["total_components"] == 3
        assert stats["active_components"] == 2
        assert stats["inactive_components"] == 1
        assert stats["max_active"] is None
        assert stats["activation_rate"] == 2 / 3


class TestDynamicActivationState:
    """Test suite for DynamicActivationState with real components."""

    def test_state_creation_with_registry(self):
        """Test creating state with embedded registry."""
        state = DynamicActivationState()

        assert isinstance(state.registry, DynamicRegistry)
        assert state.active_components == {}
        assert state.component_metadata == {}
        assert state.last_activation is None
        assert state.activation_history == []

    def test_component_activation_with_meta_state(self):
        """Test activating components with MetaStateSchema wrapping."""
        state = DynamicActivationState()

        # Create and register component
        component = TestComponent(name="processor", version="1.0")
        item = RegistryItem(
            id="proc_001",
            name="Processor",
            description="Data processor",
            component=component,
        )
        state.registry.register(item)

        # Activate component
        meta_state = state.activate_component("proc_001")

        # Verify activation
        assert meta_state is not None
        assert isinstance(meta_state, MetaStateSchema)
        assert meta_state.agent == component
        assert "proc_001" in state.active_components
        assert state.active_components["proc_001"] == meta_state

        # Verify metadata tracking
        assert "proc_001" in state.component_metadata
        assert state.component_metadata["proc_001"]["activated"] is True
        assert state.component_metadata["proc_001"]["name"] == "Processor"

        # Verify activation history
        assert len(state.activation_history) == 1
        assert state.activation_history[0]["component_id"] == "proc_001"
        assert state.activation_history[0]["action"] == "activate"

    def test_component_deactivation(self):
        """Test deactivating components."""
        state = DynamicActivationState()

        # Create and register component
        component = TestComponent(name="processor", version="1.0")
        item = RegistryItem(
            id="proc_001",
            name="Processor",
            description="Data processor",
            component=component,
        )
        state.registry.register(item)

        # Activate then deactivate
        state.activate_component("proc_001")
        success = state.deactivate_component("proc_001")

        # Verify deactivation
        assert success is True
        assert "proc_001" not in state.active_components
        assert state.component_metadata["proc_001"]["activated"] is False

        # Verify deactivation history
        assert len(state.activation_history) == 2
        assert state.activation_history[1]["component_id"] == "proc_001"
        assert state.activation_history[1]["action"] == "deactivate"

    def test_activation_stats(self):
        """Test getting activation statistics."""
        state = DynamicActivationState()

        # Create and register multiple components
        components = [
            TestComponent(name="processor_1", version="1.0"),
            TestComponent(name="processor_2", version="2.0"),
            TestComponent(name="analyzer", version="1.5"),
        ]

        for i, component in enumerate(components):
            item = RegistryItem(
                id=f"comp_{i:03d}",
                name=component.name.title(),
                description=f"Component {i}",
                component=component,
            )
            state.registry.register(item)

        # Activate some components
        state.activate_component("comp_000")
        state.activate_component("comp_001")

        # Get activation stats
        stats = state.get_activation_stats()

        assert stats["total_registered"] == 3
        assert stats["currently_active"] == 2
        assert stats["total_activations"] == 2
        assert stats["activation_rate"] == 2 / 3

        # Verify component list
        assert len(stats["active_components"]) == 2
        active_names = {comp["name"] for comp in stats["active_components"]}
        assert active_names == {"Processor_1", "Processor_2"}

    def test_component_execution_through_meta_state(self):
        """Test executing components through MetaStateSchema."""
        state = DynamicActivationState()

        # Create and register component
        component = TestComponent(name="processor", version="1.0")
        item = RegistryItem(
            id="proc_001",
            name="Processor",
            description="Data processor",
            component=component,
        )
        state.registry.register(item)

        # Activate component
        meta_state = state.activate_component("proc_001")

        # Execute through MetaStateSchema
        result = meta_state.execute_agent(
            input_data={"test": "data"}, update_state=True
        )

        # Verify execution
        assert result is not None
        assert meta_state.execution_status == "completed"
        assert meta_state.last_execution_time is not None


class TestRealComponentIntegration:
    """Integration tests with real AugLLMConfig components."""

    @pytest.fixture
    def aug_llm_config(self):
        """Create real AugLLMConfig for testing."""
        return AugLLMConfig(
            name="test_llm",
            temperature=0.1,  # Low temperature for consistent tests
            max_tokens=100,
            model="gpt-4o-mini",  # Use smaller model for faster tests
        )

    def test_registry_with_real_aug_llm_config(self, aug_llm_config):
        """Test registry with real AugLLMConfig components."""
        registry = DynamicRegistry[AugLLMConfig]()

        # Create registry item with real AugLLMConfig
        item = RegistryItem(
            id="llm_001",
            name="Test LLM",
            description="Test LLM configuration",
            component=aug_llm_config,
        )

        # Register and activate
        registry.register(item)
        success = registry.activate("llm_001")

        assert success is True
        assert "llm_001" in registry.active_items

        # Verify active component is real AugLLMConfig
        active_items = registry.get_active_items()
        assert len(active_items) == 1
        assert isinstance(active_items[0].component, AugLLMConfig)
        assert active_items[0].component.name == "test_llm"
        assert active_items[0].component.temperature == 0.1

    def test_dynamic_activation_with_real_components(self, aug_llm_config):
        """Test dynamic activation with real AugLLMConfig components."""
        state = DynamicActivationState()

        # Create registry item with real AugLLMConfig
        item = RegistryItem(
            id="llm_001",
            name="Test LLM",
            description="Test LLM configuration",
            component=aug_llm_config,
        )

        # Register and activate
        state.registry.register(item)
        meta_state = state.activate_component("llm_001")

        # Verify MetaStateSchema wrapping
        assert meta_state is not None
        assert isinstance(meta_state, MetaStateSchema)
        assert isinstance(meta_state.agent, AugLLMConfig)
        assert meta_state.agent.name == "test_llm"

        # Verify tracking
        assert meta_state.graph_context["agent_type"] == "AugLLMConfig"
        assert meta_state.graph_context["component_activation"] is True

    def test_multiple_real_components_activation(self, aug_llm_config):
        """Test activating multiple real components."""
        state = DynamicActivationState()

        # Create multiple real AugLLMConfig components
        configs = [
            AugLLMConfig(name="llm_1", temperature=0.1, model="gpt-4o-mini"),
            AugLLMConfig(name="llm_2", temperature=0.5, model="gpt-4o-mini"),
            AugLLMConfig(name="llm_3", temperature=0.9, model="gpt-4o-mini"),
        ]

        # Register all configs
        for i, config in enumerate(configs):
            item = RegistryItem(
                id=f"llm_{i:03d}",
                name=f"LLM {i}",
                description=f"LLM configuration {i}",
                component=config,
            )
            state.registry.register(item)

        # Activate all configs
        meta_states = []
        for i in range(3):
            meta_state = state.activate_component(f"llm_{i:03d}")
            meta_states.append(meta_state)

        # Verify all activations
        assert len(meta_states) == 3
        assert all(isinstance(ms, MetaStateSchema) for ms in meta_states)
        assert all(isinstance(ms.agent, AugLLMConfig) for ms in meta_states)

        # Verify different temperatures
        temps = [ms.agent.temperature for ms in meta_states]
        assert temps == [0.1, 0.5, 0.9]

        # Verify activation stats
        stats = state.get_activation_stats()
        assert stats["total_registered"] == 3
        assert stats["currently_active"] == 3
        assert stats["total_activations"] == 3
        assert stats["activation_rate"] == 1.0


class TestDynamicActivationPerformance:
    """Performance tests for dynamic activation pattern."""

    def test_large_registry_performance(self):
        """Test performance with large number of components."""
        import time

        registry = DynamicRegistry[TestComponent]()

        # Create large number of components
        num_components = 1000
        components = []

        start_time = time.time()

        for i in range(num_components):
            component = TestComponent(
                name=f"component_{i}", version="1.0", metadata={"index": i}
            )
            item = RegistryItem(
                id=f"comp_{i:04d}",
                name=f"Component {i}",
                description=f"Test component {i}",
                component=component,
            )
            registry.register(item)
            components.append(component)

        registration_time = time.time() - start_time

        # Test activation performance
        start_time = time.time()

        # Activate first 100 components
        for i in range(100):
            registry.activate(f"comp_{i:04d}")

        activation_time = time.time() - start_time

        # Verify performance benchmarks
        assert registration_time < 1.0  # Should register 1000 components in < 1 second
        assert activation_time < 0.5  # Should activate 100 components in < 0.5 seconds

        # Verify correctness
        assert len(registry.items) == num_components
        assert len(registry.active_items) == 100

        # Test stats performance
        start_time = time.time()
        stats = registry.get_stats()
        stats_time = time.time() - start_time

        assert stats_time < 0.1  # Stats should be fast
        assert stats["total_components"] == num_components
        assert stats["active_components"] == 100

    def test_activation_deactivation_cycles(self):
        """Test repeated activation/deactivation cycles."""
        state = DynamicActivationState()

        # Create components
        components = []
        for i in range(10):
            component = TestComponent(name=f"component_{i}", version="1.0")
            item = RegistryItem(
                id=f"comp_{i:03d}",
                name=f"Component {i}",
                description=f"Test component {i}",
                component=component,
            )
            state.registry.register(item)
            components.append(component)

        # Perform multiple activation/deactivation cycles
        for _cycle in range(100):
            # Activate random components
            for i in range(0, 10, 2):  # Even numbered components
                state.activate_component(f"comp_{i:03d}")

            # Deactivate them
            for i in range(0, 10, 2):
                state.deactivate_component(f"comp_{i:03d}")

        # Verify final state
        assert len(state.active_components) == 0
        assert (
            len(state.activation_history) == 2000
        )  # 100 cycles * 10 components * 2 actions

        # Verify stats
        stats = state.get_activation_stats()
        assert (
            stats["total_activations"] == 1000
        )  # 100 cycles * 5 components * 2 activations
        assert stats["currently_active"] == 0
