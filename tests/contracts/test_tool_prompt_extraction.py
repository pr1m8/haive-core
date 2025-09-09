"""Tests for tool and prompt extraction components.

Tests the new ToolConfig, PromptConfig, ToolRegistry, and PromptLibrary
components that extract functionality from AugLLMConfig.
"""

import pytest
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate as LangchainPromptTemplate
from pydantic import BaseModel

from haive.core.contracts.tool_config import (
    ToolConfig,
    ToolContract,
    ToolCapability
)
from haive.core.contracts.prompt_config import (
    PromptConfig,
    PromptContract,
    PromptVariable
)
from haive.core.contracts.tool_registry import ToolRegistry
from haive.core.contracts.prompt_library import PromptLibrary


# Test tools
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return str(eval(expression))


@tool
def word_counter(text: str) -> int:
    """Count words in text."""
    return len(text.split())


class DataProcessor(BaseModel):
    """Process data with configuration."""
    prefix: str = "Result"
    
    def __call__(self, data: str) -> str:
        """Process the data."""
        return f"{self.prefix}: {data}"


class TestToolConfig:
    """Test ToolConfig functionality."""
    
    def test_tool_config_creation(self):
        """Test creating a tool configuration."""
        config = ToolConfig(
            tools=[calculator, word_counter],
            routing_strategy="capability",
            force_tool_use=True
        )
        
        assert len(config.tools) == 2
        assert config.routing_strategy == "capability"
        assert config.force_tool_use is True
    
    def test_add_tool_with_contract(self):
        """Test adding a tool with a contract."""
        config = ToolConfig()
        
        contract = ToolContract(
            name="calculator",
            description="Math calculator",
            capabilities=ToolCapability(
                can_read_state=False,
                computational_cost="low"
            )
        )
        
        config.add_tool(calculator, contract)
        
        assert calculator in config.tools
        assert "calculator" in config.contracts
        assert config.contracts["calculator"].capabilities.computational_cost == "low"
    
    def test_get_tools_by_capability(self):
        """Test finding tools by capability."""
        config = ToolConfig()
        
        # Add tools with different capabilities
        safe_contract = ToolContract(
            name="calculator",
            description="Safe calculator",
            capabilities=ToolCapability(can_write_state=False)
        )
        
        # Use the actual class name for BaseModel classes
        unsafe_contract = ToolContract(
            name="DataProcessor",
            description="Data processor",
            capabilities=ToolCapability(can_write_state=True)
        )
        
        config.add_tool(calculator, safe_contract)
        config.add_tool(DataProcessor, unsafe_contract)
        
        # Find safe tools
        safe_tools = config.get_tools_by_capability("can_write_state", False)
        assert len(safe_tools) == 1
        assert calculator in safe_tools
        
        # Find unsafe tools
        unsafe_tools = config.get_tools_by_capability("can_write_state", True)
        assert len(unsafe_tools) == 1
        assert DataProcessor in unsafe_tools
    
    def test_validate_permissions(self):
        """Test permission validation."""
        config = ToolConfig()
        
        contract = ToolContract(
            name="processor",
            description="Requires permissions",
            required_permissions={"read_files", "write_files"}
        )
        
        config.add_tool(DataProcessor, contract)
        
        # Test with sufficient permissions
        assert config.validate_permissions(
            DataProcessor,
            {"read_files", "write_files", "execute"}
        ) is True
        
        # Test with insufficient permissions
        assert config.validate_permissions(
            DataProcessor,
            {"read_files"}
        ) is False


class TestPromptConfig:
    """Test PromptConfig functionality."""
    
    def test_prompt_config_creation(self):
        """Test creating a prompt configuration."""
        template = ChatPromptTemplate.from_template("Hello {name}")
        
        config = PromptConfig(
            prompt_template=template,
            system_message="You are a helpful assistant",
            include_examples=True
        )
        
        assert config.prompt_template == template
        assert config.system_message == "You are a helpful assistant"
        assert config.include_examples is True
    
    def test_prompt_with_contract(self):
        """Test prompt with contract and variables."""
        template = ChatPromptTemplate.from_template("Analyze {data} for {purpose}")
        
        contract = PromptContract(
            name="analysis",
            description="Data analysis prompt",
            variables=[
                PromptVariable(name="data", type="object", required=True),
                PromptVariable(name="purpose", type="string", required=True)
            ],
            output_format="json"
        )
        
        config = PromptConfig(
            prompt_template=template,
            contracts={"analysis": contract}
        )
        
        # Test validation
        errors = config.validate_variables({"data": {"value": 1}})
        assert "purpose" in errors  # Missing required variable
        
        errors = config.validate_variables({"data": {"value": 1}, "purpose": "testing"})
        assert len(errors) == 0  # All required variables provided
    
    def test_prompt_composition(self):
        """Test composing prompt configurations."""
        config1 = PromptConfig(
            prompt_template=ChatPromptTemplate.from_template("First: {input}"),
            system_message="System 1"
        )
        
        config2 = PromptConfig(
            prompt_template=ChatPromptTemplate.from_template("Second: {output}"),
            system_message="System 2"
        )
        
        # Compose configs
        composed = config1.compose_with(config2, mode="after")
        
        assert composed.system_message == "System 2"  # Second overrides
        assert len(composed.fallback_prompts) == 0
    
    def test_partial_variables(self):
        """Test applying partial variables."""
        template = LangchainPromptTemplate.from_template(
            "Process {data} with format: {format_instructions}"
        )
        
        config = PromptConfig(
            prompt_template=template,
            partial_variables={"format_instructions": "JSON output required"}
        )
        
        config.apply_partial_variables()
        
        # Template should now have format_instructions filled
        assert "format_instructions" in config.partial_variables


class TestToolRegistry:
    """Test ToolRegistry functionality."""
    
    def test_registry_creation(self):
        """Test creating a tool registry."""
        registry = ToolRegistry()
        
        assert len(registry.tools) == 0
        assert len(registry.metadata) == 0
    
    def test_register_tool(self):
        """Test registering tools."""
        registry = ToolRegistry()
        
        contract = ToolContract(
            name="calculator",
            description="Math tool",
            capabilities=ToolCapability(
                computational_cost="low",
                is_async=False
            )
        )
        
        registry.register("calculator", calculator, contract, tags={"math", "utility"})
        
        assert "calculator" in registry.tools
        assert "calculator" in registry.metadata
        assert "math" in registry.tag_index
        assert "utility" in registry.tag_index
    
    def test_find_by_capability(self):
        """Test finding tools by capability."""
        registry = ToolRegistry()
        
        # Register async tool
        async_contract = ToolContract(
            name="async_tool",
            description="Async tool",
            capabilities=ToolCapability(is_async=True)
        )
        
        # Register sync tool
        sync_contract = ToolContract(
            name="sync_tool",
            description="Sync tool",
            capabilities=ToolCapability(is_async=False)
        )
        
        registry.register("async", calculator, async_contract)
        registry.register("sync", word_counter, sync_contract)
        
        # Find async tools
        async_tools = registry.find_by_capability("is_async", True)
        assert len(async_tools) == 1
        
        # Find sync tools
        sync_tools = registry.find_by_capability("is_async", False)
        assert len(sync_tools) == 1
    
    def test_permission_validation(self):
        """Test permission validation in registry."""
        registry = ToolRegistry()
        
        contract = ToolContract(
            name="secure_tool",
            description="Requires permissions",
            required_permissions={"admin", "write"}
        )
        
        registry.register("secure", calculator, contract)
        
        # Test with sufficient permissions
        valid, missing = registry.validate_permissions(
            "secure",
            {"admin", "write", "read"}
        )
        assert valid is True
        assert len(missing) == 0
        
        # Test with insufficient permissions
        valid, missing = registry.validate_permissions(
            "secure",
            {"read"}
        )
        assert valid is False
        assert "admin" in missing
        assert "write" in missing
    
    def test_usage_tracking(self):
        """Test usage tracking."""
        registry = ToolRegistry()
        registry.register("calc", calculator)
        
        # Track usage
        registry.track_usage("calc", execution_time=0.5)
        registry.track_usage("calc", execution_time=0.3)
        
        stats = registry.get_usage_stats()
        assert stats["calc"]["usage_count"] == 2
        assert "avg_execution_time" in stats["calc"]["performance"]
        assert stats["calc"]["performance"]["avg_execution_time"] == pytest.approx(0.4)


class TestPromptLibrary:
    """Test PromptLibrary functionality."""
    
    def test_library_creation(self):
        """Test creating a prompt library."""
        library = PromptLibrary()
        
        assert len(library.templates) == 0
        assert len(library.categories) == 0
    
    def test_add_template(self):
        """Test adding templates to library."""
        library = PromptLibrary()
        
        template = ChatPromptTemplate.from_template("Hello {name}")
        contract = PromptContract(
            name="greeting",
            description="Greeting prompt",
            variables=[
                PromptVariable(name="name", type="string")
            ]
        )
        
        library.add_template(
            "greeting",
            template,
            contract,
            version="1.0.0",
            tags={"basic", "greeting"},
            category="greetings"
        )
        
        assert "greeting:1.0.0" in library.templates
        assert library.latest_versions["greeting"] == "1.0.0"
        assert "greetings" in library.categories
    
    def test_get_template_versions(self):
        """Test getting template versions."""
        library = PromptLibrary()
        
        # Add multiple versions
        template_v1 = ChatPromptTemplate.from_template("Hello {name}")
        template_v2 = ChatPromptTemplate.from_template("Hi {name}!")
        
        contract = PromptContract(name="greeting", description="Greeting")
        
        library.add_template("greeting", template_v1, contract, version="1.0.0")
        library.add_template("greeting", template_v2, contract, version="2.0.0")
        
        # Get specific version
        v1 = library.get_template("greeting", "1.0.0")
        assert v1 is not None
        
        # Get latest version
        latest = library.get_latest("greeting")
        assert latest is not None
        assert library.latest_versions["greeting"] == "2.0.0"
    
    def test_fork_template(self):
        """Test forking templates."""
        library = PromptLibrary()
        
        # Add original template
        original = ChatPromptTemplate.from_template("Original: {input}")
        contract = PromptContract(name="original", description="Original template")
        
        library.add_template("original", original, contract, version="1.0.0")
        
        # Fork template
        library.fork_template(
            "original",
            "forked",
            new_version="1.0.0"
        )
        
        assert "forked:1.0.0" in library.templates
        assert library.templates["forked:1.0.0"].parent_version == "original:1.0.0"
    
    def test_compose_templates(self):
        """Test composing templates."""
        library = PromptLibrary()
        
        # Add templates to compose
        t1 = ChatPromptTemplate.from_template("First: {input}")
        t2 = ChatPromptTemplate.from_template("Second: {middle}")
        
        c1 = PromptContract(name="first", description="First")
        c2 = PromptContract(name="second", description="Second")
        
        library.add_template("first", t1, c1)
        library.add_template("second", t2, c2)
        
        # Compose templates
        composed = library.compose_templates(
            ["first", "second"],
            "composed",
            mode="sequential"
        )
        
        assert composed is not None
        assert "composed:1.0.0" in library.templates
        assert library.composition_rules["composed"] == ["first", "second"]
    
    def test_evolution_history(self):
        """Test getting template evolution history."""
        library = PromptLibrary()
        
        contract = PromptContract(name="evolving", description="Evolving template")
        
        # Add multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0", "1.2.0"]:
            template = ChatPromptTemplate.from_template(f"Version {version}: {{input}}")
            library.add_template("evolving", template, contract, version=version)
        
        # Get evolution history
        history = library.get_evolution_history("evolving")
        
        # Should be sorted by version
        assert history == ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]