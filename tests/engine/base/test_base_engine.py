# tests/core/engine/test_engine_base.py

import pytest
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from haive_core.engine.base import Engine, EngineType, EngineRegistry, InvokableEngine
from langchain_core.runnables import RunnableConfig

# Mock models for testing
class TestInputModel(BaseModel):
    query: str
    param1: Optional[int] = None
    param2: Optional[str] = None

class TestOutputModel(BaseModel):
    result: str
    score: Optional[float] = None

# Mock engine implementation for testing
class MockEngine(InvokableEngine[TestInputModel, TestOutputModel]):
    engine_type: EngineType = EngineType.LLM
    model_name: str = "test-model"
    param_a: Optional[str] = None
    param_b: Optional[int] = None
    
    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> Any:
        # Extract config parameters
        params = self.apply_runnable_config(runnable_config)
        
        # Apply parameters if provided
        model = params.get("model_name", self.model_name)
        param_a = params.get("param_a", self.param_a)
        param_b = params.get("param_b", self.param_b)
        
        # Return a simple callable that simulates a runnable
        def mock_runnable(input_data):
            # Process the input
            query = input_data.query if isinstance(input_data, TestInputModel) else input_data.get("query", "")
            
            # Generate a result
            return TestOutputModel(
                result=f"Result for {query} using {model}, params: {param_a}, {param_b}",
                score=0.95
            )
        
        return mock_runnable

class TestEngine:
    """Test suite for the Engine base class."""
    
    def test_engine_id_generation(self):
        """Test that each engine gets a unique ID."""
        engine1 = MockEngine(name="engine1")
        engine2 = MockEngine(name="engine2")
        
        # Each engine should have a unique ID
        assert engine1.id != engine2.id
        assert isinstance(engine1.id, str)
        assert len(engine1.id) > 0
    
    def test_engine_custom_id(self):
        """Test setting a custom ID."""
        custom_id = str(uuid.uuid4())
        engine = MockEngine(id=custom_id, name="custom-id-engine")
        
        assert engine.id == custom_id
    
    def test_registry_find_by_id(self):
        """Test finding an engine by ID."""
        registry = EngineRegistry.get_instance()
        registry.clear()  # Start with a clean registry
        
        # Create and register an engine
        engine = MockEngine(name="test-engine")
        registry.register(engine)
        
        # Find by ID
        found_engine = registry.find_by_id(engine.id)
        assert found_engine is engine
        
        # Test with non-existent ID
        assert registry.find_by_id("non-existent-id") is None
    
    def test_engine_to_dict(self):
        """Test serializing an engine to a dictionary."""
        engine = MockEngine(
            name="test-engine", 
            model_name="gpt-4",
            param_a="test-value",
            param_b=42
        )
        
        # Convert to dict
        engine_dict = engine.to_dict()
        
        # Check core fields
        assert engine_dict["id"] == engine.id
        assert engine_dict["name"] == "test-engine"
        assert engine_dict["engine_type"] == EngineType.LLM
        assert engine_dict["model_name"] == "gpt-4"
        assert engine_dict["param_a"] == "test-value"
        assert engine_dict["param_b"] == 42
        
        # Check class information for reconstruction
        assert "engine_class" in engine_dict
        assert "MockEngine" in engine_dict["engine_class"]
    
    def test_engine_from_dict(self):
        """Test creating an engine from a dictionary."""
        # Create a dictionary representation
        engine_dict = {
            "id": str(uuid.uuid4()),
            "name": "reconstructed-engine",
            "engine_type": EngineType.LLM,
            "model_name": "gpt-3.5-turbo",
            "param_a": "reconstructed-value",
            "param_b": 99,
            "engine_class": f"{MockEngine.__module__}.{MockEngine.__name__}"
        }
        
        # Reconstruct engine
        engine = Engine.from_dict(engine_dict)
        
        # Check that it's the right type and has correct values
        assert isinstance(engine, MockEngine)
        assert engine.id == engine_dict["id"]
        assert engine.name == "reconstructed-engine"
        assert engine.engine_type == EngineType.LLM
        assert engine.model_name == "gpt-3.5-turbo"
        assert engine.param_a == "reconstructed-value"
        assert engine.param_b == 99
    
    def test_apply_runnable_config_by_id(self):
        """Test applying config using engine ID."""
        engine = MockEngine(
            name="test-engine", 
            model_name="default-model"
        )
        
        # Create a config targeting by ID
        config = {
            "configurable": {
                "engine_configs": {
                    engine.id: {
                        "model_name": "id-targeted-model",
                        "param_a": "id-value"
                    }
                }
            }
        }
        
        # Apply the config
        params = engine.apply_runnable_config(config)
        
        # Check extracted parameters
        assert params["model_name"] == "id-targeted-model"
        assert params["param_a"] == "id-value"
    
    def test_apply_runnable_config_by_name(self):
        """Test applying config using engine name."""
        engine = MockEngine(
            name="test-engine", 
            model_name="default-model"
        )
        
        # Create a config targeting by name
        config = {
            "configurable": {
                "engine_configs": {
                    "test-engine": {
                        "model_name": "name-targeted-model",
                        "param_a": "name-value"
                    }
                }
            }
        }
        
        # Apply the config
        params = engine.apply_runnable_config(config)
        
        # Check extracted parameters
        assert params["model_name"] == "name-targeted-model"
        assert params["param_a"] == "name-value"
    
    def test_apply_runnable_config_by_type(self):
        """Test applying config using engine type."""
        engine = MockEngine(
            name="test-engine", 
            model_name="default-model"
        )
        
        # Create a config targeting by type
        config = {
            "configurable": {
                "engine_configs": {
                    "llm_config": {
                        "model_name": "type-targeted-model",
                        "param_a": "type-value"
                    }
                }
            }
        }
        
        # Apply the config
        params = engine.apply_runnable_config(config)
        
        # Check extracted parameters
        assert params["model_name"] == "type-targeted-model"
        assert params["param_a"] == "type-value"
    
    def test_apply_runnable_config_priority(self):
        """Test priority order when applying config (ID > name > type)."""
        engine = MockEngine(
            name="test-engine", 
            model_name="default-model"
        )
        
        # Create a config with all three targets
        config = {
            "configurable": {
                "engine_configs": {
                    engine.id: {
                        "model_name": "id-targeted-model",
                        "param_a": "id-value",
                        "param_b": 1
                    },
                    "test-engine": {
                        "model_name": "name-targeted-model",
                        "param_a": "name-value",
                        "param_b": 2
                    },
                    "llm_config": {
                        "model_name": "type-targeted-model",
                        "param_a": "type-value",
                        "param_b": 3
                    }
                }
            }
        }
        
        # Apply the config
        params = engine.apply_runnable_config(config)
        
        # ID should take priority
        assert params["model_name"] == "id-targeted-model"
        assert params["param_a"] == "id-value"
        assert params["param_b"] == 1
    
    def test_engine_invoke(self):
        """Test invoking an engine."""
        engine = MockEngine(
            name="test-engine", 
            model_name="test-model",
            param_a="test-a",
            param_b=123
        )
        
        # Create input
        input_data = TestInputModel(query="test query")
        
        # Invoke the engine
        result = engine.invoke(input_data)
        
        # Check result
        assert isinstance(result, TestOutputModel)
        assert "test query" in result.result
        assert "test-model" in result.result
        assert "test-a" in result.result
        assert "123" in result.result
    
    def test_engine_invoke_with_config(self):
        """Test invoking an engine with runtime config."""
        engine = MockEngine(
            name="test-engine", 
            model_name="default-model"
        )
        
        # Create config
        config = {
            "configurable": {
                "engine_configs": {
                    engine.id: {
                        "model_name": "runtime-model",
                        "param_a": "runtime-value",
                        "param_b": 456
                    }
                }
            }
        }
        
        # Create input
        input_data = TestInputModel(query="config test")
        
        # Invoke with config
        result = engine.invoke(input_data, config)
        
        # Check result reflects runtime config
        assert isinstance(result, TestOutputModel)
        assert "config test" in result.result
        assert "runtime-model" in result.result
        assert "runtime-value" in result.result
        assert "456" in result.result
    
    def test_to_runnable_config(self):
        """Test converting an engine to a runnable config."""
        engine = MockEngine(
            name="test-engine", 
            model_name="test-model",
            param_a="test-a",
            param_b=123
        )
        
        # Convert to runnable config
        config = engine.to_runnable_config(thread_id="test-thread")
        
        # Check config structure
        assert "configurable" in config
        assert "thread_id" in config["configurable"]
        assert config["configurable"]["thread_id"] == "test-thread"
        
        # Check engine configs
        assert "engine_configs" in config["configurable"]
        
        # Check ID-based config
        assert engine.id in config["configurable"]["engine_configs"]
        assert config["configurable"]["engine_configs"][engine.id]["model_name"] == "test-model"
        assert config["configurable"]["engine_configs"][engine.id]["param_a"] == "test-a"
        assert config["configurable"]["engine_configs"][engine.id]["param_b"] == 123
        
        # Check name-based config
        assert "test-engine" in config["configurable"]["engine_configs"]
        
        # Check type-based config
        assert "llm_config" in config["configurable"]["engine_configs"]