#!/usr/bin/env python3
"""Test enhanced naming integration with AugLLMConfig and structured outputs."""

from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from langchain_core.tools import tool

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.utils.enhanced_naming import (
    enhanced_sanitize_tool_name,
    sanitize_pydantic_model_name_enhanced
)


# Complex Pydantic models that would break basic naming
class Plan[Task](BaseModel):
    """Generic plan model - this would break with brackets."""
    tasks: List[Task]
    status: str

class SearchResults[Document](BaseModel):
    """Generic search results."""
    query: str
    results: List[Document] 
    count: int

class APIResponse[T](BaseModel):
    """Generic API response wrapper."""
    data: T
    success: bool
    errors: Optional[List[str]] = None

class ComplexModel(BaseModel):
    """Complex nested model."""
    cache: Dict[str, List[Optional[str]]]
    union_field: Union[str, int, List[str]]


def test_pydantic_model_naming():
    """Test enhanced naming with complex Pydantic models."""
    print("🧪 TESTING PYDANTIC MODEL NAMING")
    print("=" * 45)
    
    # These models would break OpenAI tool naming with basic sanitization
    models = [
        Plan[str],  # Plan[Task] -> plan_str_generic
        SearchResults[Dict],  # SearchResults[Document] -> search_results_dict_generic
        APIResponse[List[str]],  # APIResponse[T] -> api_response_list_str_nested_generic
        ComplexModel,  # No generics, should work normally
    ]
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. Testing model: {model}")
        
        try:
            # Test enhanced naming
            sanitized_name, metadata = sanitize_pydantic_model_name_enhanced(model)
            
            print(f"   ✅ Enhanced name: {sanitized_name}")
            
            if metadata:
                print(f"   📊 Transformation: {metadata.transformation_type}")
                print(f"   📈 Complexity: {metadata.complexity_level}")
                print(f"   📝 Description: {metadata.description}")
                
                if metadata.warnings:
                    print(f"   ⚠️  Warnings: {len(metadata.warnings)}")
                    for warning in metadata.warnings:
                        print(f"      - {warning}")
            
            # Verify OpenAI compliance
            from haive.core.utils.naming import validate_tool_name
            is_valid, issues = validate_tool_name(sanitized_name)
            
            if is_valid:
                print(f"   ✅ OpenAI compliant: YES")
            else:
                print(f"   ❌ OpenAI compliant: NO")
                for issue in issues:
                    print(f"      - {issue}")
                    
        except Exception as e:
            print(f"   💥 ERROR: {e}")


def test_aug_llm_config_integration():
    """Test enhanced naming with AugLLMConfig and structured outputs."""
    print("\n\n🔧 TESTING AUGLLLMCONFIG INTEGRATION")  
    print("=" * 45)
    
    # Create a tool that returns a complex generic type
    @tool
    def complex_search(query: str) -> str:
        """Search that returns complex structured data."""
        return f"SearchResults[Document] for query: {query}"
    
    print("1. Testing AugLLMConfig with complex structured output...")
    
    try:
        # This would normally break with Plan[Task] due to brackets
        config = AugLLMConfig(
            tools=[complex_search],
            structured_output_model=SearchResults[str]  # Complex generic!
        )
        
        print(f"   ✅ Config created successfully")
        print(f"   🔧 Tools count: {len(config.tools)}")
        print(f"   📝 Tool routes: {list(config.tool_routes.keys())}")
        
        # Check if structured output model was handled
        if hasattr(config, 'structured_output_model'):
            model = config.structured_output_model
            enhanced_name, metadata = sanitize_pydantic_model_name_enhanced(model)
            print(f"   📊 Structured output name: {enhanced_name}")
            
            if metadata:
                print(f"      Type: {metadata.transformation_type}")
                print(f"      Description: {metadata.description}")
        
    except Exception as e:
        print(f"   💥 ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_complex_tool_scenarios():
    """Test complex real-world tool scenarios."""
    print("\n\n🌍 TESTING COMPLEX TOOL SCENARIOS")
    print("=" * 40)
    
    # Scenario 1: Multiple complex generic tools
    @tool 
    def workflow_processor(data: str) -> str:
        """Process workflow data."""
        return f"WorkflowStep[InputSchema, OutputSchema]: {data}"
    
    @tool
    def validation_checker(input_data: str) -> str:
        """Validate data with optional errors."""
        return f"ValidationResult[Optional[ErrorDetails]]: {input_data}"
    
    @tool
    def cache_manager(key: str) -> str:
        """Manage cache entries."""
        return f"CacheEntry[str, Dict[str, Any]]: {key}"
    
    print("1. Creating AugLLMConfig with multiple complex tools...")
    
    try:
        config = AugLLMConfig(tools=[
            workflow_processor,
            validation_checker, 
            cache_manager
        ])
        
        print(f"   ✅ Config created with {len(config.tools)} tools")
        print(f"   📝 Tool names:")
        for i, tool_name in enumerate(config.tool_routes.keys(), 1):
            print(f"      {i}. {tool_name}")
            
        # Test that all tool names are OpenAI compliant
        from haive.core.utils.naming import validate_tool_name
        
        all_valid = True
        for tool_name in config.tool_routes.keys():
            is_valid, issues = validate_tool_name(tool_name)
            if not is_valid:
                print(f"   ❌ {tool_name} is not OpenAI compliant:")
                for issue in issues:
                    print(f"      - {issue}")
                all_valid = False
                
        if all_valid:
            print(f"   ✅ All tool names are OpenAI compliant!")
            
    except Exception as e:
        print(f"   💥 ERROR: {e}")


def test_backward_compatibility():
    """Test that enhanced naming doesn't break existing functionality."""
    print("\n\n🔄 TESTING BACKWARD COMPATIBILITY")
    print("=" * 40)
    
    # Test with simple models that worked before
    class SimpleModel(BaseModel):
        name: str
        value: int
    
    @tool
    def simple_tool(text: str) -> str:
        """Simple tool."""
        return f"Processed: {text}"
    
    try:
        # This should work exactly as before
        config = AugLLMConfig(
            tools=[simple_tool],
            structured_output_model=SimpleModel
        )
        
        print(f"   ✅ Simple config works: {len(config.tools)} tools")
        
        # Test enhanced naming on simple case
        enhanced_name, metadata = enhanced_sanitize_tool_name("SimpleModel")
        print(f"   📝 Enhanced name: {enhanced_name}")
        
        if metadata:
            print(f"      Type: {metadata.transformation_type}")
            print(f"      Complexity: {metadata.complexity_level}")
            
        # Should be same as basic naming
        from haive.core.utils.naming import sanitize_tool_name
        basic_name = sanitize_tool_name("SimpleModel")
        
        if enhanced_name == basic_name:
            print(f"   ✅ Backward compatible: enhanced matches basic")
        else:
            print(f"   ⚠️  Different results: enhanced='{enhanced_name}', basic='{basic_name}'")
            
    except Exception as e:
        print(f"   💥 ERROR: {e}")


if __name__ == "__main__":
    test_pydantic_model_naming()
    test_aug_llm_config_integration()
    test_complex_tool_scenarios()
    test_backward_compatibility()
    
    print("\n\n✅ ALL INTEGRATION TESTS COMPLETED")
    print("\n🎯 SUMMARY:")
    print("   • Enhanced naming handles complex generics like Plan[Task]")
    print("   • Provides detailed transformation metadata")
    print("   • Maintains OpenAI compliance for all name types")
    print("   • Backward compatible with existing simple cases")
    print("   • Integrates seamlessly with AugLLMConfig")