import pytest
import logging
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.prompts.chat import ChatPromptValue

from haive_core.engine.aug_llm import AugLLMConfig, compose_runnable
from haive_core.models.llm.base import AzureLLMConfig

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DummyOutput(BaseModel):
    answer: str


def test_basic_init():
    config = AugLLMConfig(llm_config=AzureLLMConfig(model="gpt-4o"))
    logger.info(f"Basic Init: {config.llm_config}")
    print("✅ test_basic_init:", config.llm_config)
    assert config.llm_config.model == "gpt-4o"
    assert config.engine_type.value == "llm"


def test_prompt_template_input_schema():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are helpful"),
        MessagesPlaceholder(variable_name="messages")
    ])
    config = AugLLMConfig.from_prompt(prompt)
    schema = config.derive_input_schema()
    logger.info(f"Derived input schema fields: {list(schema.model_fields)}")
    print("✅ test_prompt_template_input_schema:", list(schema.model_fields))
    assert "messages" in schema.model_fields


def test_output_schema_with_structured_output():
    config = AugLLMConfig(
        llm_config=AzureLLMConfig(model="gpt-4o"),
        structured_output_model=DummyOutput
    )
    schema = config.derive_output_schema()
    logger.info(f"Derived output schema fields: {list(schema.model_fields)}")
    print("✅ test_output_schema_with_structured_output:", list(schema.model_fields))
    assert "answer" in schema.model_fields or "dummyoutput" in schema.model_fields


def test_process_input_string():
    config = AugLLMConfig(llm_config=AzureLLMConfig(model="gpt-4o"))
    result = config._process_input("Hello!")
    logger.info(f"Processed input string to message: {result}")
    print("✅ test_process_input_string:", result)
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "Hello!"


def test_invoke_runs(monkeypatch):
    def mock_llm(input):
        if isinstance(input, ChatPromptValue):
            messages = input.to_messages()
        elif isinstance(input, dict) and "messages" in input:
            messages = input["messages"]
        else:
            messages = [str(input)]
        return {"messages": messages}

    monkeypatch.setattr(
        "src.haive.core.models.llm.base.AzureLLMConfig.instantiate_llm",
        lambda self: RunnableLambda(mock_llm)
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant."),
        MessagesPlaceholder(variable_name="messages")
    ])
    config = AugLLMConfig(
        llm_config=AzureLLMConfig(model="gpt-4o"),
        prompt_template=prompt
    )

    result = config.invoke("What's the weather like?")
    logger.info(f"Invocation result: {result}")
    print("✅ test_invoke_runs:")
    print(result)
    print(type(result))
    #for m in result["messages"]:
    #@    #print(f" - {m.type}: {m.content}")
    
    assert isinstance(result, AIMessage)
    assert "content" in result.model_dump()
    # assert any(isinstance(m, HumanMessage) for m in result["messages"])

def test_compose_runnable_creates_chain(monkeypatch):
    monkeypatch.setattr(
        "src.haive.core.models.llm.base.AzureLLMConfig.instantiate_llm",
        lambda self: RunnableLambda(lambda x: x)
    )
    config = AugLLMConfig(llm_config=AzureLLMConfig(model="gpt-4o"))
    chain = compose_runnable(config)
    logger.info(f"Composed chain: {chain}")
    print("✅ test_compose_runnable_creates_chain:", chain)
    assert hasattr(chain, "invoke")


def test_apply_runnable_config_overrides(monkeypatch):
    monkeypatch.setattr(
        "src.haive.core.models.llm.base.AzureLLMConfig.instantiate_llm",
        lambda self: RunnableLambda(lambda x: x)
    )
    config = AugLLMConfig(llm_config=AzureLLMConfig(model="gpt-4o"))
    rcfg = RunnableConfig(configurable={"temperature": 0.1})
    resolved = config.apply_runnable_config(rcfg)
    logger.info(f"Resolved config overrides: {resolved}")
    print("✅ test_apply_runnable_config_overrides:", resolved)
    assert resolved.get("temperature") == 0.1



def test_invoke_runs_real_llm():
    from haive_core.models.llm.base import AzureLLMConfig

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant."),
        MessagesPlaceholder(variable_name="messages")
    ])

    config = AugLLMConfig(
        llm_config=AzureLLMConfig(model="gpt-4o"),
        prompt_template=prompt
    )

    # Actually invoke the real LLM
    result = config.invoke("What is the capital of France?")

    logger.info(f"🔮 Real LLM result: {result}")
    print("\n🔮 test_invoke_runs_real_llm output:")

    # Show the content based on actual return type
    if hasattr(result, "content"):
        print(f" - ai: {result.content}")
        assert "Paris" in result.content
    elif isinstance(result, dict) and "messages" in result:
        for m in result["messages"]:
            print(f" - {m.type}: {m.content}")
        assert any("Paris" in m.content for m in result["messages"])
    else:
        raise TypeError(f"Unexpected result format: {type(result)}")
