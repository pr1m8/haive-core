"""
Configuration file for pytest to properly handle imports and logging.
Save as tests/conftest.py
"""

import os
import sys
import logging
import pytest
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Tuple
import uuid
from datetime import datetime
from pydantic import Field
from langchain_core.runnables import RunnableConfig

# Ensure imports use the correct package structure if pytest.ini sets pythonpath="."
from haive_core.engine.base import Engine, InvokableEngine, NonInvokableEngine, EngineType
from haive_core.engine.aug_llm import AugLLMConfig
from haive_core.engine.retriever import RetrieverConfig, RetrieverType
from haive_core.engine.vectorstore import VectorStoreConfig, VectorStoreProvider
from haive_core.engine.embeddings import EmbeddingsEngineConfig
from haive_core.models.embeddings.base import HuggingFaceEmbeddingConfig
from haive_core.models.llm.base import AzureLLMConfig, OpenAILLMConfig
# Get a logger for conftest setup messages
conftest_logger = logging.getLogger("conftest")
conftest_logger.setLevel(logging.DEBUG) # Ensure conftest logs are at DEBUG level

# --------------------------------------------------------------------
# ✅ Add the project root to sys.path so imports work across project
# --------------------------------------------------------------------
def pytest_configure(config):
    """Ensure project root is in sys.path for proper imports."""
    conftest_logger.debug("--- Running pytest_configure --- ")
    # Adjust path finding relative to conftest.py location
    # Assuming conftest is in packages/haive-core/tests/
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
        conftest_logger.debug(f"✅ Added package root to sys.path: {package_root}")
    else:
        conftest_logger.debug(f"Package root already in sys.path: {package_root}")
    conftest_logger.debug(f"Current sys.path: {sys.path}")
    conftest_logger.debug("--- Finished pytest_configure ---")

# --------------------------------------------------------------------
# ✅ Dynamic per-test log file creation (mirroring test structure)
# --------------------------------------------------------------------
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Set up logging for each test with file structure mirroring test modules."""
    # Get the run directory from session
    run_dir = item.session.run_dir if hasattr(item.session, 'run_dir') else Path("logs/fallback")
    
    # Get relative path of test file
    item_path = Path(item.fspath).resolve()
    try:
        cwd = Path.cwd()
        rel_test_path = item_path.relative_to(cwd)
        
        # Create log directory mirroring test structure
        log_dir = run_dir / "tests" / rel_test_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file path
        log_file_path = log_dir / f"{rel_test_path.stem}.log"
        test_name = item.name
        log_file_path = log_dir / f"{rel_test_path.stem}.{test_name}.log"
        #setup_logger.debug(f"Calculated relative test path: {rel_test_path}")
        #setup_logger.debug(f"Target log file path: {log_file_path}")
    except ValueError as e:
        #setup_logger.warning(f"Could not determine relative path for logging item {item.nodeid}: {e}. Using default log path.")
        # Fallback path if relative path calculation fails
        log_file_path = Path("logs/tests/pytest_fallback.log")
        
        #setup_logger.debug(f"Using fallback log path: {log_file_path}")
        
    # Create directory if it doesn't exist
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        #setup_logger.debug(f"Ensured log directory exists: {log_file_path.parent}")
    except OSError as e:
        #setup_logger.error(f"Failed to create log directory {log_file_path.parent}: {e}")
        # Fallback to logging only to console if directory fails
        log_file_path = None 
        
    # Configure root logger
    root_logger = logging.getLogger() # Get the root logger
    #setup_logger.debug(f"Current root logger handlers before clearing: {root_logger.handlers}")
    
    # Clear existing handlers from the root logger ONLY
    # Avoid clearing handlers from other loggers like 'conftest'
    for handler in list(root_logger.handlers): # Iterate over a copy
        #setup_logger.debug(f"Removing handler from root logger: {handler}")
        root_logger.removeHandler(handler)
        handler.close() # Close handler to release file locks if any
        
    #setup_logger.debug(f"Root logger handlers after clearing: {root_logger.handlers}")

    # Common formatter
    log_format = "%(asctime)s [%(levelname)-8s] %(name)-25s: %(message)s"
    date_format = "%H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)
    #setup_logger.debug("Created log formatter.")

    # Set up file handler if path is valid
    if log_file_path:
        try:
            file_handler = logging.FileHandler(log_file_path, mode="w", encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            #setup_logger.debug(f"Added FileHandler: {log_file_path}")
        except Exception as e:
            #setup_logger.error(f"Failed to create or add FileHandler {log_file_path}: {e}")
            pass

    # Set up console handler (always add this)
    stream_handler = logging.StreamHandler(sys.stdout) # Log to stdout
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    setup_logger.debug("Added StreamHandler (console).")
    
    # Set root logger level (controls handlers unless they have stricter levels)
    root_logger.setLevel(logging.DEBUG)
    setup_logger.debug(f"Set root logger level to DEBUG.")
    setup_logger.debug(f"Final root logger handlers: {root_logger.handlers}")
    setup_logger.debug("--- Finished pytest_runtest_setup ---")

# Helper function for consistent naming
def generate_test_id(prefix: str) -> str:
    test_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    conftest_logger.debug(f"Generated test ID: {test_id}")
    return test_id

def pytest_sessionstart(session):
    """Set up run-specific logging directory at start of test session."""
    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up the run directory
    cwd = Path.cwd()
    run_dir = cwd / "logs" / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tests directory within run
    test_dir = run_dir / "tests"
    test_dir.mkdir(exist_ok=True)
    
    # Store the run directory in session for later use
    session.run_dir = run_dir
    
    # Update the latest symlink
    latest_link = cwd / "logs" / "latest"
    try:
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Symlinks might not work on all platforms
        pass


# --------------------------------------------------------------------
# ✅ Test Engine Classes (Simplified / Mock Implementations)
# --------------------------------------------------------------------
conftest_logger.debug("Defining Mock Engine Classes")

class MockEngine(Engine):
    """Mock engine for testing with custom ID."""
    engine_type: EngineType = EngineType.LLM
    id: str = Field(default_factory=lambda: generate_test_id("mock-engine"))
    name: str = Field(default_factory=lambda: f"mock_engine_{uuid.uuid4().hex[:4]}")

    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> Any:
        conftest_logger.debug(f"MockEngine '{self.name}' create_runnable called. Config: {runnable_config}")
        return lambda x: x # Simple pass-through runnable

class MockInvokableEngine(InvokableEngine):
    """Mock invokable engine for testing invoke/ainvoke."""
    engine_type: EngineType = EngineType.LLM
    id: str = Field(default_factory=lambda: generate_test_id("mock-invokable"))
    name: str = Field(default_factory=lambda: f"mock_invokable_engine_{uuid.uuid4().hex[:4]}")

    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> Any:
        conftest_logger.debug(f"MockInvokableEngine '{self.name}' create_runnable called. Config: {runnable_config}")
        return self # Runnable is the engine itself for testing

    def invoke(self, input_data: Any, runnable_config: Optional[RunnableConfig] = None) -> Any:
        conftest_logger.debug(f"MockInvokableEngine '{self.name}' invoke called. Input: {input_data}, Config: {runnable_config}")
        # Return input data plus a marker
        if isinstance(input_data, dict):
            result = {**input_data, "invoked_by": self.name}
        else:
            result = {"result": input_data, "invoked_by": self.name}
        conftest_logger.debug(f"MockInvokableEngine '{self.name}' invoke returning: {result}")
        return result

    async def ainvoke(self, input_data: Any, runnable_config: Optional[RunnableConfig] = None) -> Any:
        conftest_logger.debug(f"MockInvokableEngine '{self.name}' ainvoke called. Input: {input_data}, Config: {runnable_config}")
        # Async version of invoke
        result = self.invoke(input_data, runnable_config)
        conftest_logger.debug(f"MockInvokableEngine '{self.name}' ainvoke returning: {result}")
        return result

class MockNonInvokableEngine(NonInvokableEngine):
    """Mock non-invokable engine for testing instantiation."""
    engine_type: EngineType = EngineType.EMBEDDINGS
    id: str = Field(default_factory=lambda: generate_test_id("mock-non-invokable"))
    name: str = Field(default_factory=lambda: f"mock_non_invokable_engine_{uuid.uuid4().hex[:4]}")

    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> Any:
        conftest_logger.debug(f"MockNonInvokableEngine '{self.name}' create_runnable called. Config: {runnable_config}")
        # Return a simple dictionary indicating creation
        result = {"instance_created_by": self.name}
        conftest_logger.debug(f"MockNonInvokableEngine '{self.name}' create_runnable returning: {result}")
        return result

# --------------------------------------------------------------------
# ✅ Mock Engine Fixtures
# --------------------------------------------------------------------
conftest_logger.debug("Defining Mock Engine Fixtures")

@pytest.fixture
def mock_engine() -> MockEngine:
    """Provides a basic mock engine instance."""
    conftest_logger.debug("Creating mock_engine fixture")
    instance = MockEngine()
    conftest_logger.debug(f"Created mock_engine instance: {instance.id} / {instance.name}")
    return instance

@pytest.fixture
def mock_invokable_engine() -> MockInvokableEngine:
    """Provides a mock invokable engine instance."""
    conftest_logger.debug("Creating mock_invokable_engine fixture")
    instance = MockInvokableEngine()
    conftest_logger.debug(f"Created mock_invokable_engine instance: {instance.id} / {instance.name}")
    return instance

@pytest.fixture
def mock_non_invokable_engine() -> MockNonInvokableEngine:
    """Provides a mock non-invokable engine instance."""
    conftest_logger.debug("Creating mock_non_invokable_engine fixture")
    instance = MockNonInvokableEngine()
    conftest_logger.debug(f"Created mock_non_invokable_engine instance: {instance.id} / {instance.name}")
    return instance

# --------------------------------------------------------------------
# ✅ Real Engine Fixtures (Using Actual Config Classes)
# --------------------------------------------------------------------
conftest_logger.debug("Defining Real Engine Fixtures")
# These use the actual config classes but might need credentials/setup
# to fully instantiate runnables in real tests.

@pytest.fixture
def real_llm_engine():
    """Create a real LLM engine config for testing."""
    conftest_logger.debug("Creating real_llm_engine fixture (AugLLMConfig)")
    instance = AugLLMConfig(
        id=generate_test_id("real-llm"),
        name=f"real_llm_{uuid.uuid4().hex[:4]}",
        engine_type=EngineType.LLM,
        model="gpt-4o", # Using a default model, adjust if needed
        temperature=0.7,
        description="Real LLM Config for Testing (AugLLM)"
    )
    conftest_logger.debug(f"Created real_llm_engine instance: {instance.id} / {instance.name}")
    return instance

@pytest.fixture
def real_aug_llm_engine() -> AugLLMConfig:
    """Provides a real AugLLM engine config instance with base LLM."""
    conftest_logger.debug("Creating real_aug_llm_engine fixture")
    # AugLLM often wraps another LLM config
    base_llm = AzureLLMConfig(
        id=generate_test_id("aug-base-llm"),
        name=f"aug_base_llm_{uuid.uuid4().hex[:4]}",
        model="gpt-4o-mini",
        api_key="sk-test-key-for-tests", # Placeholder API key
        temperature=0.1
    )
    conftest_logger.debug(f"Created base LLM for AugLLM: {base_llm.id} / {base_llm.name}")
    instance = AugLLMConfig(
        id=generate_test_id("real-aug-llm"),
        name=f"real_aug_llm_{uuid.uuid4().hex[:4]}",
        engine_type=EngineType.LLM,
        llm_config=base_llm, # Pass the base LLM config
        temperature=0.7, # Can override base config temp
        description="Real AugLLM Config for Testing"
    )
    conftest_logger.debug(f"Created real_aug_llm_engine instance: {instance.id} / {instance.name}")
    return instance

@pytest.fixture
def real_embeddings_engine() -> EmbeddingsEngineConfig:
    """Provides a real Embeddings engine config instance."""
    conftest_logger.debug("Creating real_embeddings_engine fixture")
    # Using HuggingFace embeddings as it's often locally runnable
    hf_config = HuggingFaceEmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2")
    instance = EmbeddingsEngineConfig(
        id=generate_test_id("real-embeddings"),
        name=f"real_embeddings_{uuid.uuid4().hex[:4]}",
        engine_type=EngineType.EMBEDDINGS,
        embedding_config=hf_config,
        description="Real Embeddings Config for Testing"
    )
    conftest_logger.debug(f"Created real_embeddings_engine instance: {instance.id} / {instance.name}")
    return instance

@pytest.fixture
def real_vectorstore_engine(real_embeddings_engine: EmbeddingsEngineConfig) -> VectorStoreConfig:
    """Provides a real VectorStore engine config instance (In-Memory)."""
    conftest_logger.debug("Creating real_vectorstore_engine fixture")
    instance = VectorStoreConfig(
        id=generate_test_id("real-vs"),
        name=f"real_vectorstore_{uuid.uuid4().hex[:4]}",
        engine_type=EngineType.VECTOR_STORE,
        vector_store_provider=VectorStoreProvider.IN_MEMORY,
        embedding_model=real_embeddings_engine.embedding_config, # Reuse embedding config
        description="Real In-Memory VectorStore Config for Testing"
    )
    conftest_logger.debug(f"Created real_vectorstore_engine instance: {instance.id} / {instance.name}")
    return instance

@pytest.fixture
def real_retriever_engine(real_vectorstore_engine: VectorStoreConfig) -> RetrieverConfig:
    """Provides a real Retriever engine config instance."""
    conftest_logger.debug("Creating real_retriever_engine fixture")
    instance = RetrieverConfig(
        id=generate_test_id("real-retriever"),
        name=f"real_retriever_{uuid.uuid4().hex[:4]}",
        engine_type=EngineType.RETRIEVER,
        retriever_type=RetrieverType.VECTOR_STORE,
        vector_store_config=real_vectorstore_engine, # Use the real VS config
        k=3, # Default number of documents to retrieve
        description="Real Retriever Config for Testing"
    )
    conftest_logger.debug(f"Created real_retriever_engine instance: {instance.id} / {instance.name}")
    return instance

# --------------------------------------------------------------------
# ℹ️ Note on Test vs Real Fixtures:
# - Mock fixtures are good for testing Engine base class logic without external deps.
# - Real fixtures use actual EngineConfig subclasses, useful for integration tests.
# - The 'Test...' classes and fixtures from the original file are removed as
#   they are largely covered by the mock and real fixtures now.
# --------------------------------------------------------------------