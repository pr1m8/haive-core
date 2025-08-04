# Files That Need Import Fixes

## haive-core Package

### Files with "from core." imports (4 files):

1. **src/haive/core/persistence/store/postgres.py**
   - Line 10: `from core.persistence.postgres_config import PostgresCheckpointerConfig`
   - Should be: `from haive.core.persistence.postgres_config import PostgresCheckpointerConfig`

2. **src/haive/core/persistence/store/wrappers/memory.py**
   - Line 6: `from core.persistence.store.base import SerializableStoreWrapper`
   - Line 7: `from core.persistence.store.embeddings import EmbeddingAdapter`
   - Should use `haive.core.` prefix

3. **src/haive/core/persistence/store/wrappers/postgres.py**
   - Line 6: `from core.persistence.store.base import SerializableStoreWrapper`
   - Line 7: `from core.persistence.store.connection import ConnectionManager`
   - Line 8: `from core.persistence.store.embeddings import EmbeddingAdapter`
   - Line 9: `from core.persistence.store.types import StoreType`
   - Should use `haive.core.` prefix

4. **src/haive/core/tools/store_manager.py**
   - Line 13: `from core.persistence.store.base import SerializableStoreWrapper`
   - Line 14: `from core.persistence.store.factory import create_store`
   - Line 15: `from core.persistence.store.types import StoreType`
   - Should use `haive.core.` prefix

## Summary

Total files to fix: 4
Total import statements to fix: 10

All of these are in the persistence store and tools modules.
/home/will/Projects/haive/backend/haive/docs/source/guides/documentation/documentation_tools.rst:139: WARNING: Inline literal start-string without end-string. [docutils]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/guides/documentation/documentation_tools.rst:139: WARNING: Inline literal start-string without end-string. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:11: ERROR: Unknown directive type "exec_code".
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:11: ERROR: Unknown directive type "exec_code".
.. exec_code::
:language: python

# This code runs when building docs!

import sys
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}") [docutils]
/home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:24: ERROR: Unknown directive type "exec_code".
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:24: ERROR: Unknown directive type "exec_code".
.. exec_code::
:language: python
:hide_output:

# Import Haive components

from haive.agents.simple import SimpleAgent
from haive.core.engine.aug_llm import AugLLMConfig

# Create configuration

config = AugLLMConfig(
temperature=0.7,
max_tokens=100
)

# Create agent

agent = SimpleAgent(
name="doc_example",
engine=config
)
print(f"Agent created: {agent.name}")
print(f"Agent type: {type(agent).**name**}") [docutils]
/home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:64: WARNING: Title underline too short.
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:64: WARNING: Title underline too short.
Mathematical Computations
------------------------ [docutils]
/home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:64: WARNING: Title underline too short.
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:64: WARNING: Title underline too short.
Mathematical Computations
------------------------ [docutils]
/home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:68: ERROR: Unknown directive type "exec_code".
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:68: ERROR: Unknown directive type "exec_code".
.. exec_code::
:language: python
import numpy as np

# Create sample data

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.4f}")
print(f"Std Dev: {np.std(data):.4f}")
print(f"Min: {np.min(data):.4f}")
print(f"Max: {np.max(data):.4f}") [docutils]
/home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:86: ERROR: Unknown directive type "exec_code".
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/guides/executable_examples.rst:86: ERROR: Unknown directive type "exec_code".
.. exec_code::
:language: python
:hide_output:
try:

# This will raise an error

nox > ℹ️ # This will raise an error
result = 1 / 0
except ZeroDivisionError as e:
nox > ℹ️ except ZeroDivisionError as e:
print(f"Caught error: {e}")
nox > ℹ️ print(f"Caught error: {e}")
print("Error handling works!") [docutils]
nox > ℹ️ print("Error handling works!") [docutils]
/home/will/Projects/haive/backend/haive/docs/source/guides/index.rst:64: WARNING: toctree contains reference to nonexisting document 'guides/advanced_patterns' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/guides/index.rst:64: WARNING: toctree contains reference to nonexisting document 'guides/advanced_patterns' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/guides/index.rst:64: WARNING: toctree contains reference to nonexisting document 'guides/performance' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/guides/index.rst:64: WARNING: toctree contains reference to nonexisting document 'guides/performance' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/guides/index.rst:64: WARNING: toctree contains reference to nonexisting document 'guides/deployment' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/guides/index.rst:64: WARNING: toctree contains reference to nonexisting document 'guides/deployment' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/guides/index.rst:64: WARNING: toctree contains reference to nonexisting document 'guides/troubleshooting' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/guides/index.rst:64: WARNING: toctree contains reference to nonexisting document 'guides/troubleshooting' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/index.rst:274: WARNING: The parent of a 'grid-item' should be a 'grid-row' [design.grid] [design.grid]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/index.rst:274: WARNING: The parent of a 'grid-item' should be a 'grid-row' [design.grid] [design.grid]
/home/will/Projects/haive/backend/haive/docs/source/index.rst:283: WARNING: The parent of a 'grid-item' should be a 'grid-row' [design.grid] [design.grid]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/index.rst:283: WARNING: The parent of a 'grid-item' should be a 'grid-row' [design.grid] [design.grid]
/home/will/Projects/haive/backend/haive/docs/source/index.rst:292: WARNING: The parent of a 'grid-item' should be a 'grid-row' [design.grid] [design.grid]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/index.rst:292: WARNING: The parent of a 'grid-item' should be a 'grid-row' [design.grid] [design.grid]
/home/will/Projects/haive/backend/haive/docs/source/index.rst:397: WARNING: toctree contains reference to nonexisting document 'examples/index' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/index.rst:397: WARNING: toctree contains reference to nonexisting document 'examples/index' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/index.rst:397: WARNING: toctree contains reference to nonexisting document 'reference/index' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/index.rst:397: WARNING: toctree contains reference to nonexisting document 'reference/index' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/dynamic-mcp.rst:746: WARNING: toctree contains reference to nonexisting document 'mcp/development' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/dynamic-mcp.rst:746: WARNING: toctree contains reference to nonexisting document 'mcp/development' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/dynamic-mcp.rst:746: WARNING: toctree contains reference to nonexisting document 'mcp/troubleshooting' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/dynamic-mcp.rst:746: WARNING: toctree contains reference to nonexisting document 'mcp/troubleshooting' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst:521: WARNING: toctree contains reference to nonexisting document 'mcp/development' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst:521: WARNING: toctree contains reference to nonexisting document 'mcp/development' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst:521: WARNING: toctree contains reference to nonexisting document 'mcp/troubleshooting' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst:521: WARNING: toctree contains reference to nonexisting document 'mcp/troubleshooting' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/servers.rst:1115: WARNING: toctree contains reference to nonexisting document 'mcp/filesystem/index' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/servers.rst:1115: WARNING: toctree contains reference to nonexisting document 'mcp/filesystem/index' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/servers.rst:1115: WARNING: toctree contains reference to nonexisting document 'mcp/github/index' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/servers.rst:1115: WARNING: toctree contains reference to nonexisting document 'mcp/github/index' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/servers.rst:1115: WARNING: toctree contains reference to nonexisting document 'mcp/puppeteer/index' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/servers.rst:1115: WARNING: toctree contains reference to nonexisting document 'mcp/puppeteer/index' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:1076: WARNING: toctree contains reference to nonexisting document 'mcp/environment-config' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:1076: WARNING: toctree contains reference to nonexisting document 'mcp/environment-config' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:1076: WARNING: toctree contains reference to nonexisting document 'mcp/security-config' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:1076: WARNING: toctree contains reference to nonexisting document 'mcp/security-config' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:1076: WARNING: toctree contains reference to nonexisting document 'mcp/performance-tuning' [toc.not_readable]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:1076: WARNING: toctree contains reference to nonexisting document 'mcp/performance-tuning' [toc.not_readable]
/home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:388: WARNING: duplicate label postgresql-setup, other instance in /home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:388: WARNING: duplicate label postgresql-setup, other instance in /home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst
/home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:464: WARNING: duplicate label filesystem-setup, other instance in /home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:464: WARNING: duplicate label filesystem-setup, other instance in /home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst
/home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:522: WARNING: duplicate label github-setup, other instance in /home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/mcp/setup.rst:522: WARNING: duplicate label github-setup, other instance in /home/will/Projects/haive/backend/haive/docs/source/mcp/index.rst
/home/will/Projects/haive/backend/haive/docs/source/real_examples/base_v2_examples.md:47: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/base_v2_examples.md:47: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/battleship_examples.md:56: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/battleship_examples.md:56: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/board_examples.md:47: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/board_examples.md:47: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/checkers_examples.md:54: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/checkers_examples.md:54: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/core_examples.md:56: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/core_examples.md:56: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/custom.DemoAgent_examples.md:84: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/custom.DemoAgent_examples.md:84: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/custom.PerformanceAgent_examples.md:92: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/custom.PerformanceAgent_examples.md:92: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/custom.ReactAgent_examples.md:248: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/custom.ReactAgent_examples.md:248: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/custom.SimpleAgent_examples.md:703: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/custom.SimpleAgent_examples.md:703: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/debate_examples.md:56: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/debate_examples.md:56: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/fox_and_geese_examples.md:74: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/fox_and_geese_examples.md:74: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/go_examples.md:57: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/go_examples.md:57: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/mafia_examples.md:74: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/mafia_examples.md:74: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/mastermind_examples.md:71: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/mastermind_examples.md:71: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/multi_player_examples.md:47: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/multi_player_examples.md:47: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/nim_examples.md:57: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/nim_examples.md:57: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/react_examples.md:71: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/react_examples.md:71: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/simple_examples.md:71: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/simple_examples.md:71: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/single_player_examples.md:57: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/single_player_examples.md:57: ERROR: Document may not end with a transition. [docutils]
/home/will/Projects/haive/backend/haive/docs/source/real_examples/tic_tac_toe_examples.md:58: ERROR: Document may not end with a transition. [docutils]
nox > ❌ /home/will/Projects/haive/backend/haive/docs/source/real_examples/tic_tac_toe_examples.md:58: ERROR: Document may not end with a transition. [docutils]
WARNING: Failed to get a function signature for haive.tools.search.google_search_tool: [GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(search_engine=<googleapiclient.discovery.Resource object at 0x771038ecc5f0>, google_api_key='AIzaSyB2sJELFi9rISdvqET724eqbjVbfg7_mM0', google_cse_id='b2c7f60b2a031467d', k=10, siterestrict=False))] is not a callable object
nox > ⚠️ WARNING: Failed to get a function signature for haive.tools.search.google_search_tool: [GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(search_engine=<googleapiclient.discovery.Resource object at 0x771038ecc5f0>, google_api_key='AIzaSyB2sJELFi9rISdvqET724eqbjVbfg7_mM0', google_cse_id='b2c7f60b2a031467d', k=10, siterestrict=False))] is not a callable object
WARNING: Cannot resolve forward reference in type annotations of "haive.tools.search.tavily_search_tool": name 'Annotated' is not defined
nox > ⚠️ WARNING: Cannot resolve forward reference in type annotations of "haive.tools.search.tavily_search_tool": name 'Annotated' is not defined
/home/will/Projects/haive/backend/haive/docs/source/agents/climateresearchagent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/ClimateResearchAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/climateresearchagent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/ClimateResearchAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/conversation/collaborative.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/collaberative/example.py'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/conversation/collaborative.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/collaberative/example.py'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/conversation/debate.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/debate/example.py'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/conversation/debate.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/debate/example.py'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/conversation/directed.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/directed/example.py'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/conversation/directed.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/directed/example.py'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/conversation/round_robin.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/round_robin/example.py'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/conversation/round_robin.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/round_robin/example.py'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/conversation/social_media.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/social_media/example.py'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/conversation/social_media.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/social_media/example.py'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/conversation_examples.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/directed/example.py'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/conversation_examples.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/directed/example.py'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/conversation_examples.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/collaberative/example.py'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/conversation_examples.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/collaberative/example.py'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/conversation_examples.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/social_media/example.py'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/conversation_examples.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/packages/haive-agents/src/haive/agents/conversation/social_media/example.py'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/quantumexplaineragent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/QuantumExplainerAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/quantumexplaineragent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/QuantumExplainerAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/reactresearchagent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/ReactResearchAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/reactresearchagent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/ReactResearchAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/simpleanalysisagent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/SimpleAnalysisAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/simpleanalysisagent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/SimpleAnalysisAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
/home/will/Projects/haive/backend/haive/docs/source/agents/textsummarizeragent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/TextSummarizerAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
nox > ⚠️ /home/will/Projects/haive/backend/haive/docs/source/agents/textsummarizeragent_showcase.rst: WARNING: Dependency file PosixPath('/home/will/Projects/haive/backend/haive/docs/captures/TextSummarizerAgent_graph.png'), doesn't exist, skipping [git.dependency_not_found]
