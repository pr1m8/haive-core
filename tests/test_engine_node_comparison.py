#!/usr/bin/env python3
"""Test script to compare original vs test engine node behavior.

This shows the difference between:
1. Original: Only extracts 'messages' field for LLM engines
2. Test: Uses engine's derived input fields (messages, query, context)
"""

import sys

from langchain_core.prompts import ChatPromptTemplate

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.graph.node.engine_node import EngineNodeConfig as OriginalEngineNode
from haive.core.graph.node.engine_node_test import EngineNodeConfig as TestEngineNode

sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")
sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-agents/src")


# Create the same prompt from test_basic.py
RAG_QUERY_REFINEMENT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert. Analyze query: {query} with context: {context}"),
        ("human", "Process: {query}"),
    ]
)

# Create AugLLMConfig with this prompt
engine = AugLLMConfig(name="test_engine", prompt_template=RAG_QUERY_REFINEMENT)


# Test original engine node
original_node = OriginalEngineNode(name="original", engine=engine)

# Test new engine node
test_node = TestEngineNode(name="test", engine=engine)
