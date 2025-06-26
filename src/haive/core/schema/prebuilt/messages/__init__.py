"""
Enhanced message handling for Haive agents.

This package provides improved message handling capabilities including:
- Conversation round tracking
- Tool call deduplication and analysis
- Message transformation utilities
- Advanced filtering and analysis

Migration guide from classic MessagesState:

1. Direct replacement (basic features):
   ```python
   # Old code
   from haive.core.schema.prebuilt.messages_state import MessagesState

   # New code (compatible with old API)
   from haive.core.schema.prebuilt.messages import MessagesState
   ```

2. Using enhanced features:
   ```python
   # Import enhanced state
   from haive.core.schema.prebuilt.messages import MessagesState

   state = MessagesState()

   # New features
   rounds = state.get_conversation_rounds()
   state.deduplicate_tool_calls()
   tool_calls = state.get_completed_tool_calls()
   ```

3. Full migration to new architecture:
   ```python
   # Import enhanced implementation
   from haive.core.schema.prebuilt.messages import EnhancedMessagesState

   # Create with same API but more capabilities
   state = EnhancedMessagesState()

   # Access new features
   print(state.round_count)
   print(state.has_tool_errors)
   state.transform_ai_to_human()
   ```
"""

# Export the compatibility adapter for advanced usage
from haive.core.schema.prebuilt.messages.compatibility import MessagesStateAdapter

# Export utility classes for advanced usage
from haive.core.schema.prebuilt.messages.utils import (
    MessageRound,
    ToolCallInfo,
    extract_tool_calls,
    inject_state_into_tool_calls,
    is_real_human_message,
    is_tool_error,
)

# Re-export the original implementation with enhancements
from haive.core.schema.prebuilt.messages_state import MessagesState

# Re-export the enhanced implementation under a different name
try:
    from haive.core.schema.prebuilt.messages.messages_state import (
        MessageList,
    )
    from haive.core.schema.prebuilt.messages.messages_state import (
        MessagesState as EnhancedMessagesState,
    )
except ImportError:
    # Fallback if the enhanced implementation is not available yet
    EnhancedMessagesState = None
    MessageList = None
