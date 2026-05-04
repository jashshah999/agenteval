"""agenteval: Testing framework for AI agents."""

__version__ = "0.1.0"

from .trace import Trace, TraceStep, ToolCall, LLMResponse
from .recorder import AgentRecorder, record
from .assertions import (
    assert_tool_called,
    assert_tool_not_called,
    assert_tool_called_with,
    assert_tool_call_order,
    assert_output_contains,
    assert_output_matches,
    assert_no_hallucination,
    assert_steps_between,
    assert_llm_judge,
    assert_custom,
)
from .suite import AgentTestSuite, agent_test
from .snapshot import snapshot_match

__all__ = [
    "Trace",
    "TraceStep",
    "ToolCall",
    "LLMResponse",
    "AgentRecorder",
    "record",
    "assert_tool_called",
    "assert_tool_not_called",
    "assert_tool_called_with",
    "assert_tool_call_order",
    "assert_output_contains",
    "assert_output_matches",
    "assert_no_hallucination",
    "assert_steps_between",
    "assert_llm_judge",
    "assert_custom",
    "AgentTestSuite",
    "agent_test",
    "snapshot_match",
]
