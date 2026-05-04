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


def watch(name: str = "agent", providers: list[str] | None = None) -> AgentRecorder:
    """Auto-instrument all detected LLM providers. Zero config.

    Usage:
        import agenteval
        recorder = agenteval.watch()
        # ... use OpenAI/Anthropic as normal, everything is captured ...
        trace = recorder.trace
    """
    from .integrations.auto import watch as _watch
    return _watch(name=name, providers=providers)


__all__ = [
    "Trace",
    "TraceStep",
    "ToolCall",
    "LLMResponse",
    "AgentRecorder",
    "record",
    "watch",
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
