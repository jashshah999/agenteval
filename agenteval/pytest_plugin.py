"""Pytest plugin for agenteval.

Provides fixtures and markers for agent testing in pytest.

Usage:
    def test_my_agent(agent_recorder):
        rec = agent_recorder("my_test")
        rec.start()
        rec.record_tool_call("search", args={"q": "hello"}, result="world")
        trace = rec.stop(output="Found: world")

        assert_tool_called(trace, "search")
        assert_output_contains(trace, "world")
"""

import pytest

from .recorder import AgentRecorder


@pytest.fixture
def agent_recorder():
    """Fixture that creates AgentRecorder instances."""
    recorders = []

    def factory(name: str = "", input_text: str = "") -> AgentRecorder:
        rec = AgentRecorder(name=name, input_text=input_text)
        recorders.append(rec)
        return rec

    yield factory


@pytest.fixture
def trace_from(agent_recorder):
    """Fixture that creates a quick trace from tool calls.

    Usage:
        def test_my_agent(trace_from):
            trace = trace_from(
                name="test",
                input_text="hello",
                output="world",
                tool_calls=[("search", {"q": "hello"}, "world")],
            )
            assert_tool_called(trace, "search")
    """
    def factory(
        name: str = "",
        input_text: str = "",
        output: str = "",
        tool_calls: list[tuple] | None = None,
    ):
        rec = agent_recorder(name=name, input_text=input_text)
        rec.start()
        for tc in (tool_calls or []):
            if len(tc) == 2:
                rec.record_tool_call(tc[0], args=tc[1])
            elif len(tc) == 3:
                rec.record_tool_call(tc[0], args=tc[1], result=tc[2])
        return rec.stop(output=output)

    yield factory
