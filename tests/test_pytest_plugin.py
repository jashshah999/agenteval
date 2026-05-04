"""Tests for the pytest plugin fixtures."""

from agenteval.assertions import assert_tool_called, assert_output_contains


def test_agent_recorder_fixture(agent_recorder):
    rec = agent_recorder("my_test", input_text="hello")
    rec.start()
    rec.record_tool_call("search", args={"q": "test"}, result="found")
    trace = rec.stop(output="Found it")

    assert trace.name == "my_test"
    assert len(trace.tool_calls) == 1


def test_trace_from_fixture(trace_from):
    trace = trace_from(
        name="quick_test",
        input_text="query",
        output="answer",
        tool_calls=[
            ("search", {"q": "test"}, "found"),
            ("format", {"style": "text"}, "done"),
        ],
    )

    assert_tool_called(trace, "search")
    assert_tool_called(trace, "format")
    assert_output_contains(trace, "answer")
    assert len(trace.tool_calls) == 2


def test_trace_from_minimal(trace_from):
    trace = trace_from(output="just output")
    assert trace.output == "just output"
    assert trace.tool_calls == []
