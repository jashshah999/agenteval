"""Tests for the AgentRecorder and record context manager."""

import time
from agenteval.recorder import AgentRecorder, record


def test_recorder_basic():
    rec = AgentRecorder(name="test", input_text="hello")
    rec.start()
    rec.record_tool_call("search", args={"q": "test"}, result="found")
    trace = rec.stop(output="done")

    assert trace.name == "test"
    assert trace.input == "hello"
    assert trace.output == "done"
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "search"
    assert trace.duration_ms > 0


def test_recorder_llm_response():
    rec = AgentRecorder()
    rec.start()
    rec.record_llm_response(model="gpt-4", prompt="hi", response="hello", input_tokens=5, output_tokens=3)
    trace = rec.stop()

    assert len(trace.llm_responses) == 1
    assert trace.llm_responses[0].model == "gpt-4"
    assert trace.total_tokens == 8


def test_recorder_multiple_steps():
    rec = AgentRecorder()
    rec.start()
    rec.record_tool_call("step1", result="a")
    rec.record_llm_response(model="gpt-4", response="b")
    rec.record_tool_call("step2", result="c")
    rec.record_tool_call("step3", result="d")
    trace = rec.stop()

    assert len(trace.steps) == 4
    assert trace.tool_names == ["step1", "step2", "step3"]


def test_recorder_error():
    rec = AgentRecorder()
    rec.start()
    rec.record_error("something went wrong")
    trace = rec.stop()
    assert trace.error == "something went wrong"


def test_recorder_metadata():
    rec = AgentRecorder()
    rec.set_metadata("env", "test")
    rec.set_metadata("version", "1.0")
    assert rec.trace.metadata == {"env": "test", "version": "1.0"}


def test_record_context_manager():
    with record("my_test", input_text="query") as rec:
        rec.record_tool_call("search", result="found")

    trace = rec.trace
    assert trace.name == "my_test"
    assert trace.input == "query"
    assert len(trace.tool_calls) == 1
    assert trace.duration_ms > 0


def test_record_context_manager_error():
    try:
        with record("fail_test") as rec:
            rec.record_tool_call("step1")
            raise ValueError("boom")
    except ValueError:
        pass

    assert rec.trace.error == "boom"
    assert len(rec.trace.tool_calls) == 1


def test_wrap_tool():
    rec = AgentRecorder()

    @rec.wrap_tool("my_tool")
    def my_function(x, y):
        return x + y

    result = my_function(x=1, y=2)
    assert result == 3
    assert len(rec.trace.tool_calls) == 1
    assert rec.trace.tool_calls[0].name == "my_tool"
    assert rec.trace.tool_calls[0].result == 3
    assert rec.trace.tool_calls[0].duration_ms > 0


def test_wrap_tool_error():
    rec = AgentRecorder()

    @rec.wrap_tool("bad_tool")
    def bad_function():
        raise RuntimeError("fail")

    try:
        bad_function()
    except RuntimeError:
        pass

    assert len(rec.trace.tool_calls) == 1
    assert rec.trace.tool_calls[0].error == "fail"
