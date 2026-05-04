"""Tests for the Trace data model."""

import json
from agenteval.trace import Trace, TraceStep, ToolCall, LLMResponse


def test_tool_call_roundtrip():
    tc = ToolCall(name="search", args={"q": "hello"}, result="world")
    d = tc.to_dict()
    tc2 = ToolCall.from_dict(d)
    assert tc2.name == "search"
    assert tc2.args == {"q": "hello"}
    assert tc2.result == "world"


def test_llm_response_roundtrip():
    lr = LLMResponse(model="gpt-4", prompt="hi", response="hello", input_tokens=5, output_tokens=3)
    d = lr.to_dict()
    lr2 = LLMResponse.from_dict(d)
    assert lr2.model == "gpt-4"
    assert lr2.input_tokens == 5
    assert lr2.output_tokens == 3


def test_trace_add_steps():
    trace = Trace(name="test", input="query")
    trace.add_tool_call(ToolCall(name="search", args={"q": "test"}, result="found"))
    trace.add_llm_response(LLMResponse(model="gpt-4", response="answer"))
    trace.add_tool_call(ToolCall(name="write", args={"content": "done"}, result="ok"))

    assert len(trace.steps) == 3
    assert len(trace.tool_calls) == 2
    assert len(trace.llm_responses) == 1
    assert trace.tool_names == ["search", "write"]


def test_trace_json_roundtrip():
    trace = Trace(name="test", input="q", output="a")
    trace.add_tool_call(ToolCall(name="search", args={"q": "test"}, result="found"))
    trace.add_llm_response(LLMResponse(model="gpt-4", response="answer", input_tokens=10, output_tokens=5))

    json_str = trace.to_json()
    trace2 = Trace.from_json(json_str)

    assert trace2.name == "test"
    assert trace2.input == "q"
    assert trace2.output == "a"
    assert len(trace2.steps) == 2
    assert trace2.tool_calls[0].name == "search"
    assert trace2.llm_responses[0].model == "gpt-4"


def test_trace_total_tokens():
    trace = Trace()
    trace.add_llm_response(LLMResponse(input_tokens=100, output_tokens=50))
    trace.add_llm_response(LLMResponse(input_tokens=200, output_tokens=100))
    assert trace.total_tokens == 450


def test_trace_metadata():
    trace = Trace(metadata={"env": "test", "version": "1.0"})
    assert trace.metadata["env"] == "test"
    d = trace.to_dict()
    assert d["metadata"]["version"] == "1.0"


def test_empty_trace():
    trace = Trace()
    assert trace.tool_calls == []
    assert trace.llm_responses == []
    assert trace.tool_names == []
    assert trace.total_tokens == 0
    assert trace.error is None
