"""Tests for deterministic assertions."""

import pytest
from agenteval.trace import Trace, ToolCall, LLMResponse
from agenteval.assertions import (
    assert_tool_called,
    assert_tool_not_called,
    assert_tool_called_with,
    assert_tool_call_order,
    assert_output_contains,
    assert_output_matches,
    assert_steps_between,
    assert_custom,
    AssertionError,
)


@pytest.fixture
def sample_trace():
    trace = Trace(name="test", input="find flights to NYC", output="Found 3 flights to NYC from $200")
    trace.add_tool_call(ToolCall(name="search_flights", args={"dest": "NYC", "date": "2024-01-15"}, result=["UA100", "DL200", "AA300"]))
    trace.add_tool_call(ToolCall(name="get_prices", args={"flights": ["UA100", "DL200", "AA300"]}, result={"UA100": 200, "DL200": 250, "AA300": 300}))
    trace.add_llm_response(LLMResponse(model="gpt-4", input_tokens=500, output_tokens=100))
    trace.add_tool_call(ToolCall(name="format_response", args={"format": "text"}, result="Found 3 flights"))
    return trace


def test_assert_tool_called_pass(sample_trace):
    result = assert_tool_called(sample_trace, "search_flights")
    assert result.passed


def test_assert_tool_called_fail(sample_trace):
    with pytest.raises(AssertionError) as exc:
        assert_tool_called(sample_trace, "book_flight")
    assert "book_flight" in str(exc.value)
    assert "search_flights" in str(exc.value)


def test_assert_tool_called_min_times(sample_trace):
    result = assert_tool_called(sample_trace, "search_flights", min_times=1)
    assert result.passed

    with pytest.raises(AssertionError):
        assert_tool_called(sample_trace, "search_flights", min_times=2)


def test_assert_tool_not_called_pass(sample_trace):
    result = assert_tool_not_called(sample_trace, "delete_account")
    assert result.passed


def test_assert_tool_not_called_fail(sample_trace):
    with pytest.raises(AssertionError):
        assert_tool_not_called(sample_trace, "search_flights")


def test_assert_tool_called_with_args(sample_trace):
    result = assert_tool_called_with(sample_trace, "search_flights", args={"dest": "NYC"})
    assert result.passed


def test_assert_tool_called_with_wrong_args(sample_trace):
    with pytest.raises(AssertionError):
        assert_tool_called_with(sample_trace, "search_flights", args={"dest": "LAX"})


def test_assert_tool_called_with_result(sample_trace):
    result = assert_tool_called_with(
        sample_trace, "get_prices",
        result={"UA100": 200, "DL200": 250, "AA300": 300}
    )
    assert result.passed


def test_assert_tool_call_order_pass(sample_trace):
    result = assert_tool_call_order(sample_trace, ["search_flights", "get_prices", "format_response"])
    assert result.passed


def test_assert_tool_call_order_subsequence(sample_trace):
    result = assert_tool_call_order(sample_trace, ["search_flights", "format_response"])
    assert result.passed


def test_assert_tool_call_order_fail(sample_trace):
    with pytest.raises(AssertionError):
        assert_tool_call_order(sample_trace, ["format_response", "search_flights"])


def test_assert_output_contains_pass(sample_trace):
    result = assert_output_contains(sample_trace, "flights")
    assert result.passed


def test_assert_output_contains_case_insensitive(sample_trace):
    result = assert_output_contains(sample_trace, "NYC")
    assert result.passed
    result = assert_output_contains(sample_trace, "nyc")
    assert result.passed


def test_assert_output_contains_fail(sample_trace):
    with pytest.raises(AssertionError):
        assert_output_contains(sample_trace, "hotels")


def test_assert_output_matches_pass(sample_trace):
    result = assert_output_matches(sample_trace, r"Found \d+ flights")
    assert result.passed


def test_assert_output_matches_fail(sample_trace):
    with pytest.raises(AssertionError):
        assert_output_matches(sample_trace, r"^No flights found$")


def test_assert_steps_between_pass(sample_trace):
    result = assert_steps_between(sample_trace, min_steps=1, max_steps=10)
    assert result.passed


def test_assert_steps_between_fail(sample_trace):
    with pytest.raises(AssertionError):
        assert_steps_between(sample_trace, min_steps=10, max_steps=20)


def test_assert_custom_pass(sample_trace):
    result = assert_custom(
        sample_trace,
        fn=lambda t: len(t.tool_calls) == 3,
        message="Should have 3 tool calls",
    )
    assert result.passed


def test_assert_custom_fail(sample_trace):
    with pytest.raises(AssertionError):
        assert_custom(
            sample_trace,
            fn=lambda t: len(t.tool_calls) == 0,
            message="Should have 0 tool calls",
        )


def test_assert_custom_exception(sample_trace):
    with pytest.raises(AssertionError) as exc:
        assert_custom(sample_trace, fn=lambda t: 1 / 0)
    assert "division by zero" in str(exc.value)


def test_empty_trace_assertions():
    trace = Trace(output="hello")
    assert_tool_not_called(trace, "anything")
    assert_output_contains(trace, "hello")
    assert_steps_between(trace, min_steps=0, max_steps=0)
