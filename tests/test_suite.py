"""Tests for the test suite runner."""

from agenteval.trace import Trace, ToolCall
from agenteval.suite import AgentTestSuite
from agenteval.assertions import assert_tool_called, assert_output_contains


def test_suite_all_pass():
    suite = AgentTestSuite("test_suite")

    def test_one():
        trace = Trace(output="hello world")
        trace.add_tool_call(ToolCall(name="greet"))
        assert_tool_called(trace, "greet")
        assert_output_contains(trace, "hello")

    suite.add_test("test_one", test_one)
    result = suite.run(verbose=False)

    assert result.total == 1
    assert result.passed == 1
    assert result.failed == 0


def test_suite_with_failure():
    suite = AgentTestSuite("test_suite")

    def passing_test():
        trace = Trace(output="ok")
        assert_output_contains(trace, "ok")

    def failing_test():
        trace = Trace(output="nope")
        assert_output_contains(trace, "expected_value")

    suite.add_test("pass", passing_test)
    suite.add_test("fail", failing_test)
    result = suite.run(verbose=False)

    assert result.total == 2
    assert result.passed == 1
    assert result.failed == 1


def test_suite_with_exception():
    suite = AgentTestSuite("test_suite")

    def error_test():
        raise RuntimeError("unexpected error")

    suite.add_test("error", error_test)
    result = suite.run(verbose=False)

    assert result.total == 1
    assert result.failed == 1
    assert result.results[0].error == "unexpected error"


def test_suite_timing():
    suite = AgentTestSuite("test_suite")

    def slow_test():
        import time
        time.sleep(0.01)

    suite.add_test("slow", slow_test)
    result = suite.run(verbose=False)

    assert result.results[0].duration_ms >= 10
    assert result.duration_ms >= 10
