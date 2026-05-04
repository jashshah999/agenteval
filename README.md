# agenteval

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-51%20passed-green.svg)]()

**Testing framework for AI agents. Record traces, assert behavior, catch regressions.**

There are 11,000+ agent framework repos on GitHub. There are almost zero agent testing tools. agenteval fills that gap.

```python
from agenteval import record, assert_tool_called, assert_output_contains

with record("booking_agent", input_text="Book a flight to NYC") as rec:
    # ... run your agent ...
    rec.record_tool_call("search_flights", args={"dest": "NYC"}, result=["UA100"])
    rec.record_tool_call("book_flight", args={"flight": "UA100"}, result="confirmed")

trace = rec.trace
assert_tool_called(trace, "search_flights")
assert_tool_called(trace, "book_flight")
assert_output_contains(trace, "confirmed")
```

## Why?

You're building an AI agent. It calls tools, makes decisions, produces output. You ship it. Next week you change the prompt, upgrade the model, or refactor a tool. **Did anything break?**

Right now you either:
- Test manually (doesn't scale)
- Write brittle unit tests that mock everything (misses real failures)
- Use an observability tool like Langfuse (tells you what happened, not whether it was correct)
- Use promptfoo (JS-only, tests prompts not agents, no tool call assertions)

agenteval gives you **pytest-style testing for agent behavior**:
- Assert which tools were called (and in what order)
- Assert tool arguments and return values
- Assert output content with regex or substring matching
- Use LLM-as-judge for subjective quality checks
- Snapshot traces and diff them to catch regressions
- Works with any agent framework (LangChain, CrewAI, smolagents, or raw code)

## Install

```bash
pip install agenteval
```

## Quick Start

### 1. Record a trace

```python
from agenteval import record

with record("my_agent", input_text="What's the weather in SF?") as rec:
    # Run your agent however you want
    rec.record_tool_call("get_weather", args={"city": "SF"}, result={"temp": 65, "condition": "sunny"})
    rec.record_llm_response(model="gpt-4o", response="It's 65F and sunny in SF", input_tokens=50, output_tokens=20)

trace = rec.trace
```

Or wrap existing functions:

```python
from agenteval import AgentRecorder

rec = AgentRecorder("my_agent")

@rec.wrap_tool("get_weather")
def get_weather(city):
    return {"temp": 65, "condition": "sunny"}

get_weather(city="SF")  # automatically recorded
```

### 2. Assert behavior

```python
from agenteval import (
    assert_tool_called,
    assert_tool_not_called,
    assert_tool_called_with,
    assert_tool_call_order,
    assert_output_contains,
    assert_output_matches,
    assert_steps_between,
)

# Tool was called
assert_tool_called(trace, "get_weather")

# Tool was NOT called (safety check)
assert_tool_not_called(trace, "delete_account")

# Tool was called with specific args
assert_tool_called_with(trace, "get_weather", args={"city": "SF"})

# Tools called in correct order
assert_tool_call_order(trace, ["get_weather", "format_response"])

# Output checks
assert_output_contains(trace, "sunny")
assert_output_matches(trace, r"\d+F")

# Step count bounds
assert_steps_between(trace, min_steps=1, max_steps=10)
```

### 3. LLM-as-judge

For subjective quality checks that deterministic assertions can't cover:

```python
from agenteval import assert_llm_judge, assert_no_hallucination

# Custom criterion
assert_llm_judge(
    trace,
    criterion="The response is helpful, concise, and directly answers the user's question",
    model="gpt-4o-mini",  # or any OpenAI/Anthropic model
)

# Hallucination check
assert_no_hallucination(
    trace,
    ground_truth="San Francisco is 65F and sunny today",
)
```

### 4. Snapshot testing

Record a trace once, then catch regressions on every run:

```python
from agenteval import snapshot_match

result = snapshot_match(trace, "weather_agent_happy_path")
# First run: saves snapshot
# Subsequent runs: diffs against saved snapshot

if not result.matched:
    print(result.diffs)  # shows exactly what changed
```

### 5. Use with pytest

agenteval ships as a pytest plugin with fixtures:

```python
# test_my_agent.py

from agenteval import assert_tool_called, assert_output_contains

def test_weather_agent(trace_from):
    trace = trace_from(
        name="weather_test",
        input_text="Weather in NYC?",
        output="72F and cloudy",
        tool_calls=[
            ("get_weather", {"city": "NYC"}, {"temp": 72}),
        ],
    )
    assert_tool_called(trace, "get_weather")
    assert_output_contains(trace, "72")


def test_agent_doesnt_delete(trace_from):
    trace = trace_from(
        output="Done",
        tool_calls=[("search", {}, "result")],
    )
    from agenteval import assert_tool_not_called
    assert_tool_not_called(trace, "delete_everything")
```

Run with:
```bash
pytest test_my_agent.py -v
```

### 6. Test suite (without pytest)

```python
from agenteval import AgentTestSuite, agent_test
from agenteval import assert_tool_called, assert_output_contains
from agenteval.trace import Trace, ToolCall

@agent_test("weather happy path")
def test_weather():
    trace = Trace(output="65F sunny")
    trace.add_tool_call(ToolCall(name="get_weather", result={"temp": 65}))
    assert_tool_called(trace, "get_weather")
    assert_output_contains(trace, "sunny")

@agent_test("no dangerous tools")
def test_safety():
    trace = Trace(output="done")
    from agenteval import assert_tool_not_called
    assert_tool_not_called(trace, "rm_rf")

suite = AgentTestSuite("Weather Agent Tests")
suite.add_tests_from_registry()
result = suite.run()  # prints rich table with pass/fail
```

## CLI

```bash
# Inspect a saved trace
agenteval inspect trace.json

# List snapshots
agenteval snapshots
```

## All Assertions

| Assertion | What it checks |
|-----------|---------------|
| `assert_tool_called(trace, name)` | Tool was called at least N times |
| `assert_tool_not_called(trace, name)` | Tool was never called |
| `assert_tool_called_with(trace, name, args, result)` | Tool called with specific args/result |
| `assert_tool_call_order(trace, [names])` | Tools called in order (subsequence) |
| `assert_output_contains(trace, text)` | Output contains substring (case-insensitive) |
| `assert_output_matches(trace, pattern)` | Output matches regex |
| `assert_steps_between(trace, min, max)` | Step count within bounds |
| `assert_no_hallucination(trace, ground_truth)` | No hallucinated facts (LLM judge) |
| `assert_llm_judge(trace, criterion)` | Custom LLM-as-judge check |
| `assert_custom(trace, fn)` | Custom Python function |
| `snapshot_match(trace, name)` | Matches saved snapshot |

## Framework Integration

agenteval works with any agent framework because it operates on traces, not framework internals. Record your traces however makes sense:

- **LangChain/LangGraph**: Use callbacks to feed tool calls into `AgentRecorder`
- **CrewAI**: Wrap crew tools with `rec.wrap_tool()`
- **smolagents**: Record tool outputs in the agent loop
- **Custom agents**: Use the `record()` context manager or `AgentRecorder` directly

## License

MIT
