# agenteval

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-64%20passed-green.svg)]()

**Testing framework for AI agents. Auto-instrument, assert behavior, catch regressions.**

There are 11,000+ agent framework repos on GitHub. There are almost zero agent testing tools. agenteval fills that gap.

```python
import agenteval

# One line. Captures every OpenAI/Anthropic call automatically.
recorder = agenteval.watch()

# ... run your agent as normal ...

trace = recorder.trace
agenteval.assert_tool_called(trace, "search_flights")
agenteval.assert_tool_call_order(trace, ["search_flights", "book_flight"])
agenteval.assert_output_contains(trace, "confirmed")
```

## Why?

You're building an AI agent. It calls tools, makes decisions, produces output. You ship it. Next week you change the prompt, upgrade the model, or refactor a tool. **Did anything break?**

Right now you either:
- Test manually (doesn't scale)
- Write brittle unit tests that mock everything (misses real failures)
- Use an observability tool like Langfuse (tells you what happened, not whether it was correct)
- Use promptfoo (JS-only, tests prompts not agents, no tool call assertions)

agenteval gives you **pytest-style testing for agent behavior**:
- **Auto-instrumentation** -- patch OpenAI/Anthropic/LangChain with one line, zero manual recording
- Assert which tools were called (and in what order, with what args)
- Use LLM-as-judge for subjective quality checks
- Snapshot traces and diff them to catch regressions
- **Web dashboard** to visualize traces and diffs
- **GitHub Action** to block PRs on regressions
- Works with any agent framework

## Install

```bash
pip install agenteval            # core (no LLM deps)
pip install agenteval[openai]    # + OpenAI auto-instrumentation
pip install agenteval[anthropic] # + Anthropic auto-instrumentation
pip install agenteval[langchain] # + LangChain callback
pip install agenteval[all]       # everything
```

## Quick Start

### 1. Auto-instrument (zero code changes)

```python
import agenteval

# Patches OpenAI + Anthropic SDKs. All calls are captured automatically.
recorder = agenteval.watch()

# Use OpenAI as normal -- agenteval captures everything
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[...],
)

# Get the trace with all captured calls
trace = recorder.trace
print(f"Captured {len(trace.tool_calls)} tool calls, {len(trace.llm_responses)} LLM calls")
```

### 2. LangChain / LangGraph (callback)

```python
from agenteval.integrations.langchain_callback import AgentEvalCallback

cb = AgentEvalCallback()
agent.invoke({"input": "Book me a flight"}, config={"callbacks": [cb]})
trace = cb.trace  # all tool calls + LLM responses captured
```

### 3. Manual recording (any framework)

```python
from agenteval import record

with record("my_agent", input_text="query") as rec:
    rec.record_tool_call("search", args={"q": "hello"}, result="world")
    rec.record_llm_response(model="gpt-4o", response="answer")

trace = rec.trace
```

### 4. Assert behavior

```python
from agenteval import (
    assert_tool_called,
    assert_tool_not_called,
    assert_tool_called_with,
    assert_tool_call_order,
    assert_output_contains,
    assert_output_matches,
    assert_steps_between,
    assert_no_hallucination,
    assert_llm_judge,
)

assert_tool_called(trace, "search_flights")
assert_tool_not_called(trace, "delete_account")  # safety check
assert_tool_called_with(trace, "search_flights", args={"dest": "NYC"})
assert_tool_call_order(trace, ["search", "book", "confirm"])
assert_output_contains(trace, "confirmed")
assert_output_matches(trace, r"Flight [A-Z]{2}\d+ booked")
assert_steps_between(trace, min_steps=2, max_steps=10)

# LLM-as-judge for subjective checks
assert_llm_judge(trace, criterion="Response is helpful and concise")
assert_no_hallucination(trace, ground_truth="Flight UA100 departs at 3pm")
```

### 5. Snapshot testing

```python
from agenteval import snapshot_match

result = snapshot_match(trace, "booking_happy_path")
# First run: saves snapshot
# Next runs: diffs against saved snapshot, flags regressions

if not result.matched:
    print(result.diffs)  # exactly what changed
```

### 6. Save and visualize

```python
# Save trace to disk
trace.save()  # saves to .agenteval_traces/

# Launch web dashboard
# $ agenteval server --traces-dir .agenteval_traces
```

### 7. pytest integration

```python
# test_my_agent.py
from agenteval import assert_tool_called, assert_output_contains

def test_booking_agent(trace_from):
    trace = trace_from(
        output="Flight confirmed",
        tool_calls=[
            ("search_flights", {"dest": "NYC"}, ["UA100"]),
            ("book_flight", {"flight": "UA100"}, "confirmed"),
        ],
    )
    assert_tool_called(trace, "search_flights")
    assert_tool_called(trace, "book_flight")
    assert_output_contains(trace, "confirmed")
```

```bash
pytest test_my_agent.py -v
```

### 8. CI with GitHub Action

```yaml
# .github/workflows/agent-tests.yml
name: Agent Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jashshah999/agenteval@main
        with:
          test-path: tests/
```

## Web Dashboard

```bash
agenteval server --traces-dir .agenteval_traces
# Opens http://localhost:7600
```

View all captured traces, expand to see tool calls and LLM responses, inspect snapshots.

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

## Auto-Instrumentation Support

| Provider | Method | What's captured |
|----------|--------|-----------------|
| **OpenAI** | `agenteval.watch()` or `patch_openai()` | Chat completions, tool calls, token usage |
| **Anthropic** | `agenteval.watch()` or `patch_anthropic()` | Messages, tool use, token usage |
| **LangChain / LangGraph** | `AgentEvalCallback()` | Tool calls, LLM responses, chain I/O |
| **Any framework** | `record()` context manager | Manual recording |

## CLI

```bash
agenteval inspect trace.json    # inspect a trace file
agenteval snapshots             # list saved snapshots
agenteval server                # launch web dashboard
```

## License

MIT
