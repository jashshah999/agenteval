"""Tests for auto-instrumentation integrations."""

import pytest
from agenteval.recorder import AgentRecorder

try:
    import openai  # noqa: F401
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic  # noqa: F401
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
def test_openai_patch_and_unpatch():
    from agenteval.integrations.openai_patch import patch_openai, unpatch_openai

    rec = AgentRecorder(name="test")
    rec.start()

    recorder = patch_openai(rec)
    assert recorder is rec

    from openai.resources.chat.completions import Completions
    assert Completions.create.__name__ == "patched_create"

    unpatch_openai()


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
def test_anthropic_patch_and_unpatch():
    from agenteval.integrations.anthropic_patch import patch_anthropic, unpatch_anthropic

    rec = AgentRecorder(name="test")
    rec.start()

    recorder = patch_anthropic(rec)
    assert recorder is rec

    from anthropic.resources.messages import Messages
    assert Messages.create.__name__ == "patched_create"

    unpatch_anthropic()


def test_auto_watch_detects_installed_providers():
    from agenteval.integrations.auto import _detect_providers

    providers = _detect_providers()
    # These may or may not be installed -- just check the function works
    assert isinstance(providers, list)


def test_auto_watch_returns_recorder():
    from agenteval.integrations.auto import watch, unwatch

    rec = watch(name="test_watch")
    assert isinstance(rec, AgentRecorder)
    assert rec.trace.name == "test_watch"
    unwatch()


def test_langchain_callback_basic():
    from agenteval.integrations.langchain_callback import AgentEvalCallback

    cb = AgentEvalCallback(name="test")
    assert cb.trace.name == "test"

    # Simulate chain start
    cb.on_chain_start({}, {"input": "hello"})
    assert cb.trace.input == "hello"


def test_langchain_callback_tool_recording():
    from agenteval.integrations.langchain_callback import AgentEvalCallback
    from uuid import uuid4

    cb = AgentEvalCallback()
    run_id = uuid4()

    cb.on_tool_start({}, "search query", run_id=run_id)
    cb.on_tool_end("search result", run_id=run_id, name="search")

    assert len(cb.trace.tool_calls) == 1
    assert cb.trace.tool_calls[0].name == "search"
    assert cb.trace.tool_calls[0].result == "search result"


def test_langchain_callback_tool_error():
    from agenteval.integrations.langchain_callback import AgentEvalCallback
    from uuid import uuid4

    cb = AgentEvalCallback()
    run_id = uuid4()

    cb.on_tool_start({}, "bad query", run_id=run_id)
    cb.on_tool_error(RuntimeError("tool failed"), run_id=run_id, name="bad_tool")

    assert len(cb.trace.tool_calls) == 1
    assert cb.trace.tool_calls[0].error == "tool failed"
