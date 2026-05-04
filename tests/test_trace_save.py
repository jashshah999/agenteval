"""Tests for trace save functionality."""

import json
from pathlib import Path
from agenteval.trace import Trace, ToolCall


def test_trace_save_auto_path(tmp_path):
    trace = Trace(name="my_agent", input="q", output="a")
    trace.add_tool_call(ToolCall(name="search"))

    saved = trace.save(dir=str(tmp_path / "traces"))
    assert Path(saved).exists()
    assert "my_agent" in saved

    loaded = json.loads(Path(saved).read_text())
    assert loaded["name"] == "my_agent"
    assert len(loaded["steps"]) == 1


def test_trace_save_explicit_path(tmp_path):
    trace = Trace(name="test", output="done")
    path = str(tmp_path / "custom" / "trace.json")
    saved = trace.save(path=path)
    assert saved == path
    assert Path(path).exists()


def test_trace_save_creates_dirs(tmp_path):
    trace = Trace(name="test")
    path = str(tmp_path / "deep" / "nested" / "dir" / "trace.json")
    trace.save(path=path)
    assert Path(path).exists()
