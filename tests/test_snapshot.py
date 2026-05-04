"""Tests for snapshot matching."""

import json
import pytest
from pathlib import Path

from agenteval.trace import Trace, ToolCall
from agenteval.snapshot import snapshot_match


@pytest.fixture
def snap_dir(tmp_path):
    return str(tmp_path / "snapshots")


def _make_trace(output="result"):
    trace = Trace(name="test", input="query", output=output)
    trace.add_tool_call(ToolCall(name="search", args={"q": "test"}, result="found"))
    trace.add_tool_call(ToolCall(name="format", args={"style": "text"}, result="done"))
    return trace


def test_snapshot_new(snap_dir):
    trace = _make_trace()
    result = snapshot_match(trace, "test_snap", snapshot_dir=snap_dir)
    assert result.matched
    assert result.is_new
    assert Path(snap_dir, "test_snap.json").exists()


def test_snapshot_match(snap_dir):
    trace1 = _make_trace()
    snapshot_match(trace1, "test_snap", snapshot_dir=snap_dir)

    trace2 = _make_trace()
    result = snapshot_match(trace2, "test_snap", snapshot_dir=snap_dir)
    assert result.matched
    assert not result.is_new


def test_snapshot_mismatch_output(snap_dir):
    trace1 = _make_trace(output="result_v1")
    snapshot_match(trace1, "test_snap", snapshot_dir=snap_dir)

    trace2 = _make_trace(output="result_v2")
    result = snapshot_match(trace2, "test_snap", snapshot_dir=snap_dir)
    assert not result.matched
    assert len(result.diffs) > 0
    assert any("output" in d for d in result.diffs)


def test_snapshot_mismatch_tool_calls(snap_dir):
    trace1 = _make_trace()
    snapshot_match(trace1, "test_snap", snapshot_dir=snap_dir)

    trace2 = Trace(name="test", input="query", output="result")
    trace2.add_tool_call(ToolCall(name="different_tool", args={"q": "test"}, result="found"))
    result = snapshot_match(trace2, "test_snap", snapshot_dir=snap_dir)
    assert not result.matched


def test_snapshot_update(snap_dir):
    trace1 = _make_trace(output="v1")
    snapshot_match(trace1, "test_snap", snapshot_dir=snap_dir)

    trace2 = _make_trace(output="v2")
    result = snapshot_match(trace2, "test_snap", snapshot_dir=snap_dir, update=True)
    assert result.matched

    # Verify the snapshot was updated
    saved = json.loads(Path(snap_dir, "test_snap.json").read_text())
    assert saved["output"] == "v2"


def test_snapshot_ignores_timestamps(snap_dir):
    trace1 = _make_trace()
    trace1.timestamp = 1000.0
    snapshot_match(trace1, "test_snap", snapshot_dir=snap_dir)

    trace2 = _make_trace()
    trace2.timestamp = 2000.0
    result = snapshot_match(trace2, "test_snap", snapshot_dir=snap_dir)
    assert result.matched  # timestamps ignored by default
