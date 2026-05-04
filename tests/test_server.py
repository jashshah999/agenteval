"""Tests for the web dashboard server."""

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from agenteval.server import start_server, DashboardHandler
from agenteval.trace import Trace, ToolCall


@pytest.fixture
def traces_dir(tmp_path):
    d = tmp_path / "traces"
    d.mkdir()

    trace = Trace(name="test_trace", input="hello", output="world")
    trace.add_tool_call(ToolCall(name="search", args={"q": "test"}, result="found"))
    (d / "test_trace.json").write_text(trace.to_json())

    trace2 = Trace(name="error_trace", input="bad", error="something broke")
    (d / "error_trace.json").write_text(trace2.to_json())

    return str(d)


@pytest.fixture
def snap_dir(tmp_path):
    d = tmp_path / "snapshots"
    d.mkdir()

    trace = Trace(name="snap1", input="q", output="a")
    trace.add_tool_call(ToolCall(name="tool1"))
    (d / "snap1.json").write_text(trace.to_json())

    return str(d)


@pytest.fixture
def server_url(traces_dir, snap_dir):
    port = 17601

    DashboardHandler.traces_dir = traces_dir
    DashboardHandler.snapshots_dir = snap_dir

    from http.server import HTTPServer
    httpd = HTTPServer(("127.0.0.1", port), DashboardHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    yield f"http://127.0.0.1:{port}"

    httpd.shutdown()


def test_dashboard_html(server_url):
    resp = urllib.request.urlopen(f"{server_url}/")
    html = resp.read().decode()
    assert "agenteval" in html
    assert resp.status == 200


def test_api_traces(server_url):
    resp = urllib.request.urlopen(f"{server_url}/api/traces")
    data = json.loads(resp.read())
    assert len(data) == 2
    names = {t["name"] for t in data}
    assert "test_trace" in names
    assert "error_trace" in names


def test_api_trace_detail(server_url):
    resp = urllib.request.urlopen(f"{server_url}/api/trace?name=test_trace")
    data = json.loads(resp.read())
    assert data["name"] == "test_trace"
    assert data["input"] == "hello"
    assert len(data["steps"]) == 1


def test_api_snapshots(server_url):
    resp = urllib.request.urlopen(f"{server_url}/api/snapshots")
    data = json.loads(resp.read())
    assert len(data) == 1
    assert data[0]["name"] == "snap1"


def test_api_snapshot_detail(server_url):
    resp = urllib.request.urlopen(f"{server_url}/api/snapshot?name=snap1")
    data = json.loads(resp.read())
    assert data["name"] == "snap1"
    assert len(data["steps"]) == 1
