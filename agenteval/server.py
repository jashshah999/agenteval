"""Local web server for trace visualization and diff inspection.

Usage:
    agenteval server                     # serve traces from default dir
    agenteval server --dir ./my_traces   # serve from custom dir
"""

from __future__ import annotations

import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from .snapshot import DEFAULT_SNAPSHOT_DIR


class DashboardHandler(BaseHTTPRequestHandler):
    traces_dir: str = "."
    snapshots_dir: str = DEFAULT_SNAPSHOT_DIR

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            self._serve_dashboard()
        elif path == "/api/traces":
            self._serve_traces_list()
        elif path == "/api/trace":
            params = parse_qs(parsed.query)
            name = params.get("name", [""])[0]
            self._serve_trace(name)
        elif path == "/api/snapshots":
            self._serve_snapshots_list()
        elif path == "/api/snapshot":
            params = parse_qs(parsed.query)
            name = params.get("name", [""])[0]
            self._serve_snapshot(name)
        else:
            self._send_json({"error": "not found"}, 404)

    def _serve_dashboard(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(_DASHBOARD_HTML.encode())

    def _serve_traces_list(self) -> None:
        traces_path = Path(self.traces_dir)
        traces = []
        if traces_path.exists():
            for f in sorted(traces_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    data = json.loads(f.read_text())
                    traces.append({
                        "name": data.get("name", f.stem),
                        "file": f.name,
                        "input": str(data.get("input", ""))[:100],
                        "output": str(data.get("output", ""))[:100],
                        "steps": len(data.get("steps", [])),
                        "duration_ms": data.get("duration_ms", 0),
                        "error": data.get("error"),
                        "timestamp": data.get("timestamp", 0),
                    })
                except (json.JSONDecodeError, Exception):
                    pass
        self._send_json(traces)

    def _serve_trace(self, name: str) -> None:
        traces_path = Path(self.traces_dir)
        fpath = traces_path / f"{name}.json"
        if not fpath.exists():
            fpath = traces_path / name
        if fpath.exists():
            self._send_json(json.loads(fpath.read_text()))
        else:
            self._send_json({"error": "not found"}, 404)

    def _serve_snapshots_list(self) -> None:
        snap_path = Path(self.snapshots_dir)
        snapshots = []
        if snap_path.exists():
            for f in sorted(snap_path.glob("*.json")):
                try:
                    data = json.loads(f.read_text())
                    snapshots.append({
                        "name": f.stem,
                        "file": f.name,
                        "steps": len(data.get("steps", [])),
                        "size": f.stat().st_size,
                    })
                except (json.JSONDecodeError, Exception):
                    pass
        self._send_json(snapshots)

    def _serve_snapshot(self, name: str) -> None:
        snap_path = Path(self.snapshots_dir) / f"{name}.json"
        if snap_path.exists():
            self._send_json(json.loads(snap_path.read_text()))
        else:
            self._send_json({"error": "not found"}, 404)

    def _send_json(self, data: object, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format: str, *args: object) -> None:
        pass  # suppress default logging


def start_server(
    port: int = 7600,
    traces_dir: str = ".",
    snapshots_dir: str = DEFAULT_SNAPSHOT_DIR,
) -> None:
    DashboardHandler.traces_dir = traces_dir
    DashboardHandler.snapshots_dir = snapshots_dir
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"agenteval dashboard: http://localhost:{port}")
    print(f"Traces: {traces_dir}")
    print(f"Snapshots: {snapshots_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>agenteval dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; }
.header { padding: 20px 32px; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 18px; font-weight: 600; }
.header .badge { background: #1a1a2e; color: #7c7cf0; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
.tabs { display: flex; gap: 0; border-bottom: 1px solid #222; padding: 0 32px; }
.tab { padding: 12px 20px; cursor: pointer; color: #888; border-bottom: 2px solid transparent; font-size: 14px; }
.tab.active { color: #fff; border-bottom-color: #7c7cf0; }
.tab:hover { color: #ccc; }
.content { padding: 24px 32px; max-width: 1200px; }
.card { background: #111; border: 1px solid #222; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }
.card-header { padding: 14px 18px; display: flex; justify-content: space-between; align-items: center; cursor: pointer; }
.card-header:hover { background: #161616; }
.card-title { font-weight: 500; font-size: 14px; }
.card-meta { display: flex; gap: 16px; font-size: 12px; color: #666; }
.card-meta .pass { color: #4caf50; }
.card-meta .fail { color: #f44336; }
.card-body { padding: 0 18px 18px; display: none; }
.card.open .card-body { display: block; }
.step { padding: 10px 14px; border-left: 3px solid #333; margin: 8px 0; background: #0d0d0d; border-radius: 0 4px 4px 0; font-size: 13px; }
.step.tool_call { border-left-color: #7c7cf0; }
.step.llm_response { border-left-color: #4caf50; }
.step-type { font-size: 11px; text-transform: uppercase; color: #666; margin-bottom: 4px; }
.step-name { font-weight: 600; color: #ddd; }
.step-detail { color: #999; font-size: 12px; margin-top: 4px; font-family: monospace; white-space: pre-wrap; word-break: break-all; }
.step-error { color: #f44336; }
.empty { color: #555; text-align: center; padding: 60px 20px; }
.io-block { margin: 8px 0; padding: 10px 14px; background: #0d0d0d; border-radius: 4px; }
.io-label { font-size: 11px; text-transform: uppercase; color: #666; margin-bottom: 4px; }
.io-value { font-family: monospace; font-size: 13px; white-space: pre-wrap; word-break: break-all; }
.error-badge { background: #2a1010; color: #f44336; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
.duration { color: #666; }
</style>
</head>
<body>

<div class="header">
  <h1>agenteval</h1>
  <span class="badge" id="trace-count">loading...</span>
</div>

<div class="tabs">
  <div class="tab active" data-tab="traces">Traces</div>
  <div class="tab" data-tab="snapshots">Snapshots</div>
</div>

<div class="content" id="content"></div>

<script>
let activeTab = 'traces';

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    activeTab = tab.dataset.tab;
    loadContent();
  });
});

async function loadContent() {
  if (activeTab === 'traces') await loadTraces();
  else await loadSnapshots();
}

async function loadTraces() {
  const el = document.getElementById('content');
  try {
    const res = await fetch('/api/traces');
    const traces = await res.json();
    document.getElementById('trace-count').textContent = traces.length + ' traces';
    if (!traces.length) { el.innerHTML = '<div class="empty">No traces found. Save traces as JSON files to view them here.</div>'; return; }
    el.innerHTML = traces.map((t, i) => `
      <div class="card" onclick="toggleCard(this, '${t.file}')">
        <div class="card-header">
          <span class="card-title">${esc(t.name || t.file)}</span>
          <div class="card-meta">
            <span>${t.steps} steps</span>
            <span class="duration">${t.duration_ms ? t.duration_ms.toFixed(0) + 'ms' : ''}</span>
            ${t.error ? '<span class="fail">ERROR</span>' : '<span class="pass">OK</span>'}
          </div>
        </div>
        <div class="card-body" id="body-${i}"></div>
      </div>
    `).join('');
  } catch(e) { el.innerHTML = '<div class="empty">Failed to load traces.</div>'; }
}

async function loadSnapshots() {
  const el = document.getElementById('content');
  try {
    const res = await fetch('/api/snapshots');
    const snaps = await res.json();
    if (!snaps.length) { el.innerHTML = '<div class="empty">No snapshots found.</div>'; return; }
    el.innerHTML = snaps.map((s, i) => `
      <div class="card" onclick="toggleCard(this, '${s.name}', true)">
        <div class="card-header">
          <span class="card-title">${esc(s.name)}</span>
          <div class="card-meta">
            <span>${s.steps} steps</span>
            <span class="duration">${(s.size/1024).toFixed(1)} KB</span>
          </div>
        </div>
        <div class="card-body" id="snap-${i}"></div>
      </div>
    `).join('');
  } catch(e) { el.innerHTML = '<div class="empty">Failed to load snapshots.</div>'; }
}

async function toggleCard(card, name, isSnap) {
  card.classList.toggle('open');
  const body = card.querySelector('.card-body');
  if (card.classList.contains('open') && !body.dataset.loaded) {
    const url = isSnap ? `/api/snapshot?name=${name}` : `/api/trace?name=${name.replace('.json','')}`;
    try {
      const res = await fetch(url);
      const data = await res.json();
      body.innerHTML = renderTrace(data);
      body.dataset.loaded = '1';
    } catch(e) { body.innerHTML = '<div class="empty">Failed to load.</div>'; }
  }
}

function renderTrace(t) {
  let html = '';
  if (t.input) html += `<div class="io-block"><div class="io-label">Input</div><div class="io-value">${esc(t.input)}</div></div>`;
  if (t.output) html += `<div class="io-block"><div class="io-label">Output</div><div class="io-value">${esc(t.output)}</div></div>`;
  if (t.error) html += `<div class="io-block"><div class="io-label">Error</div><div class="io-value step-error">${esc(t.error)}</div></div>`;
  (t.steps || []).forEach((s, i) => {
    const d = s.data || {};
    let detail = '';
    if (s.step_type === 'tool_call') {
      detail = `<span class="step-name">${esc(d.name || '')}</span>`;
      if (d.args && Object.keys(d.args).length) detail += `<div class="step-detail">${esc(JSON.stringify(d.args, null, 2))}</div>`;
      if (d.result !== null && d.result !== undefined) detail += `<div class="step-detail">-> ${esc(JSON.stringify(d.result).substring(0,500))}</div>`;
      if (d.error) detail += `<div class="step-detail step-error">ERROR: ${esc(d.error)}</div>`;
    } else if (s.step_type === 'llm_response') {
      detail = `<span class="step-name">${esc(d.model || 'LLM')}</span>`;
      detail += `<div class="step-detail">${d.input_tokens||0} in + ${d.output_tokens||0} out tokens</div>`;
      if (d.response) detail += `<div class="step-detail">${esc(d.response.substring(0,500))}</div>`;
    } else {
      detail = `<div class="step-detail">${esc(JSON.stringify(d).substring(0,500))}</div>`;
    }
    html += `<div class="step ${s.step_type}"><div class="step-type">${s.step_type} #${i}</div>${detail}</div>`;
  });
  return html || '<div class="empty">Empty trace</div>';
}

function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

loadContent();
</script>
</body>
</html>"""
