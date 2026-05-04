"""Snapshot testing for agent traces.

Record a trace, save it as a snapshot. On re-run, compare to detect regressions
in tool call patterns, output structure, or behavior changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .trace import Trace


DEFAULT_SNAPSHOT_DIR = ".agenteval_snapshots"


def snapshot_match(
    trace: Trace,
    snapshot_name: str,
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
    update: bool = False,
    ignore_fields: list[str] | None = None,
) -> SnapshotResult:
    """Compare a trace against a saved snapshot.

    First run: saves the snapshot and passes.
    Subsequent runs: compares and reports diffs.

    Args:
        trace: Current trace to compare.
        snapshot_name: Name for this snapshot.
        snapshot_dir: Directory to store snapshots.
        update: If True, overwrite existing snapshot.
        ignore_fields: Fields to ignore in comparison (e.g., ["timestamp", "duration_ms"]).
    """
    snap_path = Path(snapshot_dir) / f"{snapshot_name}.json"
    ignore = set(ignore_fields or ["timestamp", "duration_ms"])

    current = _normalize(trace.to_dict(), ignore)

    if not snap_path.exists() or update:
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps(current, indent=2, default=str))
        return SnapshotResult(
            matched=True,
            message=f"Snapshot saved: {snap_path}" if not snap_path.exists() or update else "",
            is_new=True,
        )

    saved = json.loads(snap_path.read_text())
    diffs = _diff(saved, current, path="")

    if not diffs:
        return SnapshotResult(matched=True, message="Snapshot matches")

    diff_summary = "\n".join(f"  {d}" for d in diffs[:20])
    if len(diffs) > 20:
        diff_summary += f"\n  ... and {len(diffs) - 20} more"

    return SnapshotResult(
        matched=False,
        message=f"Snapshot mismatch ({len(diffs)} diffs):\n{diff_summary}",
        diffs=diffs,
    )


class SnapshotResult:
    def __init__(
        self,
        matched: bool,
        message: str = "",
        is_new: bool = False,
        diffs: list[str] | None = None,
    ):
        self.matched = matched
        self.message = message
        self.is_new = is_new
        self.diffs = diffs or []

    def __bool__(self) -> bool:
        return self.matched

    def __repr__(self) -> str:
        status = "MATCH" if self.matched else "MISMATCH"
        return f"SnapshotResult({status}: {self.message})"


def _normalize(obj: Any, ignore: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v, ignore) for k, v in obj.items() if k not in ignore}
    elif isinstance(obj, list):
        return [_normalize(v, ignore) for v in obj]
    return obj


def _diff(expected: Any, actual: Any, path: str) -> list[str]:
    diffs = []

    if not isinstance(expected, type(actual)) and not isinstance(actual, type(expected)):
        diffs.append(f"{path}: type changed {type(expected).__name__} -> {type(actual).__name__}")
        return diffs

    if isinstance(expected, dict):
        all_keys = set(expected) | set(actual)
        for k in sorted(all_keys):
            child_path = f"{path}.{k}" if path else k
            if k not in expected:
                diffs.append(f"{child_path}: added")
            elif k not in actual:
                diffs.append(f"{child_path}: removed")
            else:
                diffs.extend(_diff(expected[k], actual[k], child_path))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length changed {len(expected)} -> {len(actual)}")
        for i in range(min(len(expected), len(actual))):
            diffs.extend(_diff(expected[i], actual[i], f"{path}[{i}]"))
    elif expected != actual:
        exp_str = str(expected)[:100]
        act_str = str(actual)[:100]
        diffs.append(f"{path}: {exp_str} -> {act_str}")

    return diffs
