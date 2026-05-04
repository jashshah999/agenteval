"""CLI for agenteval."""

import json
import click
from pathlib import Path

from rich.console import Console
from rich.table import Table


@click.group()
@click.version_option()
def main():
    """agenteval: Testing framework for AI agents."""
    pass


@main.command()
@click.argument("trace_file", type=click.Path(exists=True))
def inspect(trace_file):
    """Inspect a saved trace file."""
    from .trace import Trace

    data = json.loads(Path(trace_file).read_text())
    trace = Trace.from_dict(data)
    console = Console()

    console.print(f"\n[bold]Trace: {trace.name}[/bold]")
    console.print(f"Input: {trace.input[:200]}")
    console.print(f"Output: {trace.output[:200]}")
    console.print(f"Steps: {len(trace.steps)}")
    console.print(f"Duration: {trace.duration_ms:.0f}ms")
    if trace.error:
        console.print(f"[red]Error: {trace.error}[/red]")

    table = Table(title="Steps")
    table.add_column("#", justify="right")
    table.add_column("Type")
    table.add_column("Details")

    for i, step in enumerate(trace.steps):
        if step.step_type == "tool_call":
            tc = step.data
            details = f"{tc.name}({json.dumps(tc.args, default=str)[:80]})"
            if tc.error:
                details += f" [red]ERROR: {tc.error}[/red]"
        elif step.step_type == "llm_response":
            lr = step.data
            details = f"{lr.model} ({lr.input_tokens}+{lr.output_tokens} tokens)"
        else:
            details = str(step.data)[:80]
        table.add_row(str(i), step.step_type, details)

    console.print(table)


@main.command()
@click.argument("snapshot_dir", type=click.Path(), default=".agenteval_snapshots")
def snapshots(snapshot_dir):
    """List saved snapshots."""
    console = Console()
    snap_path = Path(snapshot_dir)

    if not snap_path.exists():
        console.print("No snapshots found.")
        return

    files = sorted(snap_path.glob("*.json"))
    if not files:
        console.print("No snapshots found.")
        return

    table = Table(title="Snapshots")
    table.add_column("Name")
    table.add_column("Size", justify="right")
    table.add_column("Steps", justify="right")

    for f in files:
        data = json.loads(f.read_text())
        n_steps = len(data.get("steps", []))
        size = f"{f.stat().st_size / 1024:.1f} KB"
        table.add_row(f.stem, size, str(n_steps))

    console.print(table)


@main.command()
@click.option("--port", type=int, default=7600, help="Port to serve on.")
@click.option("--traces-dir", type=click.Path(), default=".", help="Directory with trace JSON files.")
@click.option("--snapshots-dir", type=click.Path(), default=".agenteval_snapshots", help="Snapshots directory.")
def server(port, traces_dir, snapshots_dir):
    """Launch the web dashboard for viewing traces and snapshots."""
    from .server import start_server
    start_server(port=port, traces_dir=traces_dir, snapshots_dir=snapshots_dir)


if __name__ == "__main__":
    main()
