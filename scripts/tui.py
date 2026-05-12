"""Tiny live dashboard for the uncensored-finetune one-liner.

Polls two artifacts produced by ``launch.sh``:
  - ``logs/metrics.jsonl``  — one JSON record per logging step (written by
                              ``JsonlMetricsCallback`` in finetune_uncensored.py).
  - ``logs/train.stdout``   — raw trainer stdout (we tail the last N lines).

Renders a 3-panel rich layout (header / stats / log tail). Exits cleanly on
``q`` or Ctrl-C. Exiting the TUI does NOT kill the training process — that
decision is left to launch.sh's signal trap.

Usage:
    python scripts/tui.py --metrics logs/metrics.jsonl --stdout logs/train.stdout \\
        --pid <train-pid> [--stage prepare|train|done]

Designed to degrade gracefully: if metrics.jsonl doesn't exist yet, shows
"waiting for first log step"; if rich isn't installed, falls back to plain tail.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


def _plain_fallback(stdout_path: Path) -> int:
    """If rich isn't installed, just tail the stdout file."""
    print("[tui] rich not installed — falling back to plain tail", file=sys.stderr)
    print(f"[tui] tailing {stdout_path} (Ctrl-C to exit)", file=sys.stderr)
    try:
        last_size = 0
        while True:
            if stdout_path.exists():
                size = stdout_path.stat().st_size
                if size > last_size:
                    with stdout_path.open() as f:
                        f.seek(last_size)
                        sys.stdout.write(f.read())
                        sys.stdout.flush()
                    last_size = size
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def _read_last_metric(metrics_path: Path) -> dict[str, Any] | None:
    """Read the most recent JSONL record. Returns None if file empty/missing."""
    if not metrics_path.exists():
        return None
    last_line: str | None = None
    try:
        with metrics_path.open("rb") as f:
            # Seek near end if file is large
            f.seek(0, os.SEEK_END)
            size = f.tell()
            window = min(size, 8192)
            f.seek(size - window)
            tail = f.read().decode("utf-8", errors="replace")
            for line in tail.splitlines():
                line = line.strip()
                if line:
                    last_line = line
    except OSError:
        return None
    if last_line is None:
        return None
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


def _tail_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            window = min(size, 16384)
            f.seek(size - window)
            raw = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = raw.splitlines()
    return lines[-n:]


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fmt_eta(remaining_steps: int, sec_per_step: float | None) -> str:
    if sec_per_step is None or remaining_steps <= 0:
        return "—"
    secs = int(remaining_steps * sec_per_step)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def run_rich(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.text import Text

    metrics_path = Path(args.metrics)
    stdout_path = Path(args.stdout)

    console = Console()

    stage = args.stage or "train"
    start_wall = time.time()
    # We initialize last_step lazily from the first metric record so any
    # pre-existing steps in metrics.jsonl don't get mis-attributed to "now".
    last_step: int | None = None
    last_step_wall: float | None = None
    sec_per_step_ema: float | None = None
    ema_alpha = 0.3

    progress = Progress(
        TextColumn("[bold]step[/]"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True,
    )
    task_id = progress.add_task("train", total=1, completed=0)

    def render() -> Layout:
        nonlocal last_step, last_step_wall, sec_per_step_ema
        metric = _read_last_metric(metrics_path)
        step = int(metric.get("step") or 0) if metric else 0
        max_steps = int(metric.get("max_steps") or 0) if metric else 0
        epoch = metric.get("epoch") if metric else None
        loss = metric.get("loss") if metric else None
        lr = metric.get("learning_rate") if metric else None
        grad_norm = metric.get("grad_norm") if metric else None

        now = time.time()
        # Use the metric's own wall-clock so initial backfill doesn't look
        # like it happened "now"; first observation seeds without rate calc.
        metric_wall = metric.get("wall") if metric else None
        if step > 0 and metric_wall is not None:
            if last_step is None:
                last_step = step
                last_step_wall = float(metric_wall)
            elif step > last_step:
                d_step = step - last_step
                d_wall = max(float(metric_wall) - (last_step_wall or float(metric_wall)), 1e-6)
                inst = d_wall / d_step
                sec_per_step_ema = inst if sec_per_step_ema is None else (
                    ema_alpha * inst + (1 - ema_alpha) * sec_per_step_ema
                )
                last_step = step
                last_step_wall = float(metric_wall)

        if max_steps > 0:
            progress.update(task_id, total=max_steps, completed=step)

        # Header
        alive = _pid_alive(args.pid)
        status = "[green]running[/]" if alive else "[red]exited[/]"
        elapsed = int(now - start_wall)
        eh, er = divmod(elapsed, 3600)
        em, es = divmod(er, 60)
        elapsed_s = f"{eh}h{em:02d}m{es:02d}s" if eh else f"{em}m{es:02d}s"
        header_text = Text.from_markup(
            f"[bold]lrd-uncensored[/] · stage=[cyan]{stage}[/] · pid={args.pid or '—'} · {status} · elapsed={elapsed_s}"
        )

        # Stats table
        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="dim", justify="right")
        tbl.add_column()

        def _fmt(v: Any, fmt: str = "{}") -> str:
            if v is None:
                return "—"
            try:
                return fmt.format(v)
            except (ValueError, TypeError):
                return str(v)

        eta = _fmt_eta(max_steps - step, sec_per_step_ema) if max_steps else "—"
        sps = f"{1.0/sec_per_step_ema:.2f}" if sec_per_step_ema else "—"

        tbl.add_row("loss", _fmt(loss, "{:.4f}"))
        tbl.add_row("lr", _fmt(lr, "{:.2e}"))
        tbl.add_row("grad_norm", _fmt(grad_norm, "{:.3f}"))
        tbl.add_row("epoch", _fmt(epoch, "{:.3f}"))
        tbl.add_row("steps/sec", sps)
        tbl.add_row("eta", eta)

        # Log tail
        tail = _tail_lines(stdout_path, args.tail_lines)
        log_text = "\n".join(tail) if tail else "(no stdout yet)"

        layout = Layout()
        layout.split(
            Layout(Panel(header_text, border_style="cyan"), size=3, name="header"),
            Layout(name="body"),
            Layout(Panel(progress, title="progress", border_style="green"), size=3, name="prog"),
        )
        layout["body"].split_row(
            Layout(Panel(tbl, title="metrics", border_style="magenta"), name="stats", ratio=1),
            Layout(Panel(log_text, title=f"stdout (last {args.tail_lines})", border_style="dim"),
                   name="log", ratio=3),
        )
        return layout

    # Graceful Ctrl-C: exit without killing training.
    interrupted = {"v": False}
    def _sigint(_signum, _frame):
        interrupted["v"] = True
    signal.signal(signal.SIGINT, _sigint)

    with Live(render(), console=console, refresh_per_second=2, screen=False) as live:
        while not interrupted["v"]:
            time.sleep(0.5)
            live.update(render())
            if args.pid is not None and not _pid_alive(args.pid):
                # One more refresh after the trainer dies, then exit.
                time.sleep(1.0)
                live.update(render())
                break

    console.print("[tui] detached.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="path to metrics.jsonl")
    ap.add_argument("--stdout", required=True, help="path to trainer stdout log")
    ap.add_argument("--pid", type=int, default=None, help="trainer PID (for liveness check)")
    ap.add_argument("--stage", default=None, help="stage label shown in header")
    ap.add_argument("--tail-lines", type=int, default=20)
    args = ap.parse_args()

    try:
        import rich  # noqa: F401
    except ImportError:
        return _plain_fallback(Path(args.stdout))

    return run_rich(args)


if __name__ == "__main__":
    sys.exit(main())
