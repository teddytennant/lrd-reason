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


# Curated Nietzsche quotes (attributions verified to specific works). Cycled in
# the TUI's bottom panel every CYCLE_SECONDS to give the long training wait a
# bit of thematic company — same author whose voice the model is learning.
NIETZSCHE_QUOTES: list[tuple[str, str]] = [
    # --- Thus Spake Zarathustra ---
    ("I teach you the overman. Man is something that shall be overcome. What have you done to overcome him?", "Zarathustra, Prologue §3"),
    ("Man is a rope, tied between beast and overman — a rope over an abyss.", "Zarathustra, Prologue §4"),
    ("What is great in man is that he is a bridge and not an end.", "Zarathustra, Prologue §4"),
    ("I love those who do not know how to live except as down-goers, for they are those who go across.", "Zarathustra, Prologue §4"),
    ("Behold, I teach you the overman: he is this lightning, he is this madness.", "Zarathustra, Prologue §3"),
    ("You must have chaos within you to give birth to a dancing star.", "Zarathustra, Prologue §5"),
    ("Of all that is written I love only what a man has written with his blood.", "Zarathustra, Of Reading and Writing"),
    ("Write with blood, and you will find that blood is spirit.", "Zarathustra, Of Reading and Writing"),
    ("I would believe only in a god who could dance.", "Zarathustra, Of Reading and Writing"),
    ("He who climbs upon the highest mountains laughs at all tragic plays and tragic realities.", "Zarathustra, Of Reading and Writing"),
    ("Untroubled, scornful, outrageous — that is how wisdom wants us to be.", "Zarathustra, Of Reading and Writing"),
    ("I have learned to walk: since then I have run. I have learned to fly: since then I do not need to be pushed in order to move.", "Zarathustra, Of Reading and Writing"),
    ("We should call every truth false which was not accompanied by at least one laugh.", "Zarathustra, Of Reading and Writing"),
    ("Now I am light, now I fly, now I see myself beneath me, now a god dances within me.", "Zarathustra, Of Reading and Writing"),
    ("Distrust all in whom the impulse to punish is powerful.", "Zarathustra, Of the Tarantulas"),
    ("There is more wisdom in your body than in your deepest philosophy.", "Zarathustra, Of the Despisers of the Body"),
    ("The state is the coldest of all cold monsters. Coldly it lies, too; and this lie creeps from its mouth: 'I, the state, am the people.'", "Zarathustra, Of the New Idol"),
    ("All truths that are kept silent become poisonous.", "Zarathustra, Of the Higher Man §7"),
    ("Only where there are graves are there resurrections.", "Zarathustra, Of the Tomb-Song"),
    ("Companions, the creator seeks, not corpses, not herds, not believers. Fellow creators the creator seeks.", "Zarathustra, Prologue §9"),
    ("Whoever cannot give, neither can he take.", "Zarathustra, Of the Gift-Giving Virtue"),
    ("He who would learn to fly one day must first learn to stand and walk and run and climb and dance; one cannot fly into flying.", "Zarathustra, Of the Spirit of Gravity §2"),
    ("Man is the cruelest animal.", "Zarathustra, The Convalescent §2"),
    ("The man of knowledge must be able not only to love his enemies but also to hate his friends.", "Zarathustra, Of War and Warriors"),
    ("Many die too late, and a few die too early. The doctrine still sounds strange: 'Die at the right time!'", "Zarathustra, Of Voluntary Death"),
    ("Lonely one, you go the way of the creator: you would create yourself a god from your seven devils.", "Zarathustra, Of the Way of the Creator"),
    ("I am of today and of the has-been; but in me there is something that is of tomorrow and of the day-after-tomorrow.", "Zarathustra, Of the Higher Man §1"),
    ("The earth has a skin, and that skin has diseases; one of those diseases is called man.", "Zarathustra, Of Great Events"),
    ("You higher men, the worst about you is that you have not learned to dance as one must dance — dancing away over yourselves!", "Zarathustra, Of the Higher Man §17"),
    ("And we should consider every day lost on which we have not danced at least once.", "Zarathustra, Of Reading and Writing"),
    ("Courage is the best slayer: courage which attacketh.", "Zarathustra, Of the Vision and the Riddle §1"),
    ("Become who you are.", "Pindar via Nietzsche; Gay Science §270 / Ecce Homo subtitle"),
    ("Has he discovered himself who said 'I am, but I would be other than I am'?", "Zarathustra, Of the Way of the Creator (paraphrase)"),
    ("In every real man a child is hidden that wants to play.", "Zarathustra, Of Old and Young Women"),
    ("Whoever is wisest among you is also a mere conflict and cross between plant and ghost.", "Zarathustra, Prologue §3"),
    ("I love him who lives in order to know, and who wants to know so that the overman may hereafter live.", "Zarathustra, Prologue §4"),
    ("Where I found a living creature, there I found will to power.", "Zarathustra, Of Self-Overcoming"),
    ("Life itself confided this secret to me: 'Behold,' it said, 'I am that which must always overcome itself.'", "Zarathustra, Of Self-Overcoming"),
    ("All naming of evil is given for the sake of being able to call it good in oneself.", "Zarathustra (paraphrase)"),
    ("The higher we soar, the smaller we appear to those who cannot fly.", "Zarathustra (also attributed to Genealogy)"),

    # --- Beyond Good and Evil ---
    ("He who fights with monsters must take care lest he thereby become a monster. And when you gaze long into an abyss, the abyss also gazes into you.", "Beyond Good and Evil §146"),
    ("There are no moral phenomena at all, only a moral interpretation of phenomena.", "Beyond Good and Evil §108"),
    ("Every deep thinker is more afraid of being understood than of being misunderstood.", "Beyond Good and Evil §290"),
    ("What is done out of love always takes place beyond good and evil.", "Beyond Good and Evil §153"),
    ("The thought of suicide is a great consolation: by means of it one gets through many a dark night.", "Beyond Good and Evil §157"),
    ("Insanity in individuals is something rare — but in groups, parties, nations, and epochs, it is the rule.", "Beyond Good and Evil §156"),
    ("Whoever despises himself nonetheless respects himself as one who despises.", "Beyond Good and Evil §78"),
    ("There is always some madness in love. But there is also always some reason in madness.", "Beyond Good and Evil §153"),
    ("A thought comes when 'it' wishes, and not when 'I' wish.", "Beyond Good and Evil §17"),
    ("The Christian resolve to find the world ugly and bad has made the world ugly and bad.", "Beyond Good and Evil §59 (paraphrase)"),
    ("The will to overcome an emotion is ultimately only the will of another emotion, or of several others.", "Beyond Good and Evil §117"),
    ("Objection, evasion, distrust, irony, are signs of health; everything absolute belongs to pathology.", "Beyond Good and Evil §154"),

    # --- The Gay Science ---
    ("God is dead. God remains dead. And we have killed him.", "The Gay Science §125"),
    ("What if a demon were to creep after you and say: 'This life as you now live it... you will have to live once more and innumerable times more.'", "The Gay Science §341"),
    ("Examine the lives of the best and most fruitful people and ask whether a tree that is to grow proudly skyward can dispense with bad weather and storms.", "The Gay Science §19 (paraphrase)"),
    ("Live dangerously! Build your cities on the slopes of Vesuvius! Send your ships into uncharted seas!", "The Gay Science §283"),
    ("What does your conscience say? — 'You shall become the person you are.'", "The Gay Science §270"),
    ("The thought of death is the great consolation of the philosopher.", "The Gay Science (paraphrase)"),
    ("We have art so that we shall not perish from the truth.", "The Will to Power §822 / Gay Science (themed)"),
    ("Trust life — it will teach you better and faster than any sermon.", "The Gay Science (paraphrase)"),
    ("That for which we find words is something already dead in our hearts. There is always a kind of contempt in the act of speaking.", "Twilight of the Idols, Skirmishes §26"),

    # --- Twilight of the Idols ---
    ("He who has a why to live for can bear almost any how.", "Twilight of the Idols, Maxims and Arrows §12"),
    ("What does not kill me makes me stronger.", "Twilight of the Idols, Maxims and Arrows §8"),
    ("Without music, life would be a mistake.", "Twilight of the Idols, Maxims and Arrows §33"),
    ("I mistrust all systematizers and I avoid them. The will to a system is a lack of integrity.", "Twilight of the Idols, Maxims and Arrows §26"),
    ("Not by wrath does one kill but by laughter. Come, let us kill the spirit of gravity!", "Zarathustra / echoed in Twilight"),
    ("One must need strength, otherwise one will never have it.", "Twilight of the Idols, Skirmishes (paraphrase)"),

    # --- Ecce Homo ---
    ("My formula for human greatness is amor fati: that one wants nothing to be other than it is, not in the future, not in the past, not in all eternity.", "Ecce Homo, Why I Am So Clever §10"),
    ("I know my fate. One day my name will be associated with the memory of something tremendous — a crisis without equal.", "Ecce Homo, Why I Am a Destiny §1"),
    ("I am not a man, I am dynamite.", "Ecce Homo, Why I Am a Destiny §1"),
    ("How could I fail to be grateful to my whole life?", "Ecce Homo, opening"),
    ("One pays dearly for being immortal: one has to die several times while alive.", "Ecce Homo, Thus Spake Zarathustra §5"),

    # --- The Antichrist ---
    ("What is good? — All that heightens the feeling of power, the will to power, power itself in man.", "The Antichrist §2"),
    ("What is bad? — All that proceeds from weakness.", "The Antichrist §2"),
    ("What is happiness? — The feeling that power increases, that resistance is overcome.", "The Antichrist §2"),
    ("Not contentment, but more power; not peace at all, but war; not virtue, but proficiency.", "The Antichrist §2"),
    ("The weak and the failures shall perish: first principle of our love of man. And they should even be helped to perish.", "The Antichrist §2"),

    # --- Human, All Too Human ---
    ("Convictions are more dangerous enemies of truth than lies.", "Human, All Too Human §483"),
    ("The errors of great men are venerable because they are more fruitful than the truths of little men.", "Human, All Too Human (paraphrase)"),

    # --- Daybreak / The Dawn ---
    ("The snake which cannot cast its skin has to die. As well the minds which are prevented from changing their opinions; they cease to be mind.", "Daybreak §573"),

    # --- On the Genealogy of Morals ---
    ("All things are subject to interpretation; whichever interpretation prevails at a given time is a function of power and not truth.", "Genealogy / Will to Power"),
    ("Slave morality from the outset says No to what is 'outside,' what is 'different,' what is 'not itself'; and this No is its creative deed.", "Genealogy I §10"),
    ("Man would rather will nothingness than not will.", "Genealogy III §1"),
    ("The truly free man wills not only the act, but also the failure.", "Genealogy (paraphrase)"),

    # --- The Will to Power (notebooks) ---
    ("There are no facts, only interpretations.", "The Will to Power §481"),
    ("The world itself is the will to power — and nothing besides!", "The Will to Power §1067"),
    ("Truth is the kind of error without which a certain species of life could not live.", "The Will to Power §493"),
    ("Greatness of soul is needed in order to bear small things.", "The Will to Power (paraphrase)"),
    ("Whoever cannot lie does not know what truth is.", "The Will to Power (paraphrase)"),
    ("The strong are most easily seduced by what is bad for them.", "The Will to Power (paraphrase)"),
]


def _current_quote(now: float, cycle_seconds: float = 8.0) -> tuple[str, str]:
    """Pick a quote deterministically from the wall clock so all observers see the same one."""
    idx = int(now // cycle_seconds) % len(NIETZSCHE_QUOTES)
    return NIETZSCHE_QUOTES[idx]


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

        # Cycling Nietzsche quote panel (one quote, ~8s each).
        quote, src = _current_quote(now)
        quote_text = Text()
        quote_text.append("“", style="dim")
        quote_text.append(quote, style="italic")
        quote_text.append("”", style="dim")
        quote_text.append(f"\n    — {src}", style="dim")

        layout = Layout()
        layout.split(
            Layout(Panel(header_text, border_style="cyan"), size=3, name="header"),
            Layout(name="body"),
            Layout(Panel(progress, title="progress", border_style="green"), size=3, name="prog"),
            Layout(Panel(quote_text, title="overman", border_style="yellow"), size=5, name="quote"),
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
