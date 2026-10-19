# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The event feed of the shared timing path: a stage or wait that runs long reaches the feed without per-tool code.

`stages` binds this module to the runtime file's `[events]` section at every command start. A span (command, stage
or pool dispatch, so also every slot wait and queue wait, which are stages) is slow once it has run for
`slow_seconds` and for `overrun_percent` of the wall the previous command spent in the same stage path. A slow span
is announced twice: once while it still runs (a watchdog thread looks at the open spans, so a stuck job is seen
while it is stuck) and once when it closes, with its final wall. Nothing is written for a span that stays under its
limit, so a healthy command adds no lines. Events carry no kind: they are factual `info` (slow) or `blocked` (a wait)
lines of the tool that ran the command."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from abucache import events

INTERVAL_MAX = 15.0  # seconds between the watchdog's looks at the open spans, at most

Open = tuple[str, tuple[str, ...], float, float | None, str]  # span id, path, seconds open, expected seconds, kind

path: Path | None = None  # the feed; None until a runtime file is bound
slow_seconds = 0
overrun_percent = 150
_told: set[str] = set()  # spans already announced as still running
_thread: threading.Thread | None = None
_stop = threading.Event()


def bind(runtime: Any) -> None:
    """Takes the feed and its limits from a runtime (None turns the feed off)."""
    global path, slow_seconds, overrun_percent
    path = None if runtime is None else runtime.feed
    slow_seconds = 0 if runtime is None else runtime.slow_seconds
    overrun_percent = 150 if runtime is None else runtime.overrun_percent


def limit(expected: float | None) -> float:
    """Seconds a span with this expected wall may run before it is slow."""
    return max(float(slow_seconds), (expected or 0.0) * overrun_percent / 100)


def slow(seconds: float, expected: float | None) -> bool:
    return path is not None and slow_seconds > 0 and seconds >= limit(expected)


def _waiting(stage: tuple[str, ...]) -> bool:
    return stage[-1].startswith("pool.wait")


def say(
    tool: str,
    stage: tuple[str, ...],
    seconds: float,
    expected: float | None,
    *,
    running: bool,
    detail: dict[str, Any] | None = None,
    failed: Callable[[str], None] = lambda reason: None,
) -> None:
    """One feed line for a slow span, still running or closed. A feed that cannot be written reports through `failed`
    (the command's stage tree) instead of dropping the line silently."""
    if path is None:
        return
    name = " > ".join(stage)
    state = "still running" if running else "finished"
    message = f"{name} {state} after {seconds:.0f}s"
    if expected:
        message += f" (the last command took {expected:.0f}s)"
    payload: dict[str, Any] = {
        "stage": name,
        "seconds": round(seconds, 1),
        "running": running,
        "pid": os.getpid(),
        **({"expected_seconds": round(expected, 1)} if expected else {}),
        **(detail or {}),
    }
    try:
        status = "blocked" if _waiting(stage) else "info"
        events.append(path, events.make_event(tool or "abucache", status, message, payload=payload))
    except (OSError, ValueError) as error:
        failed(f"{path}: {error}")


def start(tool: str, opened: Callable[[], Sequence[Open]], failed: Callable[[str], None] = lambda reason: None) -> None:
    """Starts the watchdog of a command: every few seconds it announces the open spans that went slow while running."""
    global _thread
    stop()
    if path is None or slow_seconds <= 0:
        return
    _told.clear()
    _stop.clear()
    interval = max(0.05, min(INTERVAL_MAX, slow_seconds / 4))

    def watch() -> None:
        while not _stop.wait(interval):
            for span, stage, seconds, expected, _ in opened():
                if slow(seconds, expected) and span not in _told:
                    _told.add(span)
                    say(tool, stage, seconds, expected, running=True, failed=failed)

    _thread = threading.Thread(target=watch, name="abucache-feed", daemon=True)
    _thread.start()


def announced(span: str) -> bool:
    """Whether the watchdog announced this span while it ran."""
    return span in _told


def stop() -> None:
    global _thread
    thread, _thread = _thread, None
    if thread is not None:
        _stop.set()
        thread.join(timeout=5)


def _after_fork() -> None:
    """A forked worker has no watchdog: it keeps no lock or thread of its parent's."""
    global _thread
    _thread = None
    _told.clear()


os.register_at_fork(after_in_child=_after_fork)
