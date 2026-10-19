# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Atomic file writes, JSONL streams and kernel locks with a poll deadline and a writer's gate."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

_CLOCK: Callable[[], float] = time.monotonic  # replaced by tests so a deadline passes without waiting
_PAUSE: Callable[[float], None] = time.sleep
_POLL_S = 0.2
_HELD_FDS: set[int] = set()  # the descriptors whose flock this process holds now


def waited(seconds: float, what: str, detail: str) -> None:
    """Called once for every lock that blocked, with the seconds it blocked: the hook that lets the stage recorder
    show what a command waited on. Replaced by `abucache.stages`; a bare use of this module drops the news."""


def write(path: Path, content: bytes) -> bool:
    """The one way a file reaches a tree: written atomically and only when its bytes differ (mtime kept). Returns
    whether the file changed."""
    try:
        if path.read_bytes() == content:
            return False
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{uuid.uuid4().hex}.part")  # the name alone may already sit near the 255-byte limit
    part.write_bytes(content)
    part.replace(path)
    return True


def scratch(root: Path) -> Path:
    """The temporary directory every child of a tree gets (TMPDIR), created when first needed."""
    path = root / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: Path, row: Any) -> None:
    """Append one JSON line as a single O_APPEND write under an exclusive lock (`<name>.lock` beside the file)."""
    line = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock(path.with_name(path.name + ".lock")), path.open("ab") as handle:
        handle.write(line.encode())


def read_jsonl(path: Path) -> list[Any]:
    """Every line of a JSONL file; a missing file is empty. A line that does not parse raises ValueError naming
    `path:line`, which a caller turns into its own finding."""
    if not path.exists():
        return []
    result: list[Any] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            result.append(json.loads(raw))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return result


def _poll(fd: int, flags: int, wait_s: float) -> bool:
    """Try the lock every _POLL_S until the deadline; False when it passed with the lock still held elsewhere."""
    deadline = _CLOCK() + wait_s
    while True:
        try:
            fcntl.flock(fd, flags | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if _CLOCK() >= deadline:
                return False
            _PAUSE(_POLL_S)


@contextmanager
def _flock(path: Path, flags: int, *, wait: bool = True, wait_s: float | None = None) -> Iterator[int | None]:
    """Hold the lock; a contended lock yields None when `wait` is false or `wait_s` ran out, else blocks and reports
    the wait."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fd = handle.fileno()
        try:
            fcntl.flock(fd, flags | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                yield None
                return
            start = time.perf_counter()
            if wait_s is None:
                fcntl.flock(fd, flags)
                taken = True
            else:
                taken = _poll(fd, flags, wait_s)
            waited(time.perf_counter() - start, "lock", path.name)
            if not taken:
                yield None
                return
        _HELD_FDS.add(fd)
        try:
            yield fd
        finally:
            _HELD_FDS.discard(fd)


@contextmanager
def lock(
    path: Path,
    *,
    shared: bool = False,
    wait: bool = True,
    deadline_s: float | None = None,
    gate: Path | None = None,
) -> Iterator[int | None]:
    """Hold the kernel lock on `path`; yields its descriptor, or None when the lock is contended and `wait` is false
    or `deadline_s` seconds passed. The gate rule: a writer that waits with a deadline first holds `gate` exclusive
    (a wait on it counts against the deadline), so readers arriving meanwhile, who pass the gate shared before asking
    for the lock, queue behind it and cannot starve it. A writer given a gate and no deadline passes no gate."""
    flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    with ExitStack() as stack:
        left = deadline_s
        if gate is not None and not shared and wait and deadline_s is not None:
            start = _CLOCK()
            if stack.enter_context(_flock(gate, fcntl.LOCK_EX, wait_s=deadline_s)) is None:
                yield None
                return
            left = max(0.0, deadline_s - (_CLOCK() - start))
        elif gate is not None and shared and gate.exists():
            with _flock(gate, fcntl.LOCK_SH):
                pass
        yield stack.enter_context(_flock(path, flags, wait=wait, wait_s=left))


def drop_inherited() -> None:
    """A forked worker lets go of the flocks it inherited without unlocking them for the parent: each descriptor is
    pointed at /dev/null, so the lock lives exactly as long as the parent holds it."""
    if not _HELD_FDS:
        return
    null = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in _HELD_FDS:
            os.dup2(null, fd)
    finally:
        os.close(null)
    _HELD_FDS.clear()
