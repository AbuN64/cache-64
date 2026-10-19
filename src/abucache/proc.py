# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Native execution with per-child accounting and process-group timeouts."""

from __future__ import annotations

import ctypes
import os
import selectors
import signal
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from time import perf_counter_ns
from typing import IO

from abucache import slots

_PR_SET_CHILD_SUBREAPER = 36  # linux/prctl.h


@dataclass(frozen=True)
class Completed:
    """What one child did. `exit` is the exit code, or None when a signal ended it (then `signal` is its number)."""

    argv: tuple[str, ...]
    cwd: str
    exit: int | None
    signal: int | None
    stdout: bytes
    stderr: bytes
    wall: float  # seconds
    cpu: float  # seconds of CPU of the child and every process it left behind
    max_rss: int  # bytes
    start_ns: int = 0  # perf_counter_ns at spawn


def _kill(pid: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


@cache
def subreaper() -> None:
    """Once per process (the setting is process-wide): orphans of a killed child come to us to be reaped."""
    ctypes.CDLL(None).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)


def _reap(group: int) -> float:
    """Kills what the child left in its process group and reaps it, returning the CPU seconds of every member: a
    killed child never waited for its own children, which are ours (subreaper) and only we can account for."""
    _kill(group)
    cpu = 0.0
    while True:
        try:
            _, _, usage = os.wait4(-group, 0)
        except ChildProcessError:
            return cpu
        cpu += usage.ru_utime + usage.ru_stime


def run(
    argv: Sequence[str | Path],
    cwd: Path,
    *,
    tmp: Path,
    stdin: bytes = b"",
    stdout_path: Path | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] = {},
    pass_fds: Sequence[int] = (),
    cores: int = 0,
) -> Completed:
    """Run one child in its own process group and session with TMPDIR=`tmp`, feeding `stdin` and collecting both
    streams (stdout goes to `stdout_path` when given). A `timeout` kills the whole group. Raises OSError when the
    child cannot start. `cores` > 0 holds that many slots (at most the host's fair share) for the child's life and
    tells it the grant as UNBAKE_CORES; control-plane calls (git, systemctl, ssh) pass 0 and hold none."""
    tmp.mkdir(parents=True, exist_ok=True)
    start = perf_counter_ns()
    subreaper()
    deadline = None if timeout is None else start + timeout * 1_000_000_000
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    with ExitStack() as stack:
        granted = {slots.CORES_ENV: str(stack.enter_context(slots.cores(cores)))} if cores > 0 else {}
        destination: int | IO[bytes] = subprocess.PIPE
        if stdout_path is not None:
            destination = stack.enter_context(stdout_path.open("wb"))
        child = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=destination,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env={**os.environ, **env, **granted, "TMPDIR": str(tmp)},
            pass_fds=tuple(pass_fds),
        )
        try:
            selector = stack.enter_context(selectors.DefaultSelector())
            pipes: dict[str, IO[bytes]] = {}
            for role, out in (("stdout", child.stdout), ("stderr", child.stderr)):
                if out is not None:
                    pipes[role] = stack.enter_context(out)
                    os.set_blocking(out.fileno(), False)
                    selector.register(out, selectors.EVENT_READ, role)
            assert child.stdin is not None  # opened with stdin=PIPE
            pipes["stdin"] = stack.enter_context(child.stdin)
            pending = memoryview(stdin)
            if pending:
                os.set_blocking(child.stdin.fileno(), False)
                selector.register(child.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                child.stdin.close()
            while selector.get_map():
                remaining = None
                if deadline is not None:
                    remaining = (deadline - perf_counter_ns()) / 1_000_000_000
                    if remaining <= 0:
                        _kill(child.pid)
                        deadline = None
                        remaining = None
                for key, _ in selector.select(remaining):
                    pipe = pipes[key.data]
                    if key.data == "stdin":
                        try:
                            pending = pending[os.write(pipe.fileno(), pending[:65536]) :]
                        except BrokenPipeError:
                            pending = pending[:0]
                        except BlockingIOError:
                            continue
                        if not pending:
                            selector.unregister(pipe)
                            pipe.close()
                    else:
                        try:
                            chunk = os.read(pipe.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if chunk:
                            streams[key.data].extend(chunk)
                        else:
                            selector.unregister(pipe)
                            pipe.close()
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
            orphans = _reap(child.pid)
        finally:
            if child.returncode is None:
                _kill(child.pid)
                _, status, _ = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
                _reap(child.pid)
    code = child.returncode
    return Completed(
        tuple(str(arg) for arg in argv),
        str(cwd),
        code if code >= 0 else None,
        -code if code < 0 else None,
        bytes(streams["stdout"]),
        bytes(streams["stderr"]),
        (perf_counter_ns() - start) / 1_000_000_000,
        usage.ru_utime + usage.ru_stime + orphans,
        usage.ru_maxrss * 1024,
        start,
    )
