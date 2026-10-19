# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Start a command inside a systemd user slice, so a host that splits cores between users of the machine splits
them by slice weight and not by how many ssh sessions happen to be busy.

When the process is not already in the slice it re-execs itself under `systemd-run --user --scope`; make, gcc and
the pools it starts inherit the slice. A service is placed by its unit's `Slice=`; only session processes are moved.
A host without systemd-run, or one that refuses, runs the command in place with one warning line."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


def in_slice(slice_name: str, cgroup: str) -> bool:
    """Whether a /proc/<pid>/cgroup text puts the process under a cgroup directory named `slice_name`."""
    return any(slice_name in line.partition(":")[2].partition(":")[2].split("/") for line in cgroup.splitlines())


def in_service(cgroup: str) -> bool:
    """Whether the unified (`0::`) cgroup path of a /proc/<pid>/cgroup text ends in a `.service` component."""
    for line in cgroup.splitlines():
        if line.startswith("0::"):
            return line[3:].rstrip("/").rpartition("/")[2].endswith(".service")
    return False


def _read_cgroup() -> str:
    try:
        return Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return ""


def _systemd_run(slice_name: str, command: Sequence[str]) -> list[str]:
    return ["systemd-run", "--user", "--scope", "--quiet", f"--slice={slice_name}", "--", *command]


def _clean(guard_env: str, slice_name: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != guard_env}
    if slice_name:
        env[guard_env] = slice_name
    return env


def enter(
    slice_name: str | None,
    argv: Sequence[str],
    *,
    guard_env: str,
    label: str,
    cgroup: Callable[[], str] = _read_cgroup,
    which: Callable[[str], str | None] = shutil.which,
    probe: Callable[[str], subprocess.CompletedProcess[str]] | None = None,
    execvpe: Callable[..., None] = os.execvpe,
    warn: Callable[[str], None] = lambda text: sys.stderr.write(text + "\n"),
) -> bool:
    """Re-execs `argv` inside `slice_name` and does not return when it does. Returns False when the command should
    run in place: no slice named, already in it, already tried by a parent (the environment variable `guard_env`
    names the slice tried, so a refusal can never loop), a service, or a host that cannot do it (one warning
    prefixed `label`). Never raises."""

    def default_probe(name: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            _systemd_run(name, ["true"]),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=_clean(guard_env, None),
        )

    here = cgroup()
    if not slice_name or os.environ.get(guard_env) == slice_name or in_slice(slice_name, here) or in_service(here):
        return False
    if which("systemd-run") is None:
        warn(f"{label}: run slice {slice_name} not entered: systemd-run is missing; running in place")
        return False
    try:
        tried = (probe or default_probe)(slice_name)
        reason = (
            "" if tried.returncode == 0 else (tried.stderr.strip().splitlines() or [f"exit {tried.returncode}"])[-1]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
    if reason:
        warn(f"{label}: run slice {slice_name} not entered: systemd-run refused ({reason}); running in place")
        return False
    command = _systemd_run(slice_name, argv)
    try:
        execvpe(command[0], command, _clean(guard_env, slice_name))
    except OSError as exc:
        warn(f"{label}: run slice {slice_name} not entered: {exc}; running in place")
    return False
