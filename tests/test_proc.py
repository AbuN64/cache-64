# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Native execution: streams, accounting, group kills and reaping."""

from __future__ import annotations

import dataclasses
import os
import signal
from pathlib import Path
from threading import Barrier, Thread, get_ident
from time import perf_counter, perf_counter_ns
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from abucache import proc, slots


def test_exit_and_streams_kept(tmp_path: Path) -> None:
    argv = ("/bin/sh", "-c", "echo a; echo b >&2; exit 3")
    result = proc.run(argv, tmp_path, tmp=tmp_path / "tmp")
    assert (result.argv, result.cwd, result.exit, result.signal) == (argv, str(tmp_path), 3, None)
    assert (result.stdout, result.stderr) == (b"a\n", b"b\n")
    assert result.wall > 0 and result.cpu >= 0 and result.max_rss > 0
    assert dataclasses.is_dataclass(result) and result.__dataclass_params__.frozen


def test_children_get_the_given_tmpdir_and_env(tmp_path: Path) -> None:
    tmp = tmp_path / "scratch"
    result = proc.run(("/bin/sh", "-c", "echo $TMPDIR $KEPT"), tmp_path, tmp=tmp, env={"KEPT": "yes"})
    assert result.stdout.decode().split() == [str(tmp), "yes"] and tmp.is_dir()


def test_a_missing_program_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        proc.run((str(tmp_path / "absent"),), tmp_path, tmp=tmp_path / "tmp")


def test_cores_hold_slots_for_the_childs_life_and_tell_it_the_grant(
    tmp_path: Path, rundir: Path, make_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slots, "default", slots.Slots(make_runtime(workers=4)))
    argv = ("/bin/sh", "-c", "echo ${UNBAKE_CORES:-none}; ls " + str(rundir / slots.POOL) + " 2>&1; true")
    plain = proc.run(argv, tmp_path, tmp=tmp_path / "tmp")
    assert plain.exit == 0 and plain.stdout.startswith(b"none\n")  # a control-plane call holds nothing
    held = proc.run(argv, tmp_path, tmp=tmp_path / "tmp", cores=2)
    assert held.exit == 0 and held.stdout.startswith(b"2\n")
    assert b"slot-0.lock" in held.stdout and b"slot-1.lock" in held.stdout
    assert not list((rundir / slots.POOL).glob("cmd-*.lock"))  # and gives them back


def test_concurrent_cpu_attributed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    barrier = Barrier(2)
    results, usages, errors = {}, {}, []
    real_wait4 = os.wait4

    def wait4(pid, options):
        value = real_wait4(pid, options)
        usages[get_ident()] = value[2].ru_utime + value[2].ru_stime
        return value

    monkeypatch.setattr(proc.os, "wait4", wait4)

    def spin(count):
        try:
            barrier.wait(timeout=5)
            script = f"i=0; while [ $i -lt {count} ]; do i=$((i+1)); done"
            result = proc.run(("/bin/sh", "-c", script), tmp_path, timeout=5, tmp=tmp_path / "tmp")
            results[count] = (result, usages[get_ident()])
        except BaseException as error:
            errors.append(error)

    threads = [Thread(target=spin, args=(count,)) for count in (30000, 180000)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=7)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    short, long = (results[count][0] for count in (30000, 180000))
    assert short.exit == long.exit == 0
    assert short.cpu > 0
    assert long.cpu > short.cpu * 2
    for result, own_cpu in results.values():
        assert result.cpu == pytest.approx(own_cpu, abs=1e-9)
    assert short.cpu + long.cpu == pytest.approx(sum(usages.values()), abs=1e-9)


@pytest.mark.parametrize("stdin", [b"", b"x" * (1 << 20)], ids=["empty-stdin", "blocked-stdin"])
def test_timeout_kills_group(tmp_path: Path, stdin: bytes) -> None:
    # The deadline starts when the shell is spawned, so under load it may expire before the shell has started its
    # children: the group is found by its unique duration, and may be empty only when the kill came first.
    marker = f"{30 + os.getpid() % 1000}.{perf_counter_ns() % 100000}"
    script = f"sleep {marker} & sleep {marker} & wait"
    started = perf_counter()
    result = proc.run(("/bin/sh", "-c", script), tmp_path, stdin=stdin, timeout=0.5, tmp=tmp_path / "tmp")
    elapsed = perf_counter() - started
    assert result.exit is None
    assert result.signal == signal.SIGKILL
    assert elapsed < 2
    assert 0.45 <= result.wall < 2
    for other in psutil.process_iter(["cmdline", "status"]):
        if marker in " ".join(other.info["cmdline"] or ()):
            assert other.info["status"] == psutil.STATUS_ZOMBIE, other.info  # zombies cannot execute


@pytest.mark.parametrize("redirect", [False, True])
def test_large_stdin_and_both_streams(tmp_path: Path, redirect: bool) -> None:
    payload = b"input\x00\xff\n" * 40000
    stdout_path = tmp_path / "stdout" if redirect else None
    result = proc.run(
        ("/bin/sh", "-c", "printf '%070000d' 0 >&2; cat"),
        tmp_path,
        stdin=payload,
        stdout_path=stdout_path,
        timeout=2,
        tmp=tmp_path / "tmp",
    )
    assert result.exit == 0 and result.signal is None
    assert result.stderr == b"0" * 70000
    if stdout_path is not None:
        assert result.stdout == b"" and stdout_path.read_bytes() == payload
    else:
        assert result.stdout == payload


def test_a_grandchild_the_child_leaves_behind_is_killed_and_reaped_by_us(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_wait4, reaped = proc.os.wait4, []

    def wait4(pid, options):
        found = real_wait4(pid, options)
        reaped.append((pid, found[0]))
        return found

    monkeypatch.setattr(proc.os, "wait4", wait4)
    result = proc.run(("/bin/sh", "-c", "sleep 300 >/dev/null 2>&1 & exit 0"), tmp_path, tmp=tmp_path / "tmp")
    child = reaped[0][0]
    assert result.exit == 0 and [pid for pid, _ in reaped] == [child, -child]


def test_wait4_accounting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_popen, real_wait4 = proc.subprocess.Popen, proc.os.wait4
    seen = {}

    def popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        child.wait = Mock(side_effect=AssertionError("the runner reaps with wait4 only"))
        seen["child"], seen["kwargs"] = child, kwargs
        return child

    def wait4(pid, options):
        if pid < 0:
            return real_wait4(pid, options)
        seen["waited"] = pid
        reaped, status, _ = real_wait4(pid, options)
        return reaped, status, SimpleNamespace(ru_utime=0.5, ru_stime=0.25, ru_maxrss=4096)

    monkeypatch.setattr(proc.subprocess, "Popen", popen)
    monkeypatch.setattr(proc.os, "wait4", wait4)
    before = perf_counter()
    result = proc.run(("/bin/sh", "-c", "exit 7"), tmp_path, tmp=tmp_path / "tmp")
    after = perf_counter()
    assert seen["kwargs"]["start_new_session"] is True and seen["kwargs"]["cwd"] == tmp_path
    assert seen["waited"] == seen["child"].pid
    assert seen["child"].returncode == result.exit == 7
    seen["child"].wait.assert_not_called()
    assert result.cpu == 0.75 and result.max_rss == 4096 * 1024
    assert before * 1e9 <= result.start_ns <= after * 1e9


def test_the_subreaper_is_set_once() -> None:
    proc.subreaper()
    proc.subreaper()
