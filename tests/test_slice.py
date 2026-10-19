# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The run slice: decide whether to re-exec under systemd-run, with every host fact mocked."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from abucache import slice as run_slice

GUARD = "FIXTURE_SLICE_ENTERED"
IN = "0::/user.slice/user-1000.slice/user@1000.service/decomp.slice/run-r1.scope\n"
OUT = "0::/user.slice/user-1000.slice/session-3.scope\n"
SERVICE = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/m2-stream.service\n"


def _done(code: int, err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, "", err)


def _enter(cgroup: str, which: bool = True, probe: Any = None) -> tuple[bool, list[Any], list[str]]:
    execs: list[Any] = []
    warned: list[str] = []
    result = run_slice.enter(
        "decomp.slice",
        ["/venv/bin/python", "-m", "tool", "make"],
        guard_env=GUARD,
        label="fixture",
        cgroup=lambda: cgroup,
        which=lambda name: "/usr/bin/systemd-run" if which else None,
        probe=probe or (lambda name: _done(0)),
        execvpe=lambda *call: execs.append(call),
        warn=warned.append,
    )
    return result, execs, warned


def test_a_process_already_in_the_slice_runs_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GUARD, raising=False)
    assert _enter(IN) == (False, [], [])


def test_a_process_outside_the_slice_re_execs_its_own_argv_under_systemd_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GUARD, raising=False)
    _, execs, warned = _enter(OUT)
    assert warned == []
    [(program, command, env)] = execs
    assert program == "systemd-run"
    assert command == [
        "systemd-run", "--user", "--scope", "--quiet", "--slice=decomp.slice", "--",
        "/venv/bin/python", "-m", "tool", "make",
    ]  # fmt: skip
    assert env[GUARD] == "decomp.slice"


def test_a_service_is_placed_by_its_unit_and_never_moved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GUARD, raising=False)
    assert _enter(SERVICE) == (False, [], [])


def test_a_slice_a_parent_already_tried_is_not_tried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GUARD, "decomp.slice")
    assert _enter(OUT) == (False, [], [])


def test_no_slice_named_runs_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GUARD, raising=False)
    assert run_slice.enter(None, ["x"], guard_env=GUARD, label="fixture") is False


def test_a_host_without_systemd_run_runs_in_place_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GUARD, raising=False)
    result, execs, warned = _enter(OUT, which=False)
    assert (result, execs) == (False, [])
    assert len(warned) == 1 and warned[0].startswith("fixture: ") and "systemd-run is missing" in warned[0]


def test_a_refusing_systemd_run_runs_in_place_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GUARD, raising=False)
    result, execs, warned = _enter(OUT, probe=lambda name: _done(1, "Failed to connect to bus\n"))
    assert (result, execs) == (False, [])
    assert len(warned) == 1 and "Failed to connect to bus" in warned[0]


def test_a_probe_that_raises_runs_in_place_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GUARD, raising=False)

    def boom(name: str) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("systemd-run", 15)

    result, _, warned = _enter(OUT, probe=boom)
    assert result is False and len(warned) == 1


def test_an_exec_that_fails_runs_in_place_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GUARD, raising=False)

    def boom(*call: Any) -> None:
        raise OSError("exec format error")

    warned: list[str] = []
    result = run_slice.enter(
        "decomp.slice", ["x"], guard_env=GUARD, label="fixture", cgroup=lambda: OUT, which=lambda n: "/x",
        probe=lambda n: _done(0), execvpe=boom, warn=warned.append,
    )  # fmt: skip
    assert result is False and len(warned) == 1


def test_in_slice_matches_a_whole_cgroup_component_only() -> None:
    assert run_slice.in_slice("decomp.slice", IN)
    assert not run_slice.in_slice("decomp.slice", OUT)
    assert not run_slice.in_slice("decomp.slice", "0::/user.slice/notdecomp.slice/x.scope\n")


def test_in_service_reads_the_unified_path_only() -> None:
    assert run_slice.in_service(SERVICE)
    assert not run_slice.in_service(OUT)
    assert not run_slice.in_service(IN)
    assert not run_slice.in_service("")
    assert not run_slice.in_service("1:name=systemd:/x/y.service\n")
