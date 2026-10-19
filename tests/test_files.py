# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Atomic writes, JSONL streams and kernel locks."""

from __future__ import annotations

import fcntl
import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from abucache import files


def _hold(path: Path, started: Path, release: Path) -> None:
    with files.lock(path, wait=False) as fd:
        started.write_text(str(fd is not None))
        while not release.exists():
            time.sleep(0.02)


def _wait(path: Path) -> None:
    for _ in range(500):
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"{path} never appeared")


def test_write_changes_a_file_only_when_its_bytes_differ(tmp_path: Path) -> None:
    path = tmp_path / "a" / "f"
    assert files.write(path, b"x") is True
    stamp = path.stat().st_mtime_ns
    assert files.write(path, b"x") is False and path.stat().st_mtime_ns == stamp
    assert files.write(path, b"y") is True and path.read_bytes() == b"y"


def test_write_keeps_a_name_near_the_filesystem_limit(tmp_path: Path) -> None:
    path = tmp_path / ("x" * 240 + ".s")  # the temporary file must not lengthen the name past 255 bytes
    assert files.write(path, b"a") and path.read_bytes() == b"a" and [p.name for p in tmp_path.iterdir()] == [path.name]


def test_write_leaves_the_other_hard_link_untouched(tmp_path: Path) -> None:
    shared, mine = tmp_path / "cache" / "a.s", tmp_path / "tree" / "a.s"
    shared.parent.mkdir()
    mine.parent.mkdir()
    shared.write_bytes(b"old\n")
    shared.chmod(0o444)
    os.link(shared, mine)
    assert files.write(mine, b"new\n")
    assert (mine.read_bytes(), shared.read_bytes(), shared.stat().st_nlink) == (b"new\n", b"old\n", 1)


def test_scratch_is_created_once_under_the_root(tmp_path: Path) -> None:
    assert files.scratch(tmp_path / "state") == tmp_path / "state" / "tmp"
    assert files.scratch(tmp_path / "state").is_dir()


def test_jsonl_roundtrip_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "s" / "f.jsonl"
    assert files.read_jsonl(path) == []
    files.append_jsonl(path, {"b": 1, "a": "é"})
    files.append_jsonl(path, {"n": [1, 2]})
    assert path.read_text().splitlines()[0] == '{"a":"\\u00e9","b":1}'
    assert files.read_jsonl(path) == [{"a": "é", "b": 1}, {"n": [1, 2]}]


def test_a_corrupt_jsonl_line_names_the_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    files.append_jsonl(path, {"ok": 1})
    with path.open("a") as handle:
        handle.write("{broken\n\n")
    with pytest.raises(ValueError, match=f"{path}:2:"):
        files.read_jsonl(path)


def test_a_second_holder_gets_none_without_waiting(tmp_path: Path) -> None:
    lock, started, release = tmp_path / "l", tmp_path / "started", tmp_path / "release"
    proc = multiprocessing.get_context("fork").Process(target=_hold, args=(lock, started, release))
    proc.start()
    try:
        _wait(started)
        assert started.read_text() == "True"
        with files.lock(lock, wait=False) as fd:
            assert fd is None
        with files.lock(lock, shared=True, wait=False) as fd:
            assert fd is None
    finally:
        release.write_text("go")
        proc.join(10)
    assert proc.exitcode == 0
    with files.lock(lock, wait=False) as fd:
        assert fd is not None


def test_a_blocked_lock_reports_its_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock, started, release = tmp_path / "l", tmp_path / "started", tmp_path / "release"
    proc = multiprocessing.get_context("fork").Process(target=_hold, args=(lock, started, release))
    proc.start()
    waits: list[tuple[float, str, str]] = []
    monkeypatch.setattr(files, "waited", lambda *news: waits.append(news))
    try:
        _wait(started)
        threading.Timer(0.3, release.write_text, ["go"]).start()
        with files.lock(lock) as fd:
            assert fd is not None
    finally:
        proc.join(10)
    assert [(what, detail) for _, what, detail in waits] == [("lock", "l")]
    assert waits[0][0] > 0


def test_a_lock_that_is_free_reports_no_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[object] = []
    monkeypatch.setattr(files, "waited", lambda *news: waits.append(news))
    with files.lock(tmp_path / "l"):
        pass
    assert waits == []


def test_a_deadline_passes_without_the_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(files, "_CLOCK", lambda: now[0])
    monkeypatch.setattr(files, "_PAUSE", lambda seconds: now.__setitem__(0, now[0] + seconds))

    def flock(fd: int, flags: int) -> None:
        raise BlockingIOError

    monkeypatch.setattr(files.fcntl, "flock", flock)
    with files.lock(tmp_path / "l", deadline_s=1.0) as fd:
        assert fd is None
    assert now[0] >= 1.0


def test_a_waiting_writer_holds_the_gate_and_a_reader_passes_it_first(tmp_path: Path) -> None:
    lock, gate = tmp_path / "l", tmp_path / "l.gate"
    with files.lock(lock, shared=True), files.lock(lock, shared=True, gate=gate) as fd:
        assert fd is not None  # no writer waits: readers share
    with files._flock(gate, fcntl.LOCK_EX), files._flock(gate, fcntl.LOCK_SH, wait=False) as probe:
        assert probe is None  # a reader passing the gate would queue behind the waiting writer
    with files.lock(lock, deadline_s=0.5, gate=gate) as fd:
        assert fd is not None
        with files._flock(gate, fcntl.LOCK_SH, wait=False) as probe:
            assert probe is None  # the writer holds its gate while it holds the lock


def test_a_writer_that_cannot_pass_the_gate_in_time_yields_none(tmp_path: Path) -> None:
    lock, gate = tmp_path / "l", tmp_path / "l.gate"
    started, release = tmp_path / "started", tmp_path / "release"
    proc = multiprocessing.get_context("fork").Process(target=_hold, args=(gate, started, release))
    proc.start()
    try:
        _wait(started)
        with files.lock(lock, deadline_s=0.3, gate=gate) as fd:
            assert fd is None
    finally:
        release.write_text("go")
        proc.join(10)


def test_a_forked_worker_drops_an_inherited_flock_without_unlocking_it(tmp_path: Path) -> None:
    path = tmp_path / "journal.lock"
    with files.lock(path, shared=True) as fd:
        assert fd in files._HELD_FDS
        files.drop_inherited()
        assert not files._HELD_FDS
        with files.lock(path, deadline_s=0.0) as taken:
            assert taken is not None  # no copy of the descriptor holds the lock any more
    assert not files._HELD_FDS


def test_the_held_descriptor_is_tracked_only_while_held(tmp_path: Path) -> None:
    with files.lock(tmp_path / "a.lock") as fd:
        assert list(files._HELD_FDS) == [fd]
    assert not files._HELD_FDS
