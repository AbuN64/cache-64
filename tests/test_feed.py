# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Slow stages and waits reach the event feed and the wait log by themselves: no tool calls anything to get there."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from abucache import events, feed, slots, stages

SECOND = 1_000_000_000


@pytest.fixture
def clock(monkeypatch):
    """The stages' clock, moved by the test: no test sleeps."""
    now = [SECOND]
    monkeypatch.setattr(stages, "time", SimpleNamespace(perf_counter_ns=lambda: now[0], time=time.time))

    def advance(seconds):
        now[0] += int(seconds * SECOND)

    return advance


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(stages, "_listeners", [])
    monkeypatch.setattr(stages, "_budgets", None)
    monkeypatch.setattr(stages, "_wait_log", None)
    monkeypatch.setattr(stages, "_walls", {})
    monkeypatch.setattr(stages, "_runtime", None)
    monkeypatch.setattr(stages, "_live", {})
    monkeypatch.setattr(stages, "_waits", {})
    monkeypatch.setattr(feed, "path", None)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)


@pytest.fixture
def host(tmp_path, monkeypatch, make_runtime):
    """A runtime file named by ABU_RUNTIME (feed beside it, slow after 60 s) and a runtime directory."""
    make_runtime(slow=60)
    monkeypatch.setenv("ABU_RUNTIME", str(tmp_path / "runtime.toml"))
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(run))
    return SimpleNamespace(feed=tmp_path / "feed.jsonl", sink=tmp_path / "state" / "stages.jsonl", run=run)


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_a_slow_stage_and_its_command_reach_the_feed_with_no_configure_call(host, clock):
    with stages.command("tool", "check", [], host.sink):
        with stages.stage("quick"):
            clock(5)
        with stages.stage("slow"):
            clock(100)
    written = lines(host.feed)
    assert [(e["actor"], e["payload"]["stage"], e["payload"]["running"]) for e in written] == [
        ("tool", "check > slow", False),
        ("tool", "check", False),
    ]
    assert written[0]["status"] == "info" and written[0]["payload"]["seconds"] == 100.0
    assert events.validate_lines(host.feed.read_text().splitlines())["findings"] == []


def test_a_healthy_command_adds_no_line(host, clock):
    with stages.command("tool", "check", [], host.sink), stages.stage("quick"):
        clock(59)
    assert lines(host.feed) == []


def test_a_stage_is_slow_only_past_its_expected_wall_times_the_overrun(host, clock):
    host.sink.parent.mkdir(parents=True)
    host.sink.with_name("walls.json").write_text(json.dumps({"check > phase": 100.0}))
    with stages.command("tool", "check", [], host.sink):
        with stages.stage("phase"):
            clock(120)  # past slow_seconds and past 100 s, but under 150 percent of it
        with stages.stage("phase"):
            clock(160)
    found = {e["payload"]["stage"]: e["payload"] for e in lines(host.feed)}
    assert (found["check > phase"]["seconds"], found["check > phase"]["expected_seconds"]) == (160.0, 100.0)
    assert [e["payload"]["seconds"] for e in lines(host.feed) if e["payload"]["stage"] == "check > phase"] == [160.0]


def test_a_slot_wait_is_a_blocked_event_with_the_command_total(host, clock):
    with stages.command("tool", "check", [], host.sink), stages.stage("pool.wait_slot"):
        clock(90)
        stages.waited(90, "slots", "held by pids 1")
    written = lines(host.feed)
    assert [(e["status"], e["payload"]["stage"]) for e in written] == [
        ("blocked", "check > pool.wait_slot"),
        ("info", "check"),
    ]
    assert written[1]["payload"]["slot_wait_seconds"] == 90.0


def test_a_lock_wait_past_the_limit_is_an_event_of_its_own(host, clock):
    with stages.command("tool", "check", [], host.sink), stages.stage("write"):
        stages.waited(61, "lock", "held by pid 7")
        stages.waited(3, "lock", "held by pid 7")
    waits = [e for e in lines(host.feed) if "wait.lock" in e["payload"]["stage"]]
    assert [(e["status"], e["payload"]["stage"], e["payload"]["detail"]) for e in waits] == [
        ("info", "check > write > wait.lock", "held by pid 7")
    ]


def test_a_wait_of_a_minute_reaches_the_slot_directorys_wait_log_with_no_configure_call(host):
    with stages.command("tool", "check", [], host.sink):
        stages.waited(61, "slots", "held by pids 9")
        stages.waited(5, "slots", "short")
    rows = lines(host.run / slots.POOL / slots.WAITS)
    assert [(r["what"], r["seconds"], r["detail"]) for r in rows] == [("slots", 61.0, "held by pids 9")]


def test_an_explicit_wait_log_wins(host, tmp_path):
    other = tmp_path / "other.jsonl"
    stages.configure(wait_log=other)
    with stages.command("tool", "check", [], host.sink):
        stages.waited(61, "lock", "")
    assert len(lines(other)) == 1 and not (host.run / slots.POOL).exists()


def test_the_watchdog_announces_a_span_still_running_once(monkeypatch, tmp_path):
    seen = threading.Event()
    told = []

    def say(tool, stage, seconds, expected, **kwargs):
        told.append((tool, stage, kwargs["running"]))
        seen.set()

    monkeypatch.setattr(feed, "say", say)
    monkeypatch.setattr(feed, "path", tmp_path / "feed.jsonl")
    monkeypatch.setattr(feed, "slow_seconds", 1)
    stuck = ("span-1", ("check", "stuck"), 90.0, None, "stage")
    fine = ("span-2", ("check", "fine"), 0.5, None, "stage")
    feed.start("tool", lambda: [stuck, fine])
    try:
        assert seen.wait(10)
        assert feed.announced("span-1") and not feed.announced("span-2")
    finally:
        feed.stop()
    assert told[0] == ("tool", ("check", "stuck"), True)
    assert len(told) == len(set(told))


def test_a_feed_that_cannot_be_written_is_a_finding_of_the_command_not_silence(host, clock):
    host.feed.mkdir()  # the feed path is a directory: opening it for append fails whoever runs the test
    with stages.command("tool", "check", [], host.sink), stages.stage("slow"):
        clock(100)
    root = stages.closed()[-1]
    keys = [f["key"] for f in root.findings]
    assert keys.count("events.feed") == 1


def test_an_unusable_runtime_file_is_a_finding_and_a_missing_one_is_not(host, tmp_path, monkeypatch, clock):
    (tmp_path / "runtime.toml").write_text("[resources\n")
    with stages.command("tool", "check", [], host.sink), stages.stage("slow"):
        clock(100)
    assert [f["key"] for f in stages.closed()[-1].findings] == ["events.runtime"]
    monkeypatch.setenv("ABU_RUNTIME", str(tmp_path / "absent.toml"))
    with stages.command("tool", "check", [], host.sink), stages.stage("slow"):
        clock(100)
    assert stages.closed()[-1].findings == () and feed.path is None


def test_a_runtime_given_to_configure_is_the_one_the_commands_use(host, clock, make_runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("ABU_RUNTIME", str(tmp_path / "absent.toml"))
    stages.configure(make_runtime(slow=10))
    with stages.command("tool", "check", [], host.sink):
        clock(11)
    assert [e["payload"]["stage"] for e in lines(host.feed)] == ["check"]


def test_a_forked_worker_has_no_watchdog_and_no_announcements(monkeypatch):
    monkeypatch.setattr(feed, "_thread", object())
    feed._told.add("x")
    feed._after_fork()
    assert feed._thread is None and not feed._told
