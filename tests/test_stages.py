# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Stage spans, accounting, the stage log and the wait log."""

import itertools
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from abucache import stages


class Refusal(Exception):
    def __init__(self, *findings):
        self.findings = findings
        super().__init__("refused")


stages.register_refusal(Refusal)

BUDGETS = stages.Budgets(serial_seconds=2.0, serial_cores=2.0, pool_fill=0.8, pool_fanout=2.0)


def done(cpu=0.0):
    """What `proc.run` returns, as far as `record_native` reads it."""
    return SimpleNamespace(wall=0.0, cpu=cpu, max_rss=0)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(stages, "_listeners", [])
    monkeypatch.setattr(stages, "_budgets", None)
    monkeypatch.setattr(stages, "_wait_log", None)
    monkeypatch.setattr(stages, "_walls", {})
    monkeypatch.setattr(stages, "_memory_limits", {})
    monkeypatch.setattr(stages, "_memory_limit_values", {})
    monkeypatch.setattr(stages, "_budget_origins", {})
    monkeypatch.setattr(stages, "_package", "")
    monkeypatch.setattr(stages, "_project", "")
    monkeypatch.setattr(stages, "parent_memory", 0)
    monkeypatch.setattr(stages, "LOG_BYTES", 64 << 20)


@pytest.fixture
def sink(tmp_path):
    return tmp_path / "state" / "stages.jsonl"


def run(sink, name="check"):
    return stages.command("tool", name, [], sink)


def test_listen_events(sink):
    events = []
    stages.listen(lambda event, body: events.append((event, body)))
    with run(sink), stages.stage("outer"), stages.capture():
        stages.record_native("tool", done(2), 0)
    assert [(event, body["path"]) for event, body in events] == [
        ("open", ["check"]),
        ("open", ["check", "outer"]),
        ("close", ["check", "outer"]),
        ("close", ["check"]),
    ]
    assert all(body["tool"] == "tool" for event, body in events if event == "open")
    for event, body in events:
        if event == "close":
            assert body["wall"] >= 0
            assert body["status"] == "ok"


def test_progress_event():
    events = []
    stages.listen(lambda event, body: events.append((event, body)))
    stages.progress("units", 2, 7)
    assert events == [("progress", {"name": "units", "done": 2, "total": 7})]


def test_closed_records(sink):
    with run(sink, "first"), stages.stage("child"):
        pass
    records = stages.closed()
    assert isinstance(records, tuple)
    assert [r.kind for r in records] == ["stage", "own", "command"]
    assert records[0].parent == records[-1].span
    assert records[1].path == ("first", "(own)")
    assert stages.tree()["span"] == records[-1].span
    with run(sink, "second"):
        pass
    assert [r.path for r in stages.closed()] == [("second",)]
    assert records[-1].invocation != stages.invocation()


def test_every_record_starts_with_its_schema(sink):
    with run(sink):
        pass
    row = json.loads(sink.read_text().splitlines()[-1])
    assert next(iter(row)) == "schema" and row["schema"] == "stage/1"
    assert stages.closed()[-1].schema == "stage/1"


def test_the_command_writes_its_log_to_the_sink_it_was_given(sink, capsys):
    with run(sink), stages.stage("one"):
        pass
    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    closed = stages.closed()
    assert {r["span"] for r in rows} <= {r.span for r in closed}
    assert rows[-1]["span"] == closed[-1].span  # the root is kept
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("event", ["open", "close", "progress"])
def test_listener_failure_is_finding(sink, event):
    def fail(kind, body):
        if kind == event:
            raise ValueError("listener broke")

    seen = []
    stages.listen(fail)
    stages.listen(lambda kind, body: seen.append(kind))
    with run(sink):
        stages.progress("work", 1, 1)
    findings = stages.closed()[-1].findings
    assert len(findings) == 1
    assert findings[0]["key"] == "effort.log"
    assert not findings[0]["blocking"]
    assert "listener broke" in findings[0]["reason"]
    assert seen == ["open", "progress", "close"]


def test_listeners_survive_commands(sink):
    events = []
    stages.listen(lambda event, body: events.append(event))
    for name in ("one", "two"):
        with run(sink, name):
            pass
    assert events == ["open", "close", "open", "close"]


@pytest.mark.parametrize(
    "exc,status",
    [
        (Refusal(stages.finding("effort.log", "stop")), "refused"),
        (ValueError("stop"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_span_status_and_stack_restore(sink, exc, status):
    with pytest.raises(type(exc)), run(sink), stages.stage("work"):
        raise exc
    assert all(r.status == status for r in stages.closed())
    assert stages._stack.get() == ()
    if isinstance(exc, Refusal):
        assert stages.closed()[-1].findings == exc.findings


def test_own_wall_and_cpu(sink, monkeypatch):
    wall = iter([0, 1_000_000_000, 3_000_000_000, 5_000_000_000])
    cpu = iter([0.0, 1.0, 3.0, 6.0])
    monkeypatch.setattr(stages.time, "perf_counter_ns", lambda: next(wall))
    monkeypatch.setattr(stages, "_cpu", lambda: next(cpu))
    with run(sink), stages.stage("work"):
        pass
    child, own, root = stages.closed()
    assert (child.wall_seconds, child.parent_cpu_seconds) == (2, 2)
    assert (own.wall_seconds, own.parent_cpu_seconds) == (3, 4)
    assert (root.wall_seconds, root.parent_cpu_seconds, root.cores) == (5, 6, 1.2)


def test_native_and_pool_cpu_cache(sink):
    with run(sink):
        stages.record_native("tool", done(3), stages.time.perf_counter_ns())
        stages.record_pool(
            "units",
            2,
            2,
            2,
            stages.time.perf_counter_ns(),
            [
                {
                    "worker_cpu": 4,
                    "native_cpu": 5,
                    "rss": 1234,
                    "pid": 10,
                    "start_ns": 0,
                    "end_ns": 2_000_000_000,
                    "cache": {"object": (2, 1)},
                    "records": [],
                    "wait_seconds": 1.0,
                },
                {
                    "worker_cpu": 6,
                    "native_cpu": 7,
                    "rss": 2345,
                    "pid": 11,
                    "start_ns": 0,
                    "end_ns": 3_000_000_000,
                    "cache": {"object": (1, 2)},
                    "records": [],
                },
            ],
        )
    root = stages.closed()[-1]
    assert (root.worker_cpu_seconds, root.native_cpu_seconds) == (10, 15)
    assert (root.items, root.jobs, root.workers_used, root.workers_admitted) == (2, 2, 2, 2)
    assert root.worker_rss_peak_bytes == 2345
    assert root.cache["object"] == (3, 3)
    held = next(r for r in stages.closed() if r.kind == "pool")
    assert held.wait_seconds == 1.0 and held.busy_seconds == 4.0


def test_pool_job_records_rebased(sink):
    with stages.capture() as captured, stages.stage("work"):
        stages.count("object", True)
    raws = [asdict(r) for r in captured]
    with run(sink):
        stages.record_pool(
            "units",
            1,
            1,
            1,
            stages.time.perf_counter_ns(),
            [
                {
                    "worker_cpu": 1,
                    "native_cpu": 0,
                    "rss": 1,
                    "pid": 1,
                    "start_ns": 0,
                    "end_ns": 1,
                    "cache": {"object": (1, 0)},
                    "records": raws,
                }
            ],
        )
    jobs = [r for r in stages.closed() if r.path[:2] == ("check", "units") and r.kind != "pool"]
    assert [r.kind for r in jobs] == [r["kind"] for r in raws]
    assert all(r.parent_cpu_seconds == 0 for r in jobs)
    assert [r.worker_cpu_seconds for r in jobs] == [r["parent_cpu_seconds"] for r in raws]
    assert all(r.path[:2] == ("check", "units") for r in jobs)
    assert all(r.invocation == stages.invocation() for r in jobs)
    assert all(r.schema == "stage/1" for r in jobs)
    assert not {r.span for r in jobs} & {r["span"] for r in raws}


def test_capture_restores_and_null_stage(sink):
    with stages.stage("outside") as span:
        span.add(items=3)
    with run(sink):
        stages.count("object", True)
        with stages.capture() as records:
            stages.count("view", False)
        assert stages.counters() == {"object": (1, 0)}
        assert records[-1].kind == "job"
    assert stages.closed()[-1].cache == {"object": (1, 0)}


def test_log_failure_is_debt(tmp_path, capsys):
    (tmp_path / "state").write_text("cannot be a directory")
    with run(tmp_path / "state" / "stages.jsonl"):
        pass
    assert any(f["key"] == "effort.log" and not f["blocking"] for f in stages.closed()[-1].findings)
    assert capsys.readouterr().err == ""


def test_nested_cpu_inclusive_spans_exclusive_own(sink, monkeypatch):
    cpu = iter([0.0, 1.0, 2.0, 4.0, 7.0, 10.0])
    monkeypatch.setattr(stages, "_cpu", lambda: next(cpu))
    with run(sink), stages.stage("outer"), stages.stage("inner"):
        pass
    rows = {r.path: r for r in stages.closed()}
    assert rows[("check",)].parent_cpu_seconds == 10
    assert rows[("check", "outer")].parent_cpu_seconds == 6
    assert rows[("check", "outer", "inner")].parent_cpu_seconds == 2
    assert rows[("check", "(own)")].parent_cpu_seconds == 4
    assert rows[("check", "outer", "(own)")].parent_cpu_seconds == 4


def test_imported_native_own_and_cpu_classification(sink):
    with stages.capture() as captured, stages.stage("work"):
        stages.record_native("tool", done(2), stages.time.perf_counter_ns())
    raw = [asdict(r) for r in captured]
    with run(sink):
        stages.record_pool(
            "units",
            1,
            1,
            1,
            stages.time.perf_counter_ns(),
            [
                {
                    "worker_cpu": 1,
                    "native_cpu": 2,
                    "rss": 1,
                    "pid": 1,
                    "start_ns": 0,
                    "end_ns": 1,
                    "cache": {},
                    "records": raw,
                }
            ],
        )
    imported = [r for r in stages.closed() if r.path[:3] == ("check", "units", "job")]
    assert [r.kind for r in imported] == [r["kind"] for r in raw]
    native = next(r for r in imported if r.kind == "native")
    assert native.native_cpu_seconds == 2 and native.worker_cpu_seconds == 0
    root = next(r for r in imported if r.kind == "job")
    assert root.worker_cpu_seconds > 0 and root.parent_cpu_seconds == 0
    assert root.parent_rss_peak_bytes == 0 and root.worker_rss_peak_bytes > 0


def test_wait_is_charged_to_the_innermost_stage_only(sink):
    with run(sink), stages.stage("outer"), stages.stage("inner"):
        stages.waited(1.5)
    waits = {r.path[-1]: r.wait_seconds for r in stages.closed() if r.kind in ("stage", "command")}
    assert waits == {"inner": 1.5, "outer": 0.0, "check": 0.0}
    root = stages.closed()[-1]
    assert (root.lock_wait_seconds, root.slot_wait_seconds) == (1.5, 0.0)


def test_a_long_wait_lands_in_the_wait_log_and_a_short_one_does_not(sink, tmp_path):
    log = tmp_path / "waits.jsonl"
    stages.configure(project="proj", wait_log=log)
    with run(sink):
        stages.waited(1.0, "slots", "short")
        stages.waited(61.0, "slots", "held by 42")
    (line,) = [json.loads(row) for row in log.read_text().splitlines()]
    assert (line["what"], line["seconds"], line["detail"], line["project"]) == ("slots", 61.0, "held by 42", "proj")
    assert set(line) == {"at", "pid", "project", "what", "seconds", "detail"}


def test_the_wait_log_keeps_only_its_newest_whole_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "WAIT_LOG_BYTES", 400)
    log = tmp_path / "waits.jsonl"
    for _ in range(30):
        stages._note_wait(log, "lock", 70.0, "x")
    rows = [json.loads(row) for row in log.read_text().splitlines()]
    assert 0 < len(rows) < 30 and log.stat().st_size <= 400 + 120


def test_a_lock_that_blocked_is_charged_through_files_to_the_one_path(sink, tmp_path):
    import fcntl
    import os

    from abucache import files

    lock = tmp_path / "x.lock"
    held = os.open(lock, os.O_CREAT | os.O_WRONLY)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        with run(sink), stages.stage("inner"), files.lock(lock, wait=True, deadline_s=0.3):
            pass
    finally:
        os.close(held)
    inner = next(r for r in stages.closed() if r.path[-1] == "inner")
    assert inner.wait_seconds > 0.2


def test_a_failed_stage_names_the_exception_and_the_frame_once(sink):
    def read_entry():
        return Path("/nonexistent/entry.json").read_text()

    stages.configure(package="tests")
    with pytest.raises(FileNotFoundError), run(sink, "land"), stages.stage("outer"), stages.stage("inner"):
        read_entry()
    failed = {r.path[-1]: r for r in stages.closed() if r.status == "failed"}
    (note,) = failed["inner"].findings
    assert note["key"] == "internal.error" and "FileNotFoundError" in note["reason"]
    assert " at test_stages:read_entry:" in note["reason"]  # the innermost frame of the named package
    assert failed["outer"].findings == ()  # the stage that raised names it, its parents do not repeat it


def test_close_events_carry_cores_and_jobs_and_open_events_the_previous_wall(sink):
    events = []
    stages.listen(lambda event, body: events.append((event, body)))
    with run(sink), stages.stage("outer"):
        pass
    with run(sink), stages.stage("outer"):
        pass
    opens = [(tuple(b["path"]), b["expected"]) for e, b in events if e == "open"]
    assert opens[0] == (("check",), None) and opens[1] == (("check", "outer"), None)  # nothing was known before
    assert all(wall is not None for _, wall in opens[2:])  # the second command is told how long the first took
    closes = [b for e, b in events if e == "close"]
    assert all({"cores", "jobs", "wall"} <= set(b) for b in closes)


def test_the_first_command_has_no_expected_wall(sink):
    events = []
    stages.listen(lambda event, body: events.append((event, body)))
    with run(sink), stages.stage("outer"):
        pass
    opens = {tuple(b["path"]): b["expected"] for e, b in events if e == "open"}
    assert opens == {("check",): None, ("check", "outer"): None}


def test_walls_persist_beside_the_stage_log_for_shallow_stages(sink, tmp_path):
    with run(sink), stages.stage("a"), stages.stage("b"), stages.stage("c"):
        pass
    saved = stages._load_walls(sink.with_name("walls.json"))
    assert set(saved) == {"check", "check > a", "check > a > b"}  # deeper than three levels is not remembered
    assert stages._load_walls(tmp_path / "missing.json") == {}


def test_the_stage_log_keeps_only_its_newest_whole_lines_past_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "LOG_BYTES", 100)
    log = tmp_path / "stages.jsonl"
    log.write_text("".join(f"line{i:03d}\n" for i in range(40)))
    stages._trim(log)
    kept = log.read_text().splitlines()
    assert kept[-1] == "line039" and all(k.startswith("line") and len(k) == 7 for k in kept) and len(kept) <= 7
    before = log.read_bytes()
    stages._trim(log)
    assert log.read_bytes() == before


def test_configure_takes_the_log_size_and_memory_from_the_runtime():
    runtime = SimpleNamespace(stage_log_bytes=123, parent_memory=7, worker_memory=9)
    stages.configure(runtime, memory_limits={"resources.parent_memory": "runtime.toml"})
    assert (stages.LOG_BYTES, stages.parent_memory) == (123, 7)
    assert stages._limits() == {
        "memory_limits": {"resources.parent_memory": "runtime.toml"},
        "memory_limit_values": {"resources.parent_memory": 7, "resources.worker_memory": 9},
    }


def test_the_stage_log_keeps_the_head_of_a_long_finding_and_the_record_keeps_it_whole(sink):
    reason = "error: " + "x" * 50_000  # a compile error quoting whole headers
    with run(sink), pytest.raises(Refusal), stages.stage("native.objects"):
        raise Refusal(stages.finding("compile.error", reason, missing=(f"version de: {reason}",)))
    row = next(json.loads(line) for line in sink.read_text().splitlines() if "native.objects" in line)
    logged = row["findings"][0]["reason"]
    assert logged.startswith("error: x") and len(logged) < stages.REASON_CHARS + 40 and "50007 chars" in logged
    (missing,) = row["findings"][0]["missing"]
    assert missing.startswith("version de: error: x") and "50019 chars" in missing
    kept = next(r for r in stages.closed() if r.path[-1] == "native.objects")
    assert kept.findings[0]["reason"] == reason


def test_a_tool_finding_type_goes_through_the_hook(sink, monkeypatch):
    class Note:
        def __init__(self, text):
            self.text = text

    monkeypatch.setattr(stages, "finding_to_dict", lambda f: stages.finding("tool.note", f.text))
    with run(sink), stages.stage("work") as span:
        span.add(findings=[Note("hello")])
    row = next(r for r in stages.closed() if r.path[-1] == "work")
    assert [(f["key"], f["reason"]) for f in row.findings] == [("tool.note", "hello")]


def _job_raw(path, wall, cpu, wait, n, kind="stage"):
    return {
        "invocation": "w",
        "span": f"s{n}",
        "parent": None,
        "path": list(path),
        "kind": kind,
        "status": "ok",
        "execution": "serial",
        "start_ns": n,
        "wall_seconds": wall,
        "parent_cpu_seconds": cpu,
        "worker_cpu_seconds": 0.0,
        "native_cpu_seconds": 0.0,
        "cores": cpu / wall,
        "items": 1,
        "jobs": 0,
        "workers_used": 0,
        "workers_admitted": 0,
        "parent_rss_peak_bytes": 0,
        "worker_rss_peak_bytes": n,
        "cache": {"view": [0, 1]},
        "findings": [],
        "wait_seconds": wait,
    }


def _envelope(records):
    return {"worker_cpu": 0, "native_cpu": 0, "rss": 1, "pid": 1, "start_ns": 0, "end_ns": 1, "records": records}


def test_sub_floor_worker_spans_of_one_path_fold_into_one_logged_record(sink):
    # 5,439 jobs each under the log floor: every span was dropped and their worker-seconds vanished from the log.
    # One path's spans fold into one record that crosses the floor.
    stages.configure(budgets=BUDGETS)
    raws = [_job_raw(("job", "native.objects"), 0.05, 0.02, 0.01, n) for n in range(5000)]
    with run(sink):
        stages.record_pool("units", 5000, 5000, 0, stages.time.perf_counter_ns(), [_envelope(raws)])
    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    folded = [r for r in rows if r["path"][-2:] == ["job", "native.objects"]]
    assert len(folded) == 1
    row = folded[0]
    assert row["spans"] == 5000 and row["items"] == 5000
    assert row["wall_seconds"] == pytest.approx(250.0) and row["worker_cpu_seconds"] == pytest.approx(100.0)
    assert row["wait_seconds"] == pytest.approx(50.0) and row["cores"] == pytest.approx(0.4)
    assert row["cache"] == {"view": [0, 5000]} and row["worker_rss_peak_bytes"] == 4999
    assert row["start_ns"] == 0 and not any(f["key"] == "budget.single_core" for f in row["findings"])


def test_a_folded_child_hangs_under_its_folded_parent_and_findings_stay_alone(sink):
    raws = []
    for n in range(3):
        job = _job_raw(("job",), 1.0, 0.5, 0.0, n, kind="job")
        child = _job_raw(("job", "view.get"), 0.5, 0.1, 0.0, 10 + n)
        child["parent"] = job["span"]
        raws += [job, child]
    raws[1]["findings"] = [{"key": "k", "reason": "r"}]
    with run(sink):
        stages.record_pool("units", 3, 3, 1, stages.time.perf_counter_ns(), [_envelope(raws)])
    shipped = [r for r in stages.closed() if r.path[:2] == ("check", "units") and r.kind != "pool"]
    job = next(r for r in shipped if r.kind == "job")
    views = [r for r in shipped if r.path[-1] == "view.get"]
    assert job.spans == 3 and job.wall_seconds == 3.0
    assert sorted(r.spans for r in views) == [1, 2]
    assert all(r.parent == job.span for r in views)


def test_a_worker_span_at_the_log_floor_stays_alone_so_a_pools_tail_is_seen(sink):
    stages.configure(budgets=BUDGETS)
    raws = [_job_raw(("job",), 0.05, 0.02, 0.0, n, kind="job") for n in range(40)]
    raws.append(_job_raw(("job",), 90.0, 89.0, 0.0, 99, kind="job"))
    with run(sink):
        stages.record_pool("units", 41, 41, 0, stages.time.perf_counter_ns(), [_envelope(raws)])
    jobs = sorted((r.spans, r.wall_seconds) for r in stages.closed() if r.kind == "job")
    assert jobs == [(1, 90.0), (40, pytest.approx(2.0))]


def test_the_watchdog_names_a_slow_single_core_stage_with_its_origin(sink):
    origin = {"key": "budgets.serial_seconds", "file": "runtime.toml", "sha256": "0" * 64, "note": ""}
    stages.configure(budgets=BUDGETS, budget_origins={"budgets.serial_seconds": origin})
    raw = _job_raw(("job", "slow"), 5.0, 1.0, 0.0, 1)
    with run(sink):
        stages.record_pool("units", 1, 1, 0, stages.time.perf_counter_ns(), [_envelope([raw])])
    slow = next(r for r in stages.closed() if r.path[-1] == "slow")
    (note,) = slow.findings
    assert note["key"] == "budget.single_core" and note["blocking"] is False and note["origin"] == origin


def test_no_budgets_means_no_watchdog_and_every_span_folds_by_path(sink):
    raws = [_job_raw(("job",), 0.05, 0.02, 0.0, n, kind="job") for n in range(3)]
    with run(sink):
        stages.record_pool("units", 3, 3, 0, stages.time.perf_counter_ns(), [_envelope(raws)])
    assert [r.spans for r in stages.closed() if r.kind == "job"] == [3]
    assert all(not r.findings for r in stages.closed())


def test_a_span_tree_equals_the_one_unbake_wrote_for_the_same_work(sink, monkeypatch):
    # tests/fixtures/unbake_stage_tree.json was captured from unbake 0.5.23's effort module running this scenario
    # under the same fake clock, cpu, ids and resident sizes. Only `schema` is new.
    clock = itertools.count(1_000_000_000, 100_000_000)
    cpu = itertools.count(0, 1)
    ids = itertools.count(1)
    monkeypatch.setattr(stages, "time", SimpleNamespace(perf_counter_ns=lambda: next(clock)))
    monkeypatch.setattr(
        stages,
        "resource",
        SimpleNamespace(
            getrusage=lambda who: SimpleNamespace(ru_utime=float(next(cpu)) / 4, ru_stime=0.0, ru_maxrss=1000),
            RUSAGE_SELF=0,
        ),
    )
    monkeypatch.setattr(stages, "uuid4", lambda: SimpleNamespace(hex=f"{next(ids):032x}"))
    monkeypatch.setattr(stages, "_rss", lambda: 4096)
    raw_job = {
        "invocation": "w", "span": "j1", "parent": None, "path": ["job"], "kind": "job", "status": "ok",
        "execution": "serial", "start_ns": 5, "wall_seconds": 0.5, "parent_cpu_seconds": 0.25,
        "worker_cpu_seconds": 0.0, "native_cpu_seconds": 0.0, "cores": 0.5, "items": 1, "jobs": 0,
        "workers_used": 0, "workers_admitted": 0, "parent_rss_peak_bytes": 0, "worker_rss_peak_bytes": 7,
        "cache": {"view": [1, 2]}, "findings": [{"key": "k", "reason": "r"}], "wait_seconds": 0.125,
    }  # fmt: skip
    native = SimpleNamespace(wall=0.5, cpu=0.75, max_rss=4096)
    with run(sink):
        with stages.stage("outer") as span:
            span.add(items=2, findings=[stages.finding("note.one", "first", path="a.c", line=3, blocking=False)])
            stages.count("object", True)
            stages.count("object", False)
            stages.record_native("cc", native, stages.time.perf_counter_ns())
            stages.waited(1.5)
        try:
            with stages.stage("refused"):
                raise Refusal(stages.finding("worker.memory", "too big", missing=("a",)))
        except Refusal:
            pass
        stages.record_pool(
            "units", 4, 4, 2, stages.time.perf_counter_ns(),
            [{"worker_cpu": 1.0, "native_cpu": 0.5, "rss": 2048, "pid": 9, "start_ns": 0, "end_ns": 1_000_000_000,
              "cache": {"object": [1, 1]}, "records": [raw_job], "wait_seconds": 0.25}],
            dispatch_seconds=0.125,
        )  # fmt: skip
    golden = json.loads((Path(__file__).parent / "fixtures" / "unbake_stage_tree.json").read_text())
    ours = json.loads(json.dumps(stages.tree(), default=list))
    assert ours.pop("schema") == "stage/1"

    def strip(node):
        node.pop("schema", None)
        for child in node["children"]:
            strip(child)

    strip(ours)
    assert ours == golden
