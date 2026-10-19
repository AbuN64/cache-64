# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The host slots: the frozen wire protocol, fair shares, memory admission, named capacities, the make jobserver and
the wait stages. Peers are made of flock files this test holds, so no timing decides a result."""

from __future__ import annotations

import contextlib
import fcntl
import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from abucache import slots, stages


def hold(path: Path, text: str = "") -> object:
    """A peer: a file this test holds an exclusive flock on, as a live command or slot holder would."""
    handle = path.open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    if text:
        handle.truncate(0)
        handle.write(text)
        handle.flush()
    return handle


def drop(handle) -> None:
    fcntl.flock(handle, fcntl.LOCK_UN)
    handle.close()


@pytest.fixture
def pool(rundir: Path) -> Path:
    directory = rundir / slots.POOL
    directory.mkdir()
    return directory


def test_the_wire_protocol_constants_and_file_formats_are_frozen(rundir, make_runtime, tmp_path):
    assert (slots.PROTOCOL, slots.POOL, slots.WAITS) == (2, "unbake-pool-2", "waits.jsonl")
    assert (slots.CORES_ENV, slots.DEAD_AFTER) == ("UNBAKE_CORES", 60.0)
    assert (slots._IN_CHANGES, slots._IN_FLAGS) == (0x8 | 0x80 | 0x100 | 0x200, 0o4000 | 0o2000000)
    assert slots.wait_log() == rundir / "unbake-pool-2" / "waits.jsonl"
    mine = slots.Slots(make_runtime(workers=4))
    with mine.cores(3) as granted:
        files = sorted(p.name for p in (rundir / slots.POOL).iterdir())
        assert granted == 3
        # one flock file per held slot, this command's lock holding "<demand> <budget>"
        assert [f for f in files if f.startswith("slot-")] == ["slot-0.lock", "slot-1.lock", "slot-2.lock"]
        assert (rundir / slots.POOL / f"cmd-{os.getpid()}.lock").read_text() == "3 4"
        assert "wake" not in files  # a first word wakes nobody
    assert not list((rundir / slots.POOL).glob("cmd-*.lock"))  # a finished command leaves no lock file
    assert not mine.slots and not mine.cmd


@pytest.mark.parametrize(
    ("others", "share"), [(0, 8), (1, 4), (2, 3), (3, 2), (7, 1), (12, 1)]
)
def test_a_command_takes_its_fair_share_of_the_budget(others, share, pool, make_runtime):
    peers = [hold(pool / f"cmd-{1 + i}.lock", "8 8") for i in range(others)]
    try:
        mine = slots.Slots(make_runtime(workers=8))
        with mine.cores() as granted:
            assert granted == share
    finally:
        for peer in peers:
            drop(peer)


def test_a_share_a_peer_cannot_use_goes_to_the_command_that_can(pool, make_runtime):
    peer = hold(pool / "cmd-1.lock", "2 8")
    try:
        with slots.Slots(make_runtime(workers=8)).cores() as granted:
            assert granted == 6
    finally:
        drop(peer)


def test_cores_never_exceeds_the_work(pool, make_runtime):
    mine = slots.Slots(make_runtime(workers=8))
    with mine.cores(2) as granted:
        assert granted == 2 and len(mine.slots) == 2
        assert (pool / f"cmd-{os.getpid()}.lock").read_text() == "2 8"  # peers are told the work, not the budget


def test_a_command_under_a_holder_uses_its_cores_and_takes_no_slots_of_its_own(pool, make_runtime, monkeypatch):
    monkeypatch.setenv(slots.CORES_ENV, "3")
    mine = slots.Slots(make_runtime(workers=8))
    with mine.cores() as granted:
        assert granted == 3
        assert not list(pool.iterdir())


def test_a_finished_or_unheld_command_file_is_no_peer(pool, make_runtime):
    (pool / "cmd-1.lock").write_text("1 8")  # nobody holds it
    gone = pool / "cmd-2.lock"
    gone.write_text("1 8")
    os.utime(gone, (0, 0))  # long finished: a look removes it
    with slots.Slots(make_runtime(workers=8)).cores() as granted:
        assert granted == 8
    assert not gone.exists()


def test_nested_holders_share_one_acquisition_and_the_last_one_releases(pool, make_runtime):
    mine = slots.Slots(make_runtime(workers=4))
    with mine.cores(2), mine.cores(2):
        assert len(mine.slots) == 2
    assert not mine.slots and not list(pool.glob("cmd-*.lock"))


class TestMemoryAdmission:
    def test_the_grant_is_the_tasks_the_free_memory_holds(self, pool, make_runtime, monkeypatch):
        monkeypatch.setattr(slots, "available_memory", lambda: 10 * 1024**3)
        mine = slots.Slots(make_runtime(workers=8, reserve=2 * 1024**3))
        with mine.cores(task_memory=2 * 1024**3) as granted:
            assert granted == 4  # (10 - 2) GiB // 2 GiB
            assert (pool / f"cmd-{os.getpid()}.lock").read_text() == "4 8"

    def test_the_work_and_the_budget_still_bound_it(self, pool, make_runtime, monkeypatch):
        monkeypatch.setattr(slots, "available_memory", lambda: 1000 * 1024**3)
        with slots.Slots(make_runtime(workers=3)).cores(task_memory=1024**3) as granted:
            assert granted == 3
        with slots.Slots(make_runtime(workers=8)).cores(2, task_memory=1024**3) as granted:
            assert granted == 2

    def test_a_command_always_gets_one_slot_however_little_memory_is_free(self, pool, make_runtime, monkeypatch):
        monkeypatch.setattr(slots, "available_memory", lambda: 0)
        with slots.Slots(make_runtime(workers=8, reserve=1024**3)).cores(task_memory=4 * 1024**3) as granted:
            assert granted == 1

    def test_no_task_memory_is_no_bound(self, make_runtime):
        assert slots.memory_slots(0, 123) == 0

    def test_the_memory_is_read_at_each_acquire(self, pool, make_runtime, monkeypatch):
        free = [8 * 1024**3]
        monkeypatch.setattr(slots, "available_memory", lambda: free[0])
        mine = slots.Slots(make_runtime(workers=8))
        with mine.cores(task_memory=1024**3) as granted:
            assert granted == 8
        free[0] = 3 * 1024**3
        with mine.cores(task_memory=1024**3) as granted:
            assert granted == 3

    def test_the_feeder_grows_only_within_the_free_memory(self, pool, make_runtime, monkeypatch):
        free = [2 * 1024**3]
        monkeypatch.setattr(slots, "available_memory", lambda: free[0])
        mine = slots.Slots(make_runtime(workers=8))
        with mine.cores(task_memory=1024**3):
            assert len(mine.slots) == 2
            assert mine._extra_slot() is None  # the third task does not fit
            free[0] = 4 * 1024**3
            assert mine._extra_slot() is not None and len(mine.slots) == 3  # a rebalance sees the memory freed


class TestCapacities:
    def test_a_capacity_of_one_admits_one_holder_at_a_time(self, pool, make_runtime):
        mine = slots.Slots(make_runtime(workers=4, capacities={"repro": 1}))
        (pool / "cap-repro").mkdir()
        peer = hold(pool / "cap-repro" / "token-0.lock", str(os.getppid()))
        released = threading.Timer(0.3, drop, [peer])
        released.start()
        with mine.capacity("repro"):  # blocks until the peer lets go
            assert not released.is_alive()
            assert (pool / "cap-repro" / "token-0.lock").read_text() == str(os.getpid())
        released.join()

    def test_capacity_waiters_are_served_in_arrival_order_and_none_is_passed_over(self, pool, make_runtime):
        runtime = make_runtime(capacities={"repro": 1})
        cap = pool / "cap-repro"
        cap.mkdir()
        peer = hold(cap / "token-0.lock", str(os.getppid()))
        order: list[str] = []
        gates = {"a": threading.Event(), "b": threading.Event(), "c": threading.Event()}

        def waiter(tag):
            with slots.Slots(runtime).capacity("repro"):
                order.append(tag)
                gates[tag].wait(30)

        def until(condition):
            deadline = time.monotonic() + 30
            while not condition():
                assert time.monotonic() < deadline
                time.sleep(0.01)

        threads = []
        for count, tag in enumerate("abc", 1):  # each is queued before the next one asks
            threads.append(threading.Thread(target=waiter, args=(tag,)))
            threads[-1].start()
            until(lambda count=count: len(list(cap.glob("wait-*"))) == count)
        drop(peer)  # the token frees: it is a's, not whoever asks next
        for tag, thread in zip("abc", threads, strict=True):
            until(lambda tag=tag: tag in order)
            assert order == list("abc")[: order.index(tag) + 1]  # nobody ahead of its turn
            gates[tag].set()
            thread.join(30)
        assert order == ["a", "b", "c"] and not list(cap.glob("wait-*"))

    def test_the_ticket_of_a_dead_capacity_waiter_holds_nobody_back(self, pool, make_runtime):
        cap = pool / "cap-repro"
        cap.mkdir()
        (cap / "wait-00000000000000000001-999999").touch()  # nobody holds its lock: its owner is gone
        with slots.Slots(make_runtime(capacities={"repro": 1})).capacity("repro"):
            assert (cap / "token-0.lock").read_text() == str(os.getpid())

    def test_a_capacity_of_two_admits_two(self, pool, make_runtime):
        mine = slots.Slots(make_runtime(capacities={"repro": 2}))
        with mine.capacity("repro"), mine.capacity("repro"):
            held = [p.name for p in (pool / "cap-repro").glob("token-*.lock")]
            assert sorted(held) == ["token-0.lock", "token-1.lock"]

    def test_a_waiter_is_a_stage_naming_the_holder_and_its_seconds_are_counted(self, pool, make_runtime, tmp_path):
        mine = slots.Slots(make_runtime(capacities={"repro": 1}))
        (pool / "cap-repro").mkdir()
        peer = hold(pool / "cap-repro" / "token-0.lock", str(os.getppid()))
        threading.Timer(0.3, drop, [peer]).start()
        sink = tmp_path / "stages.jsonl"
        with stages.command("tool", "work", [], sink), stages.capture() as records, mine.capacity("repro"):
            pass
        waits = [r for r in records if r.path[-1] == "pool.wait_capacity"]
        assert len(waits) == 1
        assert f"repro: held by pids {os.getppid()}" in waits[0].findings[0]["reason"]
        assert waits[0].findings[0]["blocking"] is False

    def test_cores_with_a_capacity_holds_the_token_for_the_block(self, pool, make_runtime):
        mine = slots.Slots(make_runtime(capacities={"repro": 1}))
        with mine.cores(2, capacity="repro") as granted:
            assert granted == 2 and (pool / "cap-repro" / "token-0.lock").exists()
            probe = slots._try(pool / "cap-repro" / "token-0.lock")
            assert probe is None  # held
        free = slots._try(pool / "cap-repro" / "token-0.lock")
        assert free is not None
        free.close()

    def test_an_unlisted_capacity_is_an_error_and_a_bad_name_is_refused(self, rundir, make_runtime):
        mine = slots.Slots(make_runtime(capacities={"repro": 1}))
        with pytest.raises(slots.UnknownCapacity), mine.capacity("other"):
            pass
        with pytest.raises(ValueError), mine.capacity("../x"):
            pass

    def test_a_process_under_a_holder_of_the_capacity_takes_no_token(self, pool, make_runtime, monkeypatch):
        monkeypatch.setenv(slots.CAPACITIES_ENV, "other,repro")
        with slots.Slots(make_runtime(capacities={"repro": 1})).capacity("repro"):
            assert not (pool / "cap-repro").exists()

    def test_a_capacity_never_touches_a_slot_file(self, pool, make_runtime):
        with slots.Slots(make_runtime(capacities={"repro": 1})).capacity("repro"):
            assert not list(pool.glob("slot-*")) and not list(pool.glob("cmd-*"))


def test_a_protocol_nobody_runs_any_more_costs_no_wait(rundir, make_runtime):
    other = rundir / "unbake-pool-1"
    other.mkdir()
    (other / "cmd-1.lock").write_text("1 1")  # nobody holds it
    with slots.Slots(make_runtime(workers=2)).cores() as granted:
        assert granted == 2


def test_a_command_waits_for_live_commands_of_another_slot_protocol_to_release(rundir, make_runtime, tmp_path):
    other = rundir / "unbake-pool-1"
    other.mkdir()
    peer = hold(other / "cmd-1.lock", "1 1")
    threading.Timer(0.3, drop, [peer]).start()
    mine = slots.Slots(make_runtime(workers=2))
    with stages.command("tool", "work", [], tmp_path / "stages.jsonl"), stages.capture() as records, mine.cores() as n:
        assert n == 2
    waits = [r for r in records if r.path[-1] == "pool.protocol"]
    assert len(waits) == 1 and "another slot protocol than 2" in waits[0].findings[0]["reason"]


def test_a_held_slot_shows_up_as_its_own_wait_stage_naming_the_holders(pool, make_runtime, tmp_path):
    cmd = hold(pool / "cmd-1.lock", "1 2")
    held = [hold(pool / f"slot-{i}.lock") for i in range(2)]

    def release() -> None:
        for handle in held:
            drop(handle)
        drop(cmd)

    threading.Timer(0.3, release).start()
    with stages.command("tool", "work", [], tmp_path / "stages.jsonl"), stages.capture() as records:
        mine = slots.Slots(make_runtime(workers=2))
        with mine.cores() as granted:
            assert granted >= 1
        assert stages._waits["slots"] > 0.0  # the seconds blocked, not the whole acquire
    waits = [r for r in records if r.path[-1] == "pool.wait_slot"]
    assert len(waits) == 1 and "held by pids 1" in waits[0].findings[0]["reason"]


def test_a_share_short_of_the_work_shows_in_the_tree_once_naming_the_holders(pool, make_runtime, tmp_path):
    cmd = hold(pool / "cmd-1.lock", "4 4")
    held = [hold(pool / f"slot-{i}.lock") for i in range(3)]
    try:
        with stages.command("tool", "work", [], tmp_path / "stages.jsonl"), stages.capture() as records:
            mine = slots.Slots(make_runtime(workers=4))
            with mine.cores(4) as granted:
                assert granted == 1
            with mine.cores(4):
                pass
        shorts = [r for r in records if r.path[-1] == "pool.short_slots"]
        assert len(shorts) == 1  # the same shortfall is reported once
        assert shorts[0].findings[0]["reason"] == "1 of 2 slots; the rest are held by pids 1"
    finally:
        for handle in (*held, cmd):
            drop(handle)


def test_a_freed_slot_goes_to_the_oldest_waiter_not_to_a_newer_command(pool, make_runtime):
    mine = slots.Slots(make_runtime(workers=2))
    older = pool / f"wait-{1:020d}-{os.getppid()}"
    older.write_text("")
    cmd = hold(pool / f"cmd-{os.getppid()}.lock", "1 2")  # the older waiter's command is live
    try:
        assert mine._queued(pool) is True  # this command has no ticket of its own: it queues behind
        mine._enter(pool)
        assert mine._queued(pool) is True  # its ticket is newer than the other
        older.unlink()
        assert mine._queued(pool) is False  # nobody is ahead
        mine._leave()
    finally:
        drop(cmd)


def test_the_ticket_of_a_dead_command_holds_nobody_back(pool, make_runtime):
    (pool / "wait-00000000000000000001-4000000").write_text("")
    assert slots.Slots(make_runtime())._queued(pool) is False
    assert not list(pool.glob("wait-*"))


def test_slots_grow_and_shrink_without_duplicate_locks_or_releasing_active_jobs(pool, make_runtime):
    mine = slots.Slots(make_runtime(workers=4))
    mine.acquire(cap=2)
    try:
        assert len(mine.slots) == 2
        mine.acquire(cap=4)  # grows
        assert len(mine.slots) == 4 and len({s.name for s in mine.slots}) == 4
        mine.acquire(cap=1, active=3)  # shrinks, but never below the chunks in flight
        assert len(mine.slots) == 3
    finally:
        mine.release()


def test_two_processes_each_holding_four_divide_a_budget_of_eight(rundir, make_runtime, tmp_path):
    runtime = make_runtime(workers=8)
    context = multiprocessing.get_context("fork")
    started, stop = context.Event(), context.Event()
    out = context.Queue()

    def child() -> None:
        with slots.Slots(runtime).cores(4) as granted:
            out.put(granted)
            started.set()
            stop.wait(30)

    process = context.Process(target=child)
    process.start()
    try:
        assert out.get(timeout=30) == 4 and started.wait(30)
        with slots.Slots(runtime).cores(4) as granted:
            assert granted == 4
            held = sorted(p.name for p in (rundir / slots.POOL).glob("slot-*.lock") if slots._try(p) is None)
            assert len(held) == 8  # both hold four, no slot twice
    finally:
        stop.set()
        process.join(30)


def test_a_waiter_woken_by_a_holders_death_takes_the_slot(pool, make_runtime):
    context = multiprocessing.get_context("fork")
    ready = context.Event()

    def holder() -> None:
        with slots.Slots(make_runtime(workers=1)).cores():
            ready.set()
            time.sleep(60)

    process = context.Process(target=holder)
    process.start()
    try:
        assert ready.wait(30)
        threading.Timer(0.3, process.kill).start()
        with slots.Slots(make_runtime(workers=1)).cores() as granted:  # sleeps on the pidfd, then runs
            assert granted == 1
    finally:
        process.kill()
        process.join(30)


class TestJobserver:
    def test_a_free_host_grants_make_every_worker_never_one_job(self, pool, make_runtime):
        with slots.Slots(make_runtime(workers=4)).jobserver() as (granted, env, fds):
            assert granted == 4
            assert env["UNBAKE_CORES"] == "4" and env["MAKEFLAGS"].startswith("-j4 --jobserver-auth=")
            read, write = fds
            assert os.read(read, 100) == b"+++"  # make itself holds the first token

    def test_the_jobserver_closes_its_pipe_and_gives_the_slots_back(self, pool, make_runtime):
        mine = slots.Slots(make_runtime(workers=3))
        with mine.jobserver() as (_, _, fds):
            pass
        for fd in fds:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert not mine.slots and not list(pool.glob("cmd-*.lock"))

    def test_without_a_slot_pool_makeflags_pins_j_to_the_grant(self, rundir, make_runtime, monkeypatch):
        monkeypatch.setenv(slots.CORES_ENV, "3")
        with slots.Slots(make_runtime(workers=8)).jobserver() as (granted, env, fds):
            assert (granted, env["MAKEFLAGS"], fds) == (3, "-j3", ())

    def test_the_feeder_writes_one_job_token_per_slot_that_frees(self, pool, make_runtime):
        cmd = hold(pool / "cmd-1.lock", "4 4")
        held = [hold(pool / f"slot-{i}.lock") for i in (2, 3)]
        mine = slots.Slots(make_runtime(workers=4))
        with mine.jobserver() as (granted, _, fds):
            assert granted == 2
            read, _ = fds
            for handle in (*held, cmd):  # the peer ends: its slots are free and the share is the whole budget
                drop(handle)
            tokens = os.read(read, 100)  # the grant's one token
            deadline = time.monotonic() + 15
            while len(tokens) < 3 and time.monotonic() < deadline:
                time.sleep(0.05)
                with contextlib.suppress(BlockingIOError):
                    tokens += os.read(read, 100)
            assert len(tokens) >= 3  # grew from 2 into the 4 workers
            assert len(mine.slots) == 4

    def test_exactly_the_slots_the_feeder_added_go_back_when_make_returns(self, pool, make_runtime):
        mine = slots.Slots(make_runtime(workers=4))
        with mine.cores(2):
            added = mine._extra_slot()
            assert added is not None and len(mine.slots) == 3 and added in mine.fed
            read, write = os.pipe()
            try:
                assert mine._shed(read, [], []) is False  # no peer to give a slot to
            finally:
                os.close(read)
                os.close(write)
        assert not mine.slots and not mine.fed


class TestRun:
    def test_a_command_without_the_count_runs_directly_and_takes_no_slot(self, pool, capfd):
        assert slots.run(["--", "/bin/sh", "-c", "ls " + str(pool) + " | wc -l"]) == 0
        assert capfd.readouterr().out.strip() == "0"

    def test_the_count_and_unbake_cores_are_the_held_share(self, rundir, make_runtime, monkeypatch, capfd, tmp_path):
        runtime = make_runtime(workers=6)
        monkeypatch.setattr(slots, "default", slots.Slots(runtime))
        monkeypatch.setattr(slots.runtime_file, "load", lambda path=None: runtime)
        assert slots.run(["--work", "2", "--", "/bin/sh", "-c", "echo {j} $UNBAKE_CORES"]) == 0
        assert capfd.readouterr().out.strip() == "2 2"

    def test_mem_and_capacity_flags_bound_the_grant_and_hold_a_token(self, pool, make_runtime, monkeypatch, capfd):
        runtime = make_runtime(workers=6, capacities={"repro": 1})
        monkeypatch.setattr(slots, "default", slots.Slots(runtime))
        monkeypatch.setattr(slots.runtime_file, "load", lambda path=None: runtime)
        monkeypatch.setattr(slots, "available_memory", lambda: 3 * 1024**3)
        argv = ["--mem", str(1024**3), "--capacity", "repro", "--", "/bin/sh", "-c", "echo {j} $ABU_CAPACITIES"]
        assert slots.run(argv) == 0
        assert capfd.readouterr().out.strip() == "3 repro"
        assert slots._try(pool / "cap-repro" / "token-0.lock") is not None  # given back

    def test_a_make_runs_on_a_jobserver_fed_from_the_slot_pool(self, pool, make_runtime, monkeypatch, capfd):
        runtime = make_runtime(workers=4)
        monkeypatch.setattr(slots, "default", slots.Slots(runtime))
        monkeypatch.setattr(slots.runtime_file, "load", lambda path=None: runtime)
        monkeypatch.setattr(slots, "_is_make", lambda program: True)
        script = 'echo "$MAKEFLAGS" | grep -c "jobserver-auth" ; echo "$@"'
        assert slots.run(["/bin/sh", "-c", script, "make", "-j{j}", "x"]) == 0
        out = capfd.readouterr().out.split("\n")
        assert out[0] == "1"

    def test_a_refused_runtime_file_runs_the_command_alone(self, rundir, tmp_path, monkeypatch, capfd):
        monkeypatch.setenv("ABU_RUNTIME", str(tmp_path / "absent.toml"))
        assert slots.run(["--", "/bin/sh", "-c", "echo {j} $UNBAKE_CORES"]) == 0
        assert capfd.readouterr().out.strip() == "1 1"
        assert not list((rundir).glob("unbake-pool-*"))

    def test_no_command_is_a_usage_error(self, capsys):
        assert slots.run(["--work", "2"]) == 2
