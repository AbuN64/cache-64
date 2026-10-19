# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The generic executor: ordered maps, fork sharing and spills, keyed groups and seals, failures and stalls. Ported
from unbake's executor tests with the shared object, the context and the refusal type registered through the seam."""

from __future__ import annotations

import io
import os
import pickle
import sys
import threading
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import pytest

from abucache import files, keys, memo, pool, slots, stages, store

_RAISED: BaseException = ValueError("unset")
_RAN: list[int] = []
_HELD: list[object] = []  # what the registered context holds in this process


def _fail(value):  # module level: items and functions travel to workers through a pickle
    raise _RAISED


def _record(value):
    _RAN.append(value)
    return value


def _name(value):
    return str(value)


def _square(value):
    return value * value


def _identify(value):
    return value, os.getpid()


class Refused(Exception):
    """The tool's own refusal: passes through."""


class Tree:
    """A shared object: big enough to be worth sharing, with the overlay-chain shape of a tool's snapshot."""

    def __init__(self, digest: str = "digest-1", base: Tree | None = None) -> None:
        self.digest, self.base = digest, base
        self.files = {f"src/m{i}.c": b"int m%d;\n" % i * 20 for i in range(2000)}


@pytest.fixture(autouse=True)
def _state(monkeypatch, tmp_path, rundir, make_runtime):
    """A fresh pool, slots and seam per test; nothing leaks into the next one on this process."""
    runtime = make_runtime(workers=2)
    monkeypatch.setattr(slots, "default", slots.Slots(runtime))
    monkeypatch.setattr(stages, "_listeners", [])
    monkeypatch.setattr(stages, "_runtime", None)
    monkeypatch.setattr(stages.feed, "path", None)
    monkeypatch.setenv("ABU_RUNTIME", str(tmp_path / "no-runtime.toml"))
    monkeypatch.setattr(pool, "_in_worker", False)
    for name, value in (
        ("_shared", []),
        ("_contexts", {}),
        ("_passthrough", ()),
        ("_warm", ()),
        ("_keep", ()),
        ("_carried", {}),
        ("_sent", {}),
        ("_SHARED", {}),
        ("_laid", []),
    ):
        monkeypatch.setattr(pool, name, value)
    monkeypatch.setattr(memo, "_taggers", [])
    monkeypatch.setattr(memo, "_memo", {})
    monkeypatch.setattr(memo, "_epochs", {})
    monkeypatch.setattr(sys.modules[__name__], "_RAN", [])
    pool.configure(runtime, spill_dir=tmp_path / "spill", identity_package="abucache")
    pool.register_passthrough(Refused)
    pool.register_shared(Tree, key=lambda tree: tree.digest, tag=lambda tree: tree.digest)
    yield
    pool._drop_executor()
    slots.default.release()


@pytest.fixture
def mocked_parallel(monkeypatch):
    """Futures that complete on submit, so a test sees how many chunks were in flight and no process is forked."""
    active: set[Future] = set()
    widths: list[int] = []

    class Executor:
        def submit(self, function, *args):
            future: Future = Future()
            future.set_result(function(*args))
            active.add(future)
            widths.append(len(active))
            return future

    def wait(pending, return_when, timeout=None):
        selected = next(reversed(pending))
        active.discard(selected)
        return {selected}, set(pending) - {selected}

    monkeypatch.setattr(pool, "_get_executor", lambda needed=(), work=0: Executor())
    monkeypatch.setattr(pool, "wait", wait)
    return widths


def command():
    return stages.command("tool", "check", [], Path(os.devnull))


def pool_record():
    return next(r for r in stages.closed() if r.kind == "pool")


@pytest.mark.parametrize("serial", [True, False])
def test_progress_counts(monkeypatch, mocked_parallel, serial):
    monkeypatch.setattr(pool, "_in_worker", serial)
    seen = []
    stages.listen(lambda event, body: seen.append(body) if event == "progress" else None)
    with command():
        assert pool.map("units", _square, [1, 2, 3, 4]) == [1, 4, 9, 16]
    assert seen == [{"name": "units", "done": n, "total": 4} for n in range(1, 5)]
    if not serial:
        assert max(mocked_parallel) == 2  # only admitted jobs hold slots


@pytest.mark.parametrize("items", [[], [3]])
def test_inline_no_executor(monkeypatch, items):
    monkeypatch.setattr(pool, "_get_executor", lambda *a, **k: pytest.fail("inline map started an executor"))
    with command():
        assert pool.map("units", _square, items) == [v * v for v in items]
    record = pool_record()
    assert record.items == record.jobs == len(items) and record.workers_admitted == 0


@pytest.mark.parametrize(
    ("raised", "kind"),
    [(MemoryError("limit"), "memory"), (ValueError("crash"), "crash")],
)
def test_a_failing_job_becomes_a_task_failure_naming_the_item(mocked_parallel, monkeypatch, raised, kind):
    monkeypatch.setattr(sys.modules[__name__], "_RAISED", raised)
    with command(), pytest.raises(pool.TaskFailed) as caught:
        pool.map("units", _fail, [1, 2])
    failure = caught.value
    assert (failure.kind, failure.stage, failure.index) == (kind, "units", 0)
    assert repr(raised) in failure.reason and failure.__cause__ is raised
    assert pool_record().jobs == 2


def test_a_registered_passthrough_type_reaches_the_caller_unchanged(mocked_parallel, monkeypatch):
    refusal = Refused("no")
    monkeypatch.setattr(sys.modules[__name__], "_RAISED", refusal)
    with command(), pytest.raises(Refused) as caught:
        pool.map("units", _fail, [1, 2])
    assert caught.value is refusal


def test_job_cache_and_metrics():
    def work(value):
        stages.count("object", True)
        stages.count("view", False)
        return value

    ok, value, envelope = pool._job(work, 7)
    assert (ok, value) == (True, 7)
    assert envelope["cache"] == {"object": (1, 0), "view": (0, 1)}
    assert envelope["worker_cpu"] >= 0 and envelope["rss"] > 0
    assert [r.kind for r in envelope["records"]] == ["job"]


def test_job_ships_every_record_the_chunk_folds_and_the_parent_floors():
    def work(value):
        with stages.stage("quick"):
            pass
        with stages.stage("waited"):
            stages.waited(0.5)
        return value

    _, _, everything = pool._job(work, 1)
    assert {r.path[-1] for r in everything["records"]} == {"quick", "waited", "job", "(own)"}


def test_a_crash_names_the_type_and_the_innermost_frame_of_the_identity_package():
    ok, error, _ = pool._job(keys.sha256, "not bytes")  # hashlib refuses inside keys.sha256
    assert not ok and error.__notes__[0].startswith("at keys:sha256:")
    with pytest.raises(pool.TaskFailed) as raised:
        pool._raise_failure("stage", 3, error)
    assert "TypeError" in raised.value.reason and raised.value.index == 3
    assert raised.value.notes[0].startswith("at keys:sha256:")  # the notes travel apart: the tool words its refusal


def test_a_broken_pool_is_dropped_and_named(mocked_parallel, monkeypatch):
    dropped = []

    def broken(pending, return_when, timeout=None):
        raise BrokenProcessPool("worker gone")

    monkeypatch.setattr(pool, "wait", broken)
    monkeypatch.setattr(pool, "_drop_executor", lambda: dropped.append(True))
    with command(), pytest.raises(pool.TaskFailed) as caught:
        pool.map("units", _square, [1, 2])
    assert caught.value.kind == "crash" and dropped == [True]


class FakeExecutor:
    made: list[FakeExecutor] = []

    def __init__(self, **kwargs):
        self.kwargs, self.shutdowns, self.seen = kwargs, [], set(pool._SHARED)
        FakeExecutor.made.append(self)

    def submit(self, function):
        future: Future = Future()
        future.set_result(None)
        return future

    def shutdown(self, **kwargs):
        self.shutdowns.append(kwargs)


@pytest.fixture
def fake_executor(monkeypatch):
    FakeExecutor.made = []
    monkeypatch.setattr(pool, "ProcessPoolExecutor", FakeExecutor)
    return FakeExecutor


def test_the_executor_is_reused_and_forked_again_only_when_the_host_changes(fake_executor, tmp_path, make_runtime):
    first = pool._get_executor()
    assert pool._get_executor() is first
    other = make_runtime(workers=3)
    pool.configure(other, spill_dir=tmp_path / "spill", identity_package="abucache")
    second = pool._get_executor()
    assert second is not first and first.shutdowns == [{"wait": True, "cancel_futures": True}]
    assert second.kwargs["max_workers"] == 3
    assert second.kwargs["mp_context"].get_start_method() == "fork"
    assert second.kwargs["initargs"] == (other.worker_memory,)


def test_threads_asking_at_once_create_one_executor(monkeypatch):
    made, got = [], []

    class Slow(FakeExecutor):
        def __init__(self, **kwargs):
            time.sleep(0.05)  # long enough for every other thread to arrive while this one is creating
            super().__init__(**kwargs)
            made.append(self)

    monkeypatch.setattr(pool, "ProcessPoolExecutor", Slow)
    threads = [threading.Thread(target=lambda: got.append(pool._get_executor())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1 and got == made * 8


def test_real_forked_map_runs_on_workers_in_order():
    seen = []
    stages.listen(lambda event, body: seen.append(body["done"]) if event == "progress" else None)
    with command():
        assert pool.map("units", _square, [2, 3, 4]) == [4, 9, 16]
    assert seen == [1, 2, 3]
    record = pool_record()
    assert record.jobs == 3 and 1 <= record.workers_used <= record.workers_admitted == 2


def test_real_batched_work_keeps_count_order_and_worker_pids():
    items = list(range(256))
    with command():
        values = pool.map("units", _identify, items)
    assert [value for value, _ in values] == items and all(pid != os.getpid() for _, pid in values)
    record = pool_record()
    assert record.jobs == record.items == len(items) and record.workers_admitted == 2
    assert record.workers_used == len({pid for _, pid in values})
    assert sum(r.spans for r in stages.closed() if r.kind == "job") == len(items)


def test_a_worker_dies_with_its_parent_and_lets_go_of_the_parents_locks(monkeypatch):
    calls, dropped = [], []
    monkeypatch.setattr(pool.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(pool.ctypes, "CDLL", lambda name: type("L", (), {"prctl": lambda self, *a: calls.append(a)})())
    monkeypatch.setattr(files, "drop_inherited", lambda: dropped.append("files"))
    monkeypatch.setattr(slots.default, "close_inherited", lambda: dropped.append("slots"))
    pool._init(1 << 30)
    assert calls == [(1, pool.signal.SIGKILL)] and dropped == ["slots", "files"]
    assert pool._in_worker


def test_a_worker_closes_the_parents_slot_and_command_locks():
    slots.default.acquire()
    held = [*slots.default.slots, *slots.default.cmd]
    assert held and not any(f.closed for f in held)
    slots.default.close_inherited()
    assert all(f.closed for f in held)
    slots.default.slots.clear()
    slots.default.cmd.clear()


# -- keyed groups -------------------------------------------------------------------------------------------------


def reads_of(monkeypatch):
    reads: list[str] = []
    real = store.Namespace.get

    def get(self, key):
        reads.append(self.kind)
        return real(self, key)

    monkeypatch.setattr(store.Namespace, "get", get)
    return reads


def test_a_warm_keyed_group_costs_one_store_read(monkeypatch):
    reads, items = reads_of(monkeypatch), [1, 2, 3, 4]
    with command():
        assert pool.map("units", _square, items, key=str) == [1, 4, 9, 16]
    assert reads.count("abucache.units") == 4  # the cold run asks for every item
    reads.clear()
    with command():
        assert pool.map("units", _square, items, key=str) == [1, 4, 9, 16]
        assert stages.counters()["units"] == (4, 0)  # all four still count as hits
    assert reads == ["abucache.units.group"]
    reads.clear()
    assert pool.map("units", _square, [1, 2, 3, 5], key=str) == [1, 4, 9, 25]  # one new item: per-item reads
    assert reads.count("abucache.units") == 4


def test_a_sealed_group_reads_back_whole_and_a_failed_one_is_never_sealed():
    pool.map("units", _square, [1, 2, 3], key=_name)
    group = store.Namespace(pool._cache(), "abucache", "units.group")
    whole = keys.digest([keys.digest((str(n), pool._identity(_square, _name))) for n in (1, 2, 3)])
    sealed = pickle.loads(group.get(whole))
    assert [pickle.loads(blob) for blob in sealed] == [1, 4, 9]
    with pytest.raises(pool.TaskFailed):
        pool.map("bad", _fail, [1, 2], key=_name)
    failing = keys.digest([keys.digest((str(n), pool._identity(_fail, _name))) for n in (1, 2)])
    assert store.Namespace(pool._cache(), "abucache", "bad.group").get(failing) is None


def test_whoever_ran_a_keyed_chunk_stores_its_results():
    outcomes = pool._chunk(_square, [2, 3, 4], ("units", ["a", None, "c"]))
    assert [o[1] for o in outcomes] == [4, 9, 16]
    units = store.Namespace(pool._cache(), "abucache", "units")
    assert pickle.loads(units.get("a")) == 4 and units.get("c") is not None
    assert units.get("b") is None  # an unkeyed item is not stored
    failed = pool._chunk(_fail, [1], ("units", ["f"]))
    assert failed[0][0] is False and units.get("f") is None
    assert [o[1] for o in pool._chunk(_square, [5], None)] == [25]


def test_a_stored_result_sits_under_the_tool_namespace_and_stage():
    pool.map("units", _square, [7, 8], key=_name)
    key = keys.digest(("7", pool._identity(_square, _name)))
    assert pool._cache().get("abucache.units", key) is not None


def test_an_equal_key_from_a_changed_module_is_cold(monkeypatch, mocked_parallel):
    monkeypatch.setattr(keys, "code_identity", lambda function: "one")
    pool.map("units", _record, [1, 2], key=_name)
    pool.map("units", _record, [1, 2], key=_name)
    assert _RAN == [1, 2]  # the second run is warm
    monkeypatch.setattr(keys, "code_identity", lambda function: "two")
    pool.map("units", _record, [1, 2], key=_name)
    assert _RAN == [1, 2, 1, 2]


def _stable_name(value):
    return str(value)


_stable_name.stable = True


def _clean(value):
    _RAN.append(value)
    return ()


def test_a_stable_key_survives_a_changed_module(monkeypatch, mocked_parallel):
    monkeypatch.setattr(keys, "code_identity", lambda function: "one")
    pool.map("units", _clean, [1, 2], key=_stable_name)
    monkeypatch.setattr(keys, "code_identity", lambda function: "two")
    pool.map("units", _clean, [1, 2], key=_stable_name)
    assert _RAN == [1, 2]


def _verdict(value):
    _RAN.append(value)
    return [f"finding {value}"] if value == 2 else ()


def test_a_stable_key_stores_only_clean_results(mocked_parallel):
    pool.map("units", _verdict, [1, 2], key=_stable_name)
    pool.map("units", _verdict, [1, 2], key=_stable_name)
    assert _RAN == [1, 2, 2]  # the clean item is warm, the one with findings runs again


def _coded_name(value):
    return str(value)


_coded_name.coded = True


def test_a_coded_key_keeps_its_results_warm_across_an_install(monkeypatch, mocked_parallel):
    monkeypatch.setattr(keys, "code_identity", lambda function: "one")
    pool.map("units", _record, [1, 2], key=_coded_name)
    monkeypatch.setattr(keys, "code_identity", lambda function: "two")
    pool.map("units", _record, [1, 2], key=_coded_name)
    assert _RAN == [1, 2]


_KEY_CHUNKS: list[int] = []


def _record_key_chunk(item):
    _KEY_CHUNKS.append(len(item[1]))
    return [str(value) for value in item[1]]


def test_the_key_pass_dispatches_chunks_and_the_parent_computes_no_key(monkeypatch, mocked_parallel):
    chunks, parent = [], []
    monkeypatch.setattr(sys.modules[__name__], "_KEY_CHUNKS", chunks)
    monkeypatch.setattr(pool, "_keys_job", _record_key_chunk)
    monkeypatch.setattr(pool, "_keys", lambda key, items, identity: parent.append(len(items)))
    pool.map("units", _square, list(range(640)), key=_name)
    assert sum(chunks) == 640 and 1 < len(chunks) < 640 and parent == []


def test_the_keys_of_a_few_items_are_computed_here_and_many_go_to_the_workers(monkeypatch):
    asked = []
    monkeypatch.setattr(
        pool,
        "gather",
        lambda groups: asked.append([g[0] for g in groups])
        or [[[_name(item) for item in group[2][0][1]]] for group in groups],
    )
    workers = pool._need().workers
    few, many = list(range(workers * pool._KEYS_INLINE)), list(range(workers * pool._KEYS_INLINE + 1))
    run = pool._Run("few", _square, few, _name)
    pool._resolve([run])
    assert asked == [] and run.keys == [keys.digest((_name(n), keys.code_identity(_square))) for n in few]
    run = pool._Run("many", _square, many, _name)
    pool._resolve([run])
    assert asked == [["many.keys"]]  # the key pass of a larger group is a dispatch of its own


# -- shared objects: fork, ship, spill ---------------------------------------------------------------------------


def test_items_travel_without_the_shared_object_and_workers_read_the_shared_copy():
    tree, needed = Tree(), set()
    stream = io.BytesIO()
    pool._Pickler(stream, needed).dump((_square, [(tree, 1), (tree, 2)]))
    assert len(stream.getvalue()) < 500 and len(needed) == 1 and list(pool._SHARED.values()) == [tree]
    function, items = pool._Unpickler(io.BytesIO(stream.getvalue())).load()
    assert function is _square and items[0][0] is tree and items[1][0] is tree  # no copy is made


def test_an_object_the_warm_workers_were_not_forked_with_travels_in_one_spill_file_and_forks_nothing(
    fake_executor, tmp_path
):
    one, two, needed = Tree("one"), Tree("two"), set()
    pool._Pickler(io.BytesIO(), needed).dump(one)
    first = pool._get_executor(frozenset(needed))
    assert pool._get_executor(frozenset(needed)) is first  # the same object reuses the workers
    blobs = []
    for _ in range(3):  # three tasks of one dispatch: one file, written once
        stream, needed = io.BytesIO(), set()
        pool._Pickler(stream, needed, {}, ship=pool._warm_pool()).dump((two, 1))
        blobs.append(stream.getvalue())
    assert len(fake_executor.made) == 1 and first.seen == {"one"}
    assert len(blobs[0]) < 500 and set(pool._SHARED) == {"one"}  # a reference, never the object
    spilled = list((tmp_path / "spill" / str(os.getpid())).iterdir())
    assert len(spilled) == 1 and len(pool._spilled) == 1
    back = pool._Unpickler(io.BytesIO(blobs[0])).load()[0]  # as a worker reads it: once
    assert back.digest == "two" and pool._Unpickler(io.BytesIO(blobs[1])).load()[0] is back
    pool._forget_spills()
    assert not list(spilled[0].parent.iterdir())


def _digest_of(item):
    tree, number = item
    return tree.digest, number, pool._SHARED[tree.digest] is tree, os.getpid()


def test_real_forked_workers_see_the_parent_object_unpickled():
    tree = Tree()
    with command():
        found = pool.map("units", _digest_of, [(tree, 1), (tree, 2), (tree, 3)])
    assert [f[:2] for f in found] == [("digest-1", 1), ("digest-1", 2), ("digest-1", 3)] and all(f[2] for f in found)


def test_two_workers_forked_holding_an_object_then_a_replacement_shipped_by_spill(tmp_path):
    one, two = Tree("one"), Tree("two")
    with command():
        first = pool.map("units", _digest_of, [(one, n) for n in range(8)])
        forked = pool._executor
        second = pool.map("units", _digest_of, [(two, n) for n in range(8)])
        assert pool._executor is forked  # warm: nothing forked again
    assert {f[0] for f in first} == {"one"} and {f[0] for f in second} == {"two"}
    assert all(f[2] for f in first + second)
    assert {f[3] for f in first} | {f[3] for f in second} <= set(forked._processes)
    assert len(pool._spilled) == 1  # the replacement travelled as one file, however many tasks named it


def test_a_spilled_object_carries_the_parents_indexes_so_warm_workers_never_build_them(monkeypatch):
    pool.register_warm(["catalog"], ["words"])
    scans: dict[str, str] = {"text-1": "scan-1"}
    pool.carry("scans", scans)
    memo.memo(("catalog", "pins-1"), lambda: {"Vec3": 1})
    blobs = []
    for name in ("one", "two"):
        stream = io.BytesIO()
        pool._Pickler(stream, set(), {}, ship=True).dump(Tree(name))
        blobs.append(stream.getvalue())
        scans["text-2"] = "scan-2"  # the parent scans more before the next round's file
    monkeypatch.setattr(memo, "_memo", {})  # a worker that holds none of it
    pool._carried["scans"] = {}
    built = []
    pool._Unpickler(io.BytesIO(blobs[0])).load()
    assert memo.memo(("catalog", "pins-1"), lambda: built.append(1)) == {"Vec3": 1}
    pool._Unpickler(io.BytesIO(blobs[1])).load()
    assert built == [] and pool._carried["scans"] == {"text-1": "scan-1", "text-2": "scan-2"}
    pool._forget_spills()


def test_a_worker_drops_what_it_built_for_a_replaced_object_but_keeps_the_kept_kinds():
    pool.register_warm(["catalog"], ["words"])
    old = Tree("old")
    pool._SHARED["old"] = old
    pool._laid.append("old")
    memo.memo(("words", "w"), lambda: 1)
    memo.memo(("index", "old"), lambda: 2)  # built on the old object: its key names the tag
    stream = io.BytesIO()
    pool._Pickler(stream, set(), {}, ship=True).dump(Tree("new"))
    pool._Unpickler(io.BytesIO(stream.getvalue())).load()
    assert "old" not in pool._SHARED and pool._laid == ["new"]
    assert memo.peek(("words", "w")) == 1 and memo.peek(("index", "old")) is None
    pool._forget_spills()


def _versions_read(item):
    return memo.memo(("versions.read", "split"), lambda: "read now")


def test_a_warm_worker_forgets_what_the_parent_forgot_before_the_next_dispatch():
    with command():
        memo.memo(("versions.read", "split"), lambda: "before")  # in the memo the workers fork holding
        assert pool.map("units", _versions_read, [1, 2]) == ["before"] * 2
        memo.forget("versions.read")
        assert pool.map("units", _versions_read, [1, 2]) == ["read now"] * 2
        assert pool.map("units", _versions_read, [1, 2]) == ["read now"] * 2


def _holding(item):
    return _HELD[-1] if _HELD else None


def test_a_registered_context_travels_in_every_task_blob_and_is_adopted_in_the_worker():
    state = ["landing-lock"]
    pool.register_context("lock", lambda: list(state), lambda held: _HELD.append(held))
    try:
        with command():
            assert pool.map("units", _holding, [1, 2]) == [["landing-lock"]] * 2
            state[:] = ["other"]
            assert pool.map("units", _holding, [1, 2]) == [["other"]] * 2
    finally:
        _HELD.clear()


# -- slots, stalls ----------------------------------------------------------------------------------------------


def test_slots_are_released_between_gathers_and_the_executor_stays():
    with command():
        pool.map("units", _square, [1, 2, 3])
        assert slots.default.slots == [] and slots.default.users == 0
        assert pool._executor is not None
        pool.map("units", _square, [4, 5, 6])
    assert not list((Path(os.environ["XDG_RUNTIME_DIR"]) / slots.POOL).glob("cmd-*.lock"))


def test_slots_are_released_after_the_executor_is_dropped():
    with command():
        pool.map("units", _square, [1, 2, 3])
    pool._drop_executor()
    assert pool._executor is None and slots.default.slots == [] and slots.default.cmd == []


def test_unconfigured_pool_says_so():
    pool._runtime = None
    with pytest.raises(RuntimeError, match="configure"):
        pool.map("units", _square, [1])


def _stall_run(monkeypatch, cpu, awaits=lambda pid, executor: True):
    class Executor:
        _processes = dict.fromkeys((1, 2))

        def submit(self, function, *args):
            return Future()  # never completes

    drops, rounds = [], []
    monkeypatch.setattr(pool, "_get_executor", lambda needed=(), work=0: Executor())
    monkeypatch.setattr(pool, "_proc_stat", lambda pid: ("S", cpu(pid)))
    monkeypatch.setattr(pool, "_awaits_work", awaits)
    monkeypatch.setattr(pool, "Path", lambda path: SimpleNamespace(read_text=lambda: "", exists=lambda: True))
    monkeypatch.setattr(pool, "_drop_executor", lambda: drops.append(True))

    def wait(pending, return_when, timeout=None):
        rounds.append(timeout)
        if len(rounds) > 5:
            raise BrokenProcessPool("test ends")
        return set(), set(pending)

    monkeypatch.setattr(pool, "wait", wait)
    with command(), pytest.raises(pool.TaskFailed) as caught:
        pool.map("units", _square, [1, 2])
    return caught.value, drops, len(rounds)


def test_a_stalled_pool_raises_task_failed_stalled_and_gives_the_slots_back(monkeypatch):
    failure, drops, rounds = _stall_run(monkeypatch, lambda pid: 1.0)
    assert (failure.kind, failure.stage, failure.index, drops, rounds) == ("stalled", "units", 0, [True], 2)
    assert "every worker waited for work without using CPU" in failure.reason
    assert slots.default.slots == []


def test_a_worker_sleeping_on_a_lock_is_not_stalled(monkeypatch):
    failure, drops, rounds = _stall_run(monkeypatch, lambda pid: 1.0, lambda pid, executor: pid != 2)
    assert (failure.kind, drops, rounds) == ("crash", [True], 6)


def test_a_busy_worker_keeps_the_wait_going(monkeypatch):
    ticks = iter(range(100))
    failure, drops, rounds = _stall_run(monkeypatch, lambda pid: next(ticks) if pid == 2 else 1.0)
    assert (failure.kind, drops, rounds) == ("crash", [True], 6)


@pytest.mark.parametrize(
    ("syscall", "expected"),
    [
        ("0 0x5 0x1", True),
        ("202 0x7000 0x109", True),
        ("0 0x9 0x1", False),
        ("202 0x8000 0x0", False),
        ("73 0x5 0x6", False),
        ("running", False),
    ],
)
def test_awaits_work_reads_the_blocking_syscall(monkeypatch, syscall, expected):
    monkeypatch.setattr(pool, "_READ_FUTEX", (0, 202))
    monkeypatch.setattr(pool, "Path", lambda path: SimpleNamespace(read_text=lambda: syscall))
    calls = SimpleNamespace(
        _reader=SimpleNamespace(fileno=lambda: 5), _rlock=SimpleNamespace(_semlock=SimpleNamespace(handle=0x7000))
    )
    assert pool._awaits_work(7, SimpleNamespace(_call_queue=calls)) is expected


def test_proc_stat_reads_state_and_cpu_of_a_live_process():
    state, cpu = pool._proc_stat(os.getpid())
    assert state in "RSD" and cpu >= 0
    with pytest.raises(OSError):
        pool._proc_stat(2**22 + 1)


def test_a_gather_is_a_pool_map_stage_so_its_timing_reaches_the_feed_with_no_tool_call(monkeypatch, tmp_path):
    told = []
    monkeypatch.setattr(stages.feed, "say", lambda tool, stage, *a, **k: told.append(stage))
    monkeypatch.setattr(stages.feed, "path", tmp_path / "feed.jsonl")
    monkeypatch.setattr(stages.feed, "slow_seconds", 1)
    monkeypatch.setattr(stages.feed, "overrun_percent", 100)
    with command():
        pool.map("units", _square, [1, 2])
    assert told == []  # fast: nothing is written
    assert any(r.path[-1] == "pool.map" for r in stages.closed())
