# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The one forked executor of every tool: bounded workers, ordered results, keyed results stored in the cache.

A command forks its workers once, at the host's worker count, and keeps them warm across rounds and stages. Work
runs in chunks; as many are in flight as the host slots this command holds allow (`abucache.slots`), and a finished
chunk is replaced at once. Objects registered with `register_shared` never travel by value: workers fork holding
them, or read them once from a spill file. Keyed groups resolve their cache hits in the parent and dispatch only the
misses; the results are stored through `store.Namespace(cache, identity_package, stage)`, a fully warm group as one
`<stage>.group` entry. Every gather is a `pool.map` stage of `abucache.stages`, so its waits, slot seconds and
stalls reach the stage log, the wait log and the event feed with no per-tool code.

A tool registers its seam once at import (`register_shared`, `register_context`, `register_passthrough`,
`register_warm`, `carry`) and calls `configure` once per command."""

from __future__ import annotations

import atexit
import ctypes
import gc
import io
import os
import pickle
import resource
import shutil
import signal
import threading
import time
import traceback
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from abucache import files, keys, memo, slots, stages, store
from abucache import runtime as runtime_file

Json = Any

__plumbing__ = True  # keys.code_identity names this code and never follows it: it shapes no result

_KEYS_INLINE = 1  # keys of up to this many items per worker are computed here: a dispatch costs more than they do
_INLINE_AT = 1  # a gather of up to this many chunks runs here: a dispatch costs more than one chunk does
_REBALANCE = 0.5  # seconds between looks at the other commands while a pool works
_WINDOW = _REBALANCE * 20  # seconds without a finished chunk before the workers are looked at
_HARD_LIMIT = 4  # a worker is refused memory at this many times the worker memory; that budget sizes the pool
_INLINE_BYTES = 64 << 10
_SPILL_KEEP = 8  # spill files kept past their last gather: the next dispatch of a round names the same object


class TaskFailed(Exception):
    """A task or the pool failed. `kind` is "crash", "memory" or "stalled"; `stage` and `index` name the item."""

    def __init__(self, kind: str, stage: str, index: int, reason: str, notes: tuple[str, ...] = ()) -> None:
        self.kind, self.stage, self.index, self.reason, self.notes = kind, stage, index, reason, notes
        super().__init__(reason)


@dataclass(frozen=True)
class _Shared:
    cls: type
    key: Callable[[Any], str]
    tag: Callable[[Any], str]


# the tool's seam: registered once at import
_shared: list[_Shared] = []
_contexts: dict[str, tuple[Callable[[], Any], Callable[[Any], None]]] = {}
_passthrough: tuple[type[BaseException], ...] = ()
_warm: tuple[str, ...] = ()  # memo kinds the parent builds for a round: a spill file carries them
_keep: tuple[str, ...] = ()  # memo kinds keyed by content alone: valid whatever object a worker lays
_carried: dict[str, dict[Any, Any]] = {}  # name -> a module's content-keyed table a spill file carries the news of

# the command's: set by `configure`
_runtime: runtime_file.Runtime | None = None
_spill_base: Path | None = None
_package = ""

# this process's executor and what travels to it
_executor: ProcessPoolExecutor | None = None
_runtime_digest: str | None = None
_in_worker = False
_creating = threading.Lock()
_size = 0  # workers of the current executor: the host's whole fan, forked once per command
_flying = 0  # chunks every gather of this process has in flight: their slots are never given back
_flying_lock = threading.Lock()
_SHARED: dict[str, Any] = {}  # the shared objects by key: workers are forked holding them, items name them by key
_spilled: dict[str, str] = {}  # object key -> the file a live pool's workers read it from, oldest first
_spill_users: dict[str, int] = {}  # spill key -> gathers of this process whose tasks name it now
_spill_lock = threading.Lock()
_laid: list[str] = []  # in a worker, the shared keys it laid again itself (not forked holding)
_sent: dict[str, set[Any]] = {}  # name -> the keys of that table a spill file of this command carried already
_forked: set[str] = set()  # the keys of `_SHARED` the current workers were forked with


# -- the seam -----------------------------------------------------------------------------------------------------


def configure(runtime: runtime_file.Runtime, *, spill_dir: Path, identity_package: str) -> None:
    """Once per command, by the tool: the host facts that size the pool, the directory spill files go to (never a RAM
    tmpfs) and the package whose name keys stored results (`<identity_package>.<stage>`) and whose frames a worker's
    failure points at."""
    global _runtime, _spill_base, _package
    _runtime, _spill_base, _package = runtime, spill_dir, identity_package
    slots.default.configure(runtime)


def register_shared(cls: type, *, key: Callable[[Any], str], tag: Callable[[Any], str]) -> None:
    """Objects of `cls` never travel by value. `key(obj)` names one across processes; `tag(obj)` names the memo entries
    built on one, released when a worker drops it."""
    _shared.append(_Shared(cls, key, tag))
    memo.register_tagger(lambda value: tag(value) if isinstance(value, cls) else None)


def register_context(name: str, capture: Callable[[], Any], adopt: Callable[[Any], None]) -> None:
    """State every task blob carries from parent to worker: `capture()` runs in the parent per dispatch, `adopt(state)`
    first in the worker before it runs the items."""
    _contexts[name] = (capture, adopt)


def register_passthrough(*types: type[BaseException]) -> None:
    """Exceptions of these types a job raises reach the caller of `map` as they are; any other becomes `TaskFailed`."""
    global _passthrough
    _passthrough = (*_passthrough, *types)


def register_warm(kinds: Sequence[str], keep: Sequence[str]) -> None:
    """`kinds`: memo kinds a spill file carries from the parent to warm workers. `keep`: kinds keyed by content alone,
    valid for every shared object, which a worker keeps when it lays another."""
    global _warm, _keep
    _warm, _keep = tuple(kinds), tuple(keep)


def carry(name: str, table: dict[Any, Any]) -> None:
    """Registers a module's content-keyed table: each spill file carries the entries the parent made since the last
    one, so warm workers never redo what the parent did."""
    _carried[name] = table


def _spec(obj: Any) -> _Shared | None:
    return next((s for s in _shared if isinstance(obj, s.cls)), None)


def _need() -> runtime_file.Runtime:
    if _runtime is None:
        raise RuntimeError("abucache.pool.configure was not called for this command")
    return _runtime


def _cache() -> store.ContentCache:
    runtime = _need()
    return store.open_cache(runtime.cache_dir, runtime.cache_bytes, stages.count)


def _namespace(kind: str) -> store.Namespace:
    return store.Namespace(_cache(), _package, kind)


# -- workers ------------------------------------------------------------------------------------------------------


def _init(limit: int) -> None:
    """Runs in each forked worker: its memory limit counts what it inherited, so it covers only what it adds. The
    worker memory is the budget that sizes the pool and trims a job's memo; the refusal sits well above it, so a job
    that needs more than the plan finishes instead of dying (the host guard freezes a runaway)."""
    global _in_worker
    inherited = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmData:")
    )
    hard = inherited + limit * _HARD_LIMIT
    resource.setrlimit(resource.RLIMIT_DATA, (hard, hard))
    memo.mark_fork(inherited, limit)
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG: a worker never outlives the process that forked it
    slots.default.close_inherited()  # the parent's locks are the parent's: a worker holding one pins it for life
    files.drop_inherited()  # a worker forked inside a held lock never holds the parent's
    _in_worker = True


def _stop_workers() -> None:
    """Ends the current executor and its workers; the slot files stay as they are."""
    global _executor, _runtime_digest
    executor, _executor = _executor, None
    _runtime_digest = None
    _forked.clear()
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


_READ_FUTEX = {"x86_64": (0, 202), "aarch64": (63, 98)}.get(os.uname().machine)


def _awaits_work(pid: int, executor: Any) -> bool:
    """Whether the worker is blocked taking its next work item: reading the call queue pipe, or queued on that
    pipe's read lock. A worker sleeping anywhere else (a store lock, sqlite, a file lock) is busy, not stalled."""
    calls: Any = getattr(executor, "_call_queue", None)
    try:
        number, first = Path(f"/proc/{pid}/syscall").read_text().split()[:2]
        reader, lock = calls._reader.fileno(), calls._rlock._semlock.handle
    except (OSError, ValueError, AttributeError):
        return False
    read, futex = _READ_FUTEX or (None, None)
    return (int(number), int(first, 16)) in {(read, reader), (futex, lock)}


def _proc_stat(pid: int) -> tuple[str, float]:
    """(state, cpu seconds) of a process from /proc; raises OSError or ValueError when it is gone or unreadable."""
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rindex(")") + 2 :].split()
    return fields[0], (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def _census(executor: Any) -> dict[int, float] | None:
    """CPU seconds of each worker, or None when any of them is doing something else than waiting for its next work
    item with no children (running, on a lock, a child, a disk) or is gone."""
    seconds: dict[int, float] = {}
    try:
        for pid in list(getattr(executor, "_processes", None) or ()):
            state, cpu = _proc_stat(pid)
            if state != "S" or Path(f"/proc/{pid}/task/{pid}/children").read_text().strip():
                return None
            if not _awaits_work(pid, executor):
                return None
            seconds[pid] = cpu
    except (OSError, ValueError):
        return None
    return seconds


def _stalled(executor: Any, seen: dict[int, float]) -> bool:
    """Whether every worker slept without using CPU since the last call (the first call only records)."""
    now = _census(executor)
    before = dict(seen)
    seen.clear()
    if now is None:
        return False
    seen.update(now)
    return bool(now) and now == before


def _drop_executor() -> None:
    """Ends the workers and gives the slots back; their spill files and what they were sent go with them."""
    s = slots.default
    _stop_workers()
    _forget_spills()
    _sent.clear()
    for mine in s.cmd:
        Path(mine.name).unlink(missing_ok=True)
        mine.close()
    for slot in s.slots:  # freed only once the workers are gone
        slot.close()
    if s.slots and s.dir:
        slots._signal(s.dir[0])
    s.slots.clear()
    s.cmd.clear()
    s.fed.clear()


def _flight(step: int) -> int:
    global _flying
    with _flying_lock:
        _flying += step
        return _flying


def _fan() -> int:
    """The workers one command forks: exactly the host's worker count, whatever its share is now."""
    return max(1, _need().workers)


def _warm_pool() -> bool:
    """Whether this command's workers are up for this host: a dispatch then ships what they lack, never reforks."""
    return _executor is not None and _runtime_digest == _need().digest


def _get_executor(needed: frozenset[str] = frozenset(), work: int = 0) -> ProcessPoolExecutor:
    """One pool per command, forked once at the host's fan and kept warm across rounds, levels and stages: a new
    shared object travels to the workers that lack it (`_Pickler`), and a share that grows or shrinks changes how many
    chunks are in flight, never the workers. Only a host change (or a broken pool) forks again. Workers fork holding
    the shared objects in `needed` that are in `_SHARED` then."""
    global _executor, _runtime_digest, _size
    runtime, s = _need(), slots.default
    held = _executor
    if held is not None and _runtime_digest == runtime.digest:
        return held
    with _creating:  # two threads asking at once make one executor, not two
        if _executor is not None and _runtime_digest != runtime.digest:
            _drop_executor()
        if _executor is None:
            if not s.slots:
                s.acquire(cap=work)
            _size = _fan()  # idle workers cost no slot and no CPU: the fan is forked once
            for key in [k for k in _SHARED if k not in needed]:  # workers fork holding only what they run against
                del _SHARED[key]
            _executor = ProcessPoolExecutor(
                max_workers=_size,
                mp_context=get_context("fork"),
                initializer=_init,
                initargs=(runtime.worker_memory,),
            )
            _runtime_digest = runtime.digest
            _forked.update(_SHARED)
            gc.freeze()  # the workers' copy-on-write pages stay shared: a collection would touch every object
            try:
                _executor.submit(int).result()  # the first job forks every worker
            finally:
                gc.unfreeze()
        return _executor


def _job(function: Callable[[Any], Any], item: Any) -> tuple[bool, Any, Json]:
    start_ns = time.monotonic_ns()
    before = resource.getrusage(resource.RUSAGE_SELF)
    spawned = resource.getrusage(resource.RUSAGE_CHILDREN)
    with stages.capture() as records:
        try:
            with memo.scoped():
                value, ok = function(item), True
        except Exception as exc:
            marker = f"/{_package}/"
            here = [f for f in traceback.extract_tb(exc.__traceback__) if marker in f.filename]
            if here:  # the traceback does not cross the process boundary, a note does
                exc.add_note(f"at {Path(here[-1].filename).stem}:{here[-1].name}:{here[-1].lineno}")
            value, ok = exc, False
    after = resource.getrusage(resource.RUSAGE_SELF)
    reaped = resource.getrusage(resource.RUSAGE_CHILDREN)
    # every program the job ran and waited for, however deep (a nested pool runs inline here and keeps only its slow
    # records): the job's native work is all of it, not only the native records that reached this capture
    native = reaped.ru_utime + reaped.ru_stime - spawned.ru_utime - spawned.ru_stime
    return (
        ok,
        value,
        {
            "pid": os.getpid(),
            "worker_cpu": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            "native_cpu": max(native, records[-1].native_cpu_seconds),
            "rss": after.ru_maxrss * 1024,
            "cache": dict(records[-1].cache),
            "wait_seconds": records[-1].wait_seconds,
            "records": list(records),  # every record: the chunk folds them by path, the parent applies the log floor
            "start_ns": start_ns,
            "end_ns": time.monotonic_ns(),
        },
    )


# -- spill files --------------------------------------------------------------------------------------------------


def _spill_dir() -> Path:
    """This command's spill files, under the tool's own state. A dead command's are removed the first time a later
    one spills."""
    if _spill_base is None:
        raise RuntimeError("abucache.pool.configure was not called for this command")
    mine = _spill_base / str(os.getpid())
    if not mine.is_dir():
        mine.mkdir(parents=True, exist_ok=True)
        atexit.register(shutil.rmtree, mine, True)
        for other in _spill_base.iterdir():
            if other != mine and other.name.isdigit() and not Path(f"/proc/{other.name}").exists():
                shutil.rmtree(other, ignore_errors=True)
    return mine


def _spill(name: str, value: Callable[[], bytes]) -> str:
    """The file holding `value()` for the workers, written once per name per command: every task names it, each
    worker reads it once."""
    with _spill_lock:
        if (path := _spilled.pop(name, None)) is None or not Path(path).exists():
            target = _spill_dir() / f"{name}.pickle"
            part = target.with_suffix(".part")
            part.write_bytes(value())
            part.replace(target)
            path = str(target)
            stages.count("pool.spill", False)
        else:
            stages.count("pool.spill", True)
        _spilled[name] = path  # newest last
        return path


def _warmth() -> dict[str, Any]:
    """What a spill file hands warm workers beside the object: the parent's indexes of the warm kinds and the news of
    every carried table."""
    news = {}
    for name, table in _carried.items():
        sent = _sent.setdefault(name, set())
        news[name] = {k: v for k, v in list(table.items()) if k not in sent}
        sent.update(news[name])
    return {"memo": memo.held(_warm), "tables": news}


def _adopt(warmth: dict[str, Any]) -> None:
    memo.adopt(warmth["memo"], _warm)
    for name, news in warmth["tables"].items():
        if name in _carried:
            for key, value in news.items():
                _carried[name].setdefault(key, value)


def _hold_spills(names: Collection[str], step: int) -> None:
    """A gather names its spill files while its tasks may read them; at its end, files no live gather names go,
    except the newest few (the next dispatch of a round names the same object)."""
    with _spill_lock:
        for name in names:
            _spill_users[name] = _spill_users.get(name, 0) + step
            if _spill_users[name] <= 0:
                del _spill_users[name]
        if step > 0:
            return
        idle = [n for n in _spilled if n not in _spill_users]
        for name in idle[: max(0, len(idle) - _SPILL_KEEP)]:
            Path(_spilled.pop(name)).unlink(missing_ok=True)


def _forget_spills() -> None:
    """Every spill file of this command goes: its workers are gone."""
    with _spill_lock:
        for path in _spilled.values():
            Path(path).unlink(missing_ok=True)
        _spilled.clear()
        _spill_users.clear()


# -- shared objects on the wire -------------------------------------------------------------------------------------


class _Pickler(pickle.Pickler):
    """Items travel without the shared objects they run against. Before the pool forks (`ship` false), an object goes
    into `_SHARED` and the workers fork holding it. Once the workers are up (`ship`), nothing forks again: an object
    they were not forked with (the next round's capture) travels as one spill file of the whole of it, written once
    and read once per worker, so rounds, levels and stages all run on the same warm workers."""

    def __init__(self, stream: io.BytesIO, needed: set[str], blobs: dict[str, Any] | None = None, ship: bool = False):
        super().__init__(stream, protocol=5)
        self.needed = needed
        self.blobs = {} if blobs is None else blobs  # per dispatch: each spill's file, once
        self.ship = ship
        self.spills: set[str] = set()  # the spill files this pickler's tasks name

    def persistent_id(self, obj: Any) -> Any:
        spec = _spec(obj)
        return None if spec is None else self._ref(obj, spec.key(obj))

    def _ref(self, obj: Any, key: str) -> Any:
        if key in _forked:
            self.needed.add(key)  # a refork after a broken pool forks holding it again
            return key
        if self.ship:  # an object of its own (a capture): the whole of it, read once per worker
            return ("snap", key, self._spill(f"s{key}", obj))
        _SHARED[key] = obj
        self.needed.add(key)
        return key

    def _spill(self, name: str, value: Any) -> str:
        """The file of one object, with the parent's warm indexes beside it."""
        self.spills.add(name)
        if (path := self.blobs.get(f"file:{name}")) is None:
            path = self.blobs[f"file:{name}"] = _spill(name, lambda: pickle.dumps((value, _warmth()), 5))
        return str(path)


def _lay(ref: Any) -> Any:
    """The object a reference names, laid in this worker once and held in `_SHARED` until another one comes."""
    if isinstance(ref, str):
        return _SHARED[ref]
    if ref[0] == "snap":
        _, key, path = ref
        if key not in _SHARED:
            _SHARED[key], warmth = pickle.loads(Path(path).read_bytes())
            _adopt(warmth)
            _laid.append(key)
        return _SHARED[key]
    raise ValueError(ref)


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> Any:
        if isinstance(pid, tuple):  # an object the worker was not forked holding: laid again once per worker
            if pid[1] not in _SHARED:
                # a new one (the next round's tree): what this worker laid before and built for it (its indexes,
                # pins and views) is never read again
                for old in [k for k in _laid if k != pid[1]]:
                    gone = _SHARED.pop(old, None)
                    spec = None if gone is None else _spec(gone)
                    if spec is not None:
                        memo.release(spec.tag(gone))
                _laid[:] = [k for k in _laid if k == pid[1]]
                memo.trim_to_fork(keep_kinds=(*_keep, *_warm))
            return _lay(pid)
        return _SHARED[pid]


def _chunk(
    function: Callable[..., Any], items: Sequence[Any], sink: tuple[Any, ...] | None
) -> list[tuple[bool, Any, Json]]:
    outcomes = [_job(function, item) for item in items]
    if sink and outcomes:
        clean = len(sink) > 2 and sink[2]  # a stable key stores only a result with nothing to report
        before = resource.getrusage(resource.RUSAGE_SELF)
        with stages.capture() as written, stages.stage("pool.sink"):
            _namespace(sink[0]).put_many(
                [
                    (key, pickle.dumps(value, protocol=5))
                    for (ok, value, _), key in zip(outcomes, sink[1], strict=True)
                    if ok and key is not None and not (clean and value)
                ]
            )
        after = resource.getrusage(resource.RUSAGE_SELF)
        ok, value, last = outcomes[-1]  # the write belongs to no job: the chunk's last envelope carries its cpu
        outcomes[-1] = (
            ok,
            value,
            {
                **last,
                "worker_cpu": last["worker_cpu"] + after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                "records": [*last["records"], *(r for r in written if r.kind == "stage")],
            },
        )
    if outcomes:  # the keys this chunk read or wrote travel back, so the parent writes one manifest per command
        outcomes[-1] = (*outcomes[-1][:2], {**outcomes[-1][2], "touched": store.take_touched()})
    records = [r for _, _, envelope in outcomes for r in envelope["records"]]
    parents = {r.parent for r in records}
    folded = [asdict(r) for r, _ in stages.fold([(r, r.span not in parents) for r in records])]
    for n, (ok, value, envelope) in enumerate(outcomes):  # the chunk's records travel once, in its last envelope
        outcomes[n] = (ok, value, {**envelope, "records": folded if n == len(outcomes) - 1 else []})
    return outcomes


def _header() -> bytes:
    """What every task blob starts with: the parent's forgets and the state of every registered context."""
    return pickle.dumps((memo.epochs(), {name: capture() for name, (capture, _) in _contexts.items()}), protocol=5)


def _follow(stream: io.BytesIO) -> None:
    """First in a worker: forget what the parent forgot (the items lay objects from the memo) and adopt each context
    the parent holds now, never the one it held when this worker was forked."""
    forgotten, held = pickle.load(stream)
    memo.follow(forgotten)
    for name, state in held.items():
        if name in _contexts:
            _contexts[name][1](state)


def _dispatch(blob: bytes) -> list[tuple[bool, Any, Json]]:
    """One chunk travels as its own pickle, so a worker only unpacks the items it runs and stores their results."""
    stream = io.BytesIO(blob)
    _follow(stream)
    function, items, sink = _Unpickler(stream).load()
    return _chunk(function, items, sink)


def _raise_failure(name: str, index: int, exc: Any) -> None:
    if isinstance(exc, _passthrough):
        raise exc
    memory = isinstance(exc, MemoryError)
    notes = tuple(getattr(exc, "__notes__", ()))
    raise TaskFailed(
        "memory" if memory else "crash",
        name,
        index,
        f"Stage {name} item {index} " + (f"exceeded worker memory: {exc!r}" if memory else f"failed: {exc!r}"),
        notes,
    ) from exc


def _stall(pending: dict[Future[Any], tuple[Any, list[int]]]) -> None:
    """Pending work that no worker is running will never come back: fail loudly and give the slots back."""
    run, picked = next(iter(pending.values()))
    for worker in list(
        (getattr(_executor, "_processes", None) or {}).values()
    ):  # idle by proof: a shutdown would wait on them
        worker.kill()
    _drop_executor()
    raise TaskFailed(
        "stalled",
        run.name,
        picked[0],
        f"Stage {run.name} item {picked[0]}: {len(pending)} chunks are pending and every worker waited "
        f"for work without using CPU for {_WINDOW:.0f} seconds",
    )


# -- keyed groups -------------------------------------------------------------------------------------------------


def _identity(run_function: Callable[..., Any], key: Callable[[Any], str | None]) -> str:
    """What a stored result also depends on: the job's code, unless its key names every input (`stable`) or names the
    code the result depends on itself (`coded`: a key narrower than everything the job reaches)."""
    return "" if getattr(key, "stable", False) or getattr(key, "coded", False) else keys.code_identity(run_function)


def _keys(key: Callable[[Any], str | None], items: Sequence[Any], identity: str) -> list[str | None]:
    return [None if (named := key(item)) is None else keys.digest((named, identity)) for item in items]


def _keys_job(item: tuple[Any, ...]) -> list[str | None]:
    return _keys(*item)


def _loads(blob: bytes | None) -> tuple[bool, Any]:
    """(True, the value) of a pickled cache entry; (False, None) when it is absent or another build's shape."""
    if not blob:
        return False, None
    try:
        return True, pickle.loads(blob)
    except Exception:
        return False, None


class _Run:
    """One group of a gather: its items, the cache keys of those, and the indexes the workers must still compute."""

    def __init__(
        self,
        name: str,
        function: Callable[..., Any],
        items: Sequence[Any],
        key: Callable[[Any], str | None] | None = None,
    ):
        self.name, self.function, self.items, self.key = name, function, items, key
        self.sealed = False
        self.outcomes: list[tuple[bool, Any, Json] | None] = [None] * len(items)
        self.keys: list[str | None] = []
        self.whole: str | None = None
        self.todo = list(range(len(items)))
        self.submitted, self.done = 0, 0
        self.first, self.last = 0, 0  # when its first chunk went out and its last came back: its own wall

    def resolve(self, keys_: list[str | None]) -> None:
        """With its keys known, the parent resolves the hits itself: a warm item never travels to a worker."""
        self.keys = keys_
        self.whole = keys.digest(keys_) if None not in keys_ else None
        results, group = _namespace(self.name), _namespace(self.name + ".group")
        blob = group.get(self.whole) if self.whole else None  # a fully warm group is one entry
        self.sealed = bool(blob)  # a group warm entry by entry is sealed too, so the next run reads one entry
        whole, rows = _loads(blob)
        if blob and not whole:
            self.sealed = False
        warm = rows if whole else [k and results.get(k) for k in keys_]
        for index, found in enumerate(warm):
            ok, value = _loads(found) if found else (False, None)
            stages.count(self.name, ok)
            if ok:
                self.outcomes[index] = (True, value, {})
        self.todo = [i for i, outcome in enumerate(self.outcomes) if outcome is None]
        self.done = len(self.items) - len(self.todo)

    def results(self) -> list[tuple[bool, Any, Json]]:
        return [o for o in self.outcomes if o is not None]

    def seal(self) -> None:
        stable = bool(getattr(self.key, "stable", False))  # a failure is never sealed under a code-free key
        if (
            self.whole
            and (self.todo or not self.sealed)
            and all(o and o[0] and not (stable and o[1]) for o in self.outcomes)
        ):
            blobs = [pickle.dumps(o[1], protocol=5) for o in self.results()]
            _namespace(self.name + ".group").put(self.whole, pickle.dumps(blobs))

    def sink(self, picked: Sequence[int]) -> tuple[Any, ...] | None:
        if not self.key:
            return None
        return (self.name, [self.keys[i] for i in picked], bool(getattr(self.key, "stable", False)))

    def settle(self, picked: Sequence[int], outcomes: Sequence[tuple[bool, Any, Json]]) -> None:
        for index, outcome in zip(picked, outcomes, strict=True):
            meta = dict(outcome[2])
            store.absorb_touched(meta.pop("touched", {}))
            self.outcomes[index] = (outcome[0], outcome[1], meta)
        self.done, self.last = self.done + len(picked), time.monotonic_ns()
        stages.progress(self.name, self.done, len(self.items))


def _resolve(runs: Sequence[_Run]) -> None:
    """The keys of every keyed group, computed by the workers in chunks (the parent when few), then the hits."""
    workers = _need().workers
    keyed = [(r, r.key) for r in runs if r.key]
    if sum(len(r.items) for r, _ in keyed) <= _KEYS_INLINE * workers or _in_worker:
        for run, key in keyed:
            run.resolve(_keys(key, run.items, _identity(run.function, key)))
        return
    groups = [
        (
            run.name + ".keys",
            _keys_job,
            [(key, run.items[lo : lo + size], _identity(run.function, key)) for lo in range(0, len(run.items), size)],
        )
        for run, key in keyed
        for size in [max(1, len(run.items) // (workers * 32))]
    ]
    for (run, _), chunks in zip(keyed, gather(groups), strict=True):
        run.resolve([key for chunk in chunks for key in chunk])


def map(  # noqa: A001
    stage: str,
    function: Callable[[Any], Any],
    items: Sequence[Any],
    key: Callable[[Any], str | None] | None = None,
) -> list[Any]:
    return gather([(stage, function, items, key)])[0]


def gather(groups: Sequence[tuple[Any, ...]]) -> list[list[Any]]:
    """Runs independent (stage, function, items[, key]) groups in one dispatch loop, so a small group never idles the
    workers. A keyed group resolves its cache hits in the parent and dispatches only its misses. Every chunk is
    ready at once: as many are in flight as the slots held allow, and a finished one is replaced at once."""
    s = slots.default
    with stages.stage("pool.map"):
        start_ns, waited, current, rebalanced = time.monotonic_ns(), 0.0, (0, 0), time.monotonic()
        held, mark = 0.0, time.monotonic()  # slot-seconds this gather held: the share it was given, as it changed
        runs = [_Run(*group) for group in groups]
        total, admitted, joined, width = 0, 0, False, 1  # width before anything can fail: the finally reads it
        pending: dict[Future[list[tuple[bool, Any, Json]]], tuple[_Run, list[int]]] = {}
        quiet: dict[int, float] = {}  # the workers' CPU at the last window that finished no chunk
        spills: set[str] = set()  # the spill files this gather's tasks name: kept while it runs
        try:
            _resolve(runs)
            total = sum(len(r.todo) for r in runs)
            inline = total <= _INLINE_AT or _in_worker  # nothing to fan out: the jobs run here, one at a time
            fan = _fan()
            size = {r: 1 if inline else max(1, len(r.todo) // (fan * 32)) for r in runs}
            needed: set[str] = set()
            tasks = [(r, r.todo[lo : lo + size[r]]) for r in runs for lo in range(0, len(r.todo), size[r])]
            blobs = [b""] * len(tasks)
            if not inline:  # the shared objects in the items are known before the workers are forked
                writes: dict[str, Any] = {}  # each spill's file is written once for every task of this dispatch
                ship = _warm_pool()  # workers up: what they lack travels to them, they never fork again
                forgotten = _header()  # read by each worker before its items
                for n, (run, picked) in enumerate(tasks):
                    stream = io.BytesIO(forgotten)
                    stream.seek(0, io.SEEK_END)
                    pickler = _Pickler(stream, needed, writes, ship)
                    pickler.dump((run.function, [run.items[i] for i in picked], run.sink(picked)))
                    blobs[n] = stream.getvalue()
                    spills |= pickler.spills
                _hold_spills(spills, 1)
                with s.lock:
                    if not s.slots:  # released after the last gather: take the share again, or wait for any free slot
                        s.acquire(cap=len(tasks))
                    s.users += 1
                    joined = True
                # the executor is taken once the slots are held: waiting for a slot stops idle workers, and a held
                # executor taken before that wait would be the one stopped
                executor = _get_executor(frozenset(needed), len(tasks))
                width = min(len(s.slots), _size or len(s.slots))
            admitted, mark = 0 if inline else width, time.monotonic()
            queue = iter(zip(tasks, blobs, strict=True))
            left = len(tasks)  # chunks not submitted yet
            while True:
                while len(pending) < width and (task := next(queue, None)):
                    (run, picked), blob = task
                    left -= 1
                    if inline:
                        future: Future[list[tuple[bool, Any, Json]]] = Future()
                        pending[future] = (run, picked)
                        future.set_result(_chunk(run.function, [run.items[i] for i in picked], run.sink(picked)))
                    else:
                        pending[executor.submit(_dispatch, blob)] = (run, picked)
                    _flight(1)
                    run.submitted, run.first = run.submitted + len(picked), run.first or time.monotonic_ns()
                if not pending:
                    break
                blocked = time.perf_counter()
                completed, _ = wait(pending, timeout=None if inline else _WINDOW, return_when=FIRST_COMPLETED)
                waited += time.perf_counter() - blocked
                if not completed and _stalled(executor, quiet):
                    _stall(pending)
                if completed:
                    quiet.clear()
                for future in completed:
                    run, picked = pending.pop(future)
                    _flight(-1)
                    current = (runs.index(run), picked[0])
                    try:
                        outcomes = future.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:
                        outcomes = [(False, exc, {})] * len(picked)
                    run.settle(picked, outcomes)
                if not inline:
                    # peers are probed now and then, not per chunk; but a chunk that ended while another command
                    # waits for a slot is a job boundary: the freed slot goes to it, never into the next chunk
                    if time.monotonic() - rebalanced >= _REBALANCE or (completed and s.waiting()):
                        rebalanced = time.monotonic()
                        with s.lock:  # never hold a slot no worker of this pool can use: more than the
                            # executor forked, or more than the chunks still to run, go back to the other commands;
                            # a share that grew is used at once by the same workers (they were forked at the fan)
                            more = len(pending) + left
                            s.acquire(_flying, max(1, min(more, _size or len(tasks))))
                            held, mark = held + width * (time.monotonic() - mark), time.monotonic()
                            width = min(len(s.slots), _size or len(s.slots))
                    admitted = max(admitted, width)
        except BrokenProcessPool as exc:
            _drop_executor()
            _raise_failure(runs[current[0]].name, current[1], exc)
        finally:
            _flight(-len(pending))
            _hold_spills(spills, -1)
            held += width * (time.monotonic() - mark)
            if joined:
                with s.lock:
                    s.users -= 1
                    if s.users == 0:
                        s.release()
            for run in runs:
                stages.record_pool(
                    run.name,
                    len(run.items),
                    jobs=run.submitted,
                    admitted=admitted,
                    start_ns=run.first or start_ns,
                    end_ns=run.last or start_ns,
                    envelopes=[o[2] for o in run.outcomes if o and o[2]],
                    dispatch_seconds=(time.monotonic_ns() - start_ns) / 1e9 - waited,
                    slot_seconds=held if len(runs) == 1 and admitted else 0.0,  # inline jobs hold no slot
                )
        for run in runs:
            run.seal()
            for index, outcome in enumerate(run.outcomes):
                if outcome is not None and not outcome[0]:
                    _raise_failure(run.name, index, outcome[1])
        return [[outcome[1] for outcome in run.results()] for run in runs]


__all__ = [
    "TaskFailed",
    "carry",
    "configure",
    "gather",
    "map",
    "register_context",
    "register_passthrough",
    "register_shared",
    "register_warm",
]
