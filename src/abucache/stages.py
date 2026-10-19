# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Invocation spans, exclusive parent work, worker and native accounting, and the stage log (schema "stage/1").

Every tool times its work through this module and nothing else: a command opens the outermost span with the path of
its stage log, stages nest under it, native children and pool dispatches are recorded into the span that ran them,
and a wait on a lock or a slot lands in the innermost span and, once long, in the host's wait log. The stage log is
JSONL, one `StageRecord` per row, and its sibling `walls.json` remembers the previous command's walls so a start line
can say how long a stage is expected to take."""

from __future__ import annotations

import json
import os
import resource
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from abucache import feed
from abucache import runtime as runtime_file

Json = Any
SCHEMA = "stage/1"

_stack: ContextVar[tuple[_Open, ...]] = ContextVar("stages_stack", default=())
_closed: list[StageRecord] = []
_counters: dict[str, list[int]] = {}
_sink: Path | None = None
_tool = ""
_invocation = ""
_dispatched: dict[str, tuple[str, ...]] = {}  # gather stage span -> the dispatches it ran
_seen: set[tuple[str, ...]] = set()  # slow stage paths already closed in this command
_listeners: list[Callable[[str, Json], None]] = []
_waits: dict[str, float] = {}  # this command's seconds waiting, by what it waited for: "slots" or "lock"
_walls: dict[str, float] = {}  # the previous command's wall seconds by stage path, for "expected" on start lines
_command_hooks: list[Callable[[], None]] = []  # run when a command starts (the memo clears itself)
_refusals: tuple[type[BaseException], ...] = ()  # exception types whose `.findings` are a refusal's findings

LOG_BYTES = 64 << 20  # the stage log keeps its newest half once it grows past this (`configure` sets it)
WAIT_NOTE = 60.0  # a wait this long (seconds) is also written, one line, to the host's wait log
WAIT_LOG_BYTES = 1 << 20  # the wait log keeps its newest half once it grows past this
WALL_DEPTH = 3  # stages this shallow report live progress and are remembered in walls.json
REASON_CHARS = 2000  # a finding's reason in the stage log; a compile error quoting whole headers is 100 KB per unit


def _plain(row: Any) -> dict[str, Any]:
    return dict(row)


finding_to_dict: Callable[[Any], dict[str, Any]] = _plain
"""Set by a tool whose spans carry its own finding type: turns one into the plain dict the log holds."""

parent_memory = 0  # the command's own memory budget in bytes, 0 = unknown (the memo's pressure reads it)
_memory_limits: dict[str, str] = {}
_memory_limit_values: dict[str, int] = {}
_budgets: Budgets | None = None
_budget_origins: dict[str, Any] = {}
_package = ""
_project = ""
_wait_log: Path | None = None
_runtime: Any = None  # the runtime `configure` was given; none means each command reads the runtime file itself
_live: dict[str, _Open] = {}  # the open command, stage and pool spans: what the feed's watchdog looks at
_root: _Open | None = None  # the command's own span: where a feed that cannot be written reports itself


@dataclass(frozen=True)
class Budgets:
    """The watchdog's rules. A leaf stage of `serial_seconds` or more must use `serial_cores`; a pool of
    `pool_fanout` jobs per slot must keep `pool_fill` of its slots busy."""

    serial_seconds: float
    serial_cores: float
    pool_fill: float
    pool_fanout: float


def configure(
    runtime: Any = None,
    *,
    budgets: Budgets | None = None,
    budget_origins: Mapping[str, Any] | None = None,
    memory_limits: Mapping[str, str] | None = None,
    memory_limit_values: Mapping[str, int] | None = None,
    parent: int | None = None,
    package: str = "",
    project: str = "",
    wait_log: Path | None = None,
) -> None:
    """Once per command, by the tool. `runtime` (an `abucache.runtime.Runtime`) gives the stage log's size, the
    parent memory and the memory limit values; each keyword overrides or adds to it. `budgets` turns the watchdog
    on (its findings carry `budget_origins["budgets.<rule>"]`); `package` names the package whose frames an internal
    error points at; `project` and `wait_log` say who the wait log's lines are for and where they go.

    Calling this is optional. Every command binds itself to the runtime file (`ABU_RUNTIME`, else the config home's):
    its `[events]` section sends slow stages and waits to the event feed (`abucache.feed`), and waits of a minute or
    more go to the slot directory's `waits.jsonl` unless `wait_log` names another file."""
    global LOG_BYTES, parent_memory, _memory_limits, _memory_limit_values, _budgets, _budget_origins
    global _package, _project, _wait_log, _runtime
    _runtime = runtime
    if runtime is not None:
        LOG_BYTES = runtime.stage_log_bytes
        parent_memory = runtime.parent_memory
        _memory_limit_values = {
            "resources.parent_memory": runtime.parent_memory,
            "resources.worker_memory": runtime.worker_memory,
        }
    if parent is not None:
        parent_memory = parent
    if memory_limit_values is not None:
        _memory_limit_values = dict(memory_limit_values)
    _memory_limits = dict(memory_limits or {})
    _budgets, _budget_origins = budgets, dict(budget_origins or {})
    _package, _project, _wait_log = package, project, wait_log


def on_command(hook: Callable[[], None]) -> None:
    """`hook()` runs whenever a command starts."""
    _command_hooks.append(hook)


def register_refusal(*types: type[BaseException]) -> None:
    """Exceptions of these types end a span as "refused" and hand their `.findings` to it."""
    global _refusals
    _refusals = (*_refusals, *types)


def finding(key: str, reason: str, **fields: Any) -> dict[str, Any]:
    """One finding row with every field a stage log row has."""
    row: dict[str, Any] = {
        "key": key,
        "reason": reason,
        "path": "",
        "line": 0,
        "unit": "",
        "versions": (),
        "missing": (),
        "origin": None,
        "blocking": True,
        "action": "",
        "symptoms": {},
    }
    row.update(fields)
    return row


@dataclass(frozen=True)
class StageRecord:
    schema: str = field(default=SCHEMA, init=False)
    invocation: str
    span: str
    parent: str | None
    path: tuple[str, ...]
    kind: str  # "command" | "stage" | "pool" | "job" | "native" | "own"
    status: str  # "ok" | "refused" | "failed" | "interrupted"
    execution: str  # "serial" | "parallel"
    start_ns: int
    wall_seconds: float
    parent_cpu_seconds: float
    worker_cpu_seconds: float
    native_cpu_seconds: float
    cores: float
    items: int
    jobs: int
    workers_used: int
    workers_admitted: int
    parent_rss_peak_bytes: int
    worker_rss_peak_bytes: int
    memory_limits: Mapping[str, str] = field(default_factory=dict)
    memory_limit_values: Mapping[str, int] = field(default_factory=dict)
    cache: Mapping[str, tuple[int, int]] = field(default_factory=dict)  # cache kind -> (hits, misses)
    findings: tuple[dict[str, Any], ...] = ()
    wait_seconds: float = 0.0  # time this stage itself blocked on a lock
    dispatch_seconds: float = 0.0  # pool: wall the parent spent outside waiting on workers
    busy_seconds: float = 0.0  # pool: wall the workers spent inside jobs, summed
    spans: int = 1  # how many worker spans of one pool this record folds, by path
    slot_seconds: float = 0.0  # pool: host slots held times the seconds held (the share peers left it), summed
    slot_wait_seconds: float = 0.0  # command: seconds spent waiting for host worker slots held by other commands
    lock_wait_seconds: float = 0.0  # command: seconds spent waiting for locks held by other commands


def expected(path: Sequence[str]) -> float | None:
    """How long the previous command spent in this stage path, None when it never ran or was never recorded."""
    return _walls.get(" > ".join(path))


def _load_walls(path: Path) -> dict[str, float]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _save_walls() -> None:
    """Keep this command's shallow stage walls beside the stage log: a few lines, not a scan of the log."""
    if _sink is None:
        return
    walls = {" > ".join(r.path): r.wall_seconds for r in _closed if len(r.path) <= WALL_DEPTH and r.kind != "own"}
    path = _sink.with_name("walls.json")
    try:
        part = path.with_name(f".{uuid4().hex}.part")
        part.write_text(json.dumps({**_load_walls(path), **walls}, sort_keys=True), encoding="utf-8")
        part.replace(path)
    except OSError:
        _log_failure()


def listen(callback: Callable[[str, Json], None]) -> None:
    _listeners.append(callback)


def _notify(event: str, body: Json) -> None:
    for callback in tuple(_listeners):
        try:
            callback(event, body)
        except Exception as exc:
            if _stack.get():
                _stack.get()[-1].findings.append(finding("effort.log", f"Listener: {exc}", blocking=False))


def say(text: str) -> None:
    """One line for the person watching the command now (stderr), outside any stage's totals."""
    _notify("say", {"text": text})


def progress(name: str, done: int, total: int) -> None:
    _notify("progress", {"name": name, "done": done, "total": total})


def closed() -> tuple[StageRecord, ...]:
    return tuple(_closed)


def _cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _rss() -> int:
    """This process's resident set in bytes, 0 when /proc does not say."""
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


class _Open:
    def __init__(self, name: str, kind: str, parent: _Open | None = None):
        self.span, self.parent = uuid4().hex, parent.span if parent else None
        self.path: tuple[str, ...] = (parent.path if parent else ()) + (name,)
        self.kind = kind
        self.start, self.cpu = time.perf_counter_ns(), _cpu()
        self.rss = _rss() if kind in ("command", "pool") else 0
        self.cache, self.items, self.wait = counters(), 0, 0.0
        self.findings: list[dict[str, Any]] = []
        self.children: list[StageRecord] = []
        self.remote_cache: dict[str, tuple[int, int]] = {}

    def add(self, *, items: int = 0, findings: Sequence[Any] = ()) -> None:
        self.items += items
        self.findings.extend(finding_to_dict(f) for f in findings)


def waited(seconds: float, what: str = "lock", detail: str = "") -> None:
    """Counts `seconds` this command blocked on `what` ("lock" or "slots"): into the innermost stage, into the
    command's total, and, past `WAIT_NOTE`, into the host's wait log with `detail` (who held it)."""
    if _stack.get():
        _stack.get()[-1].wait += seconds
    _waits[what] = _waits.get(what, 0.0) + seconds
    if seconds >= WAIT_NOTE and (log := _wait_log or _slot_wait_log()) is not None:
        _note_wait(log, what, seconds, detail)
    if what != "slots" and feed.slow(seconds, None):  # a slot wait is a stage of its own, announced as one
        here = _stack.get()[-1].path if _stack.get() else ()
        feed.say(
            _tool, (*here, f"wait.{what}"), seconds, None, running=False, detail={"detail": detail}, failed=_feed_failed
        )


def _slot_wait_log() -> Path | None:
    """The host's wait log in the slot directory, when there is a runtime directory to hold it."""
    base = os.environ.get("XDG_RUNTIME_DIR", "")
    if not base or not Path(base).is_dir():
        return None
    from abucache import slots  # slots imports this module

    return slots.wait_log()


def _feed_failed(reason: str) -> None:
    """The feed could not be written: the command's own span says so once, so the stream going blind is itself seen."""
    if _root is not None and not any(f["key"] == "events.feed" for f in _root.findings):
        _root.findings.append(finding("events.feed", f"The event feed cannot be written: {reason}", blocking=False))


def _bind() -> str:
    """Binds the feed and the log size to the runtime for this command: the one `configure` was given, else the runtime
    file. A missing file leaves the feed off; a file that cannot be read says so (the reason, else empty)."""
    global LOG_BYTES
    runtime = _runtime
    if runtime is None:
        file = runtime_file.default_path()
        if not file.is_file():
            feed.bind(None)
            return ""
        try:
            runtime = runtime_file.load(file)
        except runtime_file.RuntimeConfigError as error:
            feed.bind(None)
            return str(error)
        LOG_BYTES = runtime.stage_log_bytes
    feed.bind(runtime)
    return ""


def _open_spans() -> list[feed.Open]:
    now = time.perf_counter_ns()
    return [(o.span, o.path, (now - o.start) / 1e9, expected(o.path), o.kind) for o in _live.copy().values()]


def _note_wait(path: Path, what: str, seconds: float, detail: str) -> None:
    """Appends one line for a wait of this command to the host's wait log, as a single O_APPEND write so concurrent
    commands never interleave: who waited, for what, how long, and who held it. The log is the host's, not a
    project's: one file serves every tree on the host."""
    line = {
        "at": round(time.time(), 1),
        "pid": os.getpid(),
        "project": _project,
        "what": what,
        "seconds": round(seconds, 1),
        "detail": detail,
    }
    try:
        path.parent.mkdir(exist_ok=True)  # the slot directory may not exist yet when only a lock was waited for
        if path.exists() and path.stat().st_size > WAIT_LOG_BYTES:
            kept = path.read_bytes()[-WAIT_LOG_BYTES // 2 :].partition(b"\n")[2]
            part = path.with_name(f".{path.name}.{os.getpid()}")
            part.write_bytes(kept)
            part.replace(path)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            os.write(fd, (json.dumps(line) + "\n").encode())
        finally:
            os.close(fd)
    except OSError:  # a missing or read-only directory loses the line, never the command
        pass


def _limits() -> dict[str, Any]:
    if not (_memory_limits or _memory_limit_values):
        return {}
    return {"memory_limits": dict(_memory_limits), "memory_limit_values": dict(_memory_limit_values)}


def _record(
    opened: _Open, wall: float, parent: float = 0, worker: float = 0, native: float = 0, **values: Any
) -> StageRecord:
    cores = (parent + worker + native) / wall if wall else 0
    execution = "parallel" if _budgets and cores >= _budgets.serial_cores else "serial"
    values.setdefault("wait_seconds", opened.wait)
    return StageRecord(
        _invocation,
        opened.span,
        opened.parent,
        opened.path,
        opened.kind,
        "ok",
        execution,
        opened.start,
        wall,
        parent,
        worker,
        native,
        cores,
        values.pop("items", opened.items),
        values.pop("jobs", 0),
        values.pop("workers_used", 0),
        values.pop("workers_admitted", 0),
        values.pop("parent_rss_peak_bytes", 0),
        values.pop("worker_rss_peak_bytes", 0),
        **(_limits() if opened.parent is None else {}),
        **values,
    )


def _budget(record: StageRecord, leaf: bool, shipped: bool = False) -> StageRecord:
    """The watchdog: a stage that breaks a budget rule carries a non-blocking budget.* finding naming it."""
    if not _budgets or record.kind not in ("stage", "own", "pool"):
        return record
    rules, found, cores, pool = _budgets, [], f"{record.cores:.2f} cores", record.kind == "pool"
    slow = record.wall_seconds >= rules.serial_seconds
    alone = record.wall_seconds / record.spans >= rules.serial_seconds  # a folded record is many short spans
    hits, misses = (sum(v[i] for v in record.cache.values()) for i in (0, 1))
    if leaf and alone and record.cores < rules.serial_cores:
        found.append(("single_core", "serial_seconds", f"ran {record.wall_seconds:.1f}s at {cores}"))
    # a pool is judged against the slots it held on average: a share the other commands of the host took back
    # between rebalances is theirs, not idle workers (the most it was ever admitted overstates it)
    held = record.slot_seconds / record.wall_seconds if record.slot_seconds and record.wall_seconds else None
    share = record.workers_admitted if held is None else held
    if pool and share and record.jobs >= rules.pool_fanout * share and record.cores < rules.pool_fill * share:
        found.append(
            (
                "pool_underused",
                "pool_fill",
                f"kept {cores} of {share:.2f} slots held busy (at most {record.workers_admitted} admitted)",
            )
        )
    if pool and record.workers_admitted and record.jobs and hits and not misses:  # inline jobs dispatch nothing
        found.append(("warm_dispatch", "pool_fill", f"dispatched {record.jobs} jobs with every cache lookup warm"))
    # a job's stage repeats once per job by design; a dispatch repeats when a later phase needs the earlier results
    if slow and record.kind != "own" and not shipped:
        # a gather is the same stage only when it ran the same dispatches: a later phase needs the earlier results
        # the kind keeps a gather's key (its path, then the names of its dispatches) apart from the key of the
        # dispatch record itself, whose path is the gather's path then the same name
        key = (record.kind, *record.path, *_dispatched.get(record.span, ()))
        if key in _seen:
            found.append(("repeated", "serial_seconds", "closed twice in one command"))
        _seen.add(key)
    return replace(
        record,
        findings=(
            *record.findings,
            *(
                finding(
                    f"budget.{key}",
                    f"{' > '.join(record.path)} {why}",
                    blocking=False,
                    origin=_budget_origins.get(f"budgets.{name}"),
                )
                for key, name, why in found
            ),
        ),
    )


def _log_failure() -> dict[str, Any]:
    row = finding("effort.log", "Stage log could not be written.", blocking=False)
    if _stack.get():
        _stack.get()[0].findings.append(row)
    return row


def _logged(record: StageRecord) -> Json:
    """The stage-log row of a record: each finding's reason and missing lines cut to their head (the command's result
    keeps them whole)."""

    def cut(text: str) -> str:
        return text if len(text) <= REASON_CHARS else text[:REASON_CHARS] + f" ... ({len(text)} chars)"

    row = asdict(record)
    for row_finding in row["findings"]:
        row_finding["reason"] = cut(row_finding["reason"])
        row_finding["missing"] = [cut(m) for m in row_finding.get("missing", ())]
    return row


def _emit(record: StageRecord, leaf: bool = False, shipped: bool = False) -> StageRecord:
    record = _budget(record, leaf, shipped)
    if _sink and (
        record.parent is None
        or record.kind == "pool"
        or record.findings
        or (_budgets is not None and record.wall_seconds >= _budgets.serial_seconds / 4)
    ):
        try:
            with _sink.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_logged(record)) + "\n")
        except OSError:
            row = _log_failure()
            if record.parent is None:
                record = replace(record, findings=(*record.findings, row))
    _closed.append(record)
    return record


def _close(opened: _Open, status: str) -> StageRecord:
    end = time.perf_counter_ns()
    children = opened.children
    wall = max(0, end - opened.start) / 1e9
    parent = max(0, _cpu() - opened.cpu)
    own_cpu = max(0, parent - sum(c.parent_cpu_seconds for c in children))
    cache = {
        k: (v[0] - opened.cache.get(k, (0, 0))[0], v[1] - opened.cache.get(k, (0, 0))[1]) for k, v in counters().items()
    }
    _merge_cache(cache, opened.remote_cache)
    record = _record(
        opened,
        wall,
        parent,
        sum(c.worker_cpu_seconds for c in children),
        sum(c.native_cpu_seconds for c in children),
        cache=cache,
        items=opened.items + sum(c.items for c in children if c.kind == "pool"),
        jobs=sum(c.jobs for c in children),
        workers_used=max((c.workers_used for c in children), default=0),
        workers_admitted=max((c.workers_admitted for c in children), default=0),
        parent_rss_peak_bytes=max(
            opened.rss,
            _rss(),
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ),
        worker_rss_peak_bytes=max((c.worker_rss_peak_bytes for c in children), default=0),
        **(
            {"slot_wait_seconds": _waits.get("slots", 0.0), "lock_wait_seconds": _waits.get("lock", 0.0)}
            if opened.kind == "command"
            else {}
        ),
    )
    if children:
        covered, cursor = 0, opened.start
        for start, stop in sorted(
            (max(opened.start, c.start_ns), min(end, c.start_ns + int(c.wall_seconds * 1e9))) for c in children
        ):
            covered += max(0, stop - max(cursor, start))
            cursor = max(cursor, stop)
        own = _Open.__new__(_Open)
        own.span, own.parent, own.path = uuid4().hex, opened.span, (*opened.path, "(own)")
        own.kind, own.start, own.items, own.wait = "own", opened.start, 0, 0.0
        _emit(replace(_record(own, max(0, wall - covered / 1e9), own_cpu), status=status), True)
    return replace(record, status=status)


def _blame(exc: BaseException) -> str:
    """` at module:function:line` of the innermost frame of the tool's own package, empty when there is none."""
    if not _package:
        return ""
    marker = f"/{_package}/"
    frames = [
        f for f in traceback.extract_tb(exc.__traceback__) if marker in f.filename and Path(f.filename).stem != "stages"
    ]
    return f" at {Path(frames[-1].filename).stem}:{frames[-1].name}:{frames[-1].lineno}" if frames else ""


@contextmanager
def _scope(opened: _Open) -> Iterator[_Open]:
    token = _stack.set((*_stack.get(), opened))
    status = "ok"
    try:
        if opened.kind in ("command", "stage", "pool"):
            _live[opened.span] = opened
            _notify("open", {"path": list(opened.path), "expected": expected(opened.path), "tool": _tool})
        yield opened
    except BaseException as exc:
        refused = isinstance(exc, _refusals)
        status = "refused" if refused else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        if refused:
            opened.findings.extend(finding_to_dict(f) for f in getattr(exc, "findings", ()))
        elif status == "failed" and not vars(exc).get("_named", False):  # the stage that raised names the cause once
            opened.findings.append(finding("internal.error", f"{type(exc).__name__}: {exc}{_blame(exc)}"))
            vars(exc)["_named"] = True
        raise
    finally:
        try:
            record = _close(opened, status)
            if opened.kind in ("command", "stage", "pool"):
                _live.pop(opened.span, None)
                if feed.slow(record.wall_seconds, expected(opened.path)):
                    detail: dict[str, Any] = {"status": status, "waited_seconds": round(opened.wait, 1)}
                    if opened.kind == "command":
                        detail["slot_wait_seconds"] = round(_waits.get("slots", 0.0), 1)
                    feed.say(
                        _tool,
                        opened.path,
                        record.wall_seconds,
                        expected(opened.path),
                        running=False,
                        detail=detail,
                        failed=_feed_failed,
                    )
                _notify(
                    "close",
                    {
                        "path": list(opened.path),
                        "wall": record.wall_seconds,
                        "status": record.status,
                        "cores": record.cores,
                        "jobs": record.jobs,
                    },
                )
            _dispatched[record.span] = tuple(c.path[-1] for c in opened.children if c.kind == "pool")
            record = _emit(replace(record, findings=tuple(opened.findings)), not opened.children)
            _dispatched.pop(record.span, None)
            stack = _stack.get()
            if len(stack) > 1:
                stack[-2].children.append(record)
                _merge_cache(stack[-2].remote_cache, opened.remote_cache)
        finally:
            _stack.reset(token)


def _trim(path: Path) -> None:
    """Drops the oldest lines of a stage log past `LOG_BYTES`, keeping the newest half whole-line."""
    if path.exists() and path.stat().st_size > LOG_BYTES:
        with path.open("rb") as stream:
            stream.seek(-LOG_BYTES // 2, 2)
            stream.readline()
            kept = stream.read()
        temp = path.with_name(f"{path.name}.{os.getpid()}")
        temp.write_bytes(kept)
        temp.replace(path)


@contextmanager
def command(tool: str, name: str, argv: Sequence[str], sink: Path) -> Iterator[None]:
    """The outermost span of one invocation of `tool`. Its rows go to `sink` (the stage log) and the previous
    command's walls are read from `walls.json` beside it."""
    global _closed, _counters, _sink, _tool, _invocation, _seen, _root
    _closed, _counters, _sink, _tool, _invocation = [], {}, sink, tool, uuid4().hex
    _seen = set()
    _waits.clear()
    _walls.clear()
    _walls.update(_load_walls(sink.with_name("walls.json")))
    for hook in _command_hooks:
        hook()
    token = _stack.set(())
    try:
        try:
            sink.parent.mkdir(parents=True, exist_ok=True)
            _trim(sink)
        except OSError:
            pass  # the command's own span carries the finding when its row cannot be written
        root = _root = _Open(name, "command")
        if bad := _bind():
            reason = f"The runtime file is unusable, so no slow stage reaches the feed: {bad}"
            root.findings.append(finding("events.runtime", reason, blocking=False))
        with _scope(root):
            feed.start(tool, _open_spans, _feed_failed)
            try:
                yield None
            finally:
                feed.stop()
    finally:
        _stack.reset(token)
        _save_walls()


@contextmanager
def stage(name: str) -> Iterator[Any]:
    if not _stack.get():
        yield _Null()
    else:
        with _scope(_Open(name, "stage", _stack.get()[-1])) as span:
            yield span


def record_native(name: str, completed: Any, start_ns: int) -> None:
    """A native child (`abucache.proc.Completed`) as a span of the stage that ran it."""
    if _stack.get():
        opened = _Open(name, "native", _stack.get()[-1])
        opened.start = start_ns
        record = _emit(_record(opened, completed.wall, native=completed.cpu, worker_rss_peak_bytes=completed.max_rss))
        _stack.get()[-1].children.append(record)


def record_pool(
    name: str,
    items: int,
    jobs: int,
    admitted: int,
    start_ns: int,
    envelopes: Sequence[Json],
    dispatch_seconds: float = 0.0,
    end_ns: int | None = None,
    slot_seconds: float = 0.0,
) -> None:
    if not _stack.get():
        return
    opened = _Open(name, "pool", _stack.get()[-1])
    opened.start = start_ns
    cache: dict[str, tuple[int, int]] = {}
    shipped: list[tuple[StageRecord, bool]] = []  # every worker record, rebased, and whether it is a worker's leaf
    for envelope in envelopes:
        _merge_cache(cache, envelope.get("cache", {}))
        records = envelope.get("records", [])
        ids = {r["span"]: uuid4().hex for r in records}
        parents = {r["parent"] for r in records}  # a shipped stage nothing else hangs under is a worker's leaf
        for raw in records:
            values = {k: v for k, v in raw.items() if k != "schema"}
            values.update(
                invocation=_invocation,
                span=ids[raw["span"]],
                parent=ids.get(raw["parent"], opened.span),
                path=opened.path + tuple(raw["path"]),
                parent_cpu_seconds=0,
                worker_cpu_seconds=raw["parent_cpu_seconds"] + raw["worker_cpu_seconds"],
                parent_rss_peak_bytes=0,
                worker_rss_peak_bytes=max(raw["parent_rss_peak_bytes"], raw["worker_rss_peak_bytes"]),
            )
            values["memory_limits"] = dict(raw.get("memory_limits", {}))
            values.update(_limits())
            values["cache"] = {k: tuple(v) for k, v in raw.get("cache", {}).items()}
            values["findings"] = tuple(
                finding(**{**f, "versions": tuple(f.get("versions", ())), "missing": tuple(f.get("missing", ()))})
                for f in raw.get("findings", [])
            )
            shipped.append((StageRecord(**values), raw["span"] not in parents))
    for folded, leaf in fold(shipped):
        _emit(folded, leaf, True)
    record = _emit(
        _record(
            opened,
            max(0, (end_ns or time.perf_counter_ns()) - start_ns) / 1e9,
            worker=sum(e["worker_cpu"] for e in envelopes),
            native=sum(e["native_cpu"] for e in envelopes),
            items=items,
            jobs=jobs,
            workers_admitted=admitted,
            cache=cache,
            workers_used=len({e["pid"] for e in envelopes}),
            dispatch_seconds=dispatch_seconds,
            slot_seconds=slot_seconds,
            wait_seconds=sum(e.get("wait_seconds", 0.0) for e in envelopes),
            busy_seconds=sum((e["end_ns"] - e["start_ns"]) / 1e9 - e.get("wait_seconds", 0.0) for e in envelopes),
            worker_rss_peak_bytes=max((e["rss"] for e in envelopes), default=0),
        ),
        True,
    )
    _stack.get()[-1].children.append(record)
    _merge_cache(_stack.get()[-1].remote_cache, cache)


def fold(shipped: Sequence[tuple[StageRecord, bool]]) -> list[tuple[StageRecord, bool]]:
    """Worker records, one per path: thousands of jobs below the log floor each still add up to the stage they spent
    their time in. A worker folds its chunk's records, the parent a pool's chunks. A record with findings, a status
    other than ok or a wall at the log floor stays on its own: the slow jobs of a pool's tail are each seen."""
    groups: dict[tuple[str, tuple[str, ...]], list[tuple[StageRecord, bool]]] = {}
    alone: list[tuple[StageRecord, bool]] = []
    floor = _budgets.serial_seconds / 4 if _budgets else float("inf")
    for record, leaf in shipped:
        if record.findings or record.status != "ok" or record.wall_seconds >= floor:  # a slow one is seen alone
            alone.append((record, leaf))
        else:
            groups.setdefault((record.kind, record.path), []).append((record, leaf))
    span: dict[str | None, str | None] = {
        r.span: (members[0][0].span if len(members) > 1 else r.span) for members in groups.values() for r, _ in members
    }
    out = [(replace(r, parent=span.get(r.parent, r.parent)), leaf) for r, leaf in alone]
    for members in groups.values():
        first = members[0][0]
        if len(members) == 1:
            out.append((replace(first, parent=span.get(first.parent, first.parent)), members[0][1]))
            continue
        rows = [r for r, _ in members]
        wall = sum(r.wall_seconds for r in rows)
        own, worker = sum(r.parent_cpu_seconds for r in rows), sum(r.worker_cpu_seconds for r in rows)
        native = sum(r.native_cpu_seconds for r in rows)
        cores = (own + worker + native) / wall if wall else 0
        cache: dict[str, tuple[int, int]] = {}
        for r in rows:
            _merge_cache(cache, dict(r.cache))
        folded = replace(
            first,
            parent=span.get(first.parent, first.parent),
            execution="parallel" if _budgets and cores >= _budgets.serial_cores else "serial",
            start_ns=min(r.start_ns for r in rows),
            wall_seconds=wall,
            parent_cpu_seconds=own,
            worker_cpu_seconds=worker,
            native_cpu_seconds=native,
            cores=cores,
            items=sum(r.items for r in rows),
            jobs=sum(r.jobs for r in rows),
            workers_used=max(r.workers_used for r in rows),
            workers_admitted=max(r.workers_admitted for r in rows),
            worker_rss_peak_bytes=max(r.worker_rss_peak_bytes for r in rows),
            cache=cache,
            wait_seconds=sum(r.wait_seconds for r in rows),
            dispatch_seconds=sum(r.dispatch_seconds for r in rows),
            busy_seconds=sum(r.busy_seconds for r in rows),
            slot_seconds=sum(r.slot_seconds for r in rows),
            spans=sum(r.spans for r in rows),
        )
        out.append((folded, all(leaf for _, leaf in members)))
    return out


class _Null:
    def add(self, *, items: int = 0, findings: Sequence[Any] = ()) -> None:
        pass


def _merge_cache(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, pair in source.items():
        previous = target.get(key, (0, 0))
        target[key] = (previous[0] + pair[0], previous[1] + pair[1])


def count(kind: str, hit: bool) -> None:
    _counters.setdefault(kind, [0, 0])[0 if hit else 1] += 1


def counters() -> dict[str, tuple[int, int]]:
    return {k: (v[0], v[1]) for k, v in _counters.items()}


def cache_total(stages: Json) -> Json:
    """Hits and misses of the whole command, in total and per cache kind: the command's own (outermost) stage already
    holds every lookup made under it, so nothing is summed over the nested stages. An empty tree falls back to this
    process's counters."""
    pairs = stages.get("cache") if stages else None
    if not pairs:
        pairs = counters()
    kinds = {k: {"hits": int(v[0]), "misses": int(v[1])} for k, v in sorted(pairs.items()) if v[0] or v[1]}
    return {
        "hits": sum(v["hits"] for v in kinds.values()),
        "misses": sum(v["misses"] for v in kinds.values()),
        "kinds": kinds,
    }


@contextmanager
def capture() -> Iterator[list[StageRecord]]:
    """Spans of one job in a worker: collected in memory (no log), for the envelope the worker returns."""
    global _closed, _counters, _sink, _invocation, _seen
    saved = _closed, _counters, _sink, _invocation, _seen
    _closed, _counters, _sink, _invocation, _seen = [], {}, None, uuid4().hex, set()
    token = _stack.set(())
    try:
        with _scope(_Open("job", "job")):
            yield _closed
    finally:
        _stack.reset(token)
        _closed, _counters, _sink, _invocation, _seen = saved


def tree() -> Json:
    nodes = {r.span: {**asdict(r), "children": []} for r in _closed}
    roots: list[Any] = []
    for record in _closed:
        (nodes[record.parent]["children"] if record.parent in nodes else roots).append(nodes[record.span])
    return roots[-1] if roots else {}


def invocation() -> str:
    return _invocation
