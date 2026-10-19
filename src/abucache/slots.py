# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The host's worker slots: every heavy thing on a machine takes its share of the cores from one set of flock files,
so commands of any tool divide the machine fairly instead of each assuming it owns it.

Wire protocol 2 (frozen): `$XDG_RUNTIME_DIR/unbake-pool-2/` holds `slot-<n>.lock` (one flock per core), `cmd-<pid>.lock`
(a live command, holding its demand and budget), `wait-<ns>-<pid>` (the queue for a free slot), `wake` (the one file
whose write-close wakes every waiter) and `waits.jsonl`. `UNBAKE_CORES` hands a held share to a child. Named
capacities are new files (`cap-<name>/token-<n>.lock`) that no older process reads.

A command takes its max-min fair share of the budget (`runtime.workers`), at most the work it has and, when it names
`task_memory`, at most what the free memory holds. Waits and short shares are stages of the command's tree
(`abucache.stages`), so a slow or stuck job shows without per-tool code."""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import io
import os
import re
import select
import subprocess
import sys
import termios
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from abucache import runtime as runtime_file
from abucache import stages

PROTOCOL = 2  # the host slot protocol: a command of any other one books other files, so it never runs beside this one
POOL = f"unbake-pool-{PROTOCOL}"  # the slot directory of this protocol, under XDG_RUNTIME_DIR
WAITS = "waits.jsonl"  # in the slot directory: one line per wait of 60 s or more (stages.WAIT_NOTE), whoever waited
CORES_ENV = "UNBAKE_CORES"  # the cores a holder above this process took for the command
CAPACITIES_ENV = "ABU_CAPACITIES"  # comma separated capacity names a holder above this process holds for the command
REBALANCE = 0.5  # seconds between looks at the other commands while a make feeder works
CAPACITY_RECHECK = 5.0  # seconds: a capacity waiter also looks again in case a holder died before it wrote its pid
DEAD_AFTER = 60.0  # seconds a command's lock file must be old before a free one is known to be a finished command's
_IN_CHANGES = 0x8 | 0x80 | 0x100 | 0x200  # IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE | IN_DELETE: a release or a peer
_IN_FLAGS = 0o4000 | 0o2000000  # IN_NONBLOCK | IN_CLOEXEC
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class UnknownCapacity(KeyError):
    """A capacity name the runtime file does not list under `[capacities]`."""


def available_memory() -> int:
    """Bytes the machine can give a new process without swapping (MemAvailable)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")


def memory_slots(task_memory: int, reserve: int) -> int:
    """How many tasks of `task_memory` bytes fit in the free memory beyond `reserve`: at least 1, so a command always
    makes progress; 0 when no task memory is named (no bound)."""
    if task_memory <= 0:
        return 0
    return max(1, (available_memory() - reserve) // task_memory)


def _busy(path: Path) -> bool:
    """Whether a live command holds `path`. A free lock of a command that finished long ago is removed, so the files of
    thousands of past commands are never opened again by every later acquire."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        if time.time() - os.fstat(fd).st_mtime > DEAD_AFTER:
            path.unlink(missing_ok=True)
        return False
    finally:
        os.close(fd)


def _foreign(base: Path) -> tuple[list[Path], list[Path]]:
    """(the directories other slot protocols keep their command files in, the live command files there). Protocol 1
    kept its files flat in the runtime directory."""
    dirs = [base, *(d for d in base.glob("unbake-pool-*") if d.name != POOL and d.is_dir())]
    files = [*base.glob("unbake-cmd-*.lock"), *(p for d in dirs[1:] for p in d.glob("cmd-*.lock"))]
    return dirs, sorted(p for p in files if _busy(p))


def _pid(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[1])


def _pidfd(pid: int) -> int | None:
    try:
        return os.pidfd_open(pid)
    except (ProcessLookupError, PermissionError):  # gone already: what it held is free for the next attempt
        return None


def _inotify(*paths: Path) -> int:
    """One descriptor that turns readable on a write-close, a new file or a removal in any of `paths`."""
    libc = ctypes.CDLL(None, use_errno=True)
    fd = int(libc.inotify_init1(_IN_FLAGS))
    for path in paths:
        if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(path), _IN_CHANGES) < 0:
            number = ctypes.get_errno()
            if fd >= 0:
                os.close(fd)
            raise OSError(number, os.strerror(number), str(path))
    return fd


def _drain(base: Path) -> None:
    while True:
        dirs, _ = _foreign(base)
        watch = _inotify(*dirs)  # watched before the look: a command that ends between the two is not missed
        try:
            _, live = _foreign(base)
            if not live:
                return
            fds = [fd for p in live if (fd := _pidfd(_pid(p))) is not None]
            try:
                select.select([watch, *fds], [], [])
            finally:
                for fd in fds:
                    os.close(fd)
        finally:
            os.close(watch)


def _pool(base: Path) -> Path:
    """The slot directory of this protocol, once no command of another protocol is live: their slots are other files,
    so running beside one would book the same cores twice. Until they finish this command sleeps on their pidfds and
    on their command files (a release unlinks one), never on a timer; they never see this protocol, so they drain."""
    if live := _foreign(base)[1]:
        named = ", ".join(str(_pid(p)) for p in live)
        reason = f"pids {named} hold host slots under another slot protocol than {PROTOCOL}"
        stages.say(f"slots: waiting for {reason} to finish")
        with stages.stage("pool.protocol") as span:  # the wait is a stage of its own in the tree
            span.add(items=len(live), findings=[stages.finding("pool.protocol", reason, blocking=False)])
            began = time.perf_counter()
            _drain(base)
            stages.waited(time.perf_counter() - began, "slots", f"slot protocol of pids {named}")
    pool = base / POOL
    pool.mkdir(exist_ok=True)
    return pool


def _runtime_base() -> Path:
    base = Path(os.environ.get("XDG_RUNTIME_DIR", ""))
    if not base.is_dir():
        raise RuntimeError(f"XDG_RUNTIME_DIR is not a directory ({base}): the worker slots live there")
    return base


def wait_log() -> Path:
    """The host's wait log: pass it to `stages.configure(wait_log=...)` so a wait of a minute or more is noted there."""
    return Path(os.environ.get("XDG_RUNTIME_DIR", "")) / POOL / WAITS


def _declared(path: Path, want: int) -> tuple[int, int]:
    """(the slots a peer said it could use, its budget); a peer that said nothing yet is taken to want all of `want`."""
    try:
        words = path.read_text().split()
    except OSError:
        return want, want
    if len(words) != 2 or not all(w.isdigit() and int(w) > 0 for w in words):
        return want, want
    return int(words[0]), int(words[1])


def _fair(want: int, demands: dict[Path, int], mine: Path) -> int:
    """This command's max-min fair share of `want` slots: a peer that can use fewer than an equal split keeps only
    what it can use and the rest is split among the others, so no slot idles while a command has work for it."""
    left, order = want, sorted(demands, key=lambda p: (demands[p], p))
    for at, path in enumerate(order):
        others = order[at:]
        quotient, remainder = divmod(left, len(others))
        if demands[path] > quotient:  # every command from here on can use more than the equal split: split evenly
            ranked = sorted(others)
            return max(1, quotient + (ranked.index(mine) < remainder)) if mine in others else max(1, demands[mine])
        if path == mine:
            return max(1, demands[mine])
        left -= demands[path]
    return max(1, demands[mine])


def _signal(directory: Path) -> None:
    """Wakes every waiter of the directory: a write-close there is the one event they sleep on."""
    os.close(os.open(directory / "wake", os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC, 0o600))


def _give(slot: io.IOBase) -> None:
    fcntl.flock(slot, fcntl.LOCK_UN)
    slot.close()


def _try(path: Path) -> io.FileIO | None:
    """The slot at `path` held on a read-only descriptor of its own, or None when another command holds it. A probe
    that loses closes a read-only file, which wakes nobody."""
    try:
        slot = io.FileIO(str(path), "r")
    except FileNotFoundError:
        os.close(os.open(path, os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC, 0o600))
        slot = io.FileIO(str(path), "r")
    try:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        slot.close()
        return None
    return slot


def _until(
    directory: Path,
    attempt: Callable[[], bool],
    *,
    pids: Callable[[], list[int]] = list,
    stop: int | None = None,
    also: Callable[[], int | None] | None = None,
    interval: float | None = None,
) -> None:
    """Calls `attempt` until it is true or `stop` is readable, sleeping between attempts only until something that can
    free a slot happens: a release (the write-close every holder makes), a new or finished command (its lock file),
    or the death of a live holder (its pidfd, from `pids`), or `also` (a descriptor the caller names for the next
    wait). Ordinary waiters sleep only on events; a make feeder also rechecks at `interval` because consuming a token
    makes no readable event. Idle workers stay up through the wait: they hold no slot and their warm state is what
    the next dispatch runs on."""
    watch = _inotify(directory)  # watched before the first attempt: no release between the two is missed
    try:
        while True:
            held = [fd for pid in pids() if (fd := _pidfd(pid)) is not None]
            try:  # opened before the attempt: a death after it shows as a readable pidfd
                if attempt():
                    return
                extra = [fd for fd in (stop, also() if also is not None else None) if fd is not None]
                ready = select.select([watch, *held, *extra], [], [], interval)[0]
            finally:
                for fd in held:
                    os.close(fd)
            if stop is not None and stop in ready:
                return
            if watch in ready:
                while True:
                    try:
                        if not os.read(watch, 65536):
                            break
                    except BlockingIOError:
                        break
    finally:
        os.close(watch)


class Slots:
    """One process's holding of the host slots. The module's `default` serves every caller of a process; a test makes
    its own. The budget is `runtime.workers`, read once from the runtime file unless one is passed."""

    def __init__(self, runtime: runtime_file.Runtime | None = None) -> None:
        self._runtime = runtime
        self.slots: list[io.IOBase] = []  # the host-wide slots this process holds now: one flock each, freed on death
        self.dir: list[Path] = []  # the slot directory this process takes from; empty under a holder above it
        self.cmd: list[io.TextIOWrapper] = []  # this command's own lock file: its presence divides the slots fairly
        self.fed: list[io.IOBase] = []  # slots a jobserver feeder added for a make: a rebalance never gives them back
        self.ticket: list[Path] = []  # this command's place in the queue for a free slot while it waits for one
        self.want = 0  # this command's host budget, as the last acquisition declared it
        self.share = 0  # the share the last acquisition computed
        self.task_memory = 0  # bytes one task of the holder needs: bounds the grant by the free memory
        self.users = 0  # holders of cores in this process now: the slots are released when the last one ends
        self.lock = threading.Lock()
        self._said: dict[int, bytes] = {}  # per lock file descriptor, the demand last written into it
        self._short: list[int] = []  # (slots got, slots wanted) of the last short share reported

    @property
    def runtime(self) -> runtime_file.Runtime:
        if self._runtime is None:
            self._runtime = runtime_file.load()
        return self._runtime

    def configure(self, runtime: runtime_file.Runtime | None) -> None:
        self._runtime = runtime

    @property
    def budget(self) -> int:
        """The one host fact every caller divides: the workers the runtime file resolved to."""
        return self.runtime.workers

    # -- the command's place among its peers ---------------------------------------------------------------------

    def _register(self, pool: Path) -> Path:
        """This command's lock file, held exclusively while it has work: its presence divides the slots fairly. The
        file is unlinked on release, so one a peer unlinked in between is made again."""
        mine = pool / f"cmd-{os.getpid()}.lock"
        while True:
            if not self.cmd:
                self.cmd.append(mine.open("a"))
                self._said.pop(self.cmd[0].fileno(), None)  # a new file says nothing yet, whatever a closed one said
            fcntl.flock(self.cmd[0], fcntl.LOCK_EX)
            try:
                if os.fstat(self.cmd[0].fileno()).st_ino == mine.stat().st_ino:
                    return mine
            except FileNotFoundError:
                pass
            self.cmd.pop().close()

    @staticmethod
    def _peers(pool: Path, mine: Path) -> list[Path]:
        """The lock files of every live command of this protocol, this one's included."""
        return sorted([mine, *(p for p in pool.glob("cmd-*.lock") if p != mine and _busy(p))])

    def _demand(self, count: int, budget: int) -> None:
        """Writes the slots this command could use now and its budget into its lock file: peers leave it no more than
        the first, and the largest budget of any live command is how many slot files there are. A change wakes the
        peers' waiters (a jobserver feeder sheds what its share no longer holds)."""
        if self.cmd:
            fd, said = self.cmd[0].fileno(), f"{count} {budget}".encode()
            before = self._said.get(fd)
            if before == said:
                return
            self._said[fd] = said
            os.ftruncate(fd, 0)
            os.pwrite(fd, said, 0)
            if before is not None and self.dir:  # a first word needs none: the new lock file woke the waiters
                _signal(self.dir[0])

    def _divide(self, pool: Path, mine: Path, want: int, using: int = 0) -> tuple[list[Path], int, int]:
        """(live peers, slot files, this command's fair share). Every command of the host sees the same slot files: as
        many as the largest budget any live command declared, so a command with a small budget takes any free slot of
        them (at most its budget) and never waits on a few low slots while higher ones are free."""
        peers = self._peers(pool, mine)
        declared = {p: (using or want, want) if p == mine else _declared(p, want) for p in peers}
        size = max(budget for _, budget in declared.values())
        return peers, size, _fair(size, {p: demand for p, (demand, _) in declared.items()}, mine)

    def _memory_slots(self) -> int:
        return memory_slots(self.task_memory, self.runtime.reserve_memory)

    # -- the queue --------------------------------------------------------------------------------------------

    @staticmethod
    def _queue(pool: Path) -> list[Path]:
        """The tickets of the live commands waiting for a free slot, oldest first. A ticket whose command is gone is
        removed."""
        live = []
        for ticket in sorted(pool.glob("wait-*")):
            try:
                owner = int(ticket.name.rsplit("-", 1)[1])
            except ValueError:
                continue
            if owner == os.getpid() or _busy(pool / f"cmd-{owner}.lock"):
                live.append(ticket)
            else:
                ticket.unlink(missing_ok=True)
        return live

    def _queued(self, pool: Path) -> bool:
        """Whether a command that asked before this one still waits for a slot: free slots are its, in arrival order.
        A command takes a free slot ahead of nobody, so the longest waiter is never starved by newer commands."""
        queue = self._queue(pool)
        return any(t < self.ticket[0] for t in queue) if self.ticket else bool(queue)

    def waiting(self) -> bool:
        """Whether another command waits for a host slot now: a ticket in the queue that is not this process's own."""
        if not self.dir:
            return False
        return any(not t.name.endswith(f"-{os.getpid()}") for t in self._queue(self.dir[0]))

    def _enter(self, pool: Path) -> None:
        if not self.ticket:
            ticket = pool / f"wait-{time.time_ns():020d}-{os.getpid()}"
            os.close(os.open(ticket, os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC, 0o600))
            self.ticket.append(ticket)

    def _leave(self) -> None:
        """Gives the place in the queue up and wakes the waiters, who may now be first."""
        if self.ticket:
            ticket = self.ticket.pop()
            ticket.unlink(missing_ok=True)
            if self.dir:
                _signal(self.dir[0])

    # -- taking and giving slots -----------------------------------------------------------------------------

    def _take(self, pool: Path, size: int, take: int) -> list[io.IOBase]:
        """Takes free slots of the `size` slot files without waiting until this process holds `take`: the ones it
        got. None while an older waiter is queued."""
        if self._queued(pool):
            return []
        held = {getattr(f, "name", None) for f in self.slots}
        got: list[io.IOBase] = []
        for index in range(size):
            if len(self.slots) >= take:
                break
            path = pool / f"slot-{index}.lock"
            if str(path) in held:
                continue
            if (slot := _try(path)) is not None:
                self.slots.append(slot)
                got.append(slot)
        return got

    def _wait_free(self, pool: Path, mine: Path, want: int) -> None:
        """Blocks until any slot of the host frees and takes it."""

        def attempt() -> bool:
            return bool(self._take(pool, self._divide(pool, mine, want)[1], 1))

        _until(pool, attempt, pids=lambda: [_pid(p) for p in self._peers(pool, mine) if p != mine])

    def acquire(self, active: int = 0, cap: int = 0, task_memory: int = 0) -> None:
        """All processes of this user share the host slots (flock files): a command takes its share of the free ones
        (at most `cap`, the work it has, the tasks the free memory holds, and the budget), or blocks until any one of
        them frees."""
        want = self.budget
        self.task_memory = task_memory
        if inherited := int(os.environ.get(CORES_ENV) or 0):  # a holder above this process took them for the command
            self.share = min(want, inherited)
            self.slots[:] = [Path(os.devnull).open() for _ in range(min(self.share, cap or self.share))]  # noqa: SIM115
            return
        if not self.cmd or not self.dir:
            self.dir[:] = [_pool(_runtime_base())]
        pool, self.want = self.dir[0], want
        mine = self._register(pool)
        limit = min([n for n in (cap, self._memory_slots()) if n], default=0)  # what this command can use now
        self._demand(max(limit, active) if limit else want, want)  # told to its peers
        peers, size, share = self._divide(pool, mine, want)
        self.share, take = share, min(share, limit) if limit else share
        # In-flight chunks retain their slots until completion; never oversubscribe a released slot.
        shed = False
        while len(self.slots) > max(take, active):
            spare = next((f for f in reversed(self.slots) if f not in self.fed), None)
            if spare is None:  # what is left backs a make that still holds its token
                break
            self.slots.remove(spare)
            _give(spare)
            shed = True
        if shed:
            _signal(pool)
        self._take(pool, size, take)
        if 0 < len(self.slots) < take and self._short != [len(self.slots), take]:  # a short share shows once
            self._short[:] = [len(self.slots), take]
            with stages.stage("pool.short_slots") as span:
                others = [str(_pid(p)) for p in peers if p != mine]
                named = ", ".join(others) or "none live"
                reason = f"{len(self.slots)} of {take} slots; the rest are held by pids {named}"
                span.add(
                    items=take - len(self.slots),
                    findings=[stages.finding("pool.short_slots", reason, blocking=False)],
                )
        if not self.slots:  # every slot is held: wait for whichever frees first, as a stage of its own
            holders = [_pid(p) for p in peers if p != mine]
            self._enter(pool)  # first come, first served: a newer command takes no slot while this one waits
            try:
                with stages.stage("pool.wait_slot") as span:
                    held = "held by no live command"
                    if holders:
                        held = f"held by pids {', '.join(str(p) for p in holders)}"
                    span.add(items=len(holders), findings=[stages.finding("pool.wait_slot", held, blocking=False)])
                    began = time.perf_counter()
                    self._wait_free(pool, mine, want)
                    stages.waited(time.perf_counter() - began, "slots", held)
                _, size, share = self._divide(pool, mine, want)  # a release frees whole shares: take the rest too
                self.share = share
                self._take(pool, size, min(share, limit) if limit else share)
            finally:
                self._leave()

    def release(self) -> None:
        """A command holds slots only while its work runs: between phases other commands may take them."""
        for f in self.slots:
            _give(f)
        for f in self.cmd:
            fcntl.flock(f, fcntl.LOCK_UN)
        pooled = bool(self.slots) and bool(self.dir)
        self.slots.clear()
        self.fed.clear()
        for f in self.cmd:  # a finished command leaves no lock file behind
            self._said.pop(f.fileno(), None)
            Path(f.name).unlink(missing_ok=True)
            f.close()
        self.cmd.clear()
        if pooled:
            _signal(self.dir[0])

    def close_inherited(self) -> None:
        """A forked worker closes the parent's slot and command locks: they are the parent's, and a worker holding one
        would pin it for life."""
        for f in self.slots + self.cmd:
            f.close()

    # -- cores ---------------------------------------------------------------------------------------------------

    @contextmanager
    def cores(self, work: int = 0, *, task_memory: int = 0, capacity: str | None = None) -> Iterator[int]:
        """The one way anything heavy takes cores: holds this command's share of the host slots (at most `work`, and
        at most the tasks of `task_memory` bytes the free memory holds) for the block and yields how many to use, for
        a pool, `make -j`, a permuter or a test runner alike. `capacity` first holds one token of that named
        capacity, so cores are never held while waiting for it."""
        with contextlib.ExitStack() as stack:
            if capacity:
                stack.enter_context(self.capacity(capacity))
            with self.lock:
                if not self.slots:
                    self.acquire(cap=work, task_memory=task_memory)
                self.users += 1
            try:
                yield max(1, min(work or len(self.slots), len(self.slots)))
            finally:
                with self.lock:
                    self.users -= 1
                    if self.users == 0:
                        self.release()

    # -- named capacities -----------------------------------------------------------------------------------

    @contextmanager
    def capacity(self, name: str) -> Iterator[None]:
        """Holds one token of the named capacity (`runtime.capacities[name]` of them exist on the host) for the block,
        waiting as the stage `pool.wait_capacity` while all are held. A process under a holder that named the
        capacity (`ABU_CAPACITIES`) holds nothing more: its holder's token covers it."""
        if name in os.environ.get(CAPACITIES_ENV, "").split(","):
            yield
            return
        if not _NAME.match(name):
            raise ValueError(f"bad capacity name {name!r}")
        try:
            size = self.runtime.capacities[name]
        except KeyError:
            raise UnknownCapacity(name) from None
        directory = _pool(_runtime_base()) / f"cap-{name}"
        directory.mkdir(exist_ok=True)
        got: list[io.FileIO] = []
        ticket: list[io.TextIOWrapper] = []  # this holder's place in the queue while it waits: held locked while alive

        def ahead() -> list[Path]:
            """The live waiters that asked before this one (every live waiter when it has not asked yet)."""
            mine = Path(ticket[0].name).name if ticket else None
            return [t for t in sorted(directory.glob("wait-*")) if (mine is None or t.name < mine) and _busy(t)]

        def attempt() -> bool:
            """First come, first served: a free token is the oldest waiter's, so a newer holder never takes it ahead
            of one that waited and the longest waiter is never starved."""
            if ahead():
                return False
            for index in range(size):
                path = directory / f"token-{index}.lock"
                if (slot := _try(path)) is not None:
                    fd = os.open(path, os.O_WRONLY | os.O_CLOEXEC)  # the holder's pid, for waiters to watch
                    try:
                        os.ftruncate(fd, 0)
                        os.pwrite(fd, str(os.getpid()).encode(), 0)
                    finally:
                        os.close(fd)
                    got.append(slot)
                    return True
            return False

        def holders() -> list[int]:
            found = []
            for index in range(size):
                with contextlib.suppress(OSError, ValueError):
                    found.append(int((directory / f"token-{index}.lock").read_text()))
            return found

        def watched() -> list[int]:
            """Who can free the way: the token holders and the waiters ahead (one that dies leaves its ticket
            unlocked, which no file event shows)."""
            return [*holders(), *(_pid(t) for t in ahead())]

        if not attempt():
            held = f"held by pids {', '.join(str(p) for p in holders()) or 'none live'}"
            queue = directory / f"wait-{time.time_ns():020d}-{os.getpid()}"
            ticket.append(queue.open("a"))
            fcntl.flock(ticket[0], fcntl.LOCK_EX)
            try:
                with stages.stage("pool.wait_capacity") as span:
                    span.add(
                        items=size,
                        findings=[stages.finding("pool.wait_capacity", f"{name}: {held}", blocking=False)],
                    )
                    began = time.perf_counter()
                    _until(directory, attempt, pids=watched, interval=CAPACITY_RECHECK)
                    stages.waited(time.perf_counter() - began, "slots", f"capacity {name} {held}")
            finally:
                queue.unlink(missing_ok=True)
                ticket.pop().close()
                _signal(directory)  # the waiters behind may now be first
        try:
            yield
        finally:
            _give(got[0])
            _signal(directory)

    # -- make jobserver ------------------------------------------------------------------------------------------

    def _extra_slot(self) -> io.IOBase | None:
        """Takes one more free host slot for this process if one is free now and its fair share (and the free memory)
        has room (never waits): its file, None otherwise."""
        with self.lock:
            if not self.dir or not self.cmd or self.cmd[0].closed:  # a dropped executor closed the lock under a make
                return None
            pool = self.dir[0]
            _, size, share = self._divide(pool, Path(self.cmd[0].name), self.want)
            if self.task_memory:
                share = min(share, self._memory_slots())
            if len(self.slots) >= share:
                return None
            got = self._take(pool, size, len(self.slots) + 1)
            self.fed.extend(got)
            return got[0] if got else None

    @staticmethod
    def _idle(spare: int) -> int:
        """The job tokens lying in make's pipe now: slots this command holds that no job of make uses."""
        return int.from_bytes(fcntl.ioctl(spare, termios.FIONREAD, b"\0\0\0\0"), sys.byteorder)

    def _shed(self, spare: int, taken: list[io.IOBase], last: list[int]) -> bool:
        """Tells the peers what make uses plus one probe token, and gives back each idle token above this command's
        fair share with its slot, the feeder's own first: a build in a serial phase holds no slot another command's
        work could run on. `spare` reads the pipe through its own non-blocking description, so make's descriptor
        keeps its flags. One idle token stays within its share: consuming it tells the feeder make needs to grow,
        while a serial make never repeatedly takes and sheds an unused grant. A token is idle only when it lay in the
        pipe at the feeder's last look too (`last`): one make has not read yet (a build starting, a job just ended) is
        no idle slot. True while it still holds more than its share and a peer could use a slot: the feeder then
        wakes when a token comes back to the pipe."""
        with self.lock:
            if not self.dir or not self.cmd or self.cmd[0].closed:  # a dropped executor closed the lock under a make
                return False
            now = self._idle(spare)
            idle, last[:] = min(now, last[0] if last else 0), [now]
            using = min(self.want, max(1, len(self.slots) - idle + 1))
            self._demand(using, self.want)
            peers, _, share = self._divide(self.dir[0], Path(self.cmd[0].name), self.want, using)
            if self.task_memory:  # the free memory shrinks the share too: what no longer fits goes back
                share = min(share, self._memory_slots())
            elif len(peers) < 2:  # no other command to give a slot to: a token make has not read yet stays
                return False
            if len(self.slots) > max(1, share) and self.waiting():  # a token back in the pipe is a job that ended:
                idle = now  # while a peer waits and this build holds more than its share, it is the peer's, not make's
            given = False
            while len(self.slots) > max(1, share) and idle > 0:
                try:
                    os.read(spare, 1)
                except BlockingIOError:  # make took it first: that token is busy again
                    break
                idle -= 1
                slot = taken.pop() if taken else self.slots[-1]
                if slot in self.fed:
                    self.fed.remove(slot)
                self.slots.remove(slot)
                _give(slot)
                given = True
            if given:
                _signal(self.dir[0])
            return len(self.slots) > max(1, share)

    def _feed(self, tokens: int, spare: int, stop: int, taken: list[io.IOBase]) -> None:
        """Gives a running GNU make one more job token whenever a host slot frees, so a build that started on a small
        share grows into the cores other commands release instead of staying at its first grant, and takes idle
        tokens back when its share shrinks (`_shed`). Sleeps on the host's slot events and on `stop` (readable once
        make returned), and while it holds more than its share, on make's token pipe. Rechecks at the rebalance
        interval also notice token consumption and the free memory, which have no readable event."""
        last: list[int] = []
        over = [False]  # holding more than its share while a peer waits: a token coming back is worth a look

        def attempt() -> bool:
            over[0] = self._shed(spare, taken, last)
            if not self._idle(spare):  # make consumed every available token: only real demand grows the grant
                while (slot := self._extra_slot()) is not None:
                    taken.append(slot)  # the feeder's own: exactly these go back when make returns
                    os.write(tokens, b"+")
            return False

        mine = Path(self.cmd[0].name)
        pool = self.dir[0]
        _until(
            pool,
            attempt,
            pids=lambda: [_pid(p) for p in self._peers(pool, mine) if p != mine],
            stop=stop,
            also=lambda: spare if over[0] else None,
            interval=REBALANCE,
        )

    @contextmanager
    def jobserver(
        self, work: int = 0, *, task_memory: int = 0
    ) -> Iterator[tuple[int, dict[str, str], tuple[int, ...]]]:
        """make's share of the host slots as a GNU make jobserver: yields (first grant, env, fds to pass). The grant is
        only where the build starts: a feeder adds a job token whenever another host slot frees. With no slot files
        (no host pool) MAKEFLAGS pins -j to the grant and no fds pass. make itself must get no -j of its own."""
        with self.cores(work, task_memory=task_memory) as j:
            if not self.dir or not self.cmd or self.cmd[0].closed:  # a dropped executor closed the lock under a make
                yield j, {CORES_ENV: str(j), "MAKEFLAGS": f"-j{j}"}, ()
                return
            read, write = os.pipe()
            os.write(write, b"+" * (j - 1))  # make itself holds the first job token
            spare = os.open(f"/proc/self/fd/{read}", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            stopped, stop = os.pipe()
            taken: list[io.IOBase] = []
            feeder = threading.Thread(target=self._feed, args=(write, spare, stopped, taken), daemon=True)
            feeder.start()
            try:
                yield j, {CORES_ENV: str(j), "MAKEFLAGS": f"-j{j} --jobserver-auth={read},{write}"}, (read, write)
            finally:
                os.close(stop)  # the feeder's end reads EOF: it returns
                feeder.join()
                os.close(stopped)
                os.close(spare)
                os.close(read)
                os.close(write)
                with self.lock:  # the slots the feeder added go back as make returns, not when the command ends
                    given = False
                    for slot in taken:
                        if slot in self.fed:
                            self.fed.remove(slot)
                        if slot in self.slots:  # a pool of this process may have given it back already
                            self.slots.remove(slot)
                            _give(slot)
                            given = True
                    if given:
                        _signal(self.dir[0])


default = Slots()


def configure(runtime: runtime_file.Runtime | None) -> None:
    """Sets the runtime whose `workers`, `reserve_memory` and `capacities` the module-level calls use (the file named
    by `ABU_RUNTIME` is read when none was set)."""
    default.configure(runtime)


def cores(work: int = 0, *, task_memory: int = 0, capacity: str | None = None):
    """`Slots.cores` of this process's holding."""
    return default.cores(work, task_memory=task_memory, capacity=capacity)


def capacity(name: str):
    """`Slots.capacity` of this process's holding."""
    return default.capacity(name)


def jobserver(work: int = 0, *, task_memory: int = 0):
    """`Slots.jobserver` of this process's holding."""
    return default.jobserver(work, task_memory=task_memory)


def _is_make(program: str) -> bool:
    return Path(program).name in ("make", "gmake")


def run(argv: Sequence[str]) -> int:
    """The `unbake-run` body: `unbake-run [--work N] [--mem BYTES] [--capacity NAME] [--] CMD...`. Only a CMD whose
    arguments contain `{j}` holds a share of the host slots (as `{j}` and UNBAKE_CORES); any other CMD runs directly,
    taking none. `make -j{j}` runs on a jobserver fed from the slot pool, so a build takes its fair share, grows into
    slots other commands free and gives them back as they ask. A raw `make -jN` is invisible to the slots and starves
    every other command. `--capacity` holds one token of that named capacity for the command's life."""
    args, work, mem, named = list(argv), 0, 0, ""
    while args and args[0] in ("--work", "--mem", "--capacity") and len(args) > 1:
        flag, value, args = args[0], args[1], args[2:]
        if flag == "--capacity":
            named = value
        else:
            number = int(value)
            work, mem = (number, mem) if flag == "--work" else (work, number)
    command = args[1:] if args[:1] == ["--"] else args
    if not command:
        print(run.__doc__, file=sys.stderr)
        return 2
    if not any("{j}" in a for a in command):
        return subprocess.run(command).returncode
    try:
        runtime_file.load()
    except runtime_file.RuntimeConfigError:  # no usable runtime file: run the command alone, holding no slots
        return subprocess.run(
            [a.replace("{j}", "1") for a in command], env={**os.environ, CORES_ENV: "1"}
        ).returncode
    with contextlib.ExitStack() as stack:
        extra = {}
        if named:
            stack.enter_context(capacity(named))
            held = [n for n in os.environ.get(CAPACITIES_ENV, "").split(",") if n]
            extra[CAPACITIES_ENV] = ",".join([*held, named])
        if _is_make(command[0]) and "-j{j}" in command:  # elastic: the jobserver is fed from the slot pool
            j, env, fds = stack.enter_context(jobserver(work, task_memory=mem))
            if not fds:
                return subprocess.run(
                    [a.replace("{j}", str(j)) for a in command], env={**os.environ, **env, **extra}
                ).returncode
            return subprocess.run(
                [a for a in command if a != "-j{j}"], env={**os.environ, **env, **extra}, pass_fds=fds
            ).returncode
        j = stack.enter_context(cores(work, task_memory=mem))
        child = subprocess.run(
            [a.replace("{j}", str(j)) for a in command], env={**os.environ, CORES_ENV: str(j), **extra}
        )
    return child.returncode


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
