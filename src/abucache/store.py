# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The content cache every cached step of a project shares: kind:key to bytes, on disk, size limited, lock free."""

from __future__ import annotations

import fcntl
import importlib
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import time
import uuid
import zlib
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import diskcache

log = logging.getLogger("abucache")

_MANIFEST = "manifest:"
_MANIFEST_END = "manifest;"  # the first key past every manifest key: the key index finds them as one range
_LAST_COLLECT = "collect:last"
_FULL = "collect:full"
COLLECT_EVERY = 24 * 3600
OVER_LIMIT_EVERY = 300  # while over size_limit a command's end collects again after this long, not on every command
GRACE = 6 * 3600  # the longest a command may run before it writes its manifest: younger unreferenced entries are kept
DISK_FULL = 0.8  # fraction of the disk used above which the size backstop acts
KEEP_COMMITS = 20
KEEP_SECONDS = 7 * 24 * 3600  # a root without git history keeps its entries live this long (runtime cache.keep_seconds)
BATCH = 10_000
WORKERS = os.cpu_count() or 1
BACKSTOP_BATCH = 500  # the backstop checks the volume after every batch, so it stops close to its target
TARGET = 0.9  # the backstop deletes down to this fraction of size_limit


class CacheFull(RuntimeError):
    """The cache is over its limit, the disk is nearly full and every entry left is reachable."""


def _disk_used(directory: Path) -> float:
    """The used fraction of the disk the directory is on."""
    usage = shutil.disk_usage(directory)
    return usage.used / usage.total


def _ignore(kind: str, hit: bool) -> None:
    return None


def everything(identity: str) -> bool:
    """The default `installed`: every code identity still exists."""
    return True


def _name_of(installed: Callable[[str], bool]) -> str:
    """The import name module:qualname of `installed`, so a separate process can call it; empty for the default.
    A callable no other process can import is treated as the default there, which only keeps more."""
    if installed is everything:
        return ""
    module, qualname = getattr(installed, "__module__", ""), getattr(installed, "__qualname__", "")
    if module and qualname and "<" not in qualname:
        return f"{module}:{qualname}"
    log.warning("installed check %r has no import name: the collect keeps every code identity", installed)
    return ""


def resolve(name: str) -> Callable[[str], bool]:
    """The callable an import name of `_name_of` names."""
    if not name:
        return everything
    module, _, qualname = name.partition(":")
    found = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
    return found


class ContentCache:
    def __init__(self, directory: Path, size_limit: int, count: Callable[[str, bool], None] = _ignore) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        # Nothing ages out by size or recency: "none" is stored in an old directory's settings in place, and a read
        # then performs no write at all. size_limit is only the alarm threshold of a collect.
        self._cache = diskcache.Cache(str(directory), size_limit=size_limit, eviction_policy="none")
        self._size_limit = size_limit
        self._count = count
        self._touched: set[str] = set()
        self._full: bool | None = None  # whether the last collect left the cache refusing writes; read once per process

    def cached(self, kind: str, key: str, produce: Callable[[], bytes]) -> bytes:
        """The bytes for kind:key. A miss produces without any lock (production may fan out to a pool) and stores
        atomically, so concurrent producers of one key at worst duplicate work."""
        value = self.get(kind, key)
        hit = value is not None
        if not hit:
            value = produce()
            self.put(kind, key, value)
        self._count(kind, hit)
        return value

    def count(self, kind: str, hit: bool) -> None:
        self._count(kind, hit)

    def get(self, kind: str, key: str) -> bytes | None:
        """Read kind:key without producing or counting."""
        name = f"{kind}:{key}"
        self._touched.add(name)
        return self._cache.get(name)

    def put(self, kind: str, key: str, value: bytes) -> None:
        """Overwrite the entry for kind:key without a hit or miss count."""
        self._refuse_when_full()
        name = f"{kind}:{key}"
        self._touched.add(name)
        self._cache.set(name, value)

    def _refuse_when_full(self) -> None:
        if self._full is None:
            self._full = self._cache.get(_FULL) is not None
        if self._full:
            raise CacheFull(
                f"cache {self.directory} is over its {self._size_limit} byte limit, the disk is above "
                f"{DISK_FULL:.0%} used and every entry left is reachable: raise cache.max_bytes or free the disk"
            )

    def put_many(self, items: Iterable[tuple[str, str, bytes]]) -> None:
        """Overwrite many (kind, key, value) entries in one write transaction, as `put` would one by one."""
        self._refuse_when_full()
        with self._cache.transact():
            for kind, key, value in items:
                self.put(kind, key, value)

    def finish(
        self,
        root: Path,
        identity: str,
        installed: Callable[[str], bool] = everything,
        keep_commits: int = KEEP_COMMITS,
        keep_seconds: int = KEEP_SECONDS,
    ) -> None:
        """End of a command: record the keys it read or wrote as one manifest for this project root, code identity and
        tree digest. When the last collect is a day old, or the cache is over its limit and the last collect is a few
        minutes old, a collect starts in its own detached process: the command returns at once and never waits on it.
        `installed` says whether a code identity still exists anywhere; it must be importable by module:qualname."""
        if self._touched:
            keys, self._touched = sorted(self._touched), set()
            self._write_manifest(str(root), identity, _tree(root), keys)
        now = time.time()
        last = self._cache.get(_LAST_COLLECT)
        since = float("inf") if last is None else now - float(last)
        if since >= COLLECT_EVERY or (since >= OVER_LIMIT_EVERY and self._cache.volume() > self._size_limit):
            self._cache.set(_LAST_COLLECT, str(now).encode())
            self._start_collect(installed, keep_commits, keep_seconds)

    def _start_collect(self, installed: Callable[[str], bool], keep_commits: int, keep_seconds: int) -> None:
        """Run `python -m abucache.collect` for this cache in a new session, its output in collect.log beside the
        cache. No descriptor of this process reaches it, so neither a pipe reading this command nor a lock it holds
        waits on the collect."""
        argv = [sys.executable, "-m", "abucache.collect", str(self.directory), str(self._size_limit),
                str(keep_commits), _name_of(installed),
                str(keep_seconds)]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        with open(self.directory / "collect.log", "wb") as out:
            subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=out, env=env, close_fds=True,
                             start_new_session=True)

    def _write_manifest(self, root: str, identity: str, tree: str, keys: list[str], at: float | None = None) -> None:
        body = {"root": root, "identity": identity, "tree": tree, "at": at or time.time(), "keys": keys}
        self._cache.set(_MANIFEST + uuid.uuid4().hex, json.dumps(body).encode())

    def _manifest_keys(self) -> list[str]:
        """Every manifest key, found as one range of the cache's key index: no scan of the other keys."""
        query = "SELECT key FROM Cache WHERE key >= ? AND key < ? AND raw = 1"
        return [key for (key,) in self._cache._sql(query, (_MANIFEST, _MANIFEST_END)).fetchall()]

    def collect(
        self,
        installed: Callable[[str], bool] = everything,
        now: float | None = None,
        keep_commits: int = KEEP_COMMITS,
        keep_seconds: int = KEEP_SECONDS,
    ) -> int:
        """Delete every entry no live manifest references and return how many. A manifest is live while its root
        directory exists and its code identity is installed, and for a root the trees that count are the HEAD tree
        of the root and of each worktree of its repository, the trees of the last `keep_commits` commits of its HEAD,
        and the newest tree it recorded. A root with no git tree has no trees to count: its manifests are live while
        younger than `keep_seconds`, and its newest one is always live. Entries stored within the grace period are
        kept, as their command may not
        have finished. When that leaves the cache over its limit on a disk above DISK_FULL used, unreferenced entries
        go oldest first whatever their age, and when it is still over, writes are refused until a collect fits.
        One collect runs per cache at a time: another that finds one running returns 0 at once."""
        with open(self.directory / "collect.lock", "a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            return self._collect(installed, now or time.time(), keep_commits, keep_seconds)

    def _collect(self, installed: Callable[[str], bool], now: float, keep_commits: int, keep_seconds: int) -> int:
        manifests: dict[str, dict] = {}
        for name in self._manifest_keys():
            raw = self._cache.get(name)
            if raw is not None:
                manifests[name] = json.loads(raw)
        newest: dict[tuple[str, str], tuple[float, str, str]] = {}
        for name, body in manifests.items():
            group = (body["root"], body["identity"])
            newest[group] = max(newest.get(group, (0.0, "", "")), (body["at"], body["tree"], name))
        roots = sorted({body["root"] for body in manifests.values() if Path(body["root"]).is_dir()})
        with ThreadPoolExecutor(WORKERS) as pool:  # each root is a few git processes
            trees = dict(zip(roots, pool.map(lambda root: _live_trees(Path(root), keep_commits), roots), strict=True))
        alive_code: dict[str, bool] = {}
        live: set[str] = set()
        merged: dict[tuple[str, str, str], dict] = {}
        aged: set[str] = set()  # live manifests of roots without a git tree: kept as written, each has its own age
        for name, body in manifests.items():
            root, identity, tree = body["root"], body["identity"], body["tree"]
            if root not in trees:
                continue
            if identity not in alive_code:
                alive_code[identity] = installed(identity)
            latest = newest[(root, identity)]
            if tree:  # a git tree: one of the live trees of the root, or the newest tree it recorded
                kept = tree in trees[root] or tree == latest[1]
            else:  # no git tree to name: young, or the newest manifest of the group
                kept = now - body["at"] < keep_seconds or name == latest[2]
            if alive_code[identity] and kept:
                live.update(body["keys"])
                if not tree:
                    aged.add(name)
                    continue
                into = merged.setdefault((root, identity, tree), {**body, "keys": []})
                into["keys"] = sorted(set(into["keys"]) | set(body["keys"]))
                into["at"] = max(into["at"], body["at"])
        rows = self._cache._sql("SELECT rowid, key, store_time FROM Cache").fetchall()
        unreferenced = [
            (stored, rowid)
            for rowid, key, stored in rows
            if isinstance(key, str) and not key.startswith((_MANIFEST, _LAST_COLLECT, _FULL)) and key not in live
        ]
        with self._cache.transact():
            for name in manifests.keys() - aged:
                self._cache.delete(name)
            for body in merged.values():
                self._write_manifest(body["root"], body["identity"], body["tree"], body["keys"], body["at"])
        count = self._delete([(rowid, stored) for stored, rowid in unreferenced if now - stored >= GRACE])
        if self._cache.volume() > self._size_limit and _disk_used(self.directory) > DISK_FULL:
            young = sorted((stored, rowid) for stored, rowid in unreferenced if now - stored < GRACE)
            for start in range(0, len(young), BACKSTOP_BATCH):
                if self._cache.volume() <= self._size_limit * TARGET:
                    break
                count += self._delete([(rowid, stored) for stored, rowid in young[start : start + BACKSTOP_BATCH]])
        over = self._cache.volume() > self._size_limit
        full = over and _disk_used(self.directory) > DISK_FULL
        if full:
            self._cache.set(_FULL, b"1")
        else:
            self._cache.delete(_FULL)
        self._full = full
        if over:
            refusing = " and refuses writes" if full else ""
            log.warning("cache %s is over its %d byte limit%s", self.directory, self._size_limit, refusing)
        return count

    def _delete(self, rows: list[tuple[int, float]]) -> int:
        """Delete the (rowid, store_time) rows in batches, so one sweep never holds the writer for long, and return how
        many went. A row stored again since the scan has a new store_time and stays. Value files go after each batch
        commits, on every core."""
        count = 0
        with ThreadPoolExecutor(WORKERS) as pool:
            for start in range(0, len(rows), BATCH):
                batch = rows[start : start + BATCH]
                files: list[str] = []
                with self._cache.transact():
                    for rowid, stored in batch:
                        found = self._cache._sql(
                            "SELECT filename FROM Cache WHERE rowid = ? AND store_time = ?", (rowid, stored)
                        ).fetchall()
                        if found:
                            self._cache._sql("DELETE FROM Cache WHERE rowid = ?", (rowid,))
                            count += 1
                            files.extend(name for (name,) in found if name)
                list(pool.map(self._cache._disk.remove, files))
        return count

def _git(root: Path, *args: str) -> list[str]:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return done.stdout.splitlines() if done.returncode == 0 else []


def _live_trees(root: Path, keep_commits: int) -> set[str]:
    """The trees of a root worth keeping: the trees of the last `keep_commits` commits of its HEAD, and the HEAD tree of
    every worktree of its repository."""
    found = set(_git(root, "log", f"-{keep_commits}", "--format=%T", "HEAD"))
    for line in _git(root, "worktree", "list", "--porcelain"):
        if line.startswith("worktree ") and Path(line[9:]).is_dir():
            found.update(_git(Path(line[9:]), "rev-parse", "HEAD^{tree}"))
    return found


def _tree(root: Path) -> str:
    """The tree digest of the root's HEAD, or empty when it is not a git checkout."""
    done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


_OPEN: dict[str, ContentCache] = {}


def forget() -> None:
    """Close the open caches of this process, so the next open reads the directory again."""
    _OPEN.clear()


def open_cache(directory: Path, size_limit: int, count: Callable[[str, bool], None] = _ignore) -> ContentCache:
    """The one cache of this directory in this process."""
    if str(directory) not in _OPEN:
        _OPEN[str(directory)] = ContentCache(directory, size_limit, count)
    return _OPEN[str(directory)]


def take_touched() -> dict[str, list[str]]:
    """Drain the keys every open cache of this process read or wrote, by directory: what a pool worker sends back."""
    taken = {}
    for directory, cache in _OPEN.items():
        keys, cache._touched = sorted(cache._touched), set()
        if keys:
            taken[directory] = keys
    return taken


def absorb_touched(taken: dict[str, list[str]]) -> None:
    """Add keys a worker touched to the open caches of this process, so the one manifest of a command holds them."""
    for directory, keys in taken.items():
        if directory in _OPEN:
            _OPEN[directory]._touched.update(keys)


_PACKED = b"\x02"  # format tag of a stored value: a value without it (an older entry) reads as a miss


def _frame(value: bytes) -> bytes:
    """The stored form of bytes: the format tag, the crc32 of the bytes, then the bytes zlib level 1 compressed."""
    return _PACKED + zlib.crc32(value).to_bytes(4, "big") + zlib.compress(value, 1)


def _unframe(blob: bytes | None) -> bytes | None:
    """The bytes `_frame` packed, or None for a missing, foreign, torn or older-format blob."""
    if not blob or blob[:1] != _PACKED or len(blob) < 5:
        return None
    try:
        value = zlib.decompress(blob[5:])
    except zlib.error:
        return None
    return value if zlib.crc32(value).to_bytes(4, "big") == blob[1:5] else None


def pack(obj: object) -> bytes:
    """The stored form of an object: pickle protocol 5, framed (tag, crc32, zlib level 1). Entries older unbake
    versions wrote with their `Packed` cache have this exact form, so they read back through `unpack`."""
    return _frame(pickle.dumps(obj, protocol=5))


def unpack(blob: bytes | None) -> tuple[bool, object]:
    """(True, the object) of a `pack` blob; (False, None) when it is absent, torn, of another format, or a pickle
    this build cannot read: such an entry is a miss, never a crash."""
    raw = _unframe(blob)
    if raw is None:
        return False, None
    try:
        return True, pickle.loads(raw)
    except Exception:
        return False, None


T = TypeVar("T")


class Namespace:
    """One tool's key space of one kind in the host cache: its entries are `"<tool>.<kind>:<key>"`, so two tools (or two
    kinds) never read each other's keys however they name them. A key must name every input of the value: nothing here
    adds the code's identity. Values are stored framed and compressed; a write the full cache refuses raises
    `CacheFull`."""

    def __init__(self, cache: ContentCache, tool: str, kind: str) -> None:
        self.cache, self.kind = cache, f"{tool}.{kind}"

    def get(self, key: str) -> bytes | None:
        """The value of `key`, or None; reads count nothing."""
        return _unframe(self.cache.get(self.kind, key))

    def put(self, key: str, value: bytes) -> None:
        self.cache.put(self.kind, key, _frame(value))

    def put_many(self, rows: Iterable[tuple[str, bytes]]) -> None:
        """Entries in one write transaction: tens of thousands of single writes queue on the cache's one writer."""
        self.cache.put_many((self.kind, key, _frame(value)) for key, value in rows)

    def cached(self, key: str, produce: Callable[[], bytes]) -> bytes:
        """The value of `key`; a miss produces without any lock and stores it, counted as a hit or a miss."""
        value = self.get(key)
        hit = value is not None
        if value is None:
            value = produce()
            self.put(key, value)
        self.cache.count(self.kind, hit)
        return value

    def cached_object(self, key: str, produce: Callable[[], T]) -> T:
        """`cached` for any picklable object: an entry this build cannot unpickle is produced again."""
        found, value = unpack(self.cache.get(self.kind, key))
        self.cache.count(self.kind, found)
        if found:
            return value  # type: ignore[return-value]
        value = produce()
        self.cache.put(self.kind, key, pack(value))
        return value
