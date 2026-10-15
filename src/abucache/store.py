# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The content cache every cached step of a project shares: kind:key to bytes, on disk, size limited, lock free."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

import diskcache

log = logging.getLogger("abucache")

_MANIFEST = "manifest:"
_LAST_COLLECT = "collect:last"
_FULL = "collect:full"
COLLECT_EVERY = 24 * 3600
OVER_LIMIT_EVERY = 300  # while over size_limit a command's end collects again after this long, not on every command
GRACE = 6 * 3600  # the longest a command may run before it writes its manifest: younger unreferenced entries are kept
DISK_FULL = 0.8  # fraction of the disk used above which the size backstop acts
KEEP_COMMITS = 20
BATCH = 10_000
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
        installed: Callable[[str], bool] = lambda identity: True,
        keep_commits: int = KEEP_COMMITS,
    ) -> None:
        """End of a command: record the keys it read or wrote as one manifest for this project root, code identity and
        tree digest, then collect when the last collect is a day old, or when the cache is over its limit and the last
        collect is a few minutes old. `installed` says whether a code identity still exists anywhere."""
        if self._touched:
            keys, self._touched = sorted(self._touched), set()
            self._write_manifest(str(root), identity, _tree(root), keys)
        now = time.time()
        last = self._cache.get(_LAST_COLLECT)
        since = float("inf") if last is None else now - float(last)
        if since >= COLLECT_EVERY or (since >= OVER_LIMIT_EVERY and self._cache.volume() > self._size_limit):
            self._cache.set(_LAST_COLLECT, str(now).encode())
            self.collect(installed, now, keep_commits)

    def _write_manifest(self, root: str, identity: str, tree: str, keys: list[str], at: float | None = None) -> None:
        body = {"root": root, "identity": identity, "tree": tree, "at": at or time.time(), "keys": keys}
        self._cache.set(_MANIFEST + uuid.uuid4().hex, json.dumps(body).encode())

    def collect(
        self,
        installed: Callable[[str], bool] = lambda identity: True,
        now: float | None = None,
        keep_commits: int = KEEP_COMMITS,
    ) -> int:
        """Delete every entry no live manifest references and return how many. A manifest is live while its root
        directory exists and its code identity is installed, and for a root the trees that count are the HEAD tree
        of the root and of each worktree of its repository, the trees of the last `keep_commits` commits of its HEAD,
        and the newest tree it recorded. Entries stored within the grace period are kept, as their command may not
        have finished. When that leaves the cache over its limit on a disk above DISK_FULL used, unreferenced entries
        go oldest first whatever their age, and when it is still over, writes are refused until a collect fits."""
        now = now or time.time()
        manifests: dict[str, dict] = {}
        for name in list(self._cache.iterkeys()):
            if isinstance(name, str) and name.startswith(_MANIFEST):
                raw = self._cache.get(name)
                if raw is not None:
                    manifests[name] = json.loads(raw)
        newest: dict[tuple[str, str], tuple[float, str]] = {}
        for body in manifests.values():
            group = (body["root"], body["identity"])
            newest[group] = max(newest.get(group, (0.0, "")), (body["at"], body["tree"]))
        trees: dict[str, set[str]] = {}
        alive_code: dict[str, bool] = {}
        live: set[str] = set()
        merged: dict[tuple[str, str, str], dict] = {}
        for body in manifests.values():
            root, identity, tree = body["root"], body["identity"], body["tree"]
            if root not in trees:
                trees[root] = _live_trees(Path(root), keep_commits) if Path(root).is_dir() else set()
            if identity not in alive_code:
                alive_code[identity] = installed(identity)
            current = tree in trees[root] or tree == newest[(root, identity)][1]
            if Path(root).is_dir() and alive_code[identity] and current:
                live.update(body["keys"])
                into = merged.setdefault((root, identity, tree), {**body, "keys": []})
                into["keys"] = sorted(set(into["keys"]) | set(body["keys"]))
                into["at"] = max(into["at"], body["at"])
        rows = self._cache._sql("SELECT key, store_time FROM Cache").fetchall()
        unreferenced = [
            (stored, key)
            for key, stored in rows
            if isinstance(key, str) and not key.startswith((_MANIFEST, _LAST_COLLECT, _FULL)) and key not in live
        ]
        with self._cache.transact():
            for name in manifests:
                self._cache.delete(name)
            for body in merged.values():
                self._write_manifest(body["root"], body["identity"], body["tree"], body["keys"], body["at"])
        count = self._delete([key for stored, key in unreferenced if now - stored >= GRACE])
        if self._cache.volume() > self._size_limit and _disk_used(self.directory) > DISK_FULL:
            young = sorted((stored, key) for stored, key in unreferenced if now - stored < GRACE)
            for start in range(0, len(young), BACKSTOP_BATCH):
                if self._cache.volume() <= self._size_limit * TARGET:
                    break
                count += self._delete([key for _, key in young[start : start + BACKSTOP_BATCH]])
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

    def _delete(self, keys: list[str]) -> int:
        """Delete the keys in batches, so one sweep never holds the writer for long."""
        for start in range(0, len(keys), BATCH):
            with self._cache.transact():
                for key in keys[start : start + BATCH]:
                    self._cache.delete(key)
        return len(keys)


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
