# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The compile cache: one key for one content, whatever the versions are called, and nothing path-shaped shared."""

from __future__ import annotations

import fcntl
import json
import logging
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import abucache
from abucache import compile as cc
from abucache import store
from abucache.store import ContentCache, open_cache

FAKE = """#!/bin/sh
# fake {name}: records that it ran, and writes what it saw into the result
echo "{name}" >> "$COUNTER"
printf '%s|%s|%s' "{name}" "$1" "$(cat "$1")" > "$2"
{leak}
"""


def run(counter: Path):
    def runner(phase: str, argv, cwd: Path):
        env = {"COUNTER": str(counter), "PATH": "/usr/bin:/bin"}
        done = subprocess.run(list(argv), cwd=cwd, capture_output=True, env=env)
        return SimpleNamespace(exit=done.returncode, signal=None, stderr=done.stderr)
    return runner


@pytest.fixture
def toolchains(tmp_path):
    def make(name: str, leak: str = "") -> Path:
        path = tmp_path / f"cc-{name}"
        path.write_text(FAKE.format(name=name, leak=leak))
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path
    return make


START = ContentCache._start_collect  # the real one: the fixture below records in its place


@pytest.fixture(autouse=True)
def started(monkeypatch):
    """The collects a finish started, recorded in place of a detached process."""
    calls = []
    monkeypatch.setattr(
        ContentCache, "_start_collect", lambda self, installed, keep, seconds: calls.append((installed, keep))
    )
    return calls


@pytest.fixture
def cache(tmp_path):
    return ContentCache(tmp_path / "cache", 1 << 24)


def steps(tool: Path) -> list[cc.Step]:
    return [cc.Step("compile", "source.s", ("/bin/sh", str(tool), "{in}", "{out}")),
            cc.Step("assemble", "source.o", ("/bin/sh", "-c", 'cat "$0" > "$1"', "{in}", "{out}"))]


def unit(tmp_path: Path, version: str, text: str, name: str = "source.i") -> Path:
    home = tmp_path / "project" / "build" / version / "unit"
    home.mkdir(parents=True, exist_ok=True)
    (home / name).write_text(text)
    return home / name


def compiles(counter: Path) -> int:
    return len(counter.read_text().split())


@pytest.mark.parametrize("names", [["only"], ["a", "b"], ["us", "eu", "de-x", "jp", "rev1"]])
def test_identical_input_is_one_compile_for_any_number_of_versions(tmp_path, cache, toolchains, names):
    counter, tool = tmp_path / "counter", toolchains("A")
    counter.write_text("")
    tools = cc.identity([tool])
    outputs = {cc.build(cache, run(counter), steps(tool), unit(tmp_path, v, "int x;\n"), tools, tmp_path / "w")[0]
               for v in names}
    assert compiles(counter) == 1 and len(outputs) == 1


def test_one_byte_of_difference_is_another_compile(tmp_path, cache, toolchains):
    counter, tool = tmp_path / "counter", toolchains("A")
    counter.write_text("")
    tools = cc.identity([tool])
    cc.build(cache, run(counter), steps(tool), unit(tmp_path, "a", "int x;\n"), tools, tmp_path / "w")
    cc.build(cache, run(counter), steps(tool), unit(tmp_path, "b", "int y;\n"), tools, tmp_path / "w")
    assert compiles(counter) == 2


def test_the_name_of_the_input_is_part_of_the_key(tmp_path, cache, toolchains):
    counter, tool = tmp_path / "counter", toolchains("A")
    counter.write_text("")
    tools = cc.identity([tool])
    cc.build(cache, run(counter), steps(tool), unit(tmp_path, "a", "x", "one.i"), tools, tmp_path / "w")
    cc.build(cache, run(counter), steps(tool), unit(tmp_path, "b", "x", "two.i"), tools, tmp_path / "w")
    assert compiles(counter) == 2


def test_the_bytes_of_a_tool_change_the_key_and_a_path_does_not(tmp_path, cache, toolchains):
    counter = tmp_path / "counter"
    counter.write_text("")
    first, second = toolchains("A"), toolchains("B")
    (tmp_path / "copy").write_bytes(first.read_bytes())
    assert cc.identity([first]) == cc.identity([tmp_path / "copy"]) != cc.identity([second])
    assert cc.identity([first], ["pin"]) != cc.identity([first])
    source = unit(tmp_path, "a", "x")
    cc.build(cache, run(counter), steps(first), source, cc.identity([first]), tmp_path / "w")
    cc.build(cache, run(counter), steps(first), source, cc.identity([second]), tmp_path / "w")
    assert compiles(counter) == 2


def test_the_tools_see_no_path_of_the_project(tmp_path, cache, toolchains):
    counter, tool = tmp_path / "counter", toolchains("A")
    counter.write_text("")
    for version in ("a", "b"):
        source = unit(tmp_path, version, "int x;\n")
        value, _ = cc.build(cache, run(counter), steps(tool), source, "t", tmp_path / f"w-{version}")
        assert value == b"A|source.i|int x;"  # the tool saw the neutral name, whichever version it was for


def test_a_result_that_holds_a_path_of_the_run_is_never_stored(tmp_path, cache, toolchains):
    counter, tool = tmp_path / "counter", toolchains("L", leak='pwd >> "$2"')
    counter.write_text("")
    for version in ("a", "b"):
        cc.build(cache, run(counter), steps(tool), unit(tmp_path, version, "x"), "t", tmp_path / "w")
    assert compiles(counter) == 2


def test_a_failed_step_is_not_cached_and_the_caller_can_refuse(tmp_path, cache, toolchains):
    counter, tool = tmp_path / "counter", toolchains("F", leak="exit 3")
    counter.write_text("")
    for _ in range(2):
        with pytest.raises(cc.ChainFailed):
            cc.build(cache, run(counter), steps(tool), unit(tmp_path, "a", "x"), "t", tmp_path / "w")
    assert compiles(counter) == 2

    def refuse(step, result):
        raise ValueError(step.phase)
    with pytest.raises(ValueError, match="compile"):
        cc.build(cache, run(counter), steps(tool), unit(tmp_path, "a", "x"), "t", tmp_path / "w", after=refuse)


def test_trace_lists_every_step_in_order_and_the_verdict(tmp_path, cache, toolchains, caplog):
    counter, tool = tmp_path / "counter", toolchains("A")
    counter.write_text("")
    caplog.set_level(logging.INFO, logger="abucache")
    cc.build(cache, run(counter), steps(tool), unit(tmp_path, "a", "x"), "t", tmp_path / "w")
    cc.build(cache, run(counter), steps(tool), unit(tmp_path, "b", "x"), "t", tmp_path / "w")
    lines = [r.getMessage() for r in caplog.records]
    assert [line.split(":")[0] for line in lines] == ["compile cache", "compile", "assemble", "compile cache"]
    assert lines[0] == "compile cache: miss" and lines[3] == "compile cache: hit"
    assert lines[1] == f"compile: /bin/sh {tool} source.i source.s"
    assert lines[2].startswith("assemble: /bin/sh -c cat \"$0\" > \"$1\" source.s source.o")


def test_trace_flag_prints_to_stderr(capsys):
    abucache.trace(True)
    try:
        abucache.log.info("link: ld -EB")
    finally:
        abucache.trace(False)
    assert "abucache: link: ld -EB" in capsys.readouterr().err


def test_store_counts_and_overwrites(tmp_path):
    seen = []
    store = open_cache(tmp_path / "s", 1 << 20, lambda kind, hit: seen.append((kind, hit)))
    assert open_cache(tmp_path / "s", 1 << 20) is store
    assert store.cached("k", "a", lambda: b"1") == b"1" and store.cached("k", "a", lambda: b"2") == b"1"
    store.put("k", "a", b"3")
    assert store.get("k", "a") == b"3" and store.get("k", "b") is None
    assert seen == [("k", False), ("k", True)]


def test_put_many_stores_every_item_under_the_put_keys(tmp_path):
    store = open_cache(tmp_path / "m", 1 << 20)
    items = [("k", f"key{i}", bytes([i])) for i in range(50)] + [("j", "key0", b"other")]
    store.put_many(items)
    assert all(store.get(kind, key) == value for kind, key, value in items)
    store.put_many([("k", "key0", b"new")])
    assert store.get("k", "key0") == b"new" and store.get("j", "key0") == b"other"


def test_version_is_one_exact_string():
    assert abucache.__version__ == "0.7.0"


def test_failed_is_a_non_zero_exit_or_a_signal():
    assert not cc.failed(SimpleNamespace(exit=0, signal=None))
    assert cc.failed(SimpleNamespace(exit=1, signal=None))
    assert cc.failed(SimpleNamespace(exit=0, signal=9))


class TestReachability:
    """Entries never age out; only an entry no live root references is collected."""

    def make(self, tmp_path, limit=1 << 30):
        return ContentCache(tmp_path / "store", limit)

    def test_reads_write_nothing_and_never_evict(self, tmp_path):
        cache = self.make(tmp_path, limit=2000)
        for i in range(20):
            cache.put("k", str(i), b"x" * 500)
        assert all(cache.get("k", str(i)) == b"x" * 500 for i in range(20))
        query = "SELECT key, access_time, store_time FROM Cache ORDER BY key"
        before = cache._cache._sql(query).fetchall()
        for _ in range(50):
            cache.get("k", "3")
        assert cache._cache._sql(query).fetchall() == before

    def test_old_policy_dir_stops_evicting(self, tmp_path):
        import diskcache
        old = diskcache.Cache(str(tmp_path / "store"), size_limit=2000)
        assert old.eviction_policy == "least-recently-stored"
        old.close()
        cache = self.make(tmp_path, limit=2000)
        assert cache._cache.eviction_policy == "none"
        for i in range(20):
            cache.put("k", str(i), b"x" * 500)
        assert cache.get("k", "0") is not None

    def test_hot_entry_survives_and_dead_root_is_collected(self, tmp_path):
        cache = self.make(tmp_path)
        live, gone = tmp_path / "live", tmp_path / "gone"
        live.mkdir()
        gone.mkdir()
        cache.put("k", "hot", b"1")
        cache.finish(live, "code1")
        cache.put("k", "cold", b"2")
        cache.finish(gone, "code1")
        gone.rmdir()
        later = time.time() + 3 * 24 * 3600
        assert cache.collect(now=later) == 1
        assert cache.get("k", "hot") == b"1" and cache.get("k", "cold") is None
        assert cache.collect(now=later + 30 * 24 * 3600) == 0
        assert cache.get("k", "hot") == b"1"

    def test_uninstalled_identity_is_collected(self, tmp_path):
        cache = self.make(tmp_path)
        root = tmp_path / "p"
        root.mkdir()
        cache.put("k", "a", b"1")
        cache.finish(root, "old")
        cache.put("k", "b", b"2")
        cache.finish(root, "new")
        later = time.time() + 3 * 24 * 3600
        assert cache.collect(installed=lambda identity: identity == "new", now=later) == 1
        assert cache.get("k", "a") is None and cache.get("k", "b") == b"2"

    def test_recent_unreferenced_entry_is_kept_and_collect_runs_once_a_day(self, tmp_path):
        cache = self.make(tmp_path)
        root = tmp_path / "p"
        root.mkdir()
        cache.put("k", "fresh", b"1")
        assert cache.collect() == 0
        cache.finish(root, "c")
        marker = cache._cache.get("collect:last")
        cache.finish(root, "c")
        assert cache._cache.get("collect:last") == marker

    def test_workers_touched_keys_reach_the_one_manifest(self, tmp_path):
        from abucache.store import absorb_touched, forget, open_cache, take_touched
        forget()
        cache = open_cache(tmp_path / "store", 1 << 30)
        root = tmp_path / "p"
        root.mkdir()
        cache.put("k", "mine", b"1")
        taken = take_touched()
        assert take_touched() == {} and cache._touched == set()
        absorb_touched({**taken, "elsewhere": ["x"]})
        absorb_touched({str(tmp_path / "store"): ["k:worker"]})
        cache.finish(root, "c")
        manifests = cache._manifest_keys()
        assert len(manifests) == 1
        assert json.loads(cache._cache.get(manifests[0]))["keys"] == ["k:mine", "k:worker"]
        forget()


def git(path: Path, *args: str) -> str:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": "/usr/bin:/bin", "HOME": str(path)}
    done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True, env=env)
    return done.stdout.strip()


def commit(path: Path, text: str) -> str:
    (path / "f").write_text(text)
    git(path, "add", "f")
    git(path, "commit", "-qm", text)
    return git(path, "rev-parse", "HEAD^{tree}")


LIMIT = 500_000
BIG = b"x" * 40_000  # over diskcache's inline size, so a delete gives the bytes back


class TestSizeBackstop:
    """Over the limit: collect from finish, then oldest unreferenced first on a full disk, then refuse writes."""

    def make(self, tmp_path, monkeypatch, used):
        monkeypatch.setattr("abucache.store._disk_used", lambda directory: used[0])
        monkeypatch.setattr(store, "BACKSTOP_BATCH", 2)
        return ContentCache(tmp_path / "store", LIMIT)

    def fill(self, cache, count=20):
        now = time.time()
        for i in range(count):
            cache.put("k", f"{i:02d}", BIG)
        sql = "UPDATE Cache SET store_time = ? + rowid WHERE key LIKE 'k:%'"
        cache._cache._sql(sql, (now - 60,)).connection.commit()

    def test_over_the_limit_collects_from_finish_without_waiting_a_day(self, tmp_path, monkeypatch, started):
        cache = self.make(tmp_path, monkeypatch, [0.1])
        root = tmp_path / "p"
        root.mkdir()
        cache.finish(root, "c")  # first collect, stamps collect:last
        assert len(started) == 1
        calls = started
        calls.clear()
        cache.finish(root, "c")
        assert calls == []  # under the limit and a day not passed
        self.fill(cache)
        cache.finish(root, "c")
        assert calls == []  # over the limit but collected seconds ago
        cache._cache.set("collect:last", str(time.time() - 600).encode())
        cache.finish(root, "c")
        assert len(calls) == 1

    def test_disk_not_full_keeps_young_entries_and_never_refuses(self, tmp_path, monkeypatch):
        cache = self.make(tmp_path, monkeypatch, [0.5])
        self.fill(cache)
        assert cache.collect() == 0 and cache.get("k", "00") is not None
        cache.put("k", "more", b"1")

    def test_full_disk_deletes_unreferenced_oldest_first_and_spares_reachable(self, tmp_path, monkeypatch):
        cache = self.make(tmp_path, monkeypatch, [0.9])
        root = tmp_path / "p"
        root.mkdir()
        cache.put("k", "kept", b"y" * 40_000)
        cache.finish(root, "c")
        self.fill(cache)
        removed = cache.collect()
        assert removed > 0 and cache.get("k", "kept") == b"y" * 40_000
        gone = [i for i in range(20) if cache.get("k", f"{i:02d}") is None]
        assert gone == list(range(len(gone)))  # the oldest stored went first, in order
        assert cache._cache.volume() <= LIMIT * 0.9

    def test_still_over_with_everything_reachable_refuses_writes_until_a_collect_fits(self, tmp_path, monkeypatch):
        used = [0.9]
        cache = self.make(tmp_path, monkeypatch, used)
        root = tmp_path / "p"
        root.mkdir()
        self.fill(cache)
        cache.finish(root, "c")  # every key is now reachable
        cache.collect()
        for i in range(20):
            assert cache.get("k", f"{i:02d}") is not None  # reachable entries are never evicted
        with pytest.raises(store.CacheFull, match="raise cache.max_bytes"):
            cache.put("k", "new", b"1")
        with pytest.raises(store.CacheFull):
            cache.put_many([("k", "new", b"1")])
        fresh = ContentCache(tmp_path / "store", LIMIT)
        with pytest.raises(store.CacheFull):
            fresh.put("k", "new", b"1")  # another process sees the same refusal
        used[0] = 0.5
        cache.collect()
        cache.put("k", "new", b"1")
        assert cache.get("k", "new") == b"1"


class TestLiveness:
    def repo(self, tmp_path):
        root = tmp_path / "clone"
        root.mkdir()
        git(root, "init", "-q", "-b", "main")
        return root

    def test_a_later_commit_does_not_kill_an_earlier_trees_objects(self, tmp_path):
        cache = ContentCache(tmp_path / "store", 1 << 30)
        root = self.repo(tmp_path)
        trees = [commit(root, f"v{i}") for i in range(5)]
        for i, tree in enumerate(trees):
            cache.put("obj", str(i), b"o")
            cache._write_manifest(str(root), "c", tree, [f"obj:{i}"])
        later = time.time() + 7 * 24 * 3600
        assert cache.collect(now=later, keep_commits=3) == 2
        assert [cache.get("obj", str(i)) is not None for i in range(5)] == [False, False, True, True, True]

    def test_default_keeps_twenty_commits(self, tmp_path):
        cache = ContentCache(tmp_path / "store", 1 << 30)
        root = self.repo(tmp_path)
        for i in range(22):
            tree = commit(root, f"v{i}")
            cache.put("obj", str(i), b"o")
            cache._write_manifest(str(root), "c", tree, [f"obj:{i}"])
        assert cache.collect(now=time.time() + 7 * 24 * 3600) == 2

    def test_a_worktree_tree_outside_the_last_commits_stays_live(self, tmp_path):
        cache = ContentCache(tmp_path / "store", 1 << 30)
        root = self.repo(tmp_path)
        old = commit(root, "old")
        wt = tmp_path / "wt"
        git(root, "worktree", "add", "-q", "--detach", str(wt))
        for i in range(3):
            commit(root, f"new{i}")
        cache.put("obj", "old", b"o")
        cache._write_manifest(str(root), "c", old, ["obj:old"])
        cache._write_manifest(str(root), "c", git(root, "rev-parse", "HEAD^{tree}"), ["obj:head"], time.time() + 1)
        assert cache.collect(now=time.time() + 7 * 24 * 3600, keep_commits=2) == 0
        git(root, "worktree", "remove", "--force", str(wt))
        assert cache.collect(now=time.time() + 8 * 24 * 3600, keep_commits=2) == 1

    def test_work_is_one_git_scan_per_root_and_one_installed_check_per_identity(self, tmp_path, monkeypatch):
        cache = ContentCache(tmp_path / "store", 1 << 30)
        root = self.repo(tmp_path)
        trees = [commit(root, f"v{i}") for i in range(6)]
        for i, tree in enumerate(trees):
            cache._write_manifest(str(root), "c", tree, [f"obj:{i}"])
        scans, checks = [], []
        real = store._git
        monkeypatch.setattr(store, "_git", lambda *a: scans.append(a) or real(*a))
        cache.collect(installed=lambda identity: checks.append(identity) or True)
        assert len(scans) == 3 and checks == ["c"]  # log, worktree list and the one worktree's tree, for six manifests


def test_a_cache_marked_full_still_opens_and_reads(tmp_path):
    first = ContentCache(tmp_path / "store", LIMIT)
    first.put("k", "a", b"1")
    first._cache.set("collect:full", b"1")
    again = ContentCache(tmp_path / "store", LIMIT)
    assert again.get("k", "a") == b"1"
    assert again.cached("k", "a", lambda: b"never") == b"1"
    with pytest.raises(store.CacheFull):
        again.put("k", "b", b"2")


def module_level_check(identity: str) -> bool:
    return identity == "kept"


class TestCollectOffThePath:
    """A due collect starts detached and the command never waits on it; manifests come from the key index alone."""

    def test_a_due_finish_starts_a_detached_collect_and_never_collects_itself(self, tmp_path, monkeypatch, started):
        cache = ContentCache(tmp_path / "store", 1 << 30)
        monkeypatch.setattr(ContentCache, "collect", lambda *a, **k: pytest.fail("collect ran on the command path"))
        root = tmp_path / "p"
        root.mkdir()
        cache.put("k", "a", b"1")
        cache.finish(root, "c", module_level_check, 7)
        assert started == [(module_level_check, 7)]
        assert len(cache._manifest_keys()) == 1  # the manifest itself is written before the command returns

    def test_the_detached_process_gets_no_descriptor_and_its_own_session(self, tmp_path, monkeypatch):
        cache = ContentCache(tmp_path / "store", 1234)
        seen = []
        fake = SimpleNamespace(Popen=lambda argv, **kw: seen.append((argv, kw)), DEVNULL=subprocess.DEVNULL)
        monkeypatch.setattr(store, "subprocess", fake)
        START(cache, module_level_check, 5, 3600)
        (argv, kw), = seen
        assert argv == [sys.executable, "-m", "abucache.collect", str(tmp_path / "store"), "1234", "5",
                        f"{__name__}:module_level_check", "3600"]
        assert kw["start_new_session"] and kw["close_fds"] and kw["stdin"] == subprocess.DEVNULL
        assert kw["stdout"].name == str(tmp_path / "store" / "collect.log")
        assert sys.path[0] in kw["env"]["PYTHONPATH"].split(":") or not sys.path[0]

    @pytest.mark.parametrize(("check", "name"), [
        (store.everything, ""),
        (module_level_check, f"{__name__}:module_level_check"),
        (lambda identity: False, ""),
    ])
    def test_installed_travels_by_import_name(self, check, name):
        assert store._name_of(check) == name
        assert store.resolve(name) is (check if name else store.everything)

    def test_manifests_are_one_range_of_the_key_index_and_no_key_scan(self, tmp_path, monkeypatch):
        import diskcache
        cache = ContentCache(tmp_path / "store", 1 << 30)
        for i in range(50):
            cache.put("k", str(i), b"v")
        cache._write_manifest(str(tmp_path), "c", "", ["k:1"])
        plan = " ".join(str(row) for row in cache._cache._sql(
            "EXPLAIN QUERY PLAN SELECT key FROM Cache WHERE key >= ? AND key < ? AND raw = 1",
            ("manifest:", "manifest;")).fetchall())
        assert "Cache_key_raw" in plan and "SCAN" not in plan
        monkeypatch.setattr(diskcache.Cache, "iterkeys", lambda *a, **k: pytest.fail("full key scan"))
        assert len(cache._manifest_keys()) == 1
        assert cache.collect(now=time.time() + 3 * 24 * 3600) == 49

    def test_a_running_collect_makes_another_return_at_once(self, tmp_path):
        cache = ContentCache(tmp_path / "store", 1 << 30)
        cache._write_manifest(str(tmp_path / "gone"), "c", "", ["k:1"])
        with open(tmp_path / "store" / "collect.lock", "a+b") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            assert cache.collect(now=time.time() + 3 * 24 * 3600) == 0
            assert len(cache._manifest_keys()) == 1
        cache.put("k", "1", b"v")
        cache._cache._sql("UPDATE Cache SET store_time = 0 WHERE key = 'k:1'").connection.commit()
        assert cache.collect(now=time.time() + 3 * 24 * 3600) == 1 and cache._manifest_keys() == []

    def test_a_row_stored_again_after_the_scan_stays(self, tmp_path):
        cache = ContentCache(tmp_path / "store", 1 << 30)
        cache.put("k", "a", BIG)
        cache.put("k", "b", BIG)
        rows = cache._cache._sql("SELECT rowid, store_time FROM Cache ORDER BY key").fetchall()
        cache._cache._sql("UPDATE Cache SET store_time = store_time + 1 WHERE key = 'k:b'").connection.commit()
        assert cache._delete(rows) == 1
        assert cache.get("k", "a") is None and cache.get("k", "b") == BIG
        assert sum(1 for f in (tmp_path / "store").rglob("*.val")) == 1  # the value file went with its row

    def test_the_collect_module_resolves_installed_and_collects(self, tmp_path, capsys):
        from abucache import collect
        cache = ContentCache(tmp_path / "store", 1 << 30)
        root = tmp_path / "p"
        root.mkdir()
        cache.put("k", "old", b"1")
        cache.put("k", "new", b"2")
        cache._write_manifest(str(root), "gone", "", ["k:old"], time.time() - 10)
        cache._write_manifest(str(root), "kept", "", ["k:new"])
        cache._cache._sql("UPDATE Cache SET store_time = 0 WHERE key LIKE 'k:%'").connection.commit()
        argv = [str(tmp_path / "store"), str(1 << 30), "20", f"{__name__}:module_level_check"]
        assert collect.main(argv) == 0
        assert "collected 1 entries" in capsys.readouterr().out
        assert cache.get("k", "old") is None and cache.get("k", "new") == b"2"
        assert collect.main([]) == 2
