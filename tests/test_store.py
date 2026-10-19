# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Store namespaces, the packed value format and the liveness of manifests of roots without git history."""

from __future__ import annotations

import pickle
import time
import zlib
from pathlib import Path

import pytest

from abucache import store
from abucache.store import ContentCache, Namespace, pack, unpack

FIXTURE = Path(__file__).parent / "fixtures" / "unbake_0_5_23_packed.bin"
OBJECT = {"unit": ("a.c", 3, [1.5, None]), "names": frozenset({"x", "y"}), "blob": b"\x00\x01" * 40}


@pytest.fixture(autouse=True)
def no_detached_collect(monkeypatch):
    monkeypatch.setattr(ContentCache, "_start_collect", lambda *a: None)


@pytest.fixture
def cache(tmp_path):
    return ContentCache(tmp_path / "store", 1 << 30)


def test_a_blob_unbake_0_5_23_wrote_reads_back_so_the_rig_cache_stays_warm():
    # captured from unbake 0.5.23: its Packed cache holding pickle.dumps(OBJECT, protocol=5) under kind "unit"
    blob = FIXTURE.read_bytes()
    assert blob[:1] == b"\x02"
    assert unpack(blob) == (True, OBJECT)
    # what this writes is byte for byte what unbake wrote: the frame of the same pickle is the same blob (the pickle
    # itself is not compared: a frozenset pickles in the hash order of its process)
    assert store._frame(zlib.decompress(blob[5:])) == blob
    assert unpack(pack(OBJECT)) == (True, OBJECT)


def test_the_packed_format_is_a_tag_a_crc_and_zlib_level_one_over_pickle_five():
    raw = pickle.dumps([1, 2, 3], protocol=5)
    assert pack([1, 2, 3]) == b"\x02" + zlib.crc32(raw).to_bytes(4, "big") + zlib.compress(raw, 1)


@pytest.mark.parametrize(
    "blob",
    [
        None,
        b"",
        b"\x02",
        b"\x01abcdef",
        b"\x02\x00\x00\x00\x00notzlib",
        pack(1)[:-3],
        b"\x02" + bytes(4) + zlib.compress(b"x"),
    ],
)
def test_an_absent_torn_or_foreign_blob_is_a_miss(blob):
    assert unpack(blob) == (False, None)


def test_a_valid_frame_around_a_pickle_this_build_cannot_read_is_a_miss():
    raw = b"not a pickle"
    blob = b"\x02" + zlib.crc32(raw).to_bytes(4, "big") + zlib.compress(raw, 1)
    assert unpack(blob) == (False, None)


class TestNamespace:
    def test_keys_are_tool_dot_kind_colon_key_and_two_tools_never_meet(self, cache):
        one, other = Namespace(cache, "recipe", "draft"), Namespace(cache, "unbake", "draft")
        one.put("k", b"mine")
        assert cache.get("recipe.draft", "k") is not None
        assert other.get("k") is None and Namespace(cache, "recipe", "scan").get("k") is None
        assert one.get("k") == b"mine"

    def test_put_many_is_one_transaction_with_the_put_keys(self, cache):
        space = Namespace(cache, "t", "k")
        space.put_many((f"k{i}", bytes([i]) * 100) for i in range(50))
        assert all(space.get(f"k{i}") == bytes([i]) * 100 for i in range(50))
        space.put_many([("k0", b"new")])
        assert space.get("k0") == b"new" and space.get("k1") == b"\x01" * 100

    def test_cached_produces_once_and_counts_each_lookup(self, tmp_path):
        seen = []
        cache = ContentCache(tmp_path / "c", 1 << 20, lambda kind, hit: seen.append((kind, hit)))
        space = Namespace(cache, "t", "k")
        assert space.cached("a", lambda: b"1") == b"1"
        assert space.cached("a", lambda: pytest.fail("produced twice")) == b"1"
        assert seen == [("t.k", False), ("t.k", True)]

    def test_cached_object_round_trips_and_an_unreadable_entry_is_produced_again(self, tmp_path):
        seen = []
        cache = ContentCache(tmp_path / "c", 1 << 20, lambda kind, hit: seen.append(hit))
        space = Namespace(cache, "t", "obj")
        assert space.cached_object("a", lambda: OBJECT) == OBJECT
        assert space.cached_object("a", lambda: pytest.fail("produced twice")) == OBJECT
        cache.put("t.obj", "bad", b"raw bytes of another build")
        assert space.cached_object("bad", lambda: [7]) == [7]
        assert space.cached_object("bad", lambda: pytest.fail("produced twice")) == [7]
        assert seen == [False, True, False, True]

    def test_cached_object_reads_what_unbake_wrote_under_the_same_key(self, cache):
        cache.put("t.obj", "old", FIXTURE.read_bytes())
        assert Namespace(cache, "t", "obj").cached_object("old", lambda: pytest.fail("a warm entry")) == OBJECT

    def test_a_full_cache_refuses_the_write_and_still_reads(self, cache):
        space = Namespace(cache, "t", "k")
        space.put("a", b"1")
        cache._cache.set("collect:full", b"1")
        cache._full = None
        with pytest.raises(store.CacheFull):
            space.put("b", b"2")
        assert space.get("a") == b"1"

    def test_the_manifest_of_a_command_holds_the_namespaced_keys(self, cache, tmp_path):
        root = tmp_path / "p"
        root.mkdir()
        Namespace(cache, "t", "k").put("a", b"1")
        cache.finish(root, "c")
        (body,) = [__import__("json").loads(cache._cache.get(name)) for name in cache._manifest_keys()]
        assert body["keys"] == ["t.k:a"]


class TestRootsWithoutGit:
    """A root with no git tree has no commits to keep: its manifests are live while young, and its newest always."""

    def make(self, tmp_path):
        cache = ContentCache(tmp_path / "store", 1 << 30)
        root = tmp_path / "plain"
        root.mkdir()
        return cache, root

    def stored(self, cache, root, key, at):
        cache.put("k", key, b"v")
        cache._write_manifest(str(root), "c", "", [f"k:{key}"], at)
        cache._cache._sql("UPDATE Cache SET store_time = 0 WHERE key LIKE 'k:%'").connection.commit()

    def test_an_old_manifest_is_collected_and_the_newest_stays(self, tmp_path):
        cache, root = self.make(tmp_path)
        now = time.time()
        self.stored(cache, root, "old", now - 10_000)
        self.stored(cache, root, "newest", now - 5_000)
        assert cache.collect(now=now, keep_seconds=3600) == 1
        assert cache.get("k", "old") is None and cache.get("k", "newest") == b"v"

    def test_a_manifest_younger_than_keep_seconds_stays_beside_the_newest(self, tmp_path):
        cache, root = self.make(tmp_path)
        now = time.time()
        self.stored(cache, root, "a", now - 1000)
        self.stored(cache, root, "b", now - 500)
        self.stored(cache, root, "c", now - 100)
        assert cache.collect(now=now, keep_seconds=3600) == 0
        assert cache.collect(now=now, keep_seconds=700) == 1 and cache.get("k", "a") is None
        assert cache.collect(now=now, keep_seconds=1) == 1 and cache.get("k", "b") is None
        assert cache.get("k", "c") == b"v"  # the newest is live whatever its age

    def test_a_due_finish_hands_keep_seconds_to_the_detached_collect(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(ContentCache, "_start_collect", lambda self, *a: calls.append(a))
        cache, root = self.make(tmp_path)
        cache.put("k", "a", b"1")
        cache.finish(root, "c", keep_seconds=99)
        assert calls == [(store.everything, store.KEEP_COMMITS, 99)]

    def test_the_collect_module_takes_keep_seconds(self, tmp_path, capsys):
        from abucache import collect

        cache, root = self.make(tmp_path)
        now = time.time()
        self.stored(cache, root, "old", now - 10_000)
        self.stored(cache, root, "newest", now - 5_000)
        assert collect.main([str(tmp_path / "store"), str(1 << 30), "20", "", "3600"]) == 0
        assert "collected 1 entries" in capsys.readouterr().out
