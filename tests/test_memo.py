# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The command's memo, its forget epochs, fork trimming and memory pressure."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from abucache import memo, stages


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(memo, "_memo", {})
    monkeypatch.setattr(memo, "_inherited", frozenset())
    monkeypatch.setattr(memo, "_taggers", [])
    monkeypatch.setattr(memo, "_fork_data", -1)
    monkeypatch.setattr(stages, "parent_memory", 0)


def test_memo_computes_once_peeks_and_revalidates():
    calls = []
    assert memo.memo(("a", 1), lambda: calls.append(1) or 5) == 5
    assert memo.memo(("a", 1), lambda: calls.append(1) or 6) == 5 and calls == [1]
    assert memo.peek(("a", 1)) == 5 and memo.peek(("a", 2)) is None
    assert memo.memo_valid(("a", 1), lambda: 7, lambda v: v > 5) == 7
    assert memo.memo_valid(("a", 1), lambda: 8, lambda v: v > 5) == 7


def test_a_command_starts_with_an_empty_memo(tmp_path):
    memo.memo(("a", 1), lambda: 1)
    with stages.command("tool", "check", [], tmp_path / "stages.jsonl"):
        assert memo.peek(("a", 1)) is None


def test_forget_drops_only_the_named_kinds():
    memo.memo(("a", 1), lambda: 1)
    memo.memo(("b", 1), lambda: 2)
    memo.forget("a")
    assert memo.memo(("a", 1), lambda: 3) == 3 and memo.memo(("b", 1), lambda: 4) == 2
    assert memo.epochs() == {"a": memo.epochs()["a"]} or "a" in memo.epochs()


def test_a_warm_worker_follows_the_parents_forgets_once():
    memo.memo(("a", 1), lambda: 1)
    memo.follow({"a": memo.epochs().get("a", 0) + 1})
    assert memo.peek(("a", 1)) is None
    memo.memo(("a", 1), lambda: 2)
    memo.follow({"a": memo.epochs().get("a", 0)})  # the same count again: nothing more to forget
    assert memo.peek(("a", 1)) == 2


def test_held_and_adopt_move_content_keyed_kinds_to_a_worker():
    memo.memo(("index", "h1"), lambda: "x")
    memo.memo(("other", "h1"), lambda: "y")
    entries = memo.held(["index"])
    assert entries == {("index", "h1"): "x"}
    memo._memo.clear()
    memo.memo(("index", "old"), lambda: "stale")
    memo.adopt(entries, ["index"])
    assert set(memo._memo) == {("index", "h1")}


def test_release_drops_every_value_keyed_by_a_tag_and_every_value_that_names_it():
    laid = SimpleNamespace(digest="wave1")
    memo.register_tagger(lambda value: getattr(value, "digest", None))
    memo._memo.update(
        {
            ("parent", "wave1"): 1,
            ("pool.writes", "wave1"): b"blob",
            ("pin", "wave1", "a.h"): "x",
            ("overlay", "base", "w"): laid,  # keyed by its base: found by its tag
            ("pin", "wave2", "a.h"): "y",
            ("catalog", "/root", "k"): 2,
        }
    )
    assert memo.release("wave1") == 4
    assert set(memo._memo) == {("pin", "wave2", "a.h"), ("catalog", "/root", "k")}


def test_a_worker_trims_to_what_it_was_forked_holding_but_keeps_content_keyed_kinds():
    memo._memo.update({("catalog", "base"): 1})
    memo.mark_fork(0)
    try:
        memo._memo.update({("catalog", "wave"): 2, ("words", "h"): 3, ("spellings", "wave"): 4})
        assert memo.trim_to_fork(keep_kinds=("words",)) == 2
        assert set(memo._memo) == {("catalog", "base"), ("words", "h")}
    finally:
        memo.mark_fork(-1)


def test_a_worker_keeps_what_it_built_for_a_tag_it_still_holds():
    memo.mark_fork(0)
    try:
        memo._memo.update({("overlay", "wave1"): 1, ("overlay", "wave2"): 2})
        assert memo.trim_to_fork(tags=("wave1",)) == 1
        assert set(memo._memo) == {("overlay", "wave1")}
    finally:
        memo.mark_fork(-1)


def _pressure(monkeypatch, resident):
    monkeypatch.setattr(memo, "pressure", lambda: (resident, 1000))


@pytest.mark.parametrize(("resident", "kept"), [(400, True), (600, False)])
def test_scoped_drops_only_what_the_scope_added_and_only_under_pressure(monkeypatch, resident, kept):
    _pressure(monkeypatch, resident)
    memo._memo[("base", 1)] = 1
    with memo.scoped():
        memo.memo(("overlay", 1), lambda: 2)
    assert (("overlay", 1) in memo._memo) is kept
    assert ("base", 1) in memo._memo


def test_scoped_records_the_trim_count(monkeypatch, tmp_path):
    _pressure(monkeypatch, 900)
    with stages.command("tool", "t", [], tmp_path / "stages.jsonl"):
        with memo.scoped():
            memo.memo(("a", 1), lambda: 1)
            memo.memo(("a", 2), lambda: 2)
        trimmed = [r for r in stages.closed() if r.path[-1] == "effort.trimmed"]
    assert [r.items for r in trimmed] == [2] and not trimmed[0].findings


def _mock_data(monkeypatch, vmdata_kib, soft, fork):
    reads = []

    def status():
        reads.append("status")
        return f"Name:\tx\nRssAnon:\t500 kB\nVmData:\t{vmdata_kib} kB\nVmRSS:\t999999999 kB\n"

    real = Path.read_text
    monkeypatch.setattr(
        Path, "read_text", lambda self, *a, **k: status() if str(self) == "/proc/self/status" else real(self, *a, **k)
    )
    monkeypatch.setattr(memo.resource, "getrlimit", lambda which: (soft, soft))
    monkeypatch.setattr(memo, "_fork_data", fork)
    return reads


def test_a_worker_measures_data_growth_since_the_fork_never_resident_memory(monkeypatch):
    reads = _mock_data(monkeypatch, 3000, 1_000_000 + 2_048_000, -1)
    memo.mark_fork(2_048_000)  # forked at 2000 KiB
    assert memo.pressure() == (3000 * 1024 - 2_048_000, 1_000_000)  # VmRSS in the status is ignored
    assert reads == ["status"]  # one /proc read per check


def test_a_parent_compares_its_anonymous_resident_memory_to_the_parent_budget(monkeypatch):
    _mock_data(monkeypatch, 3000, 0, -1)
    monkeypatch.setattr(stages, "parent_memory", 2_000_000_000)
    assert memo.pressure() == (500 * 1024, 2_000_000_000)  # RssAnon, not VmData


@pytest.mark.parametrize(("vmdata_kib", "kept"), [(2000 + 400, True), (2000 + 600, False)])
def test_scoped_fires_on_growth_past_half_the_worker_limit(monkeypatch, vmdata_kib, kept):
    _mock_data(monkeypatch, vmdata_kib, 2000 * 1024 + 1000 * 1024, 2000 * 1024)
    with memo.scoped():
        memo.memo(("overlay", 1), lambda: 1)
    assert (("overlay", 1) in memo._memo) is kept


def test_a_worker_with_a_budget_trims_against_the_budget_not_the_higher_data_limit(monkeypatch):
    _mock_data(monkeypatch, 3000, 4_000_000 + 2_048_000, -1)
    memo.mark_fork(2_048_000, 1_000_000)
    try:
        assert memo.pressure() == (3000 * 1024 - 2_048_000, 1_000_000)
    finally:
        memo.mark_fork(-1)
