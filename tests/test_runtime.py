# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The runtime file: one schema, every key required, auto resolved by affinity and memory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from abucache import keys, runtime

GOOD = """
[resources]
cores = 8
workers = 4
worker_memory = 1073741824
parent_memory = 2147483648
reserve_memory = 536870912

[capacities]
repro = 1

[cache]
dir = "{cache}"
bytes = 1000000
keep_seconds = 3600

[stages]
log_bytes = 4096

[events]
feed = "{cache}/../feed.jsonl"
slow_seconds = 60
overrun_percent = 150
"""


def _text(tmp_path: Path) -> str:
    return GOOD.format(cache=tmp_path / "cache")


def _file(tmp_path: Path, text: str | None = None) -> Path:
    path = tmp_path / "runtime.toml"
    path.write_text(_text(tmp_path) if text is None else text)
    return path


def test_a_file_loads_into_one_frozen_record(tmp_path: Path) -> None:
    value = runtime.load(_file(tmp_path))
    assert (value.cores, value.workers, value.worker_memory) == (8, 4, 1 << 30)
    assert (value.parent_memory, value.reserve_memory) == (2 << 30, 1 << 29)
    assert dict(value.capacities) == {"repro": 1}
    assert (value.cache_dir, value.cache_bytes, value.cache_keep_seconds) == (tmp_path / "cache", 1000000, 3600)
    assert value.stage_log_bytes == 4096
    assert (value.feed, value.slow_seconds, value.overrun_percent) == (tmp_path / "cache/../feed.jsonl", 60, 150)
    with pytest.raises(AttributeError):
        value.cores = 1  # type: ignore[misc]


def test_the_digest_covers_the_resolved_values_and_moves_with_each(tmp_path: Path) -> None:
    first = runtime.load(_file(tmp_path))
    assert first.digest == runtime.load(_file(tmp_path)).digest
    for old, new in (
        ("workers = 4", "workers = 3"),
        ("repro = 1", "repro = 2"),
        ("log_bytes = 4096", "log_bytes = 8"),
        ("slow_seconds = 60", "slow_seconds = 5"),
    ):
        assert runtime.load(_file(tmp_path, _text(tmp_path).replace(old, new))).digest != first.digest
    assert first.digest == keys.digest(
        {
            "cores": 8,
            "workers": 4,
            "worker_memory": 1 << 30,
            "parent_memory": 2 << 30,
            "reserve_memory": 1 << 29,
            "capacities": {"repro": 1},
            "cache_dir": tmp_path / "cache",
            "cache_bytes": 1000000,
            "cache_keep_seconds": 3600,
            "stage_log_bytes": 4096,
            "feed": tmp_path / "cache/../feed.jsonl",
            "slow_seconds": 60,
            "overrun_percent": 150,
        }
    )


def test_the_environment_names_the_file_when_no_path_is_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ABU_RUNTIME", str(_file(tmp_path)))
    assert runtime.load().cores == 8


def test_the_default_file_is_under_the_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ABU_RUNTIME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".config" / "abu"
    target.mkdir(parents=True)
    (target / "runtime.toml").write_text(_text(tmp_path))
    assert runtime.load().workers == 4


def _refused(tmp_path: Path, text: str) -> runtime.RuntimeConfigError:
    with pytest.raises(runtime.RuntimeConfigError) as caught:
        runtime.load(_file(tmp_path, text))
    return caught.value


@pytest.mark.parametrize(
    "key",
    [
        "resources.cores",
        "resources.reserve_memory",
        "cache.dir",
        "cache.keep_seconds",
        "stages.log_bytes",
        "events.feed",
        "events.slow_seconds",
    ],
)
def test_a_missing_key_is_refused_by_file_and_dotted_key(tmp_path: Path, key: str) -> None:
    name = key.partition(".")[2]
    kept = [line for line in _text(tmp_path).splitlines() if not line.startswith(name + " =")]
    error = _refused(tmp_path, "\n".join(kept))
    assert str(error).startswith(f"{tmp_path / 'runtime.toml'}:{key}:") and error.key == key


def test_an_unknown_key_or_section_is_refused(tmp_path: Path) -> None:
    text = _text(tmp_path)
    assert _refused(tmp_path, text.replace("[stages]", "[stages]\nlogbytes = 1")).key == "stages.logbytes"
    assert _refused(tmp_path, text + "\n[extra]\nx = 1\n").key == "extra"


def test_a_missing_capacities_section_is_refused(tmp_path: Path) -> None:
    assert _refused(tmp_path, _text(tmp_path).replace("[capacities]\nrepro = 1\n", "")).key == "capacities"


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        ("cores = 8", 'cores = "many"', "resources.cores"),
        ("workers = 4", "workers = 9", "resources.workers"),
        ("workers = 4", "workers = 0", "resources.workers"),
        ("worker_memory = 1073741824", "worker_memory = 0", "resources.worker_memory"),
        ("repro = 1", "repro = true", "capacities.repro"),
        ("repro = 1", 'repro = "x"', "capacities.repro"),
        ("bytes = 1000000", "bytes = 1.5", "cache.bytes"),
        ("overrun_percent = 150", "overrun_percent = 99", "events.overrun_percent"),
        ("slow_seconds = 60", "slow_seconds = 0", "events.slow_seconds"),
    ],
)
def test_a_wrong_value_names_its_key(tmp_path: Path, old: str, new: str, key: str) -> None:
    assert _refused(tmp_path, _text(tmp_path).replace(old, new)).key == key


def test_a_relative_feed_is_refused(tmp_path: Path) -> None:
    text = _text(tmp_path).replace(f'feed = "{tmp_path}/cache/../feed.jsonl"', 'feed = "feed.jsonl"')
    assert _refused(tmp_path, text).key == "events.feed"


def test_a_relative_cache_dir_is_refused(tmp_path: Path) -> None:
    assert _refused(tmp_path, GOOD.format(cache="rel")).key == "cache.dir"


def test_an_unreadable_or_invalid_file_is_refused_by_file(tmp_path: Path) -> None:
    with pytest.raises(runtime.RuntimeConfigError) as caught:
        runtime.load(tmp_path / "absent.toml")
    assert caught.value.file == str(tmp_path / "absent.toml") and caught.value.key == ""
    assert _refused(tmp_path, "[resources\n").key == ""


def test_auto_cores_is_the_affinity_and_auto_workers_the_smaller_of_it_and_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(6)))
    monkeypatch.setattr(runtime, "_usable_memory", lambda: 1 << 40)
    both = _text(tmp_path).replace("cores = 8", 'cores = "auto"').replace("workers = 4", 'workers = "auto"')
    value = runtime.load(_file(tmp_path, both))
    assert (value.cores, value.workers) == (6, 6)
    only_workers = _text(tmp_path).replace("workers = 4", 'workers = "auto"')
    value = runtime.load(_file(tmp_path, only_workers))
    assert (value.cores, value.workers) == (8, 6)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)))
    assert runtime.load(_file(tmp_path, only_workers)).workers == 8  # never more than cores


def test_auto_workers_are_limited_by_memory_and_never_below_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)))
    text = _text(tmp_path).replace("cores = 8", "cores = 16").replace("workers = 4", 'workers = "auto"')
    monkeypatch.setattr(runtime, "_usable_memory", lambda: (2 << 30) + 3 * (1 << 30) + 5)
    assert runtime.load(_file(tmp_path, text)).workers == 3
    monkeypatch.setattr(runtime, "_usable_memory", lambda: 1 << 20)
    assert runtime.load(_file(tmp_path, text)).workers == 1


def test_usable_memory_is_positive() -> None:
    assert runtime._usable_memory() > 0
