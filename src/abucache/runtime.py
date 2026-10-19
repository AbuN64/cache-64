# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The one runtime file of a host: cores, workers, memory, named capacities, the cache, the stage log
and the event feed.

`ABU_RUNTIME` names the file, else `~/.config/abu/runtime.toml`. Every key is required and an unknown key is an
error, so a typo never silently falls back to a default."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from abucache import keys

_SCHEMA: dict[str, tuple[str, ...]] = {
    "resources": ("cores", "workers", "worker_memory", "parent_memory", "reserve_memory"),
    "cache": ("dir", "bytes", "keep_seconds"),
    "stages": ("log_bytes",),
    "events": ("feed", "slow_seconds", "overrun_percent"),
}


class RuntimeConfigError(Exception):
    """The runtime file is unreadable or wrong. `str(error)` starts with `file:dotted.key` (`file` alone when the
    whole file is at fault), then the reason."""

    def __init__(self, file: str, key: str, reason: str) -> None:
        self.file, self.key, self.reason = file, key, reason
        super().__init__(f"{file}:{key}: {reason}" if key else f"{file}: {reason}")


@dataclass(frozen=True)
class Runtime:
    cores: int
    workers: int
    worker_memory: int
    parent_memory: int
    reserve_memory: int
    capacities: Mapping[str, int]  # named capacities, e.g. {"repro": 1}
    cache_dir: Path
    cache_bytes: int
    cache_keep_seconds: int  # a cache root without git history keeps its entries live this long
    stage_log_bytes: int
    feed: Path  # the event feed the shared path writes slow stages, slot waits and queue time to
    slow_seconds: int  # a stage or wait this long (and over its expected time) is an event, running or closed
    overrun_percent: int  # "over its expected time": at least this percent of the previous command's wall
    digest: str  # keys.digest of every resolved value above


def _usable_memory() -> int:
    """Bytes this process may use: the cgroup limit when one is set, else the machine's RAM."""
    limit = Path("/sys/fs/cgroup/memory.max")
    text = limit.read_text(encoding="utf-8").strip() if limit.is_file() else "max"
    return int(text) if text.isdigit() else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _int(file: str, key: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeConfigError(file, key, "must be an integer")
    if value < minimum:
        raise RuntimeConfigError(file, key, f"must be at least {minimum}")
    return value


def default_path() -> Path:
    """The runtime file this process reads: `ABU_RUNTIME`, else `~/.config/abu/runtime.toml`."""
    env = os.environ.get("ABU_RUNTIME")
    return Path(env) if env else Path.home() / ".config" / "abu" / "runtime.toml"


def load(path: Path | None = None) -> Runtime:
    """Read and resolve the runtime file. `"auto"` for `cores` is the CPUs this process may run on; for `workers`
    it is the smaller of that and `cores`, limited by memory: (usable memory - parent_memory) // worker_memory,
    at least 1."""
    path = path or default_path()
    file = str(path)
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeConfigError(file, "", f"cannot be read: {error.strerror or error}") from error
    except tomllib.TOMLDecodeError as error:
        raise RuntimeConfigError(file, "", f"is not valid TOML: {error}") from error
    for section in (*_SCHEMA, "capacities"):
        if not isinstance(document.get(section), dict):
            raise RuntimeConfigError(file, section, "is missing")
    for section in document:
        if section not in _SCHEMA and section != "capacities":
            raise RuntimeConfigError(file, section, "is not a runtime section")
    for section, names in _SCHEMA.items():
        for name in names:
            if name not in document[section]:
                raise RuntimeConfigError(file, f"{section}.{name}", "is missing")
        for name in document[section]:
            if name not in names:
                raise RuntimeConfigError(file, f"{section}.{name}", "is not a runtime key")
    res, cache, stages = document["resources"], document["cache"], document["stages"]
    events = document["events"]
    worker_memory = _int(file, "resources.worker_memory", res["worker_memory"])
    parent_memory = _int(file, "resources.parent_memory", res["parent_memory"], minimum=0)
    reserve_memory = _int(file, "resources.reserve_memory", res["reserve_memory"], minimum=0)
    cores = len(os.sched_getaffinity(0)) if res["cores"] == "auto" else _int(file, "resources.cores", res["cores"])
    if res["workers"] == "auto":
        workers = min(len(os.sched_getaffinity(0)), cores)
        workers = min(workers, max(1, (_usable_memory() - parent_memory) // worker_memory))
    else:
        workers = _int(file, "resources.workers", res["workers"])
        if workers > cores:
            raise RuntimeConfigError(file, "resources.workers", "exceeds cores")
    capacities = {name: _int(file, f"capacities.{name}", value) for name, value in document["capacities"].items()}
    directory = cache["dir"]
    if not isinstance(directory, str) or not Path(directory).is_absolute():
        raise RuntimeConfigError(file, "cache.dir", "must be an absolute path")
    feed = events["feed"]
    if not isinstance(feed, str) or not Path(feed).is_absolute():
        raise RuntimeConfigError(file, "events.feed", "must be an absolute path")
    values: dict[str, Any] = {
        "cores": cores,
        "workers": workers,
        "worker_memory": worker_memory,
        "parent_memory": parent_memory,
        "reserve_memory": reserve_memory,
        "capacities": MappingProxyType(capacities),
        "cache_dir": Path(directory),
        "cache_bytes": _int(file, "cache.bytes", cache["bytes"]),
        "cache_keep_seconds": _int(file, "cache.keep_seconds", cache["keep_seconds"]),
        "stage_log_bytes": _int(file, "stages.log_bytes", stages["log_bytes"]),
        "feed": Path(feed),
        "slow_seconds": _int(file, "events.slow_seconds", events["slow_seconds"]),
        "overrun_percent": _int(file, "events.overrun_percent", events["overrun_percent"], minimum=100),
    }
    return Runtime(**values, digest=keys.digest(values))
