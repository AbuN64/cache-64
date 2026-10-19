# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Fixtures for the slot tests: a runtime directory of their own and a runtime file made from a few numbers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from abucache import runtime, slots

TEXT = """
[resources]
cores = {cores}
workers = {workers}
worker_memory = 1073741824
parent_memory = 0
reserve_memory = {reserve}

[capacities]
{capacities}

[cache]
dir = "{cache}"
bytes = 1000000
keep_seconds = 3600

[stages]
log_bytes = 4096
"""


@pytest.fixture
def rundir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A runtime directory of this test alone, with no cores or capacity handed down by a holder above it."""
    directory = tmp_path / "run"
    directory.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(directory))
    monkeypatch.delenv(slots.CORES_ENV, raising=False)
    monkeypatch.delenv(slots.CAPACITIES_ENV, raising=False)
    return directory


@pytest.fixture
def make_runtime(tmp_path: Path) -> Callable[..., runtime.Runtime]:
    def make(workers: int = 4, capacities: dict[str, int] | None = None, reserve: int = 0) -> runtime.Runtime:
        lines = "\n".join(f"{name} = {size}" for name, size in (capacities or {}).items())
        path = tmp_path / "runtime.toml"
        path.write_text(
            TEXT.format(
                cores=max(workers, 8), workers=workers, reserve=reserve, capacities=lines, cache=tmp_path / "cache"
            )
        )
        return runtime.load(path)

    return make
