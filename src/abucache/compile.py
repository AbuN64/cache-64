# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The unit compile cache. Its key is content only: the preprocessed bytes, the exact commands and the identity of the
tools. abucache knows no compiler. The caller gives the commands and the tool files, so every ROM version whose
preprocessed input is identical shares one compile, whatever the versions are called and however many there are.

The commands run in a private directory that holds only the input under its own base name. No path of the project, no
version and no output path reaches a tool, so what a tool bakes into its result is a function of the key."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abucache import __version__, log
from abucache.store import ContentCache

Runner = Callable[[str, Sequence[str], Path], Any]  # (phase, argv, cwd) -> a result with .exit and .signal
_STAT: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class Step:
    phase: str  # what the runner is told this step is
    out: str  # the file the step writes inside the private directory
    argv: tuple[str, ...]  # {in} is the previous step's file (the input for the first), {out} is `out`


class ChainFailed(Exception):
    def __init__(self, step: Step, result: Any) -> None:
        super().__init__(f"{step.phase} failed")
        self.step, self.result = step, result


def failed(result: Any) -> bool:
    """Whether a runner result is a non-zero exit or a signal."""
    return result.exit != 0 or result.signal is not None


def identity(files: Sequence[Path], pins: Sequence[str] = ()) -> str:
    """Hash the bytes of the tool files and the pins. A file is read again only when its size or mtime changed."""
    digest = hashlib.sha256(f"abucache {__version__}\n".encode())
    for pin in pins:
        digest.update(f"pin {len(pin)} {pin}\n".encode())
    for file in files:
        stat = file.stat()
        stamp = (str(file.resolve()), stat.st_mtime_ns, stat.st_size)
        if stamp not in _STAT:
            with file.open("rb") as handle:
                _STAT[stamp] = hashlib.file_digest(handle, "sha256").hexdigest()
        digest.update(f"file {stat.st_size} {_STAT[stamp]}\n".encode())
    return digest.hexdigest()


def key(source: bytes, name: str, steps: Sequence[Step], tools: str) -> str:
    """Content only. The path of the input is not in it, its base name is (a tool may bake that in)."""
    digest = hashlib.sha256(f"{tools}\n{name}\n{len(source)}\n".encode())
    digest.update(source)
    for step in steps:
        digest.update(("\x1e".join((step.phase, step.out, *step.argv)) + "\n").encode())
    return digest.hexdigest()


def build(cache: ContentCache, run: Runner, steps: Sequence[Step], source: Path, tools: str, work: Path,
          after: Callable[[Step, Any], None] | None = None) -> tuple[bytes, list[Any]]:
    """The last step's file for the input, from the cache or by running the steps. `after` sees each result and may
    refuse it by raising. A step that fails without a refusal raises ChainFailed. Nothing failed is cached.
    A result that holds the path of the private directory, of the input or of `work` is not a function of the key,
    so it is returned but never stored. Also returns the results of the steps that ran (none on a hit)."""
    data = source.read_bytes()
    full = key(data, source.name, steps, tools)
    found = cache.get("unit", full)
    cache.count("unit", found is not None)
    log.info("compile cache: %s", "hit" if found is not None else "miss")
    if found is not None:
        return found, []
    results: list[Any] = []
    work.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=".chain-", dir=work))
    try:
        (directory / source.name).write_bytes(data)
        previous = source.name
        for step in steps:
            argv = [w.replace("{in}", previous).replace("{out}", step.out) for w in step.argv]
            log.info("%s: %s", step.phase, " ".join(argv))
            result = run(step.phase, argv, directory)
            results.append(result)
            if after is not None:
                after(step, result)
            if failed(result):
                raise ChainFailed(step, result)
            previous = step.out
        value = (directory / previous).read_bytes()
        paths = (str(directory), str(source.parent), str(work))
        if any(p.encode() in value for p in paths if p not in ("", ".", "/")):
            log.info("not stored: the result holds a path of this run")
        else:
            cache.put("unit", full, value)
        return value, results
    finally:
        shutil.rmtree(directory, ignore_errors=True)
