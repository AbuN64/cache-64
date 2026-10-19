# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Caches and the link fast path of an N64 decompilation build. Callers pass paths, commands and a runner."""

import logging
import sys

__version__ = "0.6.0"
CONTRACT = 1  # the generic API of ARCH DESIGN 2.2; a change to a signature or schema there bumps it

log = logging.getLogger("abucache")


def require(n: int) -> None:
    """Every tool calls this at import: fail fast when the installed abucache speaks another contract."""
    if n != CONTRACT:
        raise RuntimeError(f"abucache contract {CONTRACT} is installed, but this tool needs contract {n}")


def trace(enabled: bool = True) -> None:
    """Print one line to stderr for every step abucache runs: the exact argv, and whether the compile cache hit."""
    for handler in [h for h in log.handlers if getattr(h, "_abucache_trace", False)]:
        log.removeHandler(handler)
    if enabled:
        handler = logging.StreamHandler(sys.stderr)
        handler._abucache_trace = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("abucache: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
