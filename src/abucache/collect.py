# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""`python -m abucache.collect DIRECTORY SIZE_LIMIT KEEP_COMMITS [INSTALLED [KEEP_SECONDS]]`: one collect of a content
cache in its own process. A command's finish starts it detached, so no command ever waits on a collect."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from abucache.store import KEEP_SECONDS, ContentCache, resolve


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4, 5):
        print(__doc__, file=sys.stderr)
        return 2
    directory, size_limit, keep_commits = Path(argv[0]), int(argv[1]), int(argv[2])
    installed = resolve(argv[3] if len(argv) == 4 else "")
    keep_seconds = int(argv[4]) if len(argv) == 5 else KEEP_SECONDS
    start = time.perf_counter()
    count = ContentCache(directory, size_limit).collect(installed, keep_commits=keep_commits, keep_seconds=keep_seconds)
    print(f"collected {count} entries of {directory} in {time.perf_counter() - start:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
