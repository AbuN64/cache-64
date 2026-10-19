# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""A pure reader's values for the running command, and the memory pressure that trims them.

The memo only holds pure values by input, so dropping one costs a recompute and never a wrong answer. A key that is
a tuple starts with its kind (`("index", digest)`), which `forget`, `held` and `trim_to_fork` select by."""

from __future__ import annotations

import resource
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from abucache import stages

T = TypeVar("T")

_memo: dict[object, Any] = {}  # pure readers' values for this command, by input digest
_epochs: dict[str, int] = {}  # kind -> how often this process forgot it
_applied: dict[str, int] = {}  # in a worker: the parent's forget counts it has followed (the fork's, then each sent)
_inherited: frozenset[Any] = frozenset()  # the memo keys a forked worker was born holding (shared with the parent)
_taggers: list[Callable[[Any], str | None]] = []  # a value's tag, for the values that name one (see `release`)
_fork_data = -1  # a worker's VmData at fork, set by the pool; -1 in the parent
_fork_budget = 0  # the growth a worker plans to stay within (the worker memory); 0 = its data limit is the plan


def _clear() -> None:
    """A new command starts with an empty memo (the epochs and the fork's state are the process's)."""
    _memo.clear()


stages.on_command(_clear)


def register_tagger(tagger: Callable[[Any], str | None]) -> None:
    """`tagger(value)` names the tag a memoised value belongs to (None for a value that has none): `release(tag)`
    drops the values whose tag it is, whatever their key says."""
    _taggers.append(tagger)


def memo(key: object, produce: Callable[[], T]) -> T:
    """A pure reader's value for the running command: computed once per key, dropped when the command ends."""
    if key not in _memo:
        _memo[key] = produce()
    value: T = _memo[key]
    return value


def peek(key: object) -> Any:
    """The memoised value of a key, None when none is kept."""
    return _memo.get(key)


def memo_valid(key: object, produce: Callable[[], T], valid: Callable[[T], bool]) -> T:
    """memo, but a stored value that no longer passes `valid` is made again."""
    if key in _memo and valid(_memo[key]):
        value: T = _memo[key]
        return value
    value = _memo[key] = produce()
    return value


def forget(*kinds: str) -> None:
    """Drops the memoised values of these kinds: the files they read were rewritten by this command."""
    for key in [k for k in _memo if isinstance(k, tuple) and k[0] in kinds]:
        del _memo[key]
    for kind in kinds:
        _epochs[kind] = _epochs.get(kind, 0) + 1


def epochs() -> dict[str, int]:
    """How often this process forgot each kind: a dispatch sends it, so warm workers forget what the parent did."""
    return dict(_epochs)


def follow(parent: Mapping[str, int]) -> None:
    """In a warm worker: forget every kind the parent forgot since the fork or the last dispatch. A forked worker
    started from the parent's memo after its forgets; a warm one keeps values read from files rewritten since."""
    stale = [kind for kind, count in parent.items() if count > _applied.get(kind, 0)]
    if stale:
        forget(*stale)
    _applied.update(parent)


def release(tag: str) -> int:
    """Drops every memoised value whose key names this tag, and every value a registered tagger gives this tag: what
    one a later stage never meets (a wave's overlay once the next one is laid) took with it. Returns how many went."""

    def tagged(value: Any) -> bool:
        return any(tagger(value) == tag for tagger in _taggers)

    gone = [k for k, v in _memo.items() if (isinstance(k, tuple) and tag in k) or tagged(v)]
    for k in gone:
        del _memo[k]
    return len(gone)


def trim_to_fork(keep_kinds: Collection[str] = (), tags: Collection[str] = ()) -> int:
    """A worker drops what it memoised since the fork, except the kinds in `keep_kinds` (small values keyed by
    content) and what it built for a tag it still holds (a key naming one of `tags`): what it built for one overlay
    is never read on the next, while the warm base the next one is laid over keeps its indexes. Returns how many
    went."""
    held_tags = set(tags)
    gone = [
        k
        for k in _memo
        if k not in _inherited
        and not (
            isinstance(k, tuple) and k and (k[0] in keep_kinds or (held_tags and any(p in held_tags for p in k[1:])))
        )
    ]
    for k in gone:
        del _memo[k]
    return len(gone)


def held(kinds: Collection[str]) -> dict[object, Any]:
    """The memoised values of these kinds (content-keyed indexes a parent hands to warm workers)."""
    return {k: v for k, v in _memo.items() if isinstance(k, tuple) and k and k[0] in kinds}


def adopt(entries: Mapping[object, Any], kinds: Collection[str]) -> None:
    """A warm worker takes the parent's indexes of these kinds: the ones it held of them before (an earlier round's)
    go, unless it was forked holding them, and what it already holds is kept."""
    for key in [k for k in held(kinds) if k not in entries and k not in _inherited]:
        del _memo[key]
    for key, value in entries.items():
        _memo.setdefault(key, value)


def mark_fork(data: int, budget: int = 0) -> None:
    """A forked worker records the data segment it inherited: its pressure counts only growth past it, against the
    planned budget (the hard data limit sits above it, so a job over budget trims its memo before it is refused)."""
    global _fork_data, _fork_budget, _inherited, _applied
    _fork_data, _fork_budget = data, budget
    _inherited = frozenset(_memo)
    _applied = dict(_epochs)  # the parent's forgets so far are in the memo it was forked holding


def _status_kib(field: str) -> int:
    """A kB field of /proc/self/status in bytes, 0 when absent."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    return 0


def pressure() -> tuple[int, int]:
    """(used, allowed): a worker's VmData growth since the fork against its data limit, else the parent's RssAnon
    against its budget (the parent has no data limit, and its VmData counts reserved arenas it never touches)."""
    if _fork_data >= 0:
        soft = resource.getrlimit(resource.RLIMIT_DATA)[0]
        allowed = _fork_budget or (0 if soft == resource.RLIM_INFINITY else soft - _fork_data)
        return _status_kib("VmData") - _fork_data, allowed
    return _status_kib("RssAnon"), stages.parent_memory


@contextmanager
def scoped(always: bool = False, keep: Collection[str] = ()) -> Iterator[None]:
    """Values memoised inside are dropped on exit when the process is past half its memory limit, or `always` (a
    job whose intermediate values no later job meets), except the kinds in `keep`. Keys present on entry (the base
    snapshot's) stay, so reuse across scopes survives while memory is fine."""
    before = set(_memo)
    try:
        yield
    finally:
        used, limit = pressure()
        if always or (limit and used > limit // 2):
            added = [k for k in _memo if k not in before and not (isinstance(k, tuple) and k and k[0] in keep)]
            for k in added:
                del _memo[k]
            if added:
                with stages.stage("effort.trimmed") as span:
                    span.add(items=len(added))
