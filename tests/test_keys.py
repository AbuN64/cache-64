# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""digest and code_identity: a key that guards a producer moves with the code the producer reaches and with nothing
else."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path

from abucache import keys
from abucache.keys import Key, code_identity, digest, identity


def _module(source: str) -> dict:
    scope = {"__name__": "fixturepkg.mod"}
    exec(compile(source, "mod.py", "exec"), scope)  # a module of its own package, rebuilt per edit
    return scope


_BASE = textwrap.dedent(
    """
    import re
    LIMIT = 3
    PATTERN = re.compile("a+")
    CACHE = {}

    def helper(x):
        return x + LIMIT

    def unrelated(x):
        return x * 2

    def produce(x):
        return [helper(i) for i in range(x)]

    def lambdas(x):
        return sorted(x, key=lambda v: -v)
"""
)


def _identity(source: str, name: str = "produce") -> str:
    scope = _module(source)
    code_identity.cache_clear()
    return code_identity(scope[name])


def test_digest_is_canonical_json_of_sorted_keys() -> None:
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})
    assert digest({"a": 1}) == "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
    assert digest({1, 2, 3}) == digest(frozenset({3, 2, 1}))
    assert digest(Path("/x")) == digest("/x")
    assert digest(b"abc") != digest(b"abd")


def test_digest_normalises_dataclasses_and_mappings() -> None:
    @dataclass
    class Row:
        a: int

    assert digest(Row(1)) == digest({"a": 1})
    assert digest(types.MappingProxyType({"a": 1})) == digest({"a": 1})


def test_an_edit_of_an_unrelated_function_leaves_the_identity_equal() -> None:
    assert _identity(_BASE) == _identity(_BASE.replace("return x * 2", "return x * 3 + 1"))


def test_line_shifts_alone_leave_the_identity_equal() -> None:
    assert _identity(_BASE) == _identity("# a comment\n\n\n" + _BASE.replace("def helper", "\n\ndef helper"))


def test_an_edit_of_a_callee_a_constant_or_a_nested_lambda_changes_it() -> None:
    base = _identity(_BASE)
    assert _identity(_BASE.replace("return x + LIMIT", "return x - LIMIT")) != base
    assert _identity(_BASE.replace("LIMIT = 3", "LIMIT = 4")) != base
    assert _identity(_BASE.replace("range(x)", "range(x + 1)")) != base
    assert _identity(_BASE, "lambdas") != _identity(_BASE.replace("-v", "v"), "lambdas")


def test_mutable_module_state_is_named_by_its_type_only() -> None:
    scope = _module(_BASE + "\ndef reads():\n    return CACHE\n")
    code_identity.cache_clear()
    before = code_identity(scope["reads"])
    scope["CACHE"]["filled"] = [1, 2, 3]
    code_identity.cache_clear()
    assert code_identity(scope["reads"]) == before


def test_a_module_root_covers_code_reached_only_by_name() -> None:
    scope = _module(_BASE)
    module = types.ModuleType("fixturepkg.mod")
    module.__dict__.update(scope)
    code_identity.cache_clear()
    first = code_identity(module)
    scope["unrelated"] = _module("def unrelated(x):\n    return 0\n")["unrelated"]
    module.__dict__.update(scope)
    code_identity.cache_clear()
    assert code_identity(module) != first


def test_the_package_argument_names_the_code_that_counts() -> None:
    helper = types.ModuleType("otherpkg.helper")

    def identity_of(body: str, package: str | None) -> str:
        exec(compile(f"def reached(x):\n    return {body}\n", "helper.py", "exec"), vars(helper))
        helper.reached.__module__ = "otherpkg.helper"
        code_identity.cache_clear()
        return code_identity(helper.reached, package=package)

    assert identity_of("x", None) != identity_of("x + 1", None)  # the default package is the first function's
    assert identity_of("x", "otherpkg") != identity_of("x + 1", "otherpkg")
    assert identity_of("x", "fixturepkg") == identity_of("x + 1", "fixturepkg")  # not ours: not followed


def test_a_method_called_on_a_record_is_followed_through_the_modules_classes() -> None:
    source = (
        "class Record:\n    def read(self):\n        return 1\n\n"
        "def produce(record):\n    return record.read()\n"
    )
    scope = {"__name__": "fixturepkg.methods"}
    exec(compile(source, "methods.py", "exec"), scope)
    scope["Record"].__module__ = scope["produce"].__module__ = "fixturepkg.methods"
    module = types.ModuleType("fixturepkg.methods")
    module.__dict__.update(scope)
    sys.modules["fixturepkg.methods"] = module
    try:
        code_identity.cache_clear()
        first = code_identity(scope["produce"])
        scope2 = {"__name__": "fixturepkg.methods"}
        exec(compile(source.replace("return 1", "return 2"), "methods.py", "exec"), scope2)
        scope2["Record"].__module__ = scope2["produce"].__module__ = "fixturepkg.methods"
        module.__dict__.update(scope2)
        code_identity.cache_clear()
        assert code_identity(scope2["produce"]) != first
    finally:
        del sys.modules["fixturepkg.methods"]
        code_identity.cache_clear()


def test_the_identity_of_a_producer_is_equal_in_every_process() -> None:
    program = "from abucache import keys\nprint(keys.code_identity(keys.identity, keys.digest))"
    runs = {
        subprocess.run(
            [sys.executable, "-c", program],
            env={"PYTHONHASHSEED": seed, "PYTHONPATH": ":".join(sys.path)},
            capture_output=True,
            check=True,
            text=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(runs) == 1  # a hash seed or a set order must never move a cache key


def test_code_of_a_plumbing_module_is_named_and_never_followed(monkeypatch) -> None:
    def build(plumbing: bool, body: str) -> str:
        helper = types.ModuleType("fixturepkg.helper")
        helper.__plumbing__ = plumbing
        exec(compile(f"def schedule(x):\n    return {body}\n", "helper.py", "exec"), vars(helper))
        helper.schedule.__module__ = "fixturepkg.helper"
        monkeypatch.setitem(sys.modules, "fixturepkg.helper", helper)
        scope = _module("def produce(x):\n    return schedule(x) + 1\n")
        scope["schedule"] = helper.schedule
        code_identity.cache_clear()
        return code_identity(scope["produce"])

    assert build(True, "x") == build(True, "[x, x]")  # a scheduler edit leaves stored results warm
    assert build(False, "x") != build(False, "[x, x]")
    assert build(True, "x") != build(False, "x")


def test_a_stable_or_coded_key_carries_no_code_identity() -> None:
    def job(item: int) -> int:
        return item

    named = Key(lambda item: str(item))
    assert named(3) == "3" and Key(lambda item: None)(3) is None
    assert identity(job, named) == keys.code_identity(job) != ""
    assert identity(job, Key(str, stable=True)) == ""
    assert identity(job, Key(str, coded=True)) == ""
