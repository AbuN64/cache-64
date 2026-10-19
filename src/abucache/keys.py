# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Canonical digests, the code identity of a producer, and the Key that says what a stored result depends on."""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import re
import sys
import types
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    """sha256 hex of canonical JSON (sorted keys, no spaces); dataclasses, paths, bytes and sets are normalised."""

    def norm(item: Any) -> Any:  # JSON writes the rest natively
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        if isinstance(item, Mapping):
            return {str(k): v for k, v in item.items()}
        if isinstance(item, (frozenset, set)):
            return sorted(item, key=lambda v: json.dumps(v, sort_keys=True, default=norm))
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, bytes):
            return sha256(item)
        raise TypeError(f"cannot digest {type(item).__name__}")

    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=norm)
    return sha256(text.encode())


# ---------------------------------------------------------------- code identity
_PLAIN = (str, bytes, int, float, bool, type(None))


def _plain(value: Any) -> str | None:
    """A canonical text for an immutable constant, else None: mutable containers (caches, memos) differ per process."""
    if isinstance(value, _PLAIN):
        return repr(value)
    if isinstance(value, re.Pattern):
        return repr((value.pattern, value.flags))
    if isinstance(value, tuple):
        parts = [_plain(v) for v in value]
        return None if None in parts else f"tuple({','.join(p or '' for p in parts)})"
    if isinstance(value, frozenset):
        parts = [_plain(v) for v in value]
        return None if None in parts else f"frozenset({','.join(sorted(p or '' for p in parts))})"
    return None


@functools.cache
def _assigned(module: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Per top-level name of a module: its assigned expression as a syntax tree, and the names that expression reads."""
    path = getattr(sys.modules.get(module), "__file__", None)
    found: dict[str, tuple[str, tuple[str, ...]]] = {}
    if path:
        for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                reads = tuple(sorted({n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}))
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(target, ast.Name):
                        found[target.id] = (ast.dump(node.value), reads)
    return found


def _home(value: Any, name: str, home: str, package: str) -> str:
    """The module that assigns `name`: `home` itself, or for a `from x import NAME` the package module holding it."""
    if name in _assigned(home):
        return home
    for module_name, module in list(sys.modules.items()):
        ours = module_name == package or module_name.startswith(package + ".")
        if ours and getattr(module, name, None) is value and name in _assigned(module_name):
            return module_name
    return home


def _methods(modules: Sequence[str]) -> dict[str, list[Any]]:
    """Every method of the classes the given modules define, by name: a call on a record (snapshot.read) names no
    global to follow."""
    found: dict[str, list[Any]] = {}
    for module in modules:
        for value in list(vars(sys.modules[module]).values()) if module in sys.modules else ():
            if not (inspect.isclass(value) and value.__module__ == module):
                continue
            for name, member in vars(value).items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    member = member.fget
                if isinstance(member, types.FunctionType):
                    found.setdefault(name, []).append(member)
    return found


class _Walk:
    def __init__(self, package: str, modules: Sequence[str] = ()) -> None:
        self.package = package
        self.hasher = hashlib.sha256()
        self.seen: set[Any] = set()
        self.queue: deque[tuple[Callable[[Any], None], Any]] = deque()
        self.methods = _methods(modules)

    def add(self, *parts: str) -> None:
        for part in parts:
            self.hasher.update(part.encode())
            self.hasher.update(b"\0")

    def ours(self, value: Any) -> bool:
        module = getattr(value, "__module__", None) or ""
        return module == self.package or module.startswith(self.package + ".")

    def plumbing(self, value: Any) -> bool:
        """Code of a module that declares `__plumbing__ = True`: it schedules, times or memoises work and never shapes
        a result, so it is named and not followed (a pool or timing edit leaves every stored result warm)."""
        module = sys.modules.get(getattr(value, "__module__", None) or "")
        if getattr(module, "__plumbing__", False):
            self.add("plumbing", module.__name__, getattr(value, "__qualname__", ""))  # type: ignore[union-attr]
            return True
        return False

    def value(self, value: Any, name: str, home: str = "", declared: bool = False) -> None:
        """A global or attribute the code names: followed when it is code of the package, else its constants."""
        while isinstance(getattr(value, "__wrapped__", None), types.FunctionType):
            value = value.__wrapped__
        if isinstance(value, types.ModuleType):
            self.add("module", value.__name__)
        elif isinstance(value, types.FunctionType) and self.ours(value):
            if not self.plumbing(value):
                self.queue.append((self.function, value))
        elif inspect.isclass(value) and self.ours(value):
            if not self.plumbing(value):
                self.queue.append((self.klass, value))
        elif declared and name in _assigned(_home(value, name, home, self.package)):
            self.data(value, name, home, True)  # a global a module assigns: its expression, not what a run left in it
        elif (text := _plain(value)) is not None:
            self.add("const", name, text)
        else:
            self.data(value, name, home)

    def data(self, value: Any, name: str, home: str, declared: bool = False) -> None:
        """The expression that built the global at import, never what it holds now (memos fill up per process, and
        a global a run reassigns would give every process another identity)."""
        home = _home(value, name, home, self.package)
        text, reads = _assigned(home).get(name, ("", ()))
        self.add("data", name, "global" if declared else type(value).__name__, text)
        if ("data", home, name) not in self.seen:  # the module names that expression reads (HARD reads _ABI)
            self.seen.add(("data", home, name))
            scope = vars(sys.modules[home]) if home in sys.modules else {}
            for read in reads:
                if read in scope and read != name:
                    self.value(scope[read], read, home, True)

    def klass(self, cls: type) -> None:
        if id(cls) in self.seen:
            return
        self.seen.add(id(cls))
        self.add("class", cls.__qualname__, repr(sorted(getattr(cls, "__annotations__", {}).items())))
        for base in cls.__bases__:
            if self.ours(base):
                self.queue.append((self.klass, base))
        for name, member in vars(cls).items():
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            elif isinstance(member, property):
                member = member.fget
            if isinstance(member, types.FunctionType):
                self.queue.append((self.function, member))
            elif not name.startswith("__"):
                self.value(member, name, cls.__module__)

    def function(self, function: Any) -> None:
        code = getattr(function, "__code__", None)
        if code is None:
            self.add(
                "opaque",
                getattr(function, "__module__", None) or "builtins",
                getattr(function, "__qualname__", repr(type(function))),
            )
            return
        if id(function) in self.seen:
            return
        self.seen.add(id(function))
        self.add(
            "function",
            function.__qualname__,
            _plain(function.__defaults__) or "",
            _plain(function.__kwdefaults__) or "",
        )
        for cell in function.__closure__ or ():
            with suppress(ValueError):  # an empty cell
                self.value(cell.cell_contents, "closure")
        self.code(code, function.__globals__)

    def code(self, code: types.CodeType, scope: dict[str, Any]) -> None:
        self.add("code", code.co_code.hex(), *code.co_varnames, *code.co_names)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                self.code(const, scope)
            else:
                self.add("const", _plain(const) or type(const).__name__)
        for name in code.co_names:
            if name in scope:
                value = scope[name]
                if isinstance(value, types.ModuleType) and (value.__name__ + ".").startswith(self.package + "."):
                    for attribute in code.co_names:
                        if attribute != name and hasattr(value, attribute):
                            self.value(getattr(value, attribute), attribute, value.__name__, True)
                else:
                    self.value(value, name, scope.get("__name__", ""), True)
            for method in self.methods.get(name, ()):
                self.queue.append((self.function, method))


@functools.cache
def code_identity(*functions: Any, package: str | None = None) -> str:
    """A digest of the code these functions reach: bytecode, constants, names and every package function, class and
    module attribute they name, so an edit anywhere else (line shifts included) leaves it equal. Computed once per
    process. Strings picked by name at run time and adapters picked by config are not seen: pass them in. `package`
    names the code that counts as ours (default: the top package of the first function); a method called on a record
    is followed through the classes of the modules the functions live in."""
    first = functions[0] if functions else None
    package = package or (
        getattr(first, "__name__" if isinstance(first, types.ModuleType) else "__module__", None) or ""
    ).partition(".")[0]
    homes = sorted(
        {
            f.__name__ if isinstance(f, types.ModuleType) else getattr(f, "__module__", None) or ""
            for f in functions
        }
        - {""}
    )
    walk = _Walk(package, homes)
    for function in functions:
        if isinstance(function, types.ModuleType):  # every function and class it defines, for code picked by name
            for name, member in sorted(vars(function).items()):
                if getattr(member, "__module__", None) == function.__name__:
                    walk.value(member, name)
        else:
            walk.value(function, "root")
    while walk.queue:
        visit, item = walk.queue.popleft()
        visit(item)
    return walk.hasher.hexdigest()


@dataclass(frozen=True)
class Key:
    """How a job's items are named in the store: `name(item)` is the key text, or None for an item never stored.
    `stable` says the name already covers every input; `coded` says it covers the code the result depends on."""

    name: Callable[[Any], str | None]
    stable: bool = False
    coded: bool = False

    def __call__(self, item: Any) -> str | None:
        return self.name(item)


def identity(function: Callable[..., Any], key: Key) -> str:
    """What a stored result also depends on: the job's code, unless its key names every input (`stable`) or names the
    code the result depends on itself (`coded`: a key narrower than everything the job reaches)."""
    return "" if key.stable or key.coded else code_identity(function)
