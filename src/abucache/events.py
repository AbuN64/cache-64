# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Append-only factual events and incremental reading.

This module is the one writer of the event format, for every tool. Schema 2 is additive over schema 1: every
schema 1 line stays valid and is read as written; only new lines use schema 2.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = 2
SCHEMA_FILE = Path(__file__).with_name("event.schema.json")
SCHEMAS = frozenset({1, SCHEMA})

KINDS = ("RUN-FAIL", "PROBE", "MATCH", "HANDOFF", "LAND", "LANE", "DECISION")
# Schema 1 names with a schema 2 replacement. Old lines keep their bytes; readers map them.
SCHEMA1_KINDS = {"PORT-FAIL": "RUN-FAIL", "DECOMP": "MATCH", "COMMIT": "LAND"}
MISSION_KINDS = ("RUN-FAIL", "PROBE", "MATCH", "HANDOFF", "LAND")

STATUSES = frozenset({"info", "started", "passed", "failed", "blocked", "decision", "handoff"})

SUBJECT_KEYS = frozenset({"repo", "ref", "version", "function", "address"})
LANE_KEYS = frozenset({"name", "kind", "model", "track"})
LANE_KINDS = frozenset({"tooling", "impl"})
LANE_TRACKS = frozenset({"official", "experiment"})

_REQUIRED = ("schema", "time", "actor", "status", "message", "evidence")
_OPTIONAL = {1: frozenset({"kind", "payload"}), 2: frozenset({"kind", "payload", "subject", "lane"})}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ADDRESS = re.compile(r"0x[0-9A-Fa-f]+")


def canonical_kind(value: dict[str, Any]) -> str | None:
    """The schema 2 kind of any readable record; schema 1 names map to their replacement."""
    kind = value.get("kind")
    if kind is None:
        return None
    return SCHEMA1_KINDS.get(kind, kind) if value.get("schema") == 1 else kind


def parse_kinds(text: str) -> set[str]:
    """A comma-separated kind filter in schema 2 names only."""
    kinds = {kind.strip() for kind in text.split(",") if kind.strip()}
    for kind in sorted(kinds):
        if kind in SCHEMA1_KINDS:
            raise ValueError(f"{kind} is a schema 1 kind name; filter on {SCHEMA1_KINDS[kind]}")
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind}; expected one of {', '.join(KINDS)}")
    return kinds


def evidence_of(path: Path) -> dict[str, str]:
    """The schema 2 evidence object: the file path and the sha256 of its bytes now."""
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate(value: Any) -> list[str]:
    """Every reason one decoded line is not a valid event of its own schema; empty when valid."""
    if not isinstance(value, dict):
        return ["event is not a JSON object"]
    schema = value.get("schema")
    if type(schema) is not int or schema not in SCHEMAS:
        return [f"unsupported schema {schema!r}"]
    problems = [f"missing {key}" for key in _REQUIRED if key not in value]
    extra = value.keys() - set(_REQUIRED) - _OPTIONAL[schema]
    problems.extend(f"unknown key {key} in schema {schema}" for key in sorted(extra))
    if "time" in value:
        try:
            if datetime.fromisoformat(str(value["time"])).tzinfo is None:
                problems.append("time has no UTC offset")
        except ValueError:
            problems.append("time is not ISO-8601")
    for key in ("actor", "message"):
        if key in value and not _text(value[key]):
            problems.append(f"{key} must be a nonempty string")
    if "status" in value and value["status"] not in STATUSES:
        problems.append(f"unknown status {value['status']!r}")
    if "payload" in value and not isinstance(value["payload"], dict):
        problems.append("payload must be an object")
    if "kind" in value:
        kind = value["kind"]
        if not _text(kind):
            problems.append("kind must be a nonempty string")
        elif schema == SCHEMA and kind not in KINDS:
            problems.append(f"unknown schema 2 kind {kind!r}")
    evidence = value.get("evidence")
    if isinstance(evidence, dict) and schema == SCHEMA:
        if set(evidence) != {"path", "sha256"} or not _text(evidence.get("path")):
            problems.append("evidence object needs exactly path and sha256")
        elif not isinstance(evidence["sha256"], str) or not _SHA256.fullmatch(evidence["sha256"]):
            problems.append("evidence sha256 must be 64 lowercase hex digits")
    elif evidence is not None and not _text(evidence):
        problems.append("evidence must be a path string, null, or (schema 2) a path/sha256 object")
    if schema == SCHEMA:
        problems.extend(_table_problems("subject", value.get("subject"), SUBJECT_KEYS))
        problems.extend(_table_problems("lane", value.get("lane"), LANE_KEYS))
        subject, lane = value.get("subject"), value.get("lane")
        if isinstance(subject, dict) and "address" in subject and not _ADDRESS.fullmatch(str(subject["address"])):
            problems.append("subject.address must be 0x-prefixed hex")
        if isinstance(lane, dict):
            if "name" not in lane:
                problems.append("lane.name is required")
            if "kind" in lane and lane["kind"] not in LANE_KINDS:
                problems.append(f"unknown lane.kind {lane['kind']!r}")
            if "track" in lane and lane["track"] not in LANE_TRACKS:
                problems.append(f"unknown lane.track {lane['track']!r}")
    return problems


def _table_problems(name: str, table: Any, keys: frozenset[str]) -> list[str]:
    if table is None:
        return []
    if not isinstance(table, dict) or not table:
        return [f"{name} must be a nonempty object"]
    problems = [f"unknown key {name}.{key}" for key in sorted(table.keys() - keys)]
    problems.extend(f"{name}.{key} must be a nonempty string" for key, item in table.items() if not _text(item))
    return problems


def validate_lines(lines: Iterable[bytes | str]) -> dict[str, Any]:
    """Validate a JSONL feed; findings carry 1-based line numbers. Blank lines are not events."""
    findings: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    total = 0
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        total += 1
        try:
            value = json.loads(raw)
        except ValueError as exc:
            findings.append({"line": number, "message": f"invalid JSON: {exc}"})
            continue
        for problem in validate(value):
            findings.append({"line": number, "message": problem})
        if isinstance(value, dict):
            key = f"schema{value.get('schema')}"
            counts[key] = counts.get(key, 0) + 1
    return {"events": total, "schemas": counts, "findings": findings}


def make_event(
    actor: str,
    status: str,
    message: str,
    evidence: str | dict[str, str] | None = None,
    *,
    timestamp: datetime | None = None,
    kind: str | None = None,
    payload: dict[str, Any] | None = None,
    subject: dict[str, str] | None = None,
    lane: dict[str, str] | None = None,
) -> dict[str, Any]:
    actor, status, message = actor.strip(), status.strip(), message.strip()
    if not actor or not message:
        raise ValueError("actor and message must be nonempty")
    if status not in STATUSES:
        raise ValueError("unknown event status: " + status)
    if any(ord(char) < 32 for char in actor + status):
        raise ValueError("actor/status contain control characters")
    when = timestamp or datetime.now(UTC)
    extra: dict[str, Any] = {}
    if kind:
        kind = kind.strip()
        if kind in SCHEMA1_KINDS:
            raise ValueError(f"{kind} is a schema 1 kind name; write {SCHEMA1_KINDS[kind]}")
        extra["kind"] = kind
    if payload:
        extra["payload"] = payload
    if subject:
        extra["subject"] = subject
    if lane:
        extra["lane"] = lane
    record = {
        **extra,
        "schema": SCHEMA,
        "time": when.astimezone(UTC).isoformat(),
        "actor": actor,
        "status": status,
        "message": message,
        "evidence": evidence or None,
    }
    problems = validate(record)
    if problems:
        raise ValueError("invalid event: " + "; ".join(problems))
    return record


def append(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    """One event (from `make_event`) as one line at the end of the feed at `path`, durable before it returns."""
    problems = validate(event)
    if problems:
        raise ValueError("invalid event: " + "; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)
    return {"event": event, "path": str(path)}


def read(path: Path) -> list[dict[str, Any]]:
    """Every event of the feed at `path`, in order; empty when there is none."""
    return EventReader(path).read_new()


class EventReader:
    """Read appended JSONL once; callers wake it on filesystem events."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.partial = b""
        self.identity: tuple[int, int] | None = None
        self.count = 0

    def read_new(self) -> list[dict[str, Any]]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return []
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.identity, self.offset, self.partial, self.count = identity, 0, b"", 0
        if stat.st_size == self.offset:
            return []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
            self.offset = stream.tell()
        pieces = (self.partial + data).split(bytes([10]))
        self.partial = pieces.pop()
        records: list[dict[str, Any]] = []
        for raw in pieces:
            if not raw:
                continue
            value = json.loads(raw)
            if value.get("schema") not in SCHEMAS:
                raise ValueError("unsupported event schema")
            records.append(value)
            self.count += 1
        return records


def format_event(value: dict[str, Any]) -> list[str]:
    stamp = datetime.fromisoformat(str(value["time"])).astimezone().strftime("%H:%M:%S")
    actor, status = str(value["actor"]), str(value["status"]).upper()
    message = str(value["message"]).replace("\n", " ")
    kind = canonical_kind(value)
    label = f"{kind}/{actor}/{status}" if kind else f"{actor}/{status}"
    lines = [f"{stamp} [{label}] {message}"]
    payload = value.get("payload") or {}
    if kind in {"PROBE", "MATCH"}:
        owner = payload.get("owner_address", "unknown")
        function = payload.get("function", "unknown")
        source = payload.get("decomp_path", "unknown")
        match = payload.get("match_state", "unknown")
        lines.append(f"    owner {owner} {function} | decomp {source} [{match}]")
        if payload.get("result"):
            lines.append("    result: " + str(payload["result"]))
    evidence = value.get("evidence")
    if isinstance(evidence, dict):
        lines.append(f"    evidence: {evidence.get('path')} sha256 {str(evidence.get('sha256'))[:12]}")
    elif evidence:
        lines.append("    evidence: " + str(evidence))
    return lines
