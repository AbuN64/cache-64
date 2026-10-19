# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The one event writer: vocabulary, schema file, validation, append and incremental reading."""

import hashlib
import json
from pathlib import Path

import pytest

from abucache import events
from abucache.events import (
    KINDS,
    LANE_KINDS,
    LANE_TRACKS,
    SCHEMA1_KINDS,
    SCHEMA_FILE,
    STATUSES,
    SUBJECT_KEYS,
    EventReader,
    canonical_kind,
    evidence_of,
    format_event,
    make_event,
    validate,
    validate_lines,
)

FIXTURES = Path(__file__).parent / "fixtures" / "events"


def test_schema_file_states_the_same_vocabulary_as_the_writer():
    assert events.SCHEMA == 2
    document = json.loads(SCHEMA_FILE.read_text())
    defs = document["$defs"]
    two = defs["schema2"]["properties"]
    assert tuple(two["kind"]["enum"]) == KINDS
    assert set(defs["status"]["enum"]) == STATUSES
    assert set(two["subject"]["properties"]) == SUBJECT_KEYS
    assert set(two["lane"]["properties"]["kind"]["enum"]) == LANE_KINDS
    assert set(two["lane"]["properties"]["track"]["enum"]) == LANE_TRACKS
    assert document["x-schema1-kinds"] == SCHEMA1_KINDS
    assert defs["schema1"]["required"] == defs["schema2"]["required"]
    assert set(defs["schema1"]["properties"]) < set(two)


def test_live_schema1_feed_shapes_stay_valid_as_written():
    with (FIXTURES / "schema1-live.jsonl").open("rb") as stream:
        report = validate_lines(stream)
    assert report["findings"] == []
    assert report["schemas"] == {"schema1": report["events"]}
    records = EventReader(FIXTURES / "schema1-live.jsonl").read_new()
    kinds = {record.get("kind"): canonical_kind(record) for record in records}
    wanted = ("RUN-FAIL", "MATCH", "LAND", "RUNG")
    assert (kinds["PORT-FAIL"], kinds["DECOMP"], kinds["COMMIT"], kinds["RUNG"]) == wanted
    assert all(format_event(record) for record in records)


def test_schema2_fixture_covers_every_kind_and_is_valid():
    with (FIXTURES / "schema2.jsonl").open("rb") as stream:
        report = validate_lines(stream)
    assert report["findings"] == []
    records = EventReader(FIXTURES / "schema2.jsonl").read_new()
    assert {r["kind"] for r in records if "kind" in r} == set(KINDS)
    assert "sha256 3f5a0c9e1b7d" in format_event(records[0])[1]


def test_writer_emits_schema2_and_refuses_schema1_names():
    record = make_event("a", "info", "m", kind="LAND", subject={"repo": "r", "ref": "abc"})
    assert record["schema"] == 2 and validate(record) == []
    for name, replacement in SCHEMA1_KINDS.items():
        with pytest.raises(ValueError, match=f"{name} is a schema 1 kind name; write {replacement}"):
            make_event("a", "info", "m", kind=name)


@pytest.mark.parametrize(
    "extra",
    [
        {"kind": "RUNG"},
        {"subject": {"address": "80253908"}},
        {"subject": {"unknown": "x"}},
        {"lane": {"model": "sol"}},
        {"lane": {"name": "L", "track": "side"}},
        {"evidence": {"path": "p", "sha256": "ABC"}},
    ],
)
def test_writer_refuses_a_malformed_event(extra):
    with pytest.raises(ValueError):
        make_event(
            "a",
            "info",
            "m",
            extra.get("evidence"),
            kind=extra.get("kind"),
            subject=extra.get("subject"),
            lane=extra.get("lane"),
        )


def test_validator_reports_each_problem_with_its_line():
    lines = [
        '{"schema":1,"time":"2026-01-01T00:00:00+00:00","actor":"a","status":"info","message":"m","evidence":null}',
        "",
        '{"schema":1,"time":"2026-01-01T00:00:00+00:00","actor":"a","status":"info","message":"m",'
        '"evidence":null,"lane":{"name":"x"}}',
        '{"schema":3}',
        "not json",
        '{"schema":2,"time":"yesterday","actor":"","status":"loud","message":"m","evidence":null}',
    ]
    report = validate_lines(lines)
    assert report["events"] == 5
    by_line: dict[int, list[str]] = {}
    for finding in report["findings"]:
        by_line.setdefault(finding["line"], []).append(finding["message"])
    assert by_line[3] == ["unknown key lane in schema 1"]
    assert by_line[4] == ["unsupported schema 3"]
    assert by_line[5][0].startswith("invalid JSON")
    assert by_line[6] == ["time is not ISO-8601", "actor must be a nonempty string", "unknown status 'loud'"]


def test_append_writes_hashed_evidence_and_read_returns_the_feed(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    assert evidence_of(receipt) == {"path": str(receipt.resolve()), "sha256": digest}
    feed = tmp_path / "state" / "events.jsonl"
    first = events.append(feed, make_event("lane", "handoff", "packet sent", evidence_of(receipt), kind="HANDOFF"))
    assert first["path"] == str(feed) and first["event"]["evidence"]["sha256"] == digest
    reader = EventReader(feed)
    assert reader.read_new() == [first["event"]] and reader.read_new() == []
    events.append(feed, make_event("build", "passed", "Gate passed", "receipt.txt"))
    assert [e["status"] for e in events.read(feed)] == ["handoff", "passed"]
    assert [e["status"] for e in reader.read_new()] == ["passed"]
    assert events.read(tmp_path / "none.jsonl") == []


def test_append_refuses_an_invalid_event_and_writes_nothing(tmp_path):
    feed = tmp_path / "events.jsonl"
    with pytest.raises(ValueError, match="invalid event"):
        events.append(feed, {"schema": 2})
    assert not feed.exists()
