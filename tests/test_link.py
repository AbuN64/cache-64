# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""The link fast path: the symbols a unit needs, the script, the commands and running them."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from abucache import link

TABLE = {"text": {"inputs": [".text", ".text.*"], "zero": False}, "bss": {"inputs": [".bss"], "zero": True}}
TOOLS = {"mips_objcopy": "OBJCOPY", "mips_ld": "LD"}


@dataclass
class Owned:
    section: str
    vram: int
    rom_start: int
    size: int


@dataclass
class Window:
    address: int
    start: int
    end: int


def elf(names: list[str]) -> bytes:
    """An ELF32 big-endian object with just a symbol table."""
    strtab = b"\0" + b"\0".join(n.encode() for n in names) + b"\0"
    offsets, at = [], 1
    for n in names:
        offsets.append(at)
        at += len(n) + 1
    symtab = b"\0" * 16 + b"".join(struct.pack(">IIIBBH", o, 0, 0, 0x10, 0, 0) for o in offsets)
    shstr = b"\0.symtab\0.strtab\0.shstrtab\0"
    body = strtab + symtab + shstr
    base = 52
    sections = [(0,) * 10,
                (1, 2, 0, 0, base + len(strtab), len(symtab), 2, 0, 4, 16),
                (9, 3, 0, 0, base, len(strtab), 0, 0, 1, 0),
                (17, 3, 0, 0, base + len(strtab) + len(symtab), len(shstr), 0, 0, 1, 0)]
    fields = struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, base + len(body), 0, 52, 0, 0, 40, 4, 3)
    header = b"\x7fELF\x01\x02\x01" + b"\0" * 9 + fields
    return header + body + b"".join(struct.pack(">10I", *s) for s in sections)


SYMBOLS = (b"PROVIDE(alpha = 0x80000010);\nPROVIDE(foo = 0x80000020);\nPROVIDE(foobar = 0x80000030);\n"
           b"beta = 0x80000040;\nthis line is not valid\n  PROVIDE(gamma = 0x80000050);\n")


def test_object_symbols_reads_the_table(tmp_path):
    (tmp_path / "a.o").write_bytes(elf(["foo", "beta", "gamma"]))
    assert link.object_symbols(tmp_path / "a.o") == {b"foo", b"beta", b"gamma"}


def test_needed_symbols_keeps_only_the_referenced_lines_verbatim(tmp_path):
    (tmp_path / "a.o").write_bytes(elf(["foo", "beta", "gamma", "absent"]))
    assert link.needed_symbols(tmp_path / "a.o", SYMBOLS) == (
        b"PROVIDE(foo = 0x80000020);\nbeta = 0x80000040;\nPROVIDE(gamma = 0x80000050);\n")  # foo does not take foobar


def test_a_file_that_is_not_an_elf_object_is_refused_by_name(tmp_path):
    (tmp_path / "a.o").write_bytes(b"nope")
    with pytest.raises(link.LinkFailed, match="a.o: not an ELF32 big-endian object"):
        link.object_symbols(tmp_path / "a.o")


def test_link_script_owns_and_sizes_every_section():
    script = link.link_script([Owned(".text", 0x80000400, 0x1000, 8)], "needed.ld", TABLE)
    assert script.startswith('INCLUDE "needed.ld"\nSECTIONS {\n.text 0x80000400  : SUBALIGN(4) { *(.text .text.*) }')
    assert 'ASSERT(SIZEOF(.text) >= 0x8' in link.link_script([Owned(".text", 1, 0, 8)], "n", TABLE, trim=True)
    assert 'ASSERT(SIZEOF(.text) == 0x8, "owned .text size")' in script
    assert '.bss 0x0 (NOLOAD) : { *(.bss) }' in script
    assert 'ASSERT(SIZEOF(.unowned) == 0, "unowned section")' in script


def test_link_script_subaligns_an_owned_section_at_an_address_that_is_not_16_aligned():
    table = {**TABLE, "data": {"inputs": [".data"], "zero": False}}
    script = link.link_script([Owned(".data", 0x800CA1EC, 0, 0xA0)], "n", table)
    assert ".data 0x800CA1EC  : SUBALIGN(4) { *(.data) }" in script
    eight = link.link_script([Owned(".data", 0x800CA1E8, 0, 1)], "n", table)
    assert ".data 0x800CA1E8  : SUBALIGN(8) { *(.data) }" in eight
    aligned = link.link_script([Owned(".data", 0x800CA1F0, 0, 0xA0)], "n", table)
    assert ".data 0x800CA1F0  : { *(.data) }" in aligned
    text = link.link_script([Owned(".text", 0x80000402, 0, 8)], "n", table)
    assert ".text 0x80000402  : SUBALIGN(4) { *(.text .text.*) }" in text


def test_link_script_keeps_noload_before_the_colon_when_a_zero_section_is_subaligned():
    script = link.link_script([Owned(".bss", 0x800CA1E8, 0, 0x10)], "n", TABLE)
    assert ".bss 0x800CA1E8 (NOLOAD) : SUBALIGN(8) { *(.bss) }" in script


@pytest.mark.parametrize(("trim", "score"), [(False, False), (True, True)])
def test_link_commands_are_ld_and_one_objcopy_per_section(trim, score):
    sections = [Owned(".text", 0x1000, 0x400, 8), Owned(".rodata", 0x2000, 0x500, 8), Owned(".bss", 0x3000, 0, 16)]
    commands = link.link_commands(TOOLS, Path("source.o"), sections, Path("work"), trim, score=score)
    ld, text, *rest = commands
    trimmed, (rodata,) = [c for c in rest if c[0] == "sh"], [c for c in rest if c[0] == "OBJCOPY"]
    assert ld == ("LD", "-EB", *(("--noinhibit-exec",) if score else ()), "-T", "work/unit.ld", "-o", "work/linked.elf",
                  "source.o")
    assert text == ("OBJCOPY", "-O", "binary", "-j", ".text", "work/linked.elf", "work/text.bin")
    assert rodata[:3] == ("OBJCOPY", "-O", "binary") and rodata[rodata.index("--set-section-flags") + 1] == (
        ".rodata=alloc,load,contents")
    assert len(trimmed) == (1 if trim else 0) and (not trim or commands[2] == trimmed[0])  # right after the code
    assert not any("place" in w for c in commands for w in c) and not any(".bss" in c for c in commands)


def test_the_trim_command_cuts_zero_padding_and_refuses_other_bytes(tmp_path):
    import subprocess
    commands = link.link_commands(TOOLS, Path("source.o"), [Owned(".text", 0x1000, 0, 4)], tmp_path, True)
    cut = commands[-1][2]
    (tmp_path / "text.bin").write_bytes(b"\x01\x02\x03\x04\0\0\0\0")
    assert subprocess.run(["sh", "-c", cut]).returncode == 0
    assert (tmp_path / "text.bin").read_bytes() == b"\x01\x02\x03\x04"
    (tmp_path / "text.bin").write_bytes(b"\x01\x02\x03\x04\0\x05")
    assert subprocess.run(["sh", "-c", cut]).returncode != 0
    assert (tmp_path / "text.bin").read_bytes() == b"\x01\x02\x03\x04\0\x05"


class Fake:
    """Runs the one shell that holds the link chain without a toolchain: it writes the files the chain writes."""

    def __init__(self, mode: str = "exact") -> None:
        self.mode, self.calls = mode, []

    def __call__(self, phase, argv, cwd):
        assert argv[:2] == ["/bin/sh", "-c"]
        self.calls.append(argv[2])
        if self.mode == "failed":
            return SimpleNamespace(exit=2, signal=None, stderr=b"boom")
        (cwd / "linked.elf").write_bytes(b"elf")
        (cwd / "text.bin").write_bytes(b"x" * (7 if self.mode == "wrong-size" else 8))
        return SimpleNamespace(exit=0, signal=None, stderr=b"")


def run_link(tmp_path, mode, score=False):
    (tmp_path / "a.o").write_bytes(elf(["foo"]))
    (tmp_path / "symbols.ld").write_bytes(SYMBOLS)
    sections = [Owned(".text", 0x1000, 0x400, 8), Owned(".bss", 0x2000, 0, 16)]
    fake = Fake(mode)
    done = link.link(fake, TOOLS, tmp_path / "a.o", sections, tmp_path / "symbols.ld", tmp_path / "work", False, TABLE,
                     score=score)
    return done, fake


def test_link_gives_the_linker_only_the_needed_symbols(tmp_path):
    done, fake = run_link(tmp_path, "exact")
    work = tmp_path / "work"
    assert (work / "needed.ld").read_text() == "PROVIDE(foo = 0x80000020);\n"
    script = (work / "unit.ld").read_text()
    assert f'INCLUDE "{work}/needed.ld"' in script and ".text 0x1000  : SUBALIGN(4)" in script
    assert set(done.outputs) == {"elf", ".text", ".bss"} and (work / "bss.size").read_text() == "16"
    assert len(fake.calls) == 1 and fake.calls[0].count(" && ") == 1  # ld, then objcopy: one process


@pytest.mark.parametrize("mode", ["failed", "wrong-size"])
def test_a_broken_link_is_refused(tmp_path, mode):
    with pytest.raises(link.LinkFailed):
        run_link(tmp_path, mode)


def test_a_draft_of_the_wrong_size_is_scored_not_refused(tmp_path):
    done, _ = run_link(tmp_path, "wrong-size", score=True)
    assert done.outputs[".text"].stat().st_size == 7


def test_trace_lists_every_command_in_order(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="abucache")
    run_link(tmp_path, "exact")
    lines = [r.getMessage() for r in caplog.records]
    assert [line.split()[1] for line in lines] == ["LD", "OBJCOPY"]
    assert all(line.startswith("link: ") for line in lines)


def test_link_script_discards_the_mips_options_section_with_the_other_metadata():
    script = link.link_script([Owned(".text", 0x80000400, 0x1000, 8)], "n", TABLE)
    discard = ".MIPS.abiflags .reginfo .pdr .mdebug* .options .debug* .comment .note* .gnu.attributes"
    assert f"/DISCARD/ : {{ *({discard}) }}" in script
