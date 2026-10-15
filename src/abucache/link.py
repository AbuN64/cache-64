# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 mo
"""Linking one unit: the symbols it needs, the linker script, the commands, and running them. A unit is linked with
ld and one objcopy per section. Nothing reads the ROM."""

from __future__ import annotations

import re
import shlex
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from abucache import log
from abucache.compile import Runner, failed


class Owned(Protocol):
    section: str
    vram: int
    rom_start: int
    size: int


class LinkFailed(Exception):
    def __init__(self, reason: str, result: Any = None) -> None:
        super().__init__(reason)
        self.reason, self.result = reason, result


@dataclass(frozen=True)
class Linked:
    result: Any  # the result of the last command
    outputs: dict[str, Path]  # "elf" and each owned section to its file


def section_file(section: str) -> str:
    return section[1:] + (".size" if section == ".bss" else ".bin")


def link_script(sections: Sequence[Owned], symbols: str, table: Mapping[str, Mapping[str, Any]],
                *, trim: bool = False) -> str:
    """All owned sections are sourced and sized, so unowned allocatable bytes cannot hide in a base-ROM copy.
    TABLE is the section vocabulary: name -> {inputs, zero}. The code is placed at its owned address with 4-byte
    alignment (an input aligned wider would move it up). Any other section at an address that is not 16-aligned gets
    SUBALIGN of the lowest set bit of that address, because gcc aligns inputs such as .data to 16 and ld would
    otherwise move the section up and pad its front. ld takes SUBALIGN after the colon and (NOLOAD) before it.
    A TRIM unit may carry zero padding past its row, which `link_commands` cuts, so its size assert is a lower
    bound."""
    owned = {p.section: p for p in sections}
    lines = [f'INCLUDE "{symbols}"', "SECTIONS {"]
    for name, row in table.items():
        section = "." + name
        p = owned.get(section)
        address, size = (p.vram, p.size) if p else (0, 0)
        inputs = " ".join(row["inputs"])
        if section == ".text":
            wide = " : SUBALIGN(4)"
        elif address % 16:
            wide = f" : SUBALIGN({address & -address})"
        else:
            wide = " :"
        zero = "(NOLOAD)" if row["zero"] else ""
        lines.append(f"{section} 0x{address:X} {zero}{wide} {{ *({inputs}) }}")
        compare = ">=" if trim and section == ".text" else "=="
        lines.append(f'ASSERT(SIZEOF({section}) {compare} 0x{size:X}, "owned {section} size")')
    discard = ".MIPS.abiflags .reginfo .pdr .mdebug* .options .debug* .comment .note* .gnu.attributes"
    lines += [f"/DISCARD/ : {{ *({discard}) }}",
              '.unowned 0 : { *(*) }', 'ASSERT(SIZEOF(.unowned) == 0, "unowned section")', "}"]
    return "\n".join(lines) + "\n"


def link_commands(tools: Mapping[str, str], obj: Path, sections: Sequence[Owned], work: Path, trim: bool,
                  *, score: bool = False) -> tuple[tuple[str, ...], ...]:
    """The commands that turn a unit's object into its section files: ld, then one objcopy per section. The
    constant sections are made allocatable on the objcopy that extracts them. SCORE keeps the ELF a failed size
    assert would otherwise stop. A TRIM unit's code loses its trailing zero padding down to the row."""
    elf, commands = work / "linked.elf", []
    commands.append([tools["mips_ld"], "-EB", *(["--noinhibit-exec"] if score else []), "-T", str(work / "unit.ld"),
                     "-o", str(elf), str(obj)])
    for p in sections:
        if p.section == ".bss":
            continue
        flags = [] if p.section == ".text" else ["--set-section-flags", f"{p.section}=alloc,load,contents"]
        out = work / (p.section[1:] + ".bin")
        commands.append([tools["mips_objcopy"], "-O", "binary", "-j", p.section, *flags, str(elf), str(out)])
        if trim and p.section == ".text":
            cut = (f"tail -c +{p.size + 1} {out} | tr -d '\\000' | cmp -s - /dev/null "
                   f"&& head -c {p.size} {out} > {out}.trim && mv {out}.trim {out}")
            commands.append(["sh", "-c", cut])
    return tuple(tuple(argv) for argv in commands)


def object_symbols(obj: Path) -> set[bytes]:
    """The names in the symbol table of an ELF32 big-endian object, defined or not."""
    data = obj.read_bytes()
    if data[:6] != b"\x7fELF\x01\x02":
        raise LinkFailed(f"{obj}: not an ELF32 big-endian object")
    table, entry, count = struct.unpack_from(">I", data, 32)[0], *struct.unpack_from(">HH", data, 46)
    names: set[bytes] = set()
    for index in range(count):
        _, kind, _, _, offset, size, link, _, _, entsize = struct.unpack_from(">10I", data, table + index * entry)
        if kind != 2 or not entsize:
            continue
        strings = struct.unpack_from(">10I", data, table + link * entry)
        blob = data[strings[4]:strings[4] + strings[5]]
        for at in range(offset, offset + size, entsize):
            start = struct.unpack_from(">I", data, at)[0]
            if start:
                names.add(blob[start:blob.index(b"\0", start)])
    return names


def needed_symbols(obj: Path, symbols: bytes) -> bytes:
    """The lines of a symbol file that name a symbol of the object, verbatim. The linker and place read nothing else."""
    names = sorted(object_symbols(obj), key=len, reverse=True)
    if not names:
        return b""
    pattern = re.compile(rb"^[ \t]*(?:PROVIDE\()?[ \t]*(?:" + b"|".join(map(re.escape, names)) + rb")[ \t]*=[^\n]*",
                         re.MULTILINE)
    return b"".join(match.group(0).strip() + b"\n" for match in pattern.finditer(symbols))


def link(run: Runner, tools: Mapping[str, str], obj: Path, sections: Sequence[Owned], symbols: Path, work: Path,
         trim: bool, table: Mapping[str, Mapping[str, Any]], *, score: bool = False) -> Linked:
    """Link a unit: ld and the section files, with only the needed symbols. A failed command, a signal or a
    missing or wrongly sized output raises LinkFailed (with SCORE a size assert is not a failure, the sizes are)."""
    work.mkdir(parents=True, exist_ok=True)
    needed = work / "needed.ld"
    needed.write_bytes(needed_symbols(obj, symbols.read_bytes()))
    (work / "unit.ld").write_text(link_script(sections, str(needed), table, trim=trim))
    commands = link_commands(tools, obj, sections, work, trim, score=score)
    for argv in commands:
        log.info("link: %s", " ".join(argv))
    # One process runs the whole chain: three spawns from the caller cost more than the tools do.
    result = run("link", ["/bin/sh", "-c", " && ".join(shlex.join(argv) for argv in commands)], work)
    if failed(result):
        raise LinkFailed("link failed", result)
    outputs: dict[str, Path] = {"elf": work / "linked.elf"}
    if not outputs["elf"].is_file():
        raise LinkFailed("linked ELF is missing", result)
    for p in sections:
        binary = work / section_file(p.section)
        if p.section == ".bss":
            binary.write_text(str(p.size))  # ld has asserted the actual memory extent
        elif not binary.is_file() or (not score and binary.stat().st_size != p.size):
            raise LinkFailed(f"owned {p.section} output is missing or incorrectly sized", result)
        outputs[p.section] = binary
    return Linked(result, outputs)
