"""Surface enumeration — sizes the field the mower is mowing.

Two questions this module answers:

  PTX surface     : every (op, type-tuple) cell our compiler's dispatcher
                    knows how to lower.  Parsed once from openptxas/sass/
                    isel.py — it's the canonical authority for "things we
                    can codegen".

  SASS surface    : every distinct SASS opcode (low 12 bits of bytes 0-1)
                    that has appeared in cubins emitted so far.  Walked
                    out of the probe DB's cubin store.

Coverage = how much of each surface our probes have actually exercised.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Iterable

from .db import ProbeDB


# ---------------------------------------------------------------------------
# PTX surface — extract (op, type-key) cells from isel.py dispatcher
# ---------------------------------------------------------------------------

_ELIF_RE = re.compile(
    r"^\s+elif\s+op\s*==\s*'([^']+)'\s*(.*?)\s*:\s*$",
    re.MULTILINE,
)
# Inner-clause patterns we recognize after `op == 'X'`:
#   and typ in (...)            -> set of types
#   and typ == 'X'              -> single type
#   and 'X' in instr.types      -> modifier set
_TYP_IN_RE     = re.compile(r"typ\s+in\s+\(([^)]+)\)")
_TYP_EQ_RE     = re.compile(r"typ\s*==\s*'([^']+)'")
_MOD_IN_TYPES  = re.compile(r"'([^']+)'\s+in\s+instr\.types")


def _extract_string_tuple(src: str) -> tuple[str, ...]:
    """Pull "'a', 'b', 'c'" out of `( 'a', 'b', 'c' )`."""
    return tuple(re.findall(r"'([^']+)'", src))


def enumerate_ptx_surface(isel_path: str | Path) -> set[tuple[str, frozenset]]:
    """Walk isel.py and return the canonical set of dispatcher cells.

    Each cell is (op_name, frozenset_of_qualifiers) where qualifiers
    are types/modifiers from the elif clause.  An elif with no type
    qualifier yields cell (op, frozenset()).  An elif with `typ in
    ('a','b','c')` yields three cells (op, {'a'}), (op, {'b'}),
    (op, {'c'}).

    This is what we'd call "the front of the field": every shape our
    isel handles is a cell, regardless of operand-imm subdivisions.
    """
    cells: set[tuple[str, frozenset]] = set()
    src = Path(isel_path).read_text(encoding="utf-8")
    for m in _ELIF_RE.finditer(src):
        op = m.group(1)
        tail = m.group(2)

        # Pull the type set from `typ in (...)` or `typ == 'X'`
        types: list[str] = []
        for tin in _TYP_IN_RE.findall(tail):
            types.extend(_extract_string_tuple(f"({tin})"))
        types.extend(_TYP_EQ_RE.findall(tail))

        # Modifiers from `'X' in instr.types`.  Treated as required-
        # qualifiers attached to every cell (split per type if there
        # are types; otherwise a single cell).
        mods = list(_MOD_IN_TYPES.findall(tail))

        if types:
            for t in types:
                quals = frozenset([t, *mods])
                cells.add((op, quals))
        else:
            cells.add((op, frozenset(mods)))
    return cells


def ptx_surface_summary(cells: set[tuple[str, frozenset]]) -> dict:
    """Group cells by op name; report counts and per-op breakdown."""
    by_op: dict[str, list[frozenset]] = {}
    for op, quals in cells:
        by_op.setdefault(op, []).append(quals)
    return {
        "total_cells": len(cells),
        "distinct_ops": len(by_op),
        "by_op": {op: len(qs) for op, qs in sorted(by_op.items())},
    }


# ---------------------------------------------------------------------------
# SASS surface — distinct opcodes seen in cubins
# ---------------------------------------------------------------------------

def _decode_opcode(raw16: bytes) -> int:
    """Low 12 bits of (b0 | b1<<8) — the SASS opcode key."""
    if len(raw16) < 2:
        return 0
    return (raw16[0] | (raw16[1] << 8)) & 0xFFF


def _walk_text_section(cubin: bytes) -> Iterable[bytes]:
    """Yield each 16-byte instruction in every .text.<symbol> section."""
    if len(cubin) < 64:
        return
    e_shoff = struct.unpack_from("<Q", cubin, 40)[0]
    e_shnum = struct.unpack_from("<H", cubin, 60)[0]
    e_shstrndx = struct.unpack_from("<H", cubin, 62)[0]
    sh = e_shoff + e_shstrndx * 64
    sh_offset = struct.unpack_from("<Q", cubin, sh + 24)[0]
    sh_size   = struct.unpack_from("<Q", cubin, sh + 32)[0]
    if sh_offset + sh_size > len(cubin):
        return
    shstrtab = cubin[sh_offset:sh_offset + sh_size]
    for i in range(e_shnum):
        soff = e_shoff + i * 64
        if soff + 64 > len(cubin):
            break
        name_off = struct.unpack_from("<I", cubin, soff)[0]
        end = shstrtab.find(b"\x00", name_off)
        name = bytes(shstrtab[name_off:end if end >= 0 else len(shstrtab)])
        if not name.startswith(b".text."):
            continue
        off = struct.unpack_from("<Q", cubin, soff + 24)[0]
        sz  = struct.unpack_from("<Q", cubin, soff + 32)[0]
        if off + sz > len(cubin) or sz % 16 != 0:
            continue
        for k in range(sz // 16):
            yield cubin[off + k * 16:off + k * 16 + 16]


def enumerate_sass_opcodes_seen(db: ProbeDB) -> dict[int, dict]:
    """Return {opcode: {ours_count, ptxas_count, first_probe_id}} from cubins."""
    seen: dict[int, dict] = {}

    rows = db.query(
        "SELECT probe_id, ours_cubin_sha, ptxas_cubin_sha FROM probes "
        "WHERE error IS NULL ORDER BY probe_id"
    )
    for probe_id, ours_sha, ptxas_sha in rows:
        for kind, sha in (("ours", ours_sha), ("ptxas", ptxas_sha)):
            if not sha:
                continue
            cubin = db.get_cubin(sha)
            if cubin is None:
                continue
            for raw in _walk_text_section(cubin):
                opc = _decode_opcode(raw)
                if opc == 0:
                    continue
                rec = seen.setdefault(
                    opc, {"ours_count": 0, "ptxas_count": 0,
                          "first_probe_id": probe_id})
                rec[f"{kind}_count"] += 1
    return seen


# ---------------------------------------------------------------------------
# Probe -> dispatcher cell mapping (which cells have we hit?)
# ---------------------------------------------------------------------------

def probe_cells_targeted(db: ProbeDB) -> set[tuple[str, frozenset]]:
    """Cells the probes specifically TARGET via target_op.  This is the
    focused testing surface — the ops we're zooming in on with parametric
    sweeps."""
    hit: set[tuple[str, frozenset]] = set()
    rows = db.query("SELECT target_op FROM probes WHERE error IS NULL")
    for (target_op,) in rows:
        parts = target_op.split(".")
        if not parts:
            continue
        op_root = parts[0]
        quals = [p for p in parts[1:] if p not in ("global", "shared")]
        hit.add((op_root, frozenset(quals)))
    return hit


_PTX_INSTR_RE = re.compile(
    # Match a PTX statement: skip leading whitespace and optional pred,
    # then capture opcode (lowercase identifier with dot-separated mods).
    # Statements can be multiple-per-line separated by `;`, so we don't
    # anchor to line boundaries — just take an opcode-shaped token after
    # a delimiter (start of line, `;`, or `}`).  Filtered downstream by
    # known_ops to drop directive matches.
    r"(?:^|[;{}])\s*(?:@!?%\w+\s+)?"
    r"([a-z][a-z0-9]*(?:\.[a-z0-9_]+)*)\b",
    re.MULTILINE,
)
# PTX directives / non-instructions that share the leading `.identifier`
# shape in some matches.  Filter by op-root being in the dispatcher's set.
_DIRECTIVE_PREFIXES = (
    "version", "target", "address_size", "visible", "entry", "param",
    "reg", "shared", "extern", "func", "weak", "global", "local",
    "pragma", "loc", "file", "section",
)


def probe_cells_exercised(db: ProbeDB,
                          known_ops: set[str]) -> set[tuple[str, frozenset]]:
    """Cells the probes exercise — every PTX op present in any probe's PTX
    text, whether targeted or incidentally compiled.  Filtered to `known_ops`
    (the dispatcher's op set) so directive-shaped lines are ignored."""
    hit: set[tuple[str, frozenset]] = set()
    rows = db.query("SELECT DISTINCT ptx_sha FROM probes WHERE error IS NULL")
    for (sha,) in rows:
        ptx = db.get_ptx(sha)
        if not ptx:
            continue
        for m in _PTX_INSTR_RE.finditer(ptx):
            tok = m.group(1)
            if tok.startswith("."):
                continue
            parts = tok.split(".")
            op_root = parts[0]
            if op_root in _DIRECTIVE_PREFIXES:
                continue
            if op_root not in known_ops:
                continue
            quals = [p for p in parts[1:] if p not in ("global", "shared")]
            hit.add((op_root, frozenset(quals)))
    return hit


# ---------------------------------------------------------------------------
# Encoder-catalog audit — list every encode_* function and its opcode,
# then cross-reference against opcodes actually seen in cubins.  Surfaces
# "we have this encoder but never call it" gaps.
# ---------------------------------------------------------------------------

def enumerate_encoders(modules: tuple = (
    "sass.encoding.sm_120_opcodes",
    "sass.encoding.sm_120_encode",
)) -> dict[str, dict]:
    """Walk encoder modules, call each `encode_*` with safe sample args,
    and return {encoder_name: {opcode, module, sample_bytes, error}}."""
    import importlib, inspect
    out: dict[str, dict] = {}
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            continue
        for name in dir(mod):
            if not name.startswith("encode_"):
                continue
            fn = getattr(mod, name, None)
            if not callable(fn):
                continue
            sig = None
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                pass
            # Build sample args: 0 for first 4 positional, 0xFF (RZ) for
            # remaining register slots, 0 elsewhere.  Catches signatures
            # we can fill safely; skip if it errors.
            args: list = []
            if sig:
                for i, (pname, param) in enumerate(sig.parameters.items()):
                    if param.default is not inspect.Parameter.empty:
                        break  # rest have defaults; stop adding positional
                    # First 4 positional get 0; later positional get 0xFF (RZ).
                    args.append(0 if i < 4 else 0xFF)
            try:
                raw = fn(*args)
                if isinstance(raw, (bytes, bytearray)) and len(raw) >= 2:
                    opc = (raw[0] | (raw[1] << 8)) & 0xFFF
                    out[name] = {
                        "opcode": opc,
                        "module": mod_name.rsplit(".", 1)[-1],
                        "sample": bytes(raw),
                        "error": None,
                    }
                else:
                    out[name] = {
                        "opcode": None, "module": mod_name.rsplit(".", 1)[-1],
                        "sample": None, "error": "not bytes",
                    }
            except Exception as e:
                out[name] = {
                    "opcode": None,
                    "module": mod_name.rsplit(".", 1)[-1],
                    "sample": None,
                    "error": f"{type(e).__name__}: {e}",
                }
    return out


def encoder_audit(db: ProbeDB) -> dict:
    """Cross-reference encoders with opcodes seen in cubins."""
    encoders = enumerate_encoders()
    seen = enumerate_sass_opcodes_seen(db)
    seen_opcodes = set(seen.keys())

    by_opcode: dict[int, list[str]] = {}
    for name, info in encoders.items():
        opc = info["opcode"]
        if opc is None:
            continue
        by_opcode.setdefault(opc, []).append(name)

    covered = []
    uncovered = []
    errored = []
    for name, info in sorted(encoders.items()):
        if info["error"]:
            errored.append((name, info["error"]))
            continue
        opc = info["opcode"]
        if opc in seen_opcodes:
            covered.append((name, opc))
        else:
            uncovered.append((name, opc))

    return {
        "encoders_total": len(encoders),
        "encoders_callable": len(encoders) - len(errored),
        "covered": covered,
        "uncovered": uncovered,
        "errored": errored,
        "by_opcode": by_opcode,
        "seen_opcodes": sorted(seen_opcodes),
    }


# ---------------------------------------------------------------------------
# Top-level survey — used by the `probe-survey` CLI command
# ---------------------------------------------------------------------------

def _count_covered(cells: set[tuple[str, frozenset]],
                   hit:   set[tuple[str, frozenset]]) -> set[tuple[str, frozenset]]:
    """Match dispatcher cells against probe-hit cells using subset semantics."""
    hit_by_op: dict[str, list[frozenset]] = {}
    for op, quals in hit:
        hit_by_op.setdefault(op, []).append(quals)
    covered = set()
    for op, quals in cells:
        for hquals in hit_by_op.get(op, []):
            if not quals or quals.issubset(hquals) or hquals.issubset(quals):
                covered.add((op, quals))
                break
    return covered


def survey(db: ProbeDB, isel_path: str | Path) -> dict:
    """Produce the field-size report.  Pure data, no printing."""
    ptx_cells = enumerate_ptx_surface(isel_path)
    summary   = ptx_surface_summary(ptx_cells)
    known_ops = set(summary["by_op"].keys())

    targeted  = probe_cells_targeted(db)
    exercised = probe_cells_exercised(db, known_ops)

    targeted_ops  = {op for op, _ in targeted}
    exercised_ops = {op for op, _ in exercised}

    targeted_cells  = _count_covered(ptx_cells, targeted)
    exercised_cells = _count_covered(ptx_cells, exercised)

    sass_seen = enumerate_sass_opcodes_seen(db)

    return {
        "ptx_surface": {
            "total_cells":     summary["total_cells"],
            "distinct_ops":    summary["distinct_ops"],
            "targeted_cells":  len(targeted_cells),
            "targeted_ops":    len(targeted_ops),
            "exercised_cells": len(exercised_cells),
            "exercised_ops":   len(exercised_ops),
            "by_op":           summary["by_op"],
            "untargeted_ops":  sorted(known_ops - targeted_ops),
            "unexercised_ops": sorted(known_ops - exercised_ops),
        },
        "sass_surface": {
            "distinct_opcodes": len(sass_seen),
            "opcodes_seen":     sorted(sass_seen.keys()),
            "details":          sass_seen,
        },
    }
