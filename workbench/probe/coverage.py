"""Coverage axes: a probe spec maps to a set of (axis, bin_key) tuples.
The scheduler pulls unfilled bins from the DB and asks an axis-specific
synthesizer to materialize a probe spec for that bin.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable

from .generator import ProbeSpec
from . import surface


# ---------------------------------------------------------------------------
# Axis 1: opcode_x_imm_class
#   Bins: {op}/{imm_class}/{acc_alias}
#     op           : PTX op label (e.g. mad.lo.u32)
#     imm_class    : zero | one | pow2 | non_pow2_small | large | negative
#     acc_alias    : acc_self | fresh
# ---------------------------------------------------------------------------

_IMM_CLASSES = {
    "zero":             0,
    "one":              1,
    "pow2_2":           2,
    "pow2_4":           4,
    "pow2_8":           8,
    "non_pow2_3":       3,
    "non_pow2_5":       5,
    "non_pow2_6":       6,
    "non_pow2_7":       7,
    "non_pow2_15":     15,
    "non_pow2_127":   127,
    "small_255":      255,
    "large_65535":  65535,
    "large_0x10000": 0x10000,
}

_ALU_OPS = (
    "mad.lo.u32", "mul.lo.u32",
    "add.u32", "sub.u32",
    "and.b32", "or.b32", "xor.b32",
    "shl.b32", "shr.b32",
)


def axis_opcode_imm_acc_bins() -> list[str]:
    """All bins on the opcode × imm-class × acc-alias axis."""
    bins = []
    for op in _ALU_OPS:
        for imm_label in _IMM_CLASSES:
            for acc in ("acc_self", "fresh"):
                bins.append(f"{op}/{imm_label}/{acc}")
    return bins


def synthesize_opcode_imm_acc(bin_key: str) -> ProbeSpec | None:
    """Reverse-map a bin_key to a ProbeSpec.  Skips ops that don't fit
    the (dst, src0, imm, [src2]) shape natively."""
    parts = bin_key.split("/")
    if len(parts) != 3:
        return None
    op, imm_label, acc = parts
    imm = _IMM_CLASSES.get(imm_label)
    if imm is None:
        return None
    if op == "mad.lo.u32":
        return ProbeSpec(
            template_id="alu_acc_self" if acc == "acc_self" else "alu_single",
            target_op="mad.lo.u32",
            operand_spec=(
                {"op": "mad.lo.u32", "imm": imm, "init_acc": 7}
                if acc == "acc_self"
                else {"op_text": f"mad.lo.u32 %r2, %r0, {imm}, %r3", "init_acc": 0}
            ),
        )
    # Non-mad ops: use alu_single template with appropriate op_text.
    if acc == "acc_self":
        # `add.u32 %r2, %r2, K` etc.
        op_text = f"{op} %r2, %r2, {imm}"
    else:
        op_text = f"{op} %r2, %r0, {imm}"
    return ProbeSpec(
        template_id="alu_single",
        target_op=op,
        operand_spec={"op_text": op_text, "init_acc": 0},
    )


# ---------------------------------------------------------------------------
# Axis 2: hazard_pair_x_distance
#   Bins: {op_a}->{op_b}/dist={N}
#   Probes: alu chain that exercises a writer/reader pair separated by N nops.
# ---------------------------------------------------------------------------

_HAZARD_PAIRS = [
    # (writer_op, reader_op_template_using_%r2_as_input)
    ("mad.lo.u32 %r2, %r0, 6, %r2", "add.u32 %r2, %r2, 1"),
    ("mul.lo.u32 %r2, %r0, 6",       "add.u32 %r2, %r2, 1"),
    ("add.u32 %r2, %r0, 1",          "xor.b32 %r2, %r2, 0xff"),
    ("xor.b32 %r2, %r0, 0xaa",       "and.b32 %r2, %r2, 0xff"),
    ("shl.b32 %r2, %r0, 2",          "or.b32 %r2, %r2, 1"),
]
_HAZARD_DISTS = [0, 1, 2, 3, 4, 5, 8, 16]


def axis_hazard_pair_dist_bins() -> list[str]:
    bins = []
    for op_a, op_b in _HAZARD_PAIRS:
        a_label = op_a.split()[0]
        b_label = op_b.split()[0]
        for d in _HAZARD_DISTS:
            bins.append(f"{a_label}->{b_label}/dist={d}")
    return bins


def synthesize_hazard_pair_dist(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 2:
        return None
    pair_label, dist_part = parts
    a_label, b_label = pair_label.split("->")
    if not dist_part.startswith("dist="):
        return None
    gap = int(dist_part[5:])
    # find the canonical pair for this label
    for op_a, op_b in _HAZARD_PAIRS:
        if op_a.split()[0] == a_label and op_b.split()[0] == b_label:
            return ProbeSpec(
                template_id="pair_distance",
                target_op=a_label,
                operand_spec={"op_a": op_a, "op_b": op_b, "gap": gap},
            )
    return None


# ---------------------------------------------------------------------------
# Axis 3: latency_writer_x_gap
#   Bins: {writer_label}/gap={N}
#   Probes: writer with N nops, then read.  Find the minimum N where
#   the GPU output is correct.
# ---------------------------------------------------------------------------

_LATENCY_WRITERS = [
    ("mad.lo.u32 %r2, %r0, 6, %r2", "mad_lo_K6"),
    ("mad.lo.u32 %r2, %r0, 4, %r2", "mad_lo_K4"),
    ("mad.lo.u32 %r2, %r0, 7, %r2", "mad_lo_K7"),
    ("mul.lo.u32 %r2, %r0, 13",     "mul_lo_K13"),
    ("add.u32 %r2, %r0, 100",       "add_K100"),
    ("xor.b32 %r2, %r0, 0xaa",      "xor_Kaa"),
    ("shl.b32 %r2, %r0, 3",         "shl_K3"),
]
_LATENCY_GAPS = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16]


def axis_latency_writer_gap_bins() -> list[str]:
    bins = []
    for writer, label in _LATENCY_WRITERS:
        for gap in _LATENCY_GAPS:
            bins.append(f"{label}/gap={gap}")
    return bins


def synthesize_latency_writer_gap(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 2:
        return None
    label, gap_part = parts
    if not gap_part.startswith("gap="):
        return None
    gap = int(gap_part[4:])
    for writer, lbl in _LATENCY_WRITERS:
        if lbl == label:
            target_op = writer.split()[0]
            return ProbeSpec(
                template_id="latency_sweep",
                target_op=target_op,
                operand_spec={"writer": writer, "init": 7, "gap": gap},
            )
    return None


# ---------------------------------------------------------------------------
# Registry — the scheduler iterates this.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Axis 4: alu_64bit_x_imm
#   Bins: {op}/{imm_class}
# ---------------------------------------------------------------------------

_ALU_64_OPS = ("add.u64", "sub.u64", "and.b64", "or.b64", "xor.b64", "shl.b64", "shr.u64")


def axis_alu_64bit_imm_bins() -> list[str]:
    bins = []
    for op in _ALU_64_OPS:
        for imm_label in _IMM_CLASSES:
            bins.append(f"{op}/{imm_label}")
    return bins


def synthesize_alu_64bit_imm(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 2:
        return None
    op, imm_label = parts
    imm = _IMM_CLASSES.get(imm_label)
    if imm is None:
        return None
    return ProbeSpec(
        template_id="alu_64bit",
        target_op=op,
        operand_spec={"op_text": f"{op} %rd2, %rd1, {imm}"},
    )


# ---------------------------------------------------------------------------
# Axis 5: load_consume_x_gap
#   Bins: {consume_op}/gap={N}
# ---------------------------------------------------------------------------

_LOAD_CONSUME_OPS = (
    "add.u32 %r2, %r2, 5",
    "xor.b32 %r2, %r2, 0xff",
    "mul.lo.u32 %r2, %r2, 7",
    "shl.b32 %r2, %r2, 2",
)
_LOAD_CONSUME_GAPS = [0, 1, 2, 3, 4, 6, 8, 12]


def axis_load_consume_gap_bins() -> list[str]:
    bins = []
    for op in _LOAD_CONSUME_OPS:
        label = op.split()[0]
        for gap in _LOAD_CONSUME_GAPS:
            bins.append(f"{label}/gap={gap}")
    return bins


def synthesize_load_consume_gap(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 2:
        return None
    label, gap_part = parts
    if not gap_part.startswith("gap="):
        return None
    gap = int(gap_part[4:])
    for op in _LOAD_CONSUME_OPS:
        if op.split()[0] == label:
            return ProbeSpec(
                template_id="load_consume",
                target_op=label,
                operand_spec={"consume_op": op, "gap": gap},
            )
    return None


# ---------------------------------------------------------------------------
# Axis 6: predicated_alu_x_op
#   Bins: {op}/{cond}/thr={N}
# ---------------------------------------------------------------------------

_PRED_OPS = (
    "add.u32 %r2, %r2, 1",
    "xor.b32 %r2, %r2, 0xff",
    "mul.lo.u32 %r2, %r2, 3",
)
_PRED_CONDS = ("lt", "gt", "eq", "ne")
_PRED_THRS = (0, 32, 64, 127)


def axis_predicated_alu_bins() -> list[str]:
    bins = []
    for op in _PRED_OPS:
        label = op.split()[0]
        for cond in _PRED_CONDS:
            for thr in _PRED_THRS:
                bins.append(f"{label}/{cond}/thr={thr}")
    return bins


def synthesize_predicated_alu(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 3:
        return None
    label, cond, thr_part = parts
    if not thr_part.startswith("thr="):
        return None
    thr = int(thr_part[4:])
    for op in _PRED_OPS:
        if op.split()[0] == label:
            return ProbeSpec(
                template_id="predicated_alu",
                target_op=label,
                operand_spec={"op_text": op, "pred_cond": cond, "pred_thr": thr},
            )
    return None


# ---------------------------------------------------------------------------
# Axis 7: atomic_x_op
#   Bins: atom.{op}/init={N}/arg={M}
# ---------------------------------------------------------------------------

_ATOMIC_OPS = ("add.u32", "and.b32", "or.b32", "xor.b32", "min.u32", "max.u32")


def axis_atomic_op_bins() -> list[str]:
    bins = []
    for op in _ATOMIC_OPS:
        for init_val in (0, 1, 0xff):
            for arg in (1, 7, 0xff):
                bins.append(f"atom.{op}/init={init_val}/arg={arg}")
    return bins


def synthesize_atomic_op(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 3:
        return None
    op_part, init_part, arg_part = parts
    if not op_part.startswith("atom."):
        return None
    op = op_part[5:]
    init_val = int(init_part.split("=", 1)[1], 0)
    arg = int(arg_part.split("=", 1)[1], 0)
    return ProbeSpec(
        template_id="atomic_op",
        target_op=f"atom.global.{op}",
        operand_spec={"op": op, "init_val": init_val, "arg": arg},
    )


# ---------------------------------------------------------------------------
# Axis 8: f32_alu
#   Bins: {op}/{operand_class}
#   operand_class:
#     fresh_const  — %f2 = op(%f1=tid_f, K)
#     acc_self     — %f2 = op(%f2, K)  (writer reads its own dest)
#     two_src      — %f2 = op(%f1, %f3=0)
# ---------------------------------------------------------------------------

# PTX float constants (hex bit-pattern). 0f3F800000 = 1.0, 0fC0000000 = -2.0,
# 0f40400000 = 3.0, 0f7F800000 = +inf.
_F32_CONSTS = {
    "1.0":   "0f3F800000",
    "0.5":   "0f3F000000",
    "2.0":   "0f40000000",
    "3.0":   "0f40400000",
    "neg1":  "0fBF800000",
    "tiny":  "0f00800000",   # smallest normal
    "inf":   "0f7F800000",
}

_F32_OPS = ("add.f32", "sub.f32", "mul.f32", "min.f32", "max.f32")


def axis_f32_alu_bins() -> list[str]:
    bins = []
    for op in _F32_OPS:
        for k_label in _F32_CONSTS:
            for shape in ("fresh_const", "acc_self"):
                bins.append(f"{op}/{k_label}/{shape}")
    return bins


def synthesize_f32_alu(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 3:
        return None
    op, k_label, shape = parts
    k = _F32_CONSTS.get(k_label)
    if k is None:
        return None
    if shape == "fresh_const":
        op_text = f"{op} %f2, %f1, {k}"
    else:  # acc_self
        op_text = f"{op} %f2, %f2, {k}"
    return ProbeSpec(
        template_id="alu_f32",
        target_op=op,
        operand_spec={"op_text": op_text},
    )


# ---------------------------------------------------------------------------
# Axis 9: cvt_op
#   Bins: {dst_type}_from_{src_type}/{rounding}
#   Matrix of common conversions. Some entries skip rounding when N/A.
# ---------------------------------------------------------------------------

# Each entry: (cvt_text, pre_context_lines)
# %r0 = tid (u32 already), %r2 = result (u32 stored).
_CVT_CASES = {
    "u32_from_s32":   ("cvt.u32.s32 %r2, %s1",         ["mov.s32 %s1, %r0;"]),
    "s32_from_u32":   ("cvt.s32.u32 %r2, %r0",         []),
    "u64_round_trip": ("cvt.u64.u32 %sd1, %r0; cvt.u32.u64 %r2, %sd1", []),
    "f32_from_u32":   ("cvt.rn.f32.u32 %f1, %r0; mov.b32 %r2, %f1", []),
    "u32_from_f32":   ("cvt.rn.f32.u32 %f1, %r0; cvt.rzi.u32.f32 %r2, %f1", []),
    "s32_from_f32":   ("cvt.rn.f32.u32 %f1, %r0; cvt.rzi.s32.f32 %r2, %f1", []),
    "f32_from_s32":   ("cvt.s32.u32 %s1, %r0; cvt.rn.f32.s32 %f1, %s1; mov.b32 %r2, %f1", []),
}


def axis_cvt_op_bins() -> list[str]:
    return list(_CVT_CASES.keys())


def synthesize_cvt_op(bin_key: str) -> ProbeSpec | None:
    if bin_key not in _CVT_CASES:
        return None
    cvt_text, pre = _CVT_CASES[bin_key]
    target_op = cvt_text.split(";")[0].split()[0]
    return ProbeSpec(
        template_id="cvt_op",
        target_op=target_op,
        operand_spec={"cvt_text": cvt_text},
        pre_context=pre,
    )


# ---------------------------------------------------------------------------
# Axis 10: alu_unary
#   Bins: {op}.{typ}
# ---------------------------------------------------------------------------

_UNARY_OPS = (
    ("not.b32",   "not.b32 %r2, %r0"),
    ("not.b64",   "not.b64 %rd2, %rd1"),  # placeholder; emit handled by 64-bit branch
    ("neg.s32",   "neg.s32 %r2, %r0"),
    ("neg.s64",   "neg.s64 %rd2, %rd1"),
    ("abs.s32",   "abs.s32 %r2, %r0"),
    ("abs.s64",   "abs.s64 %rd2, %rd1"),
    ("popc.b32",  "popc.b32 %r2, %r0"),
    ("clz.b32",   "clz.b32 %r2, %r0"),
    ("brev.b32",  "brev.b32 %r2, %r0"),
    ("bfind.u32", "bfind.u32 %r2, %r0"),
)


def axis_alu_unary_bins() -> list[str]:
    return [label for label, _ in _UNARY_OPS]


def synthesize_alu_unary(bin_key: str) -> ProbeSpec | None:
    for label, op_text in _UNARY_OPS:
        if label == bin_key:
            # 64-bit unary ops use alu_64bit shape; rest fit alu_unary
            if "64" in label and label != "popc.b64":
                return ProbeSpec(
                    template_id="alu_64bit",
                    target_op=label,
                    operand_spec={"op_text": op_text},
                )
            return ProbeSpec(
                template_id="alu_unary",
                target_op=label,
                operand_spec={"op_text": op_text},
            )
    return None


# ---------------------------------------------------------------------------
# Axis 11: bitfield (bfe / bfi)
#   Bins: {op}/{shape}
#   shape: imm_pos_imm_len  | reg_pos_reg_len  | edge_high_bit
# ---------------------------------------------------------------------------

_BITFIELD_BINS = {
    "bfe.u32/imm_8_8":      "bfe.u32 %r2, %r5, 8, 8",
    "bfe.u32/imm_0_4":      "bfe.u32 %r2, %r5, 0, 4",
    "bfe.u32/imm_28_4":     "bfe.u32 %r2, %r6, 28, 4",
    "bfe.u32/reg_pos_imm":  "bfe.u32 %r2, %r5, %r3, 8",
    "bfe.s32/imm_8_8":      "bfe.s32 %r2, %r6, 8, 8",
    "bfe.s32/imm_28_4":     "bfe.s32 %r2, %r6, 28, 4",
    "bfi.b32/imm_8_8":      "bfi.b32 %r2, %r0, %r5, 8, 8",
    "bfi.b32/imm_0_4":      "bfi.b32 %r2, %r0, %r5, 0, 4",
    "bfi.b32/imm_24_8":     "bfi.b32 %r2, %r0, %r5, 24, 8",
}


def axis_bitfield_bins() -> list[str]:
    return list(_BITFIELD_BINS.keys())


def synthesize_bitfield(bin_key: str) -> ProbeSpec | None:
    op_text = _BITFIELD_BINS.get(bin_key)
    if op_text is None:
        return None
    target_op = op_text.split()[0]
    return ProbeSpec(
        template_id="bitfield",
        target_op=target_op,
        operand_spec={"op_text": op_text},
    )


# ---------------------------------------------------------------------------
# Axis 12: selp_op
#   Bins: {typ}/thr={N}
# ---------------------------------------------------------------------------

_SELP_TYPES = ("b32", "u32", "s32", "f32")
_SELP_THRS  = (0, 1, 32, 64, 127)


def axis_selp_op_bins() -> list[str]:
    return [f"{t}/thr={n}" for t in _SELP_TYPES for n in _SELP_THRS]


def synthesize_selp_op(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 2:
        return None
    typ, thr_part = parts
    if not thr_part.startswith("thr="):
        return None
    thr = int(thr_part[4:])
    return ProbeSpec(
        template_id="selp_op",
        target_op=f"selp.{typ}",
        operand_spec={"typ": typ, "pred_thr": thr,
                      "a_val": 0xaaaaaaaa, "b_val": 0x55555555},
    )


# ---------------------------------------------------------------------------
# Axis 13: fma_op
#   Bins: {typ}/{k1_label}_{k2_label}
# ---------------------------------------------------------------------------

_FMA_K1S = {
    "1.0":   "0f3F800000",
    "2.0":   "0f40000000",
    "0.5":   "0f3F000000",
    "tiny":  "0f00800000",
}
_FMA_K2S = {
    "0.0":   "0f00000000",
    "1.0":   "0f3F800000",
    "neg1":  "0fBF800000",
}


def axis_fma_op_bins() -> list[str]:
    bins = []
    for k1 in _FMA_K1S:
        for k2 in _FMA_K2S:
            bins.append(f"f32/{k1}_{k2}")
    return bins


def synthesize_fma_op(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 2:
        return None
    typ, k_part = parts
    if "_" not in k_part:
        return None
    k1_lbl, k2_lbl = k_part.split("_", 1)
    k1 = _FMA_K1S.get(k1_lbl)
    k2 = _FMA_K2S.get(k2_lbl)
    if k1 is None or k2 is None:
        return None
    return ProbeSpec(
        template_id="fma_op",
        target_op=f"fma.rn.{typ}",
        operand_spec={"typ": typ, "k1": k1, "k2": k2},
    )


# ---------------------------------------------------------------------------
# Axis 14: auto_dispatch — coverage-driven, generated from sass/isel.py
#
# Walks the dispatcher's (op, type-tuple) cells and synthesizes a default
# probe for each one not already covered by hand-coded axes.  This is the
# "BelAZ scoop": for every shape our isel claims to handle, at least ONE
# probe lands.
#
# Probe-shape mapping is heuristic — the cleanest match per opcode family.
# Cells with no clean match are silently skipped (they need a custom
# template).
# ---------------------------------------------------------------------------

# Default operand-spec generators per opcode family.  Each takes (op, quals)
# where quals is a frozenset of types/modifiers.  Returns ProbeSpec or None.

# 32-bit binary ALU: op.{typ} %r2, %r0, %r3
_AUTO_BINARY_32 = {"add", "sub", "mul", "and", "or", "xor", "min", "max"}
# 32-bit unary: op.{typ} %r2, %r0
_AUTO_UNARY_32  = {"not", "neg", "abs", "popc", "clz", "brev", "bfind"}
# Shift: op.{typ} %r2, %r0, 4
_AUTO_SHIFT_32  = {"shl", "shr"}
# Float binary: op.{typ} %f2, %f1, 0f3F800000  (uses alu_f32 template)
_AUTO_FBIN     = {"add", "sub", "mul", "min", "max", "div"}
# Conversion: op.{dst}.{src} %r2, %r0
_AUTO_CVT      = {"cvt"}

_VALID_INT32_TYPES = {"u32", "s32", "b32"}
_VALID_INT64_TYPES = {"u64", "s64", "b64"}
_VALID_FLOAT_TYPES = {"f32", "f64"}


def _quals_have(quals, *needles):
    return any(n in quals for n in needles)


def _pick_int_type(quals):
    for t in ("u32", "s32", "b32", "u64", "s64", "b64"):
        if t in quals:
            return t
    return None


def _autogen_spec(op: str, quals: frozenset) -> ProbeSpec | None:
    """Default probe for a (op, quals) cell.  None if no template fits."""
    typ = _pick_int_type(quals)
    has_f32 = "f32" in quals
    has_f64 = "f64" in quals
    has_lo  = "lo"  in quals
    has_wide = "wide" in quals
    has_hi  = "hi"  in quals

    # ---- 32-bit unary integer ----
    if op in _AUTO_UNARY_32 and typ in _VALID_INT32_TYPES:
        return ProbeSpec(
            template_id="alu_unary",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %r2, %r0"},
        )

    # ---- 64-bit unary integer ----
    if op in _AUTO_UNARY_32 and typ in _VALID_INT64_TYPES:
        if op in {"popc", "clz", "brev", "bfind"}:
            return None  # 64-bit variants have different shapes; skip
        return ProbeSpec(
            template_id="alu_64bit",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %rd2, %rd1"},
        )

    # ---- 32-bit binary integer (reg-reg variant) ----
    if op in _AUTO_BINARY_32 and typ in _VALID_INT32_TYPES:
        return ProbeSpec(
            template_id="alu_single",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %r2, %r0, %r3", "init_acc": 1},
        )

    # ---- 64-bit binary integer ----
    if op in _AUTO_BINARY_32 and typ in _VALID_INT64_TYPES:
        return ProbeSpec(
            template_id="alu_64bit",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %rd2, %rd1, 5"},
        )

    # ---- shifts (reg-imm variant) ----
    if op in _AUTO_SHIFT_32 and typ in _VALID_INT32_TYPES:
        return ProbeSpec(
            template_id="alu_single",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %r2, %r0, 3", "init_acc": 0},
        )
    if op in _AUTO_SHIFT_32 and typ in _VALID_INT64_TYPES:
        return ProbeSpec(
            template_id="alu_64bit",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %rd2, %rd1, 3"},
        )

    # ---- mad/mul .lo variants ----
    if op in {"mad", "mul"} and has_lo and typ in _VALID_INT32_TYPES:
        if op == "mad":
            text = f"mad.lo.{typ} %r2, %r0, 6, %r3"
        else:
            text = f"mul.lo.{typ} %r2, %r0, %r3"
        return ProbeSpec(
            template_id="alu_single",
            target_op=f"{op}.lo.{typ}",
            operand_spec={"op_text": text, "init_acc": 1},
        )

    # ---- f32 binary float ----
    if op in _AUTO_FBIN and has_f32:
        return ProbeSpec(
            template_id="alu_f32",
            target_op=f"{op}.f32",
            operand_spec={"op_text": f"{op}.f32 %f2, %f1, 0f3F800000"},
        )

    # ---- fma f32 (3-src) ----
    if op == "fma" and has_f32:
        return ProbeSpec(
            template_id="fma_op",
            target_op="fma.rn.f32",
            operand_spec={"typ": "f32",
                          "k1": "0f3F800000",
                          "k2": "0f00000000"},
        )

    # ---- selp (predicated select) ----
    if op == "selp":
        sel_typ = "b32"
        if "u32" in quals: sel_typ = "u32"
        elif "s32" in quals: sel_typ = "s32"
        elif "f32" in quals: sel_typ = "f32"
        return ProbeSpec(
            template_id="selp_op",
            target_op=f"selp.{sel_typ}",
            operand_spec={"typ": sel_typ, "pred_thr": 64,
                          "a_val": 0xaaaaaaaa, "b_val": 0x55555555},
        )

    return None  # no template fits — needs custom work (atom, mma, ld/st...)


def _isel_path() -> str:
    """Locate openptxas/sass/isel.py — required for surface enumeration."""
    return os.environ.get(
        "OPENPTXAS_ISEL",
        os.path.expandvars(r"C:\Users\kraken\openptxas\sass\isel.py"),
    )


def _ptx_cells() -> list[tuple[str, frozenset]]:
    """Cached list of dispatcher cells from isel.py."""
    return sorted(surface.enumerate_ptx_surface(_isel_path()))


def _bin_key(op: str, quals: frozenset) -> str:
    """Stable string key for an (op, quals) cell."""
    qs = "_".join(sorted(quals)) if quals else "noquals"
    return f"{op}/{qs}"


def axis_auto_dispatch_bins() -> list[str]:
    bins: list[str] = []
    for op, quals in _ptx_cells():
        if _autogen_spec(op, quals) is None:
            continue
        bins.append(_bin_key(op, quals))
    return sorted(set(bins))


def synthesize_auto_dispatch(bin_key: str) -> ProbeSpec | None:
    for op, quals in _ptx_cells():
        if _bin_key(op, quals) == bin_key:
            return _autogen_spec(op, quals)
    return None


AXES: dict[str, tuple[Callable[[], list[str]],
                       Callable[[str], ProbeSpec | None]]] = {
    "opcode_imm_acc":     (axis_opcode_imm_acc_bins,  synthesize_opcode_imm_acc),
    "hazard_pair_dist":   (axis_hazard_pair_dist_bins, synthesize_hazard_pair_dist),
    "latency_writer_gap": (axis_latency_writer_gap_bins, synthesize_latency_writer_gap),
    "alu_64bit_imm":      (axis_alu_64bit_imm_bins,    synthesize_alu_64bit_imm),
    "load_consume_gap":   (axis_load_consume_gap_bins, synthesize_load_consume_gap),
    "predicated_alu":     (axis_predicated_alu_bins,   synthesize_predicated_alu),
    "atomic_op":          (axis_atomic_op_bins,        synthesize_atomic_op),
    "f32_alu":            (axis_f32_alu_bins,          synthesize_f32_alu),
    "cvt_op":             (axis_cvt_op_bins,           synthesize_cvt_op),
    "alu_unary":          (axis_alu_unary_bins,        synthesize_alu_unary),
    "bitfield":           (axis_bitfield_bins,         synthesize_bitfield),
    "selp_op":            (axis_selp_op_bins,          synthesize_selp_op),
    "fma_op":             (axis_fma_op_bins,           synthesize_fma_op),
    "auto_dispatch":      (axis_auto_dispatch_bins,    synthesize_auto_dispatch),
}


def all_axis_bins() -> dict[str, list[str]]:
    """Materialize every axis's full bin set.  Used by `seed_coverage`."""
    return {name: bins_fn() for name, (bins_fn, _) in AXES.items()}


def synthesize(axis: str, bin_key: str) -> ProbeSpec | None:
    if axis not in AXES:
        return None
    _, syn_fn = AXES[axis]
    return syn_fn(bin_key)
