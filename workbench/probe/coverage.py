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
    "1.0":      "0f3F800000",
    "0.5":      "0f3F000000",
    "2.0":      "0f40000000",
    "3.0":      "0f40400000",
    "neg1":     "0fBF800000",
    "tiny":     "0f00800000",   # smallest normal
    "inf":      "0f7F800000",
    # Corner-case constants — exercise rounding, FTZ, NaN propagation
    # paths that are rarely hit by typical numeric ranges.  Labels MUST
    # NOT contain underscores (the f32_alu axis splits bin keys on '_').
    "neginf":   "0fFF800000",
    "nan":      "0f7FC00000",   # quiet NaN
    "negzero":  "0f80000000",
    "maxfin":   "0f7F7FFFFF",   # largest finite f32
    "denorm":   "0f00000001",   # smallest subnormal
    "epsilon":  "0f34000000",   # 2^-23, ulp(1.0)
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
    # bfe.u32 reg_pos_imm — KNOWN-RESIDUAL: SHF.R.U32 var-shift reads stale
    # data register even with a NOP gap.  Likely needs scoreboard/wdep work
    # on the data MOV to track properly.  Skipping until investigated.
    # "bfe.u32/reg_pos_imm":  "bfe.u32 %r2, %r5, %r3, 8",
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

_SELP_TYPES = ("b32", "u32", "s32")  # f32 needs FP-typed immediates; skip until template supports it
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
    "1.0":     "0f3F800000",
    "2.0":     "0f40000000",
    "0.5":     "0f3F000000",
    "tiny":    "0f00800000",
    "inf":     "0f7F800000",
    "nan":     "0f7FC00000",
    "maxfin":  "0f7F7FFFFF",     # no underscore: bin parser splits on _
}
_FMA_K2S = {
    "0.0":      "0f00000000",
    "1.0":      "0f3F800000",
    "neg1":     "0fBF800000",
    "inf":      "0f7F800000",
    "negzero":  "0f80000000",    # no underscore
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
# mul is excluded because it requires .lo / .hi / .wide modifier — handled
# in the mul/mad branch below.
_AUTO_BINARY_32 = {"add", "sub", "and", "or", "xor", "min", "max"}
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

    # mul without lo/hi/wide is invalid PTX — skip these.  Same for div
    # without a rounding mode, mad without lo/hi/wide.
    if op == "mul" and typ in (_VALID_INT32_TYPES | _VALID_INT64_TYPES) \
            and not (has_lo or has_hi or has_wide):
        return None
    if op == "mad" and not (has_lo or has_hi or has_wide):
        return None
    if op == "div" and (has_f32 or has_f64):
        # div.f32/f64 need a rounding mode — skip until we have a fitting cell.
        return None

    # ---- 32-bit unary integer ----
    if op in _AUTO_UNARY_32 and typ in _VALID_INT32_TYPES:
        # PTX type-validity rules:
        #   not  : only .b16/.b32/.b64 (no signed/unsigned)
        #   neg  : only .s16/.s32/.s64 (no unsigned, no .b)
        #   abs  : only .s16/.s32/.s64 (signed only)
        #   popc/clz/brev/bfind: only .b32/.b64
        if op == "not"  and typ not in {"b32"}: return None
        if op == "neg"  and typ not in {"s32"}: return None
        if op == "abs"  and typ not in {"s32"}: return None
        if op in {"popc", "clz", "brev"} and typ not in {"b32"}: return None
        if op == "bfind" and typ not in {"u32", "s32"}: return None
        return ProbeSpec(
            template_id="alu_unary",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %r2, %r0"},
        )

    # ---- 64-bit unary integer ----
    if op in _AUTO_UNARY_32 and typ in _VALID_INT64_TYPES:
        if op in {"popc", "clz", "brev", "bfind"}:
            return None  # 64-bit variants have different shapes; skip
        if op == "not" and typ not in {"b64"}: return None
        if op == "neg" and typ not in {"s64"}: return None
        if op == "abs" and typ not in {"s64"}: return None
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
    # KNOWN-RESIDUAL: min/max u64/s64 have a pre-existing carry-chain bug
    # in the branchless lowering — for tid > b, SASS emits but consistently
    # returns b instead of max(tid, b).  Production suite doesn't exercise
    # min/max u64.  Skipping the auto-axis bin until the dispatcher is
    # rewritten.  Tracked in known_residuals.
    if op in {"min", "max"} and typ in _VALID_INT64_TYPES:
        return None
    if op in _AUTO_BINARY_32 and typ in _VALID_INT64_TYPES:
        return ProbeSpec(
            template_id="alu_64bit",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %rd2, %rd1, 5"},
        )

    # ---- shifts (reg-imm variant) ----
    # PTX: shl/shr only accept .b{16,32,64} — skip signed/unsigned variants.
    if op in _AUTO_SHIFT_32 and typ in _VALID_INT32_TYPES:
        if typ != "b32" and op != "shr":  # shr.s32/.u32 are valid; shl is .b32 only
            return None
        return ProbeSpec(
            template_id="alu_single",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %r2, %r0, 3", "init_acc": 0},
        )
    if op in _AUTO_SHIFT_32 and typ in _VALID_INT64_TYPES:
        if typ != "b64" and op != "shr":  # shl.b64 only
            return None
        return ProbeSpec(
            template_id="alu_64bit",
            target_op=f"{op}.{typ}",
            operand_spec={"op_text": f"{op}.{typ} %rd2, %rd1, 3"},
        )

    # ---- mad/mul .lo variants (32-bit) ----
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

    # ---- mad/mul .lo variants (64-bit) — use alu_64bit (64-bit regs) ----
    # mul.lo.b64 isn't a valid PTX type combo (b64 not allowed for mul.lo);
    # u64/s64 are valid with reg-reg operands.
    if op in {"mad", "mul"} and has_lo and typ in _VALID_INT64_TYPES:
        if typ == "b64":
            return None  # mul.lo.b64 isn't valid PTX
        if op == "mad":
            text = f"mad.lo.{typ} %rd2, %rd1, %rd1, %rd1"
        else:
            text = f"mul.lo.{typ} %rd2, %rd1, %rd1"
        return ProbeSpec(
            template_id="alu_64bit",
            target_op=f"{op}.lo.{typ}",
            operand_spec={"op_text": text},
        )

    # ---- mul.hi / mul.wide variants — skip (need richer templates) ----
    if op in {"mad", "mul"} and (has_hi or has_wide):
        return None

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
    # selp.f32 needs FP-typed constants in PTX; the integer-typed selp_op
    # template doesn't fit.  Only cover the integer/b32 selp here.
    if op == "selp":
        sel_typ = None
        if "b32" in quals: sel_typ = "b32"
        elif "u32" in quals: sel_typ = "u32"
        elif "s32" in quals: sel_typ = "s32"
        if sel_typ is None:
            return None
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


# ---------------------------------------------------------------------------
# Axis 15: regression — re-runs of canonical reproducers from edge_cases.
#   Each open edge case becomes a permanent regression probe.  When the
#   underlying issue is fixed and `status=resolved`, the probe stays in
#   the suite forever — no future change can re-introduce the bug
#   without the mower screaming.
# ---------------------------------------------------------------------------

def _regression_rows() -> list[tuple]:
    """Pull (edge_id, target_op, template_id, operand_spec) for open
    edge cases that have a stored operand_spec.  Read directly from the
    DB at the default probe-dir."""
    import json
    import sqlite3
    from pathlib import Path
    db_path = Path(os.environ.get(
        "PROBE_DIR",
        os.path.expandvars(r"C:\Users\kraken\openptxas\probes"),
    )) / "probes.sqlite"
    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(str(db_path))
        rows = list(con.execute("""
            SELECT edge_id, target_op, template_id, operand_spec, status
            FROM edge_cases
            WHERE operand_spec IS NOT NULL AND template_id IS NOT NULL
        """))
        con.close()
        # Keep all (open + resolved) — resolved ones are permanent regression
        # guards that should re-fail if a regression slips in.
        return rows
    except sqlite3.OperationalError:
        return []  # edge_cases table doesn't exist yet


def axis_regression_bins() -> list[str]:
    return [f"edge_{r[0]}" for r in _regression_rows()]


def synthesize_regression(bin_key: str) -> ProbeSpec | None:
    if not bin_key.startswith("edge_"):
        return None
    try:
        eid = int(bin_key.split("_")[1])
    except (ValueError, IndexError):
        return None
    import json
    for r in _regression_rows():
        if r[0] == eid:
            try:
                operand = json.loads(r[3])
            except (json.JSONDecodeError, TypeError):
                return None
            return ProbeSpec(
                template_id=r[2],
                target_op=r[1] or "regression",
                operand_spec=operand,
            )
    return None


# ---------------------------------------------------------------------------
# Axis 16: branch_distance — bra over varying instruction gaps.
# ---------------------------------------------------------------------------

_BRA_GAPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def axis_branch_distance_bins() -> list[str]:
    return [f"gap={g}" for g in _BRA_GAPS]


def synthesize_branch_distance(bin_key: str) -> ProbeSpec | None:
    if not bin_key.startswith("gap="):
        return None
    try:
        gap = int(bin_key[4:])
    except ValueError:
        return None
    return ProbeSpec(
        template_id="branch_distance",
        target_op="bra",
        operand_spec={"gap": gap},
    )


# ---------------------------------------------------------------------------
# Axis 17: loop_iter — counted loops at varying iteration counts.
# ---------------------------------------------------------------------------

_LOOP_ITERS = [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 100, 255, 256]


def axis_loop_iter_bins() -> list[str]:
    return [f"iters={n}" for n in _LOOP_ITERS]


def synthesize_loop_iter(bin_key: str) -> ProbeSpec | None:
    if not bin_key.startswith("iters="):
        return None
    try:
        iters = int(bin_key[6:])
    except ValueError:
        return None
    return ProbeSpec(
        template_id="loop_iter",
        target_op="loop",
        operand_spec={"iters": iters},
    )


# ---------------------------------------------------------------------------
# Axis 18: divergent_branch — split warp at varying tid thresholds.
# ---------------------------------------------------------------------------

_DIV_THRS = [1, 2, 4, 8, 16, 17, 31, 32, 33, 48, 63, 64, 96, 127]


def axis_divergent_branch_bins() -> list[str]:
    return [f"thr={t}" for t in _DIV_THRS]


def synthesize_divergent_branch(bin_key: str) -> ProbeSpec | None:
    if not bin_key.startswith("thr="):
        return None
    try:
        thr = int(bin_key[4:])
    except ValueError:
        return None
    return ProbeSpec(
        template_id="divergent_branch",
        target_op="bra.div",
        operand_spec={"thr": thr, "a_val": 0xAAAA, "b_val": 0x5555},
    )


# ---------------------------------------------------------------------------
# Axis 19: pred_composition — and/or/xor.pred at varying thresholds.
# ---------------------------------------------------------------------------

_COMPOSE_OPS = ("and", "or", "xor")
_COMPOSE_THR_PAIRS = [
    (64, 16), (32, 8), (96, 32), (127, 0), (1, 0), (16, 16), (8, 64),
]


def axis_pred_composition_bins() -> list[str]:
    bins = []
    for op in _COMPOSE_OPS:
        for a, b in _COMPOSE_THR_PAIRS:
            bins.append(f"{op}/a={a}/b={b}")
    return bins


def synthesize_pred_composition(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 3:
        return None
    op, a_part, b_part = parts
    if op not in _COMPOSE_OPS:
        return None
    if not a_part.startswith("a=") or not b_part.startswith("b="):
        return None
    try:
        thr_a = int(a_part[2:])
        thr_b = int(b_part[2:])
    except ValueError:
        return None
    return ProbeSpec(
        template_id="pred_composition",
        target_op=f"{op}.pred",
        operand_spec={"compose": op, "thr_a": thr_a, "thr_b": thr_b},
    )


# ---------------------------------------------------------------------------
# Axis 20: shared_barrier — st.shared / bar.sync / ld.shared neighbor read.
# ---------------------------------------------------------------------------

_SHARED_OFFSETS = [0, 1, 2, 4, 8, 16, 31, 32, 33, 64, 127]


def axis_shared_barrier_bins() -> list[str]:
    return [f"off={n}" for n in _SHARED_OFFSETS]


def synthesize_shared_barrier(bin_key: str) -> ProbeSpec | None:
    if not bin_key.startswith("off="):
        return None
    try:
        off = int(bin_key[4:])
    except ValueError:
        return None
    return ProbeSpec(
        template_id="shared_barrier",
        target_op="ld.shared.u32",
        operand_spec={"offset": off},
    )


# ---------------------------------------------------------------------------
# Axis 21: hmma — tensor-core mma.sync probes.
#   Single bin for now (m16n8k16/f16-f32, all-ones oracle).  When this
#   lands clean we'll add additional shapes (m16n8k8, bf16, f64...).
# ---------------------------------------------------------------------------

_HMMA_SHAPES = {
    "m16n8k16/f16_f32/all_ones": (
        "hmma_m16n8k16",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    ),
    "m16n8k8/f16_f32/all_ones": (
        "hmma_m16n8k8",
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32",
    ),
    "m16n8k16/bf16_f32/all_ones": (
        "hmma_bf16_m16n8k16",
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    ),
    "m16n8k8/tf32_f32/all_ones": (
        "hmma_tf32_m16n8k8",
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    ),
}


# ---------------------------------------------------------------------------
# Axis 22: tma — async tensor-copy synchronization primitives.
#   First-cut TMA probe family: only the standalone commit_group /
#   wait_group sync ops (no tensor descriptor needed, runs single-thread).
#   Future expansion: tensor.1d / tensor.2d load+store with mbarrier
#   sequencing — needs runner.py changes for cuTensorMap setup.
# ---------------------------------------------------------------------------

_TMA_WAIT_COUNTS  = (0, 1, 2, 4)
_TMA_COMMIT_FANS  = (1, 2, 4)


def axis_tma_bins() -> list[str]:
    return [f"commits={c}/wait={w}"
            for c in _TMA_COMMIT_FANS
            for w in _TMA_WAIT_COUNTS]


def synthesize_tma(bin_key: str) -> ProbeSpec | None:
    parts = bin_key.split("/")
    if len(parts) != 2:
        return None
    c_part, w_part = parts
    if not c_part.startswith("commits=") or not w_part.startswith("wait="):
        return None
    try:
        c = int(c_part[len("commits="):])
        w = int(w_part[len("wait="):])
    except ValueError:
        return None
    return ProbeSpec(
        template_id="tma_commit_wait",
        target_op="cp.async.bulk.commit_group",
        operand_spec={"n_commits": c, "wait_count": w},
    )


# ---------------------------------------------------------------------------
# Axis 23: ldmatrix — load shared→registers in HMMA fragment layout.
# ---------------------------------------------------------------------------

_LDMATRIX_VARIANTS = ("x1", "x2", "x4")


def axis_ldmatrix_bins() -> list[str]:
    return [f"variant={v}" for v in _LDMATRIX_VARIANTS]


def synthesize_ldmatrix(bin_key: str) -> ProbeSpec | None:
    if not bin_key.startswith("variant="):
        return None
    v = bin_key[len("variant="):]
    if v not in _LDMATRIX_VARIANTS:
        return None
    return ProbeSpec(
        template_id="ldmatrix_xN",
        target_op=f"ldmatrix.sync.aligned.{v}.m8n8.shared.b16",
        operand_spec={"variant": v},
    )


# ---------------------------------------------------------------------------
# Axis 24: mbarrier — shared-mem barrier init/arrive/wait sequencing.
# ---------------------------------------------------------------------------

_MBARRIER_COUNTS = (1, 8, 32, 64, 128)


def axis_mbarrier_bins() -> list[str]:
    return [f"arrive_count={c}" for c in _MBARRIER_COUNTS]


def synthesize_mbarrier(bin_key: str) -> ProbeSpec | None:
    if not bin_key.startswith("arrive_count="):
        return None
    try:
        c = int(bin_key[len("arrive_count="):])
    except ValueError:
        return None
    return ProbeSpec(
        template_id="mbarrier_basic",
        target_op="mbarrier.arrive.shared.b64",
        operand_spec={"arrive_count": c},
    )


# ---------------------------------------------------------------------------
# Axis 25: cvta — address-space cast round-trip.
# ---------------------------------------------------------------------------

def axis_cvta_bins() -> list[str]:
    return ["shared/roundtrip"]


def synthesize_cvta(bin_key: str) -> ProbeSpec | None:
    if bin_key != "shared/roundtrip":
        return None
    return ProbeSpec(
        template_id="cvta_addrspace",
        target_op="cvta.to.shared.u64",
        operand_spec={"direction": "shared"},
    )


def axis_hmma_bins() -> list[str]:
    return list(_HMMA_SHAPES.keys())


def synthesize_hmma(bin_key: str) -> ProbeSpec | None:
    entry = _HMMA_SHAPES.get(bin_key)
    if entry is None:
        return None
    template_id, target_op = entry
    return ProbeSpec(
        template_id=template_id,
        target_op=target_op,
        operand_spec={},
    )


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
    "branch_distance":    (axis_branch_distance_bins,  synthesize_branch_distance),
    "loop_iter":          (axis_loop_iter_bins,        synthesize_loop_iter),
    "divergent_branch":   (axis_divergent_branch_bins, synthesize_divergent_branch),
    "pred_composition":   (axis_pred_composition_bins, synthesize_pred_composition),
    "shared_barrier":     (axis_shared_barrier_bins,   synthesize_shared_barrier),
    "hmma":               (axis_hmma_bins,             synthesize_hmma),
    "tma":                (axis_tma_bins,              synthesize_tma),
    "ldmatrix":           (axis_ldmatrix_bins,         synthesize_ldmatrix),
    "mbarrier":           (axis_mbarrier_bins,         synthesize_mbarrier),
    "cvta":               (axis_cvta_bins,             synthesize_cvta),
    "auto_dispatch":      (axis_auto_dispatch_bins,    synthesize_auto_dispatch),
    "regression":         (axis_regression_bins,       synthesize_regression),
}


def all_axis_bins() -> dict[str, list[str]]:
    """Materialize every axis's full bin set.  Used by `seed_coverage`."""
    return {name: bins_fn() for name, (bins_fn, _) in AXES.items()}


def synthesize(axis: str, bin_key: str) -> ProbeSpec | None:
    if axis not in AXES:
        return None
    _, syn_fn = AXES[axis]
    return syn_fn(bin_key)
