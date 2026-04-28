"""Coverage axes: a probe spec maps to a set of (axis, bin_key) tuples.
The scheduler pulls unfilled bins from the DB and asks an axis-specific
synthesizer to materialize a probe spec for that bin.
"""
from __future__ import annotations

from typing import Callable, Iterable

from .generator import ProbeSpec


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


AXES: dict[str, tuple[Callable[[], list[str]],
                       Callable[[str], ProbeSpec | None]]] = {
    "opcode_imm_acc":     (axis_opcode_imm_acc_bins,  synthesize_opcode_imm_acc),
    "hazard_pair_dist":   (axis_hazard_pair_dist_bins, synthesize_hazard_pair_dist),
    "latency_writer_gap": (axis_latency_writer_gap_bins, synthesize_latency_writer_gap),
    "alu_64bit_imm":      (axis_alu_64bit_imm_bins,    synthesize_alu_64bit_imm),
    "load_consume_gap":   (axis_load_consume_gap_bins, synthesize_load_consume_gap),
    "predicated_alu":     (axis_predicated_alu_bins,   synthesize_predicated_alu),
    "atomic_op":          (axis_atomic_op_bins,        synthesize_atomic_op),
}


def all_axis_bins() -> dict[str, list[str]]:
    """Materialize every axis's full bin set.  Used by `seed_coverage`."""
    return {name: bins_fn() for name, (bins_fn, _) in AXES.items()}


def synthesize(axis: str, bin_key: str) -> ProbeSpec | None:
    if axis not in AXES:
        return None
    _, syn_fn = AXES[axis]
    return syn_fn(bin_key)
