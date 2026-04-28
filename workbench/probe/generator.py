"""PTX probe generator.

A probe spec is a small data class.  Templates are Python functions
that take a spec and return a complete, compileable PTX kernel string.

Each template documents what bins it covers in the coverage axes —
the scheduler pulls bins from the DB and asks the registered template
for that axis to materialize a probe spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ProbeSpec:
    template_id: str
    target_op: str                     # PTX op label, e.g. 'mad.lo.u32'
    operand_spec: dict = field(default_factory=dict)
    pre_context: list[str] = field(default_factory=list)
    post_context: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Template: alu_single
#
#   Single ALU op with a known input register.  Stores result.  Catches
#   plain encoding bugs and basic correctness.
#
#   operand_spec keys:
#     op_text  : full PTX instruction text (without trailing semicolon)
#                e.g. "mad.lo.u32 %r2, %r0, 6, %r2"
#     init_acc : initial value for %r2 (default: 0)
# ---------------------------------------------------------------------------

def template_alu_single(spec: ProbeSpec) -> str:
    init_acc = spec.operand_spec.get("init_acc", 0)
    op_text = spec.operand_spec["op_text"]
    pre = "\n    ".join(spec.pre_context)
    post = "\n    ".join(spec.post_context)
    # Init %r2..%r6 to deterministic values so probes can use any of them
    # as src/dest without uninitialized-read issues.  %r0 is %tid.x; %r1
    # is the n param (used by the early-exit setp); the rest start at
    # known small constants.
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, {init_acc};
    mov.u32 %r3, 0;
    mov.u32 %r4, 0;
    mov.u32 %r5, 0;
    mov.u32 %r6, 0;
    {pre}
    {op_text};
    {post}
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: alu_acc_self
#
#   Specifically for fused-mad-acc-alias bug class.  Target op writes
#   to %r2 AND reads %r2 as src2.  Catches IMAD acc-alias-with-non-pow2-K
#   bug we found earlier.
#
#   operand_spec keys:
#     op       : 'mad.lo.u32' | 'mad.lo.s32' | etc.
#     a_reg    : src0 register index (default: 0 → %r0)
#     imm      : K immediate value
#     init_acc : initial value for %r2 (default: 7)
# ---------------------------------------------------------------------------

def template_alu_acc_self(spec: ProbeSpec) -> str:
    op = spec.operand_spec["op"]
    a_reg = spec.operand_spec.get("a_reg", 0)
    imm = spec.operand_spec["imm"]
    init_acc = spec.operand_spec.get("init_acc", 7)
    pre = "\n    ".join(spec.pre_context)
    post = "\n    ".join(spec.post_context)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, {init_acc};
    {pre}
    {op} %r2, %r{a_reg}, {imm}, %r2;
    {post}
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: pair_distance
#
#   Two ops separated by N filler instructions.  For hazard probing.
#   The second op typically reads the first op's destination, so any
#   missing-NOP hazard surfaces as wrong GPU output.
#
#   operand_spec keys:
#     op_a   : first PTX instr writing to %r2
#     op_b   : second PTX instr reading %r2 and writing to %r2
#     gap    : number of nops between them
# ---------------------------------------------------------------------------

def template_pair_distance(spec: ProbeSpec) -> str:
    op_a = spec.operand_spec["op_a"]
    op_b = spec.operand_spec["op_b"]
    gap = spec.operand_spec.get("gap", 0)
    # PTX has no native `nop` opcode — use add.u32 %r3, %r3, 0 as a slot-
    # filling op that doesn't read or write %r2 (the probe's data register).
    nops = "\n    ".join(["add.u32 %r3, %r3, 0;"] * gap)
    pre = "\n    ".join(spec.pre_context)
    post = "\n    ".join(spec.post_context)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
    {pre}
    {op_a};
    {nops}
    {op_b};
    {post}
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: latency_sweep
#
#   Hardware-truth latency probe.  Writer op writes %r2 with a known
#   value.  After N nops, reader reads %r2 and stores.  If output is
#   wrong, latency requirement > N.  Sweep N to find first correct.
#
#   operand_spec keys:
#     writer  : full PTX instr that writes %r2 (e.g. "mad.lo.u32 %r2, %r0, 6, %r2")
#     init    : initial %r2 value before the writer
#     gap     : number of nops between writer and reader
# ---------------------------------------------------------------------------

def template_latency_sweep(spec: ProbeSpec) -> str:
    writer = spec.operand_spec["writer"]
    init = spec.operand_spec.get("init", 0)
    gap = spec.operand_spec.get("gap", 0)
    # filler: add.u32 %r3, %r3, 0 — slot-filling, no interaction with %r2.
    nops = "\n    ".join(["add.u32 %r3, %r3, 0;"] * gap)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, {init};
    mov.u32 %r3, 0;
    {writer};
    {nops}
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# Registry: template_id -> (template_fn, expected_output_fn)
# expected_output_fn: takes the ProbeSpec and tid, returns expected u32 value.
# Used to verify GPU correctness without relying on ptxas as oracle.

def expected_alu_acc_self(spec: ProbeSpec, tid: int) -> int:
    """Expected output for mad.lo: init_acc + a_reg_value * imm.
    a_reg_value = tid for a_reg=0; otherwise undefined (return None)."""
    op = spec.operand_spec["op"]
    a_reg = spec.operand_spec.get("a_reg", 0)
    imm = spec.operand_spec["imm"]
    init_acc = spec.operand_spec.get("init_acc", 7)
    if a_reg == 0 and op.startswith("mad.lo"):
        return (init_acc + tid * imm) & 0xFFFFFFFF
    return None  # unknown


def expected_latency_sweep(spec: ProbeSpec, tid: int) -> int:
    """For a latency probe, the writer's output is what we expect
    AFTER it lands.  This depends on the writer.  Hardcoded for the
    common case: writer is `mad.lo.u32 %r2, %r0, K, %r2` with init init_acc."""
    writer = spec.operand_spec["writer"]
    init = spec.operand_spec.get("init", 0)
    if writer.startswith("mad.lo.u32 %r2, %r0,"):
        # parse the K
        try:
            after = writer.split(",")[2].strip()
            k = int(after, 0)
            return (init + tid * k) & 0xFFFFFFFF
        except Exception:
            return None
    return None


TEMPLATES: dict[str, tuple[Callable[[ProbeSpec], str],
                           Callable[[ProbeSpec, int], int | None] | None]] = {
    "alu_single":     (template_alu_single,    None),
    "alu_acc_self":   (template_alu_acc_self,  expected_alu_acc_self),
    "pair_distance":  (template_pair_distance, None),
    "latency_sweep":  (template_latency_sweep, expected_latency_sweep),
}


def materialize(spec: ProbeSpec) -> str:
    """Return PTX text for a given probe spec."""
    fn, _ = TEMPLATES[spec.template_id]
    return fn(spec)


def expected_output(spec: ProbeSpec, tid: int) -> int | None:
    """Return expected u32 stored at out[tid], or None if undetermined."""
    _, exp = TEMPLATES.get(spec.template_id, (None, None))
    return exp(spec, tid) if exp else None
