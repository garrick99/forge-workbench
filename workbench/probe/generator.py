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


# ---------------------------------------------------------------------------
# Template: alu_64bit
#   64-bit ALU op probing (add.u64, sub.u64, and.b64, etc.)
#   operand_spec keys:
#     op_text  : full PTX line, e.g. "add.u64 %rd2, %rd1, 100"
#     init_lo  : low 32 bits of initial %rd1
#     init_hi  : high 32 bits of initial %rd1
# ---------------------------------------------------------------------------

def template_alu_64bit(spec: ProbeSpec) -> str:
    op_text = spec.operand_spec["op_text"]
    init_lo = spec.operand_spec.get("init_lo", 0)
    init_hi = spec.operand_spec.get("init_hi", 0)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<6>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r0;       // %rd1 = tid (zero-ext)
    add.u64 %rd1, %rd1, {(init_hi << 32) | init_lo};
    {op_text};
    cvt.u32.u64 %r2, %rd2;
    cvt.u64.u32 %rd3, %r0; shl.b64 %rd3, %rd3, 2;
    add.u64 %rd4, %rd0, %rd3;
    st.global.u32 [%rd4], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: load_consume
#   Load a value from a constant input array, consume via ALU op, store.
#   Probes LDG → ALU dependency hazards.
#
#   operand_spec keys:
#     consume_op : ALU op consuming the load result, e.g. "add.u32 %r2, %r2, 5"
#     gap        : number of filler ops between LDG and consumer
# ---------------------------------------------------------------------------

def template_load_consume(spec: ProbeSpec) -> str:
    consume_op = spec.operand_spec["consume_op"]
    gap = spec.operand_spec.get("gap", 0)
    nops = "\n    ".join(["add.u32 %r3, %r3, 0;"] * gap)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u64 p_in, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<5>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    ld.param.u64 %rd1, [p_in];
    mov.u32 %r3, 0;
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    {nops}
    {consume_op};
    add.u64 %rd4, %rd0, %rd2;
    st.global.u32 [%rd4], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: predicated_alu
#   ALU op under a @P0 predicate.  Probes predicate-tracking and
#   conditional-execution paths in our scoreboard / scheduler.
#
#   operand_spec keys:
#     op_text   : ALU op to predicate (e.g. "add.u32 %r2, %r2, 1")
#     pred_cond : "lt" / "gt" / "eq" / etc. determining when @P0 fires
#     pred_thr  : threshold (compares tid with this)
# ---------------------------------------------------------------------------

def template_predicated_alu(spec: ProbeSpec) -> str:
    op_text = spec.operand_spec["op_text"]
    pred_cond = spec.operand_spec.get("pred_cond", "lt")
    pred_thr = spec.operand_spec.get("pred_thr", 64)
    init_acc = spec.operand_spec.get("init_acc", 0)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, {init_acc};
    setp.{pred_cond}.u32 %p1, %r0, {pred_thr};
    @%p1 {op_text};
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: atomic_op
#   atom.global.<op> probing.  Single-threaded (we set n=1) to avoid
#   the multi-thread atomic chaos; what we're checking is the atomic
#   instruction's encoding and ctrl-byte assignment.
#
#   operand_spec keys:
#     op       : 'add.u32' | 'or.b32' | 'xor.b32' | 'cas' | etc.
#     init_val : initial value at the atomic location
#     arg      : the operand being applied
# ---------------------------------------------------------------------------

def template_atomic_op(spec: ProbeSpec) -> str:
    op = spec.operand_spec["op"]
    arg = spec.operand_spec.get("arg", 1)
    init_val = spec.operand_spec.get("init_val", 0)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, {init_val};
    st.global.u32 [%rd2], %r2;
    bar.sync 0;
    atom.global.{op} %r3, [%rd2], {arg};
    bar.sync 0;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: alu_f32
#   Single f32 ALU op.  Inputs come from tid (converted to f32) and a
#   constant.  Result is bit-cast to u32 and stored.
#
#   operand_spec keys:
#     op_text  : full PTX line, e.g. "add.f32 %f2, %f1, 0f3F800000"
#                (operands must use %f1 = tid_f, %f2 = result, %f3..%f5 init=0)
# ---------------------------------------------------------------------------

def template_alu_f32(spec: ProbeSpec) -> str:
    op_text = spec.operand_spec["op_text"]
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<4>; .reg .u64 %rd<3>; .reg .pred %p0;
    .reg .f32 %f<6>;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    cvt.rn.f32.u32 %f1, %r0;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.f32 %f5, 0f00000000;
    {op_text};
    mov.b32 %r2, %f2;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: cvt_op
#   cvt.{dst_type}.{src_type} probing.  Source comes from tid (via mov);
#   result stored as u32 (bit-cast for f32).
#
#   operand_spec keys:
#     cvt_text : full PTX line, e.g. "cvt.u32.s32 %r2, %r0"
#                Output must end in %r2 (u32).  For f32→u32, do mov.b32.
# ---------------------------------------------------------------------------

def template_cvt_op(spec: ProbeSpec) -> str:
    cvt_text = spec.operand_spec["cvt_text"]
    pre = "\n    ".join(spec.pre_context)
    # Compute the destination address FIRST (using %r0 = tid), then run
    # the cvt operation that produces %r2.  Avoids regalloc trying to
    # reuse %r0's GPR for %r2.  %rd2 holds the address; the final st
    # uses %rd2 directly.
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<4>; .reg .u64 %rd<4>; .reg .pred %p0;
    .reg .f32 %f<4>; .reg .s32 %s<3>; .reg .s64 %sd<3>;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    {pre}
    {cvt_text};
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: alu_unary
#   Single unary op of shape `op.<typ> %r2, %r0;` — for not, neg, abs,
#   popc, clz, brev, bfind, etc.  Reuses alu_single's structure but
#   isolates intent (and gives a place to specialize if needed).
#
#   operand_spec keys:
#     op_text  : full PTX line, e.g. "popc.b32 %r2, %r0"
# ---------------------------------------------------------------------------

def template_alu_unary(spec: ProbeSpec) -> str:
    op_text = spec.operand_spec["op_text"]
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    {op_text};
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: bitfield
#   bfe / bfi probes — inputs from tid + per-thread varying state so
#   different thread lanes see different bit positions / masks.
#
#   operand_spec keys:
#     op_text  : full PTX line.  May reference %r0 (tid), %r3 (=tid&31
#                = bit position), %r4 (=8, length), %r5 (=0xa5a5a5a5 mask)
# ---------------------------------------------------------------------------

def template_bitfield(spec: ProbeSpec) -> str:
    op_text = spec.operand_spec["op_text"]
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    and.b32 %r3, %r0, 31;
    mov.u32 %r4, 8;
    mov.u32 %r5, 0xa5a5a5a5;
    mov.u32 %r6, 0xdeadbeef;
    {op_text};
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: selp_op
#   selp.<typ> %r2, A, B, %p0 — predicated select.  The probe sets %p0
#   from setp on tid so each thread takes a different branch.
#
#   operand_spec keys:
#     typ      : 'b32' / 'u32' / 'f32' (the selp type)
#     a_val    : true-branch value
#     b_val    : false-branch value
#     pred_thr : tid threshold for setp.lt
# ---------------------------------------------------------------------------

def template_selp_op(spec: ProbeSpec) -> str:
    typ = spec.operand_spec.get("typ", "b32")
    a_val = spec.operand_spec.get("a_val", 0xaaaaaaaa)
    b_val = spec.operand_spec.get("b_val", 0x55555555)
    pred_thr = spec.operand_spec.get("pred_thr", 64)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    setp.lt.u32 %p1, %r0, {pred_thr};
    selp.{typ} %r2, {a_val}, {b_val}, %p1;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


# ---------------------------------------------------------------------------
# Template: fma_op
#   fma.rn.f32 %f2, %f1, %f3, %f4 — fused multiply-add.  Inputs from tid
#   plus float constants.  Result bit-cast back to u32 for storage.
#
#   operand_spec keys:
#     typ      : 'f32' / 'f64'
#     k1       : multiplier constant (PTX float literal, e.g. "0f3F800000")
#     k2       : addend constant
# ---------------------------------------------------------------------------

def template_fma_op(spec: ProbeSpec) -> str:
    typ = spec.operand_spec.get("typ", "f32")
    k1 = spec.operand_spec.get("k1", "0f3F800000")
    k2 = spec.operand_spec.get("k2", "0f00000000")
    if typ == "f32":
        body = f"""    cvt.rn.f32.u32 %f1, %r0;
    fma.rn.f32 %f2, %f1, {k1}, {k2};
    mov.b32 %r2, %f2;"""
    else:  # f64
        body = f"""    cvt.rn.f64.u32 %fd1, %r0;
    fma.rn.f64 %fd2, %fd1, 0d3FF0000000000000, 0d0000000000000000;
    cvt.rn.f32.f64 %f2, %fd2;
    mov.b32 %r2, %f2;"""
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<4>; .reg .u64 %rd<3>; .reg .pred %p0;
    .reg .f32 %f<4>; .reg .f64 %fd<3>;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
{body}
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


TEMPLATES: dict[str, tuple[Callable[[ProbeSpec], str],
                           Callable[[ProbeSpec, int], int | None] | None]] = {
    "alu_single":      (template_alu_single,      None),
    "alu_acc_self":    (template_alu_acc_self,    expected_alu_acc_self),
    "pair_distance":   (template_pair_distance,   None),
    "latency_sweep":   (template_latency_sweep,   expected_latency_sweep),
    "alu_64bit":       (template_alu_64bit,       None),
    "load_consume":    (template_load_consume,    None),
    "predicated_alu":  (template_predicated_alu,  None),
    "atomic_op":       (template_atomic_op,       None),
    "alu_f32":         (template_alu_f32,         None),
    "cvt_op":          (template_cvt_op,          None),
    "alu_unary":       (template_alu_unary,       None),
    "bitfield":        (template_bitfield,        None),
    "selp_op":         (template_selp_op,         None),
    "fma_op":          (template_fma_op,          None),
}


def materialize(spec: ProbeSpec) -> str:
    """Return PTX text for a given probe spec."""
    fn, _ = TEMPLATES[spec.template_id]
    return fn(spec)


def expected_output(spec: ProbeSpec, tid: int) -> int | None:
    """Return expected u32 stored at out[tid], or None if undetermined."""
    _, exp = TEMPLATES.get(spec.template_id, (None, None))
    return exp(spec, tid) if exp else None
