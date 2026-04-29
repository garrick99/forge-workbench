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


# ---------------------------------------------------------------------------
# Template: branch_distance
#   bra to a label N instructions away.  Probes branch-target encoding
#   (BRA imm field) at varying distances.  All threads take the branch
#   (uniform), so divergence is not a factor here.
#
#   operand_spec keys:
#     gap : number of filler instructions between bra and target label
# ---------------------------------------------------------------------------

def template_branch_distance(spec: ProbeSpec) -> str:
    gap = spec.operand_spec.get("gap", 0)
    # Filler ops modify %r3, never touch %r2.
    fillers = "\n    ".join(["add.u32 %r3, %r3, 1;"] * gap)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, 0xDEAD;
    mov.u32 %r3, 0;
    bra L_target;
    mov.u32 %r2, 0xBAD0;       // unreachable; if hit, output is wrong
    {fillers}
L_target:
    add.u32 %r2, %r2, 1;       // 0xDEAE if branch taken correctly
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


def expected_branch_distance(spec: ProbeSpec, tid: int) -> int:
    return 0xDEAE


# ---------------------------------------------------------------------------
# Template: loop_iter
#   Counted loop running N iterations.  Tests back-branch encoding,
#   loop-counter decrement chain, and predicate-based exit.
#
#   operand_spec keys:
#     iters : iteration count (uniform across all threads)
# ---------------------------------------------------------------------------

def template_loop_iter(spec: ProbeSpec) -> str:
    iters = spec.operand_spec.get("iters", 4)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, 0;
    mov.u32 %r3, {iters};
L_loop:
    add.u32 %r2, %r2, 1;
    sub.u32 %r3, %r3, 1;
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra L_loop;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


def expected_loop_iter(spec: ProbeSpec, tid: int) -> int:
    return spec.operand_spec.get("iters", 4) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Template: divergent_branch
#   Half of threads take one path, half take another.  Each path writes
#   a distinct value to %r2.  Tests divergence handling (BSSY/BSYNC /
#   reconvergence stack) and ensures the warp-level scheduler is correct.
#
#   operand_spec keys:
#     thr     : tid threshold for setp.lt (default 32)
#     a_val   : value if predicate true
#     b_val   : value if predicate false
# ---------------------------------------------------------------------------

def template_divergent_branch(spec: ProbeSpec) -> str:
    thr = spec.operand_spec.get("thr", 32)
    a_val = spec.operand_spec.get("a_val", 0xAAAA)
    b_val = spec.operand_spec.get("b_val", 0x5555)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, 0;
    setp.lt.u32 %p1, %r0, {thr};
    @%p1 bra L_taken;
    mov.u32 %r2, {b_val};
    bra L_join;
L_taken:
    mov.u32 %r2, {a_val};
L_join:
    add.u32 %r2, %r2, 1;       // both sides converge here
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


def expected_divergent_branch(spec: ProbeSpec, tid: int) -> int:
    thr = spec.operand_spec.get("thr", 32)
    a_val = spec.operand_spec.get("a_val", 0xAAAA)
    b_val = spec.operand_spec.get("b_val", 0x5555)
    base = a_val if tid < thr else b_val
    return (base + 1) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Template: pred_composition
#   Compose two predicates with and.pred / or.pred / xor.pred, then use
#   the result to conditionally execute.  Probes PSETP encoding and
#   predicate-register tracking.
#
#   operand_spec keys:
#     compose : 'and' | 'or' | 'xor'
#     thr_a   : threshold for first setp.lt
#     thr_b   : threshold for second setp.gt
# ---------------------------------------------------------------------------

def template_pred_composition(spec: ProbeSpec) -> str:
    compose = spec.operand_spec.get("compose", "and")
    thr_a = spec.operand_spec.get("thr_a", 64)
    thr_b = spec.operand_spec.get("thr_b", 16)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0, %p1, %p2, %p3;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.u32 %r2, 0;
    setp.lt.u32 %p1, %r0, {thr_a};
    setp.gt.u32 %p2, %r0, {thr_b};
    {compose}.pred %p3, %p1, %p2;
    @%p3 mov.u32 %r2, 0xCAFE;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


def expected_pred_composition(spec: ProbeSpec, tid: int) -> int:
    compose = spec.operand_spec.get("compose", "and")
    thr_a = spec.operand_spec.get("thr_a", 64)
    thr_b = spec.operand_spec.get("thr_b", 16)
    p1 = tid < thr_a
    p2 = tid > thr_b
    if compose == "and":
        p3 = p1 and p2
    elif compose == "or":
        p3 = p1 or p2
    elif compose == "xor":
        p3 = p1 != p2
    else:
        return None
    return 0xCAFE if p3 else 0


# ---------------------------------------------------------------------------
# Template: shared_barrier
#   Each thread writes its tid to shared[tid], barrier, then reads
#   shared[(tid+offset) % blockDim].  Probes ld.shared / st.shared
#   encoding, bar.sync, and shared-memory address arithmetic.
#
#   The expected output for thread tid is (tid + offset) % blockDim.
#   We use blockDim known via param `n` — but easier: assume blockDim
#   == 128 and clamp via &.
#
#   operand_spec keys:
#     offset : neighbor offset to read (default 1)
# ---------------------------------------------------------------------------

def template_shared_barrier(spec: ProbeSpec) -> str:
    offset = spec.operand_spec.get("offset", 1) & 0x7F  # mask to [0,127]
    # Static shared allocation: 128 u32 slots = 512 bytes.  Avoids the
    # need for the launcher to pass dynamic shared-mem bytes.
    return f""".version 9.0
.target sm_120
.address_size 64
.shared .align 4 .b8 buf[512];
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<5>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // st.shared.u32 [&buf[tid*4]], tid
    mov.u64 %rd1, buf;
    cvt.u64.u32 %rd2, %r0;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd1, %rd2;
    st.shared.u32 [%rd3], %r0;
    bar.sync 0;
    // read shared[(tid + offset) & 0x7F]
    add.u32 %r3, %r0, {offset};
    and.b32 %r3, %r3, 0x7F;
    cvt.u64.u32 %rd4, %r3;
    shl.b64 %rd4, %rd4, 2;
    add.u64 %rd3, %rd1, %rd4;
    ld.shared.u32 %r2, [%rd3];
    bar.sync 0;
    cvt.u64.u32 %rd2, %r0; shl.b64 %rd2, %rd2, 2;
    add.u64 %rd3, %rd0, %rd2;
    st.global.u32 [%rd3], %r2;
    ret;
}}
"""


def expected_shared_barrier(spec: ProbeSpec, tid: int) -> int:
    offset = spec.operand_spec.get("offset", 1) & 0x7F
    return (tid + offset) & 0x7F


# ---------------------------------------------------------------------------
# Template: hmma_m16n8k16
#   Tensor-core mma.sync at m16n8k16 with f16 inputs and f32 accumulator.
#   This is a one-warp probe — n must be exactly 32.
#
#   We pick A = identity-shape (a[i][j] = 1.0 iff i==j on the k-axis),
#   B = identity-shape, C = 0, so D should equal A * B = identity-block.
#   This gives a known-output oracle without depending on ptxas.
#
#   For SM_120 m16n8k16 row/col layout (PTX ISA 8.x doc):
#     - Each thread provides 4 .b32 a-fragments (each holds 2 f16s)
#     - Each thread provides 2 .b32 b-fragments
#     - Each thread provides 4 .f32 c-fragments
#     - Each thread receives 4 .f32 d-fragments
#
#   To keep things simple, we use init_ones=True: every fragment piece
#   is 1.0 (in f16: 0x3C00).  For a 16x16 (A) * 16x8 (B) matmul where
#   all entries are 1.0, every output entry = 16.0.  Trivial oracle.
#
#   operand_spec keys: (none — fixed shape)
# ---------------------------------------------------------------------------

def template_hmma_m16n8k16(spec: ProbeSpec) -> str:
    # mma.sync requires all 32 lanes of a warp to be active.  The runner
    # launches N_THREADS=128 (4 warps); all warps perform an independent
    # mma with the same all-ones inputs, so every thread sees a 16.0
    # output element.  Each thread writes ONE u32 (=bit-pattern of d0)
    # to out[tid] — fitting the standard 4-bytes-per-tid layout.
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    .reg .b32 %a<4>;            // A fragment: 4 .b32 each = 2 f16
    .reg .b32 %b<2>;            // B fragment: 2 .b32 each = 2 f16
    .reg .f32 %c<4>;            // C accumulator
    .reg .f32 %d<4>;            // D output

    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];

    // 1.0 as f16 = 0x3C00; packed (1.0, 1.0) = 0x3C003C00
    mov.b32 %a0, 0x3C003C00;
    mov.b32 %a1, 0x3C003C00;
    mov.b32 %a2, 0x3C003C00;
    mov.b32 %a3, 0x3C003C00;
    mov.b32 %b0, 0x3C003C00;
    mov.b32 %b1, 0x3C003C00;
    mov.f32 %c0, 0f00000000;
    mov.f32 %c1, 0f00000000;
    mov.f32 %c2, 0f00000000;
    mov.f32 %c3, 0f00000000;

    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{ %d0, %d1, %d2, %d3 }},
        {{ %a0, %a1, %a2, %a3 }},
        {{ %b0, %b1 }},
        {{ %c0, %c1, %c2, %c3 }};

    mov.b32 %r2, %d0;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


def expected_hmma_m16n8k16(spec: ProbeSpec, tid: int) -> int:
    """Each thread writes 4 f32 outputs.  When A=all-1.0(f16), B=all-1.0(f16),
    C=0, every entry of D is 16.0 (sum of 16 products of 1.0*1.0).

    This expected_output returns the bit-pattern of one f32 slot — we
    use slot 0.  The HMMA template's runner-side check should compare
    all four 4-byte slots per thread; for now we use slot 0 = 16.0 as
    a simple oracle marker.
    """
    # f32 16.0 = 0x41800000
    return 0x41800000


# ---------------------------------------------------------------------------
# Template: hmma_m16n8k8
#   Smaller k=8 variant of HMMA (FP16 inputs, FP32 accumulator).
#     A: 2 .b32 (4 f16)
#     B: 1 .b32 (2 f16)
#     C/D: 4 f32
#
#   With A=B=all-1.0(f16), C=0, every D entry = 8.0 = 0x41000000.
# ---------------------------------------------------------------------------

def template_hmma_m16n8k8(spec: ProbeSpec) -> str:
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    .reg .b32 %a<2>; .reg .b32 %b<1>;
    .reg .f32 %c<4>; .reg .f32 %d<4>;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.b32 %a0, 0x3C003C00; mov.b32 %a1, 0x3C003C00;
    mov.b32 %b0, 0x3C003C00;
    mov.f32 %c0, 0f00000000; mov.f32 %c1, 0f00000000;
    mov.f32 %c2, 0f00000000; mov.f32 %c3, 0f00000000;
    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
        {{ %d0, %d1, %d2, %d3 }},
        {{ %a0, %a1 }},
        {{ %b0 }},
        {{ %c0, %c1, %c2, %c3 }};
    mov.b32 %r2, %d0;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


def expected_hmma_m16n8k8(spec: ProbeSpec, tid: int) -> int:
    return 0x41000000  # f32 8.0


# ---------------------------------------------------------------------------
# Template: hmma_bf16_m16n8k16
#   bf16 inputs (1.0 = 0x3F80), f32 accumulator.  Same shape as f16
#   m16n8k16, expected D = 16.0.
# ---------------------------------------------------------------------------

def template_hmma_bf16_m16n8k16(spec: ProbeSpec) -> str:
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    .reg .b32 %a<4>; .reg .b32 %b<2>;
    .reg .f32 %c<4>; .reg .f32 %d<4>;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    // bf16 1.0 = 0x3F80; packed (1.0,1.0) = 0x3F803F80
    mov.b32 %a0, 0x3F803F80; mov.b32 %a1, 0x3F803F80;
    mov.b32 %a2, 0x3F803F80; mov.b32 %a3, 0x3F803F80;
    mov.b32 %b0, 0x3F803F80; mov.b32 %b1, 0x3F803F80;
    mov.f32 %c0, 0f00000000; mov.f32 %c1, 0f00000000;
    mov.f32 %c2, 0f00000000; mov.f32 %c3, 0f00000000;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {{ %d0, %d1, %d2, %d3 }},
        {{ %a0, %a1, %a2, %a3 }},
        {{ %b0, %b1 }},
        {{ %c0, %c1, %c2, %c3 }};
    mov.b32 %r2, %d0;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


def expected_hmma_bf16_m16n8k16(spec: ProbeSpec, tid: int) -> int:
    return 0x41800000  # f32 16.0


# ---------------------------------------------------------------------------
# Template: hmma_tf32_m16n8k8
#   tf32 m16n8k8: A 4 .b32 (4 tf32), B 2 .b32 (2 tf32), C/D 4 f32.
#   tf32 1.0 = 0x3F800000 (f32 representation, top 19 bits used).
#   With A=B=1.0, C=0: D = 8.0.
# ---------------------------------------------------------------------------

def template_hmma_tf32_m16n8k8(spec: ProbeSpec) -> str:
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<8>; .reg .u64 %rd<3>; .reg .pred %p0;
    .reg .b32 %a<4>; .reg .b32 %b<2>;
    .reg .f32 %c<4>; .reg .f32 %d<4>;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    mov.b32 %a0, 0x3F800000; mov.b32 %a1, 0x3F800000;
    mov.b32 %a2, 0x3F800000; mov.b32 %a3, 0x3F800000;
    mov.b32 %b0, 0x3F800000; mov.b32 %b1, 0x3F800000;
    mov.f32 %c0, 0f00000000; mov.f32 %c1, 0f00000000;
    mov.f32 %c2, 0f00000000; mov.f32 %c3, 0f00000000;
    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
        {{ %d0, %d1, %d2, %d3 }},
        {{ %a0, %a1, %a2, %a3 }},
        {{ %b0, %b1 }},
        {{ %c0, %c1, %c2, %c3 }};
    mov.b32 %r2, %d0;
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


def expected_hmma_tf32_m16n8k8(spec: ProbeSpec, tid: int) -> int:
    return 0x41000000  # f32 8.0


TEMPLATES: dict[str, tuple[Callable[[ProbeSpec], str],
                           Callable[[ProbeSpec, int], int | None] | None]] = {
    "alu_single":         (template_alu_single,         None),
    "alu_acc_self":       (template_alu_acc_self,       expected_alu_acc_self),
    "pair_distance":      (template_pair_distance,      None),
    "latency_sweep":      (template_latency_sweep,      expected_latency_sweep),
    "alu_64bit":          (template_alu_64bit,          None),
    "load_consume":       (template_load_consume,       None),
    "predicated_alu":     (template_predicated_alu,     None),
    "atomic_op":          (template_atomic_op,          None),
    "alu_f32":            (template_alu_f32,            None),
    "cvt_op":             (template_cvt_op,             None),
    "alu_unary":          (template_alu_unary,          None),
    "bitfield":           (template_bitfield,           None),
    "selp_op":            (template_selp_op,            None),
    "fma_op":             (template_fma_op,             None),
    "branch_distance":    (template_branch_distance,    expected_branch_distance),
    "loop_iter":          (template_loop_iter,          expected_loop_iter),
    "divergent_branch":   (template_divergent_branch,   expected_divergent_branch),
    "pred_composition":   (template_pred_composition,   expected_pred_composition),
    "shared_barrier":     (template_shared_barrier,     expected_shared_barrier),
    "hmma_m16n8k16":      (template_hmma_m16n8k16,      expected_hmma_m16n8k16),
    "hmma_m16n8k8":       (template_hmma_m16n8k8,       expected_hmma_m16n8k8),
    "hmma_bf16_m16n8k16": (template_hmma_bf16_m16n8k16, expected_hmma_bf16_m16n8k16),
    "hmma_tf32_m16n8k8":  (template_hmma_tf32_m16n8k8,  expected_hmma_tf32_m16n8k8),
    "tma_commit_wait":    (None,                        None),  # filled in below
}


# ---------------------------------------------------------------------------
# Template: tma_commit_wait
#
# Tensor Memory Accelerator (TMA) async-copy synchronization primitives.
# This is the FIRST CUT of the TMA probe family — it exercises the
# cheap standalone sync ops (no tensor descriptor / mbarrier setup
# required, runs single-thread):
#
#   cp.async.bulk.commit_group  →  UTMACMDFLUSH (opcode 0x9b7)
#   cp.async.bulk.wait_group N  →  DEPBAR.LE  SB0, N
#
# Verification path is byte-match against ptxas: there's no observable
# GPU output so the oracle is structural (does our cubin match ptxas's,
# does it compile clean, does the expected opcode appear in the SASS).
#
# Future TMA templates (tensor.1d/2d load/store) need:
#   - a tensor-map kernel parameter (.param .u64 tma_desc)
#   - host-side cuTensorMapEncodeTiled() in the runner
#   - mbarrier.init / arrive_expect_tx / wait_parity for ordering
# Tracked separately; this template lays the foundation.
#
#   operand_spec keys:
#     wait_count : N for cp.async.bulk.wait_group N (default 0)
#     n_commits  : how many commit_group invocations to chain
#                  before the wait (default 1)
# ---------------------------------------------------------------------------

def template_tma_commit_wait(spec: ProbeSpec) -> str:
    wait_count = spec.operand_spec.get("wait_count", 0)
    n_commits = max(1, spec.operand_spec.get("n_commits", 1))
    commits = "\n    ".join(["cp.async.bulk.commit_group;"] * n_commits)
    return f""".version 9.0
.target sm_120
.address_size 64
.visible .entry probe(.param .u64 p_out, .param .u32 n) {{
    .reg .u32 %r<4>; .reg .u64 %rd<3>; .reg .pred %p0;
    mov.u32 %r0, %tid.x;
    ld.param.u32 %r1, [n]; setp.ge.u32 %p0, %r0, %r1; @%p0 ret;
    ld.param.u64 %rd0, [p_out];
    {commits}
    cp.async.bulk.wait_group {wait_count};
    mov.u32 %r2, {0xCAFE + wait_count};
    cvt.u64.u32 %rd1, %r0; shl.b64 %rd1, %rd1, 2;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}}
"""


def expected_tma_commit_wait(spec: ProbeSpec, tid: int) -> int:
    # The TMA sync ops have no observable side-effect on this kernel
    # (no actual data transfer is in flight).  We tag the output with
    # 0xCAFE + wait_count so different bins write distinguishable
    # values, providing a baseline correctness check (kernel ran to
    # completion, the post-sync mov + store landed).
    return 0xCAFE + spec.operand_spec.get("wait_count", 0)


TEMPLATES["tma_commit_wait"] = (template_tma_commit_wait, expected_tma_commit_wait)


def materialize(spec: ProbeSpec) -> str:
    """Return PTX text for a given probe spec."""
    fn, _ = TEMPLATES[spec.template_id]
    return fn(spec)


def expected_output(spec: ProbeSpec, tid: int) -> int | None:
    """Return expected u32 stored at out[tid], or None if undetermined."""
    _, exp = TEMPLATES.get(spec.template_id, (None, None))
    return exp(spec, tid) if exp else None
