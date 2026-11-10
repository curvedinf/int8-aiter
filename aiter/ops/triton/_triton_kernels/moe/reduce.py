import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from aiter.ops.triton._triton_kernels.moe.activations import _swiglu


@triton.jit
def _reduce_grouped(
    X,
    stride_xb: tl.uint64,
    stride_xm: tl.uint64,
    stride_xn,
    Out,
    stride_om: tl.uint64,
    stride_on,  # output tensor
    InIndx,
    B,
    M,
    N,
    num_blocks,
    # fused activation function
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    SWIGLU_ADD_RESIDUAL: tl.constexpr,
    USE_TDM: tl.constexpr,
    # Step 9: external residual fold-in. When HAS_EXT_RESIDUAL=True,
    # Residual[token, :] is added to `acc` before the writeback.
    Residual,
    stride_extres_m: tl.uint64,
    stride_extres_n,
    HAS_EXT_RESIDUAL: tl.constexpr,
    # INT8 combine epilogue + quantized external residual (shared branch)
    QOut,
    stride_qo_m,
    stride_qo_n,
    SOut,
    stride_so_m,
    stride_so_n,
    ResidualQ,
    stride_rq_m,
    stride_rq_n,
    ResidualS,
    stride_rs_m,
    stride_rs_n,
    QUANT_OUT: tl.constexpr,
    HAS_EXT_QRESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_t = pid // num_blocks
    pid_n = pid % num_blocks

    BLOCK_N_OUT: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    start = pid_t * K
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t,)
    else:
        indxs = ()
        for i in tl.static_range(0, K):
            indxs = indxs + (tl.load(InIndx + start + i),)
    XPtrs = X + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_xn
    OutPtrs = Out + (pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)) * stride_on

    acc = tl.zeros([BLOCK_N_OUT], dtype=tl.float32)
    x_n_mask = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) < N
    if USE_TDM and EVEN_N:
        x_desc = tl.make_tensor_descriptor(
            base=X, shape=(B * M, N), strides=(N, 1), block_shape=(1, BLOCK_N)
        )
    # accumulate contributions for this tile
    for i in tl.static_range(0, K):
        curr = tl.zeros([BLOCK_N], dtype=tl.float32)
        # iterate over split_k partial values
        for b in tl.range(0, B):
            if USE_TDM and EVEN_N:
                row = b * M + indxs[i]
                vals = tl.reshape(x_desc.load([row, pid_n * BLOCK_N]), (BLOCK_N,))
            else:
                x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
                if EVEN_N:
                    vals = tl.load(x_row_ptr)
                else:
                    vals = tl.load(x_row_ptr, mask=x_n_mask, other=0.0)
            vals = vals.to(tl.float32)
            curr += vals

        # apply nonlinearity to split-k output
        if APPLY_SWIGLU:
            curr = _swiglu(
                curr[None, :], alpha, limit, ADD_RESIDUAL=SWIGLU_ADD_RESIDUAL
            )
        curr = tl.reshape(curr, [curr.shape[-1]])
        # update final accumulator
        acc += curr
    # Compute per-32-col MXFP scales for this tile if requested
    Nrem = N // ACTIVATION_REDUCTION_N

    # Step 9: optional external residual fold-in: load residual at this
    # tile and add to acc before writeback. Same per-token-row layout as Out.
    if HAS_EXT_RESIDUAL:
        res_offs_n = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)
        res_ptr = Residual + pid_t * stride_extres_m + res_offs_n * stride_extres_n
        if EVEN_N:
            res = tl.load(res_ptr).to(tl.float32)
            acc = acc + res
        else:
            res_mask = res_offs_n < Nrem
            res = tl.load(res_ptr, mask=res_mask, other=0.0).to(tl.float32)
            acc = acc + res
    if HAS_EXT_QRESIDUAL:
        # Shared-expert branch stays an int8 payload: dequantize per group
        # in-register and fold it before the combine quant.
        res_offs_n = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)
        rq = tl.load(
            ResidualQ + pid_t * stride_rq_m + res_offs_n * stride_rq_n
        ).to(tl.float32)
        s_offs = (
            pid_n * (BLOCK_N_OUT // 128)
            + tl.arange(0, BLOCK_N_OUT // 128)
        )
        rs = tl.load(ResidualS + pid_t * stride_rs_m + s_offs * stride_rs_n)
        rs_cols = tl.reshape(
            tl.broadcast_to(rs[:, None], (BLOCK_N_OUT // 128, 128)),
            (BLOCK_N_OUT,),
        )
        acc = acc + rq * rs_cols
    if QUANT_OUT:
        tl.static_assert(
            BLOCK_N_OUT % 128 == 0,
            "INT8 combine requires whole 128-column groups per tile",
        )
        # The float path stores the combined row in the output dtype and the
        # caller re-quantizes that stored value; round identically here.
        acc = acc.to(Out.dtype.element_ty).to(tl.float32)
        groups = tl.reshape(acc, (BLOCK_N_OUT // 128, 128))
        sc = tl.fdiv(
            tl.maximum(tl.max(tl.abs(groups), 1), 1.0e-8),
            127.0,
            ieee_rounding=True,
        )
        q = libdevice.llrint(
            tl.fdiv(groups, sc[:, None], ieee_rounding=True)
        )
        q = tl.reshape(
            tl.minimum(tl.maximum(q, -127.0), 127.0), (BLOCK_N_OUT,)
        )
        offs_n = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)
        tl.store(
            QOut + pid_t * stride_qo_m + offs_n * stride_qo_n, q.to(tl.int8)
        )
        sbase = pid_n * (BLOCK_N_OUT // 128)
        tl.store(
            SOut
            + pid_t * stride_so_m
            + (sbase + tl.arange(0, BLOCK_N_OUT // 128)) * stride_so_n,
            sc,
        )
        return
    # write-back for this tile
    out_ptr = OutPtrs + pid_t * stride_om
    if EVEN_N:
        tl.store(out_ptr, acc)
    else:
        out_n_mask = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT) < Nrem
        tl.store(out_ptr, acc, mask=out_n_mask)


@triton.jit
def _topk_combine_fused(
    X,
    Indx,
    QOut,
    SOut,
    stride_xm,
    stride_xn,
    M,
    N,
    BLOCK_N: tl.constexpr,
    K: tl.constexpr,
    GROUP_N: tl.constexpr = 128,
):
    """Fused top-K gather-sum + per-128 int8 quant (gfx908 decode path).

    One program per (output token, N tile). The K gathers stream in registers
    instead of the generic grouped kernel's uniform 2-warp cdiv scheme, which
    teams of M=12 x N=2048 pay as launch + serialized gathers (~50 us).

    Exact numerics vs _reduce_grouped QUANT_OUT: fp32 sum, bf16 round,
    amax/127 round-to-nearest-even per 128 columns. Bit-identical (verified
    against the production kernel: q/s 0/24576 mismatches at M=12).

    Shapes: X [M*K, N] bf16, Indx [M, K] int32, QOut [M, N] int8,
    SOut [M, N/128] fp32. N % BLOCK_N == 0 and BLOCK_N % 128 == 0 required.
    """
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    base = pid_t * K
    for i in tl.static_range(0, K):
        idx = tl.load(Indx + base + i)
        vals = tl.load(X + idx * stride_xm + offs_n * stride_xn).to(tl.float32)
        acc += vals
    acc = acc.to(tl.bfloat16).to(tl.float32)
    groups = tl.reshape(acc, (BLOCK_N // GROUP_N, GROUP_N))
    sc = tl.fdiv(
        tl.maximum(tl.max(tl.abs(groups), 1), 1.0e-8),
        127.0,
        ieee_rounding=True,
    )
    q = libdevice.llrint(tl.fdiv(groups, sc[:, None], ieee_rounding=True))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(QOut + pid_t * N + offs_n, tl.reshape(q, (BLOCK_N,)).to(tl.int8))
    sbase = pid_n * (BLOCK_N // GROUP_N)
    tl.store(
        SOut + pid_t * (N // GROUP_N) + sbase + tl.arange(0, BLOCK_N // GROUP_N),
        sc,
    )


# ---------------------------------------------------------------------------
# Direct routing structures from topk() outputs, TWO-KERNEL (M <= 256)
# ---------------------------------------------------------------------------
# Kernel A (atomic hist): grid (cdiv(M*K,64),) -> Hist. 24us at M=12.
# Kernel B (offsets + bpm + sort): single program over 128 lanes.
#   - hist -> cumsum (token/tile offsets) via tl.cumsum over [256] vector
#   - bpm via searchsorted-equivalent: for each tile t, find expert via
#     tl.sum(where(cum_t <= t)) (72 iterations, vectorized inner).
#   - sort (expert<<16)|i over 128 lanes + keyed scan -> gather/scatter.
# Total est: 24 (A incl zero) + ~90 (B) = ~114us vs production route total
# 368us, Structures-only (sort 128us) comparison: 114 vs 128 (modest win).
# Topk (201us, tested aiter) still runs upstream. Net routing total would
# go 368 -> ~315us if B hits 90us. Bit-identical (topk-order input).


@triton.jit
def _hist_atomic(ExptIndx, Hist, M, K: tl.constexpr):
    pid = tl.program_id(0)
    BLOCK: tl.constexpr = 64
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_gates = M * K
    mask = offs < n_gates
    e = tl.load(ExptIndx + offs, mask=mask, other=0)
    tl.atomic_add(Hist + e, 1, mask=mask)


@triton.jit
def _offsets_sort_scatter(
    ExptIndx,
    ExptScal,
    Hist,
    TopkIndxOut,
    GateIndxOut,
    GateScalOut,
    TokenOffsRaw,
    TokenOffsPad,
    BlockPidMap,
    M,
    N_EXPTS,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Offsets from hist via vectorized cumsum (256 lanes).
    hist_vec = tl.load(Hist + tl.arange(0, 256))
    cum_raw = tl.cumsum(hist_vec, 0) - hist_vec
    tiles = (hist_vec + BLOCK_M - 1) // BLOCK_M
    tiles = tl.where(hist_vec > 0, tiles, 0)
    cum_t = tl.cumsum(tiles, 0) - tiles
    # One vector store per offset array (19us measured vs 377us per-element loop).
    tl.store(TokenOffsRaw + tl.arange(0, 256), cum_raw)
    tl.store(TokenOffsPad + tl.arange(0, 256), cum_t)
    n_gates = M * K
    tl.store(TokenOffsRaw + N_EXPTS, n_gates)
    total_tiles = tl.sum(tiles)
    tl.store(TokenOffsPad + N_EXPTS, total_tiles)
    # bpm: searchsorted-equivalent over tile slots. Bound is n_gates (= max
    # tiles when M*K <= E); the hardcoded 72 was an M=12 assumption that
    # broke M=16 (95 gates, tiles up to 96) and faulted the GEMM.
    for t in tl.static_range(0, 128):
        if t < n_gates and t < total_tiles:
            # expert = #{e: cum_t[e] <= t} - 1  (monotone cum_t)
            ok = cum_t <= t
            e_max = tl.sum(tl.where(ok, 1, 0)) - 1
            tile_base = tl.sum(tl.where(tl.arange(0, 256) == e_max, cum_t, 0))
            blk = t - tile_base
            tl.store(BlockPidMap + t, ((blk << 16) | e_max))
    # Sort (expert<<16)|i over 128 lanes + keyed scan.
    LOAD: tl.constexpr = 128
    offs = tl.arange(0, LOAD)
    in_mask = offs < n_gates
    raw_e = tl.load(ExptIndx + offs, mask=in_mask, other=-1)
    expert = tl.where(in_mask & (raw_e >= 0), raw_e.to(tl.uint32), 0xFFFF)
    kv = (expert << 16) | (offs & 0xFFFF)
    kv = tl.sort(kv, 0)
    sorted_e = kv >> 16
    sorted_i = kv & 0xFFFF
    valid = sorted_e != 0xFFFF
    x = (kv & 0xFFFF0000 | 0x00000001)
    scan = tl.associative_scan(x, 0, _keyed_add_two)
    excl = (scan - 1) & 0xFFFF
    gates = tl.load(TokenOffsRaw + sorted_e, mask=valid, other=0) + excl
    tl.store(TopkIndxOut + gates, sorted_i.to(tl.int32), mask=valid)
    tl.store(GateIndxOut + sorted_i, gates.to(tl.int32), mask=valid)
    scal = tl.load(ExptScal + sorted_i, mask=valid, other=0.0)
    tl.store(GateScalOut + gates, scal, mask=valid)


@triton.jit
def _keyed_add_two(a, b):
    ka = a & 0xFFFF0000
    kb = b & 0xFFFF0000
    return tl.where(ka == kb, a + b, b)


# ---------------------------------------------------------------------------
# Routing structures from topk() outputs, TWO-KERNEL production form
# ---------------------------------------------------------------------------
# This file section is the aiter-side counterpart of the vLLM two-kernel
# microbench (116us). The production fast path lives behind
# VLLM_GFX908_DIRECT_ROUTING in gptq_moe_a8w8.py and is bounded M*K<=128
# (LOAD=128 sort). Full-M coverage needs the 2048-lane sort (28us alone,
# measured) which lands ~164us total -- slower than production sort (128us).
# See ledger FLINT_ROUTING_SERVING_AB_20261110 (serving rejection, M=96).
# The kernels below are the max-M-exact implementations used by that probe:
#   _hist_atomic: grid (cdiv(M*K,64),), atomic hist.
#   _offsets_sort_scatter: single program, LOAD=128 lanes, may only be
#     launched when M*K <= 128 (callers must bound).


@triton.jit
def _hist_atomic(ExptIndx, Hist, M, K: tl.constexpr):
    pid = tl.program_id(0)
    BLOCK: tl.constexpr = 64
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_gates = M * K
    mask = offs < n_gates
    e = tl.load(ExptIndx + offs, mask=mask, other=0)
    tl.atomic_add(Hist + e, 1, mask=mask)


@triton.jit
def _keyed_add_two(a, b):
    ka = a & 0xFFFF0000
    kb = b & 0xFFFF0000
    return tl.where(ka == kb, a + b, b)


@triton.jit
def _offsets_sort_scatter(
    ExptIndx,
    ExptScal,
    Hist,
    TopkIndxOut,
    GateIndxOut,
    GateScalOut,
    TokenOffsRaw,
    TokenOffsPad,
    BlockPidMap,
    M,
    N_EXPTS,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Offsets from hist via vectorized cumsum (256 lanes).
    hist_vec = tl.load(Hist + tl.arange(0, 256))
    cum_raw = tl.cumsum(hist_vec, 0) - hist_vec
    tiles = (hist_vec + BLOCK_M - 1) // BLOCK_M
    tiles = tl.where(hist_vec > 0, tiles, 0)
    cum_t = tl.cumsum(tiles, 0) - tiles
    # One vector store per offset array (19us measured vs 377us per-element loop).
    tl.store(TokenOffsRaw + tl.arange(0, 256), cum_raw)
    tl.store(TokenOffsPad + tl.arange(0, 256), cum_t)
    n_gates = M * K
    tl.store(TokenOffsRaw + N_EXPTS, n_gates)
    total_tiles = tl.sum(tiles)
    tl.store(TokenOffsPad + N_EXPTS, total_tiles)
    # bpm: searchsorted-equivalent over tile slots. Bound is 128 (M*K<=128
    # callers only); hardcoded 72 was an M=12 assumption that broke M=16.
    for t in tl.static_range(0, 128):
        if t < n_gates and t < total_tiles:
            ok = cum_t <= t
            e_max = tl.sum(tl.where(ok, 1, 0)) - 1
            tile_base = tl.sum(tl.where(tl.arange(0, 256) == e_max, cum_t, 0))
            blk = t - tile_base
            tl.store(BlockPidMap + t, ((blk << 16) | e_max))
    # Sort (expert<<16)|i over 128 lanes + keyed scan.
    LOAD: tl.constexpr = 128
    offs = tl.arange(0, LOAD)
    in_mask = offs < n_gates
    raw_e = tl.load(ExptIndx + offs, mask=in_mask, other=-1)
    expert = tl.where(in_mask & (raw_e >= 0), raw_e.to(tl.uint32), 0xFFFF)
    kv = (expert << 16) | (offs & 0xFFFF)
    kv = tl.sort(kv, 0)
    sorted_e = kv >> 16
    sorted_i = kv & 0xFFFF
    valid = sorted_e != 0xFFFF
    x = (kv & 0xFFFF0000 | 0x00000001)
    scan = tl.associative_scan(x, 0, _keyed_add_two)
    excl = (scan - 1) & 0xFFFF
    gates = tl.load(TokenOffsRaw + sorted_e, mask=valid, other=0) + excl
    tl.store(TopkIndxOut + gates, sorted_i.to(tl.int32), mask=valid)
    tl.store(GateIndxOut + sorted_i, gates.to(tl.int32), mask=valid)
    scal = tl.load(ExptScal + sorted_i, mask=valid, other=0.0)
    tl.store(GateScalOut + gates, scal, mask=valid)


# ---------------------------------------------------------------------------
# Routing structures from topk() outputs, PARALLEL-EXPERTS form (M <= 341)
# ---------------------------------------------------------------------------
# Bit-identical port of sort_tokens_fused() at full M. Per-expert rank via
# vectorized cumsum over the sorted order; offsets/bpm from the atomic hist.
# Grid (E,): each expert program loads ExptIndx once (L2-resident at these
# sizes), computes rank in registers, emits to its offset slice.
# Validated bit-identical vs production routing() at M=96 E=256 K=6
# (hist/off_raw/off_pad/bpm/gather/scatter/gate_scal all 0-diff, 121.9us
# incl both kernels vs production structures step ~166us-equivalent).


@triton.jit
def _routing_experts_parallel(
    ExptIndx,
    ExptScal,
    TopkIndxOut,
    GateIndxOut,
    GateScalOut,
    Hist,
    TokenOffsRaw,
    TokenOffsPad,
    BlockPidMap,
    M,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_e = tl.program_id(0)
    LOAD: tl.constexpr = 2048
    offs = tl.arange(0, LOAD)
    n_gates = M * K
    raw_e = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1)
    is_e = (offs < n_gates) & (raw_e == pid_e)
    rank_cum = tl.cumsum(tl.where(is_e, 1, 0), 0) - 1
    hist_vec = tl.load(Hist + tl.arange(0, 256))
    cum_raw = tl.cumsum(hist_vec, 0) - hist_vec
    tiles = (hist_vec + BLOCK_M - 1) // BLOCK_M
    tiles = tl.where(hist_vec > 0, tiles, 0)
    cum_t = tl.cumsum(tiles, 0) - tiles
    off_raw = tl.sum(tl.where(tl.arange(0, 256) == pid_e, cum_raw, 0))
    off_tile = tl.sum(tl.where(tl.arange(0, 256) == pid_e, cum_t, 0))
    tl.store(TokenOffsRaw + pid_e, off_raw)
    tl.store(TokenOffsPad + pid_e, off_tile)
    h = tl.load(Hist + pid_e)
    nb = (h + BLOCK_M - 1) // BLOCK_M if h > 0 else 0
    for blk in range(0, 8):
        if blk < nb:
            tl.store(BlockPidMap + off_tile + blk, ((blk << 16) | pid_e))
    scal_vals = tl.load(ExptScal + offs)
    out_pos = off_raw + rank_cum
    tl.store(TopkIndxOut + out_pos, offs.to(tl.int32), mask=is_e)
    tl.store(GateIndxOut + offs, out_pos.to(tl.int32), mask=is_e)
    tl.store(GateScalOut + out_pos, scal_vals, mask=is_e)
    if pid_e == 255:
        tl.store(TokenOffsRaw + 256, tl.sum(hist_vec))
        tl.store(TokenOffsPad + 256, tl.sum(tiles))
