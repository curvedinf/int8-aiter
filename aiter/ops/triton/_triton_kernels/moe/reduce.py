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
