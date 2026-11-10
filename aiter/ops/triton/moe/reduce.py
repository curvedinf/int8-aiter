import os

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.reduce import (
    _reduce_grouped,
    _topk_combine_fused,
)
from aiter.ops.triton.utils._triton.arch_info import is_tdm_avail

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.moe.reduce import (
        reduce_grouped_gluon as _reduce_grouped_gluon,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.moe.reduce import (
        reduce_grouped_gluon_num_warps as _reduce_grouped_gluon_num_warps,
    )
except (ImportError, ModuleNotFoundError):
    _reduce_grouped_gluon = None
    _reduce_grouped_gluon_num_warps = None


def reduce_grouped(
    x: torch.Tensor,
    indx: torch.Tensor,
    out: torch.Tensor,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    reduction_n=1,
    out_dtype=None,
    swiglu_add_residual: bool = True,
    residual: torch.Tensor | None = None,
    quant_out: bool = False,
    residual_q: torch.Tensor | None = None,
    residual_s: torch.Tensor | None = None,
):
    """
    Grouped row reduction used during moe scatter and also compatible with split-k reduce.

    Arguments
    - x: Tensor[AnyFloat] of shape [(num_groups * K), N]
    - indx: Tensor[Int] of shape [num_groups, K]

    Description
    For each group g in [0, num_groups), this routine sums the K rows of `x`
    specified by `indx[g, :]`. Accumulation is performed
    in float32 for numerical stability, and the result is written back in the
    dtype of `x`.

    Performance notes
    - Memory traffic per group is approximately (valid_rows_read + 1) * N * sizeof(x),
      plus index reads. With no invalid entries, this becomes (K + 1) reads/writes
      of length N per group.

    Returns
    - The output tensor `out`.
    """

    if indx is None and x.shape[0] == 1:
        assert residual is None, (
            "reduce_grouped early-return path can't apply external residual; "
            "either rebuild routing with K>=1 or skip residual fold for this call"
        )
        return x.squeeze(0)
    if indx is not None:
        num_groups = indx.shape[0]
    else:
        num_groups = x.shape[-2]
    K = 1 if indx is None else indx.shape[1]
    out_dtype = x.dtype if out_dtype is None else out_dtype
    assert x.shape[-1] % reduction_n == 0

    # Gluon path on gfx1250 for the plain grouped combine; swiglu-fused (MoE1 split-k) reductions, reduction_n != 1, and non-contiguous inputs stay on the Triton _reduce_grouped.
    use_gluon = (
        is_tdm_avail()
        and indx is not None
        and not apply_swiglu
        and reduction_n == 1
        and x.ndim == 3
        and x.is_contiguous()
        and indx.is_contiguous()
        and not quant_out
        and residual_q is None
    )
    if use_gluon:
        B, M, N = x.shape[0], x.shape[1], x.shape[2]
        npad = triton.next_power_of_2(N)
        has_ext_residual = residual is not None
        if has_ext_residual:
            assert residual.shape == out.shape, (
                f"residual.shape {tuple(residual.shape)} must match "
                f"out.shape {tuple(out.shape)}"
            )
        gluon_num_warps = _reduce_grouped_gluon_num_warps(npad)
        _reduce_grouped_gluon[(num_groups,)](
            X=x,
            Out=out,
            InIndx=indx,
            Residual=residual if has_ext_residual else out,
            stride_xm=x.stride(1),
            stride_om=out.stride(0),
            stride_on=out.stride(1),
            stride_res_m=residual.stride(0) if has_ext_residual else 0,
            stride_res_n=residual.stride(1) if has_ext_residual else 0,
            M=M,
            N=N,
            NPAD=npad,
            B=B,
            K=K,
            NUM_WARPS=gluon_num_warps,
            HAS_EXT_RESIDUAL=has_ext_residual,
            num_warps=gluon_num_warps,
        )
        return out

    BLOCK_N = 512
    num_blocks = triton.cdiv(x.shape[-1], BLOCK_N)
    if quant_out:
        if x.shape[-1] % BLOCK_N:
            raise ValueError("INT8 combine requires N divisible by 512")
        q_out = torch.empty(
            (num_groups, x.shape[-1]), dtype=torch.int8, device=x.device
        )
        s_out = torch.empty(
            (num_groups, x.shape[-1] // 128),
            dtype=torch.float32,
            device=x.device,
        )
        # gfx908 decode fast path: fused top-K gather-sum + quant. One
        # 2-D program grid per token/tile instead of the generic flattened
        # 2-warp sweep; measured 27 us vs ~54 us at M=12 (bit-identical).
        # Only for the exact MoE combine shape (split-k B=1, dense x) AND
        # no external quantized residual: the shared-expert fold-in stays
        # on the generic path (the fused kernel has no residual input).
        if (
            os.environ.get("VLLM_GFX908_TOPK_FUSED_COMBINE", "1") == "1"
            and x.ndim == 3
            and x.shape[0] <= 1
            and indx is not None
            and residual_q is None
        ):
            from aiter.ops.triton._triton_kernels.moe.reduce import (
                _topk_combine_fused as _fused,
            )

            grid = (num_groups, num_blocks)
            _fused[grid](
                x.reshape(x.shape[1], x.shape[2]),
                indx,
                q_out,
                s_out,
                x.stride(1),
                x.stride(2),
                x.shape[1],
                x.shape[2],
                BLOCK_N=BLOCK_N,
                K=K,
            )
            return q_out, s_out
    else:
        q_out = s_out = out
    if residual_q is not None:
        if (
            residual_q.dtype != torch.int8
            or residual_s is None
            or residual_q.shape != (num_groups, x.shape[-1])
            or residual_s.shape != (num_groups, x.shape[-1] // 128)
        ):
            raise ValueError("Invalid quantized external residual")

    # Step 9: prep external residual buffer + strides for the kernel.
    if residual is not None:
        assert (
            residual.shape == out.shape
        ), f"residual.shape {tuple(residual.shape)} must match out.shape {tuple(out.shape)}"
        res_stride_m = residual.stride(0)
        res_stride_n = residual.stride(1)
        has_ext_residual = True
    else:
        res_stride_m = 0
        res_stride_n = 0
        has_ext_residual = False
    _reduce_grouped[(num_blocks * num_groups,)](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out,
        out.stride(0),
        out.stride(1),
        indx,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        num_blocks,
        apply_swiglu,
        alpha,
        limit,
        reduction_n,
        BLOCK_N=BLOCK_N,
        EVEN_N=(x.shape[-1] % BLOCK_N == 0),
        K=K,
        SWIGLU_ADD_RESIDUAL=swiglu_add_residual,
        USE_TDM=is_tdm_avail(),
        Residual=residual,
        stride_extres_m=res_stride_m,
        stride_extres_n=res_stride_n,
        HAS_EXT_RESIDUAL=has_ext_residual,
        QOut=q_out,
        stride_qo_m=q_out.stride(0) if quant_out else 0,
        stride_qo_n=q_out.stride(1) if quant_out else 0,
        SOut=s_out,
        stride_so_m=s_out.stride(0) if quant_out else 0,
        stride_so_n=s_out.stride(1) if quant_out else 0,
        ResidualQ=residual_q if residual_q is not None else q_out,
        stride_rq_m=residual_q.stride(0) if residual_q is not None else 0,
        stride_rq_n=residual_q.stride(1) if residual_q is not None else 0,
        ResidualS=residual_s if residual_s is not None else s_out,
        stride_rs_m=residual_s.stride(0) if residual_s is not None else 0,
        stride_rs_n=residual_s.stride(1) if residual_s is not None else 0,
        QUANT_OUT=quant_out,
        HAS_EXT_QRESIDUAL=residual_q is not None,
        num_warps=2,
    )
    if quant_out:
        return q_out, s_out
    return out
