# SPDX-License-Identifier: Apache-2.0
"""Reference-dtype attention output projection with a GS128 INT8 epilogue."""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as lib


@triton.jit
def _gemm(
    X,
    W,
    BIAS,
    Y,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    kk = tl.arange(0, 32)
    acc = tl.zeros((BM, 128), tl.float32)
    for start in range(tl.cdiv(K, 32)):
        ks = start * 32 + kk
        x = tl.load(
            X + rows[:, None] * K + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            0,
        )
        w = tl.load(W + cols[None, :] * K + ks[:, None], ks[:, None] < K, 0)
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(BIAS + cols)[None, :].to(tl.float32)
    # Preserve the output rounding point of the reference-dtype projection.
    acc = acc.to(X.dtype.element_ty).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(acc), 1), 127.0), 1.0e-30)
    q = lib.nearbyint(tl.div_rn(acc, scale[:, None])).to(tl.int8)
    tl.store(Y + rows[:, None] * N + cols[None, :], q, rows[:, None] < M)
    tl.store(S + rows * (N // 128) + tl.program_id(1), scale, rows < M)


def gemm_float_quant_out(x, weight, bias=None):
    m, k = x.shape
    n = weight.shape[0]
    if (
        x.dtype not in (torch.bfloat16, torch.float16)
        or weight.dtype != x.dtype
        or weight.shape != (n, k)
        or n % 128
        or not x.is_cuda
        or weight.device != x.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("Expected contiguous reference-dtype attention matrices")
    if bias is not None and (
        bias.shape != (n,) or bias.device != x.device or not bias.is_contiguous()
    ):
        raise ValueError("Invalid attention output bias")
    q = torch.empty((m, n), device=x.device, dtype=torch.int8)
    s = torch.empty((m, n // 128), device=x.device, dtype=torch.float32)
    if m:
        _gemm[(triton.cdiv(m, 16), n // 128)](
            x,
            weight,
            bias,
            q,
            s,
            m,
            n,
            k,
            bias is not None,
            16,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, s
