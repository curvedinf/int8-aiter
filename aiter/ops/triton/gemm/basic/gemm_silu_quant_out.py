# SPDX-License-Identifier: Apache-2.0
"""GS128 W8A8 projection with SiLU(/multiply) and INT8 output in one kernel."""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as lib


@triton.jit
def _gemm(
    X,
    XS,
    W,
    WS,
    BIAS,
    Y,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GATED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BF16: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    kk = tl.arange(0, 128)
    gate = tl.zeros((BM, 128), tl.float32)
    if GATED:
        up = tl.zeros((BM, 128), tl.float32)
    for group in range(tl.cdiv(K, 128)):
        ks = group * 128 + kk
        x = tl.load(
            X + rows[:, None] * K + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            0,
        )
        w = tl.load(
            W + cols[None, :] * K + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            0,
        )
        sx = tl.load(XS + rows * tl.cdiv(K, 128) + group, rows < M, 0)
        sw = tl.load(WS + cols * tl.cdiv(K, 128) + group, cols < N, 0)
        gate += tl.dot(x, w).to(tl.float32) * sx[:, None] * sw[None, :]
        if GATED:
            wu = tl.load(
                W + (N + cols[None, :]) * K + ks[:, None],
                (cols[None, :] < N) & (ks[:, None] < K),
                0,
            )
            su = tl.load(WS + (N + cols) * tl.cdiv(K, 128) + group, cols < N, 0)
            up += tl.dot(x, wu).to(tl.float32) * sx[:, None] * su[None, :]
    if HAS_BIAS:
        gate += tl.load(BIAS + cols, cols < N, 0)[None, :].to(tl.float32)
        if GATED:
            up += tl.load(BIAS + N + cols, cols < N, 0)[None, :].to(tl.float32)
    # Reference GEMM output and activation rounding points, in registers only.
    if BF16:
        gate = gate.to(tl.bfloat16).to(tl.float32)
        if GATED:
            up = up.to(tl.bfloat16).to(tl.float32)
    value = gate / (1.0 + tl.exp(-gate))
    if GATED:
        value *= up
    if BF16:
        value = value.to(tl.bfloat16).to(tl.float32)
    value = tl.where(cols[None, :] < N, value, 0.0)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(value), 1), 127.0), 1.0e-30)
    codes = lib.nearbyint(tl.div_rn(value, scale[:, None])).to(tl.int8)
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        codes,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(S + rows * tl.cdiv(N, 128) + tl.program_id(1), scale, rows < M)


def gemm_silu_quant_out(
    q, scales, weight, groups, *, gated=False, bias=None, dtype=torch.bfloat16
):
    m, k = q.shape
    n = weight.shape[0] // (2 if gated else 1)
    if (
        q.dtype != torch.int8
        or weight.dtype != torch.int8
        or scales.dtype != torch.float32
        or groups.dtype != torch.float32
        or weight.shape != (n * (2 if gated else 1), k)
        or scales.shape != (m, triton.cdiv(k, 128))
        or groups.shape != (1, weight.shape[0], triton.cdiv(k, 128))
        or dtype not in (torch.bfloat16, torch.float32)
        or not all(
            t.is_cuda and t.is_contiguous() and t.device == q.device
            for t in (q, scales, weight, groups)
        )
    ):
        raise ValueError("Invalid GS128 SiLU GEMM payload")
    if bias is not None and (
        bias.shape != (weight.shape[0],)
        or not bias.is_contiguous()
        or bias.device != q.device
    ):
        raise ValueError("Invalid SiLU projection bias")
    out = torch.empty((m, n), device=q.device, dtype=torch.int8)
    s = torch.empty((m, triton.cdiv(n, 128)), device=q.device, dtype=torch.float32)
    bm = 16 if m <= 64 else 64
    if m:
        _gemm[(triton.cdiv(m, bm), triton.cdiv(n, 128))](
            q,
            scales,
            weight,
            groups,
            bias,
            out,
            s,
            m,
            n,
            k,
            gated,
            bias is not None,
            dtype == torch.bfloat16,
            bm,
            num_warps=4 if m <= 64 else 8,
            num_stages=1,
            matrix_instr_nonkdim=16,
            enable_fp_fusion=False,
        )
    return out, s
