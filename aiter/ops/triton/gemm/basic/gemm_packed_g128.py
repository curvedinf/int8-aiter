# SPDX-License-Identifier: Apache-2.0
"""Packed-weight GS128 INT8 GEMMs; no expanded weight cache."""

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_packed_g128 import (
    _embedding,
    _gemm,
    _quant,
)


def make_grid(device):
    from aiter.ops.triton._triton_kernels.gemm.basic.iq2_xxs_grid import IQ2_XXS_GRID

    return torch.tensor(
        [v >> (8 * j) & 255 for v in IQ2_XXS_GRID for j in range(8)],
        dtype=torch.uint8,
        device=device,
    )


def quantize(x):
    if (
        x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or not x.is_floating_point()
    ):
        raise ValueError("Expected contiguous floating-point GPU activation matrix")
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((m, triton.cdiv(k, 128)), device=x.device, dtype=torch.float32)
    _quant[(m, triton.cdiv(k, 128))](x, q, s, m, k)
    return q, s


def packed_gemm(
    q,
    scales,
    packed,
    grid,
    *,
    start=0,
    relu2=False,
    quant_out=False,
    codec="iq2_xxs",
    weight_scales=None,
    global_scale=None,
):
    """Multiply a logical K slice, including slices crossing source blocks.

    IQ2 uses [N, K/256, 66], Q8_0 [N, K/32, 34], NVFP4 [N, K/2].
    The activation matrix q has [M, local_K].
    Input and output activation groups use local (not global) coordinates.
    """
    if codec not in ("iq2_xxs", "q8_0", "nvfp4"):
        raise ValueError("Unsupported packed codec")
    if codec == "nvfp4":
        if packed.ndim != 2 or weight_scales is None or global_scale is None:
            raise ValueError("NVFP4 requires packed matrix and both scale tensors")
        n, half_k = packed.shape
        full_k = half_k * 2
        if (
            full_k % 16
            or weight_scales.shape != (n, full_k // 16)
            or weight_scales.dtype != torch.float8_e4m3fn
            or global_scale.numel() not in (1, n)
            or global_scale.dtype != torch.float32
        ):
            raise ValueError("Invalid NVFP4 scale geometry")
        ws = weight_scales.view(torch.uint8)
        wg = global_scale
    else:
        block_bytes, block_size = (66, 256) if codec == "iq2_xxs" else (34, 32)
        if packed.ndim != 3 or packed.shape[-1] != block_bytes:
            raise ValueError("Invalid packed block geometry")
        n, blocks, _ = packed.shape
        full_k = blocks * block_size
        ws, wg = scales, scales  # Unused pointers for codecs with inline scales.
    m, length = q.shape
    if (
        q.dtype != torch.int8
        or packed.dtype != torch.uint8
        or scales.dtype != torch.float32
        or grid.dtype != torch.uint8
        or scales.shape != (m, triton.cdiv(length, 128))
        or grid.numel() != 2048
        or start < 0
        or start + length > full_k
        or min(m, n, length) < 1
        or not all(
            t.is_cuda and t.is_contiguous() and t.device == q.device
            for t in (q, scales, packed, grid, ws, wg)
        )
    ):
        raise ValueError("Unsupported packed IQ2 G128 GEMM contract")
    y = torch.empty(
        (m, n), device=q.device, dtype=torch.int8 if quant_out else torch.float32
    )
    ys = torch.empty((m, triton.cdiv(n, 128)), device=q.device, dtype=torch.float32)
    # Amortize packed decoding across tokens without changing GS128 groups.
    bm = (256 if m >= 256 else 32) if codec == "nvfp4" else 16
    _gemm[(triton.cdiv(m, bm), triton.cdiv(n, 128))](
        q,
        scales,
        packed,
        grid,
        ws,
        wg,
        y,
        ys,
        m,
        n,
        full_k,
        start,
        length,
        relu2,
        quant_out,
        bm,
        codec,
        WG_PER_ROW=codec == "nvfp4" and wg.numel() != 1,
        num_warps=8 if bm == 256 else 4,
        num_stages=1,
        matrix_instr_nonkdim=16,
        enable_fp_fusion=False,
    )
    return (y, ys) if quant_out else y


def packed_embedding(weight, ids, width, dtype=torch.bfloat16):
    out = torch.empty((*ids.shape, width), device=weight.device, dtype=dtype)
    _embedding[(ids.numel(), triton.cdiv(width, 256))](
        weight, ids.contiguous(), out, ids.numel(), width, weight.shape[0], 256
    )
    return out


def packed_experts(q, scales, packed, grid, ids, *, start=0, down=False, plan=None):
    m, topk = ids.shape
    routes = m * topk
    n, blocks = packed.shape[1:3]
    length = q.shape[-1]
    y = torch.empty(
        (routes, n), device=q.device, dtype=torch.float32 if down else torch.int8
    )
    ys = torch.empty(
        (routes, triton.cdiv(n, 128)), device=q.device, dtype=torch.float32
    )
    sorted_ids, experts, count = (None, ids, None) if plan is None else plan
    tiles = routes if plan is None else experts.numel()
    _gemm[(tiles, triton.cdiv(n, 128))](
        q,
        scales,
        packed,
        grid,
        scales,
        scales,
        y,
        ys,
        routes,
        n,
        blocks * 256,
        start,
        length,
        not down,
        not down,
        16,
        "iq2_xxs",
        experts,
        topk,
        True,
        down,
        packed.stride(0),
        sorted_ids,
        count,
        plan is not None,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
        enable_fp_fusion=False,
    )
    return y if down else (y, ys)
