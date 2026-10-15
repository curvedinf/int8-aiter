# SPDX-License-Identifier: Apache-2.0
"""Bounded GS128 INT8 copies for immutable packed dense serving weights."""

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as lib

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_packed_g128 import (
    _decode,
    _decode_nvfp4,
    _decode_q8,
)
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

# Reloading tuning code must not allocate duplicate copies of immutable weights.
_CACHE = globals().get("_CACHE", {})
_BYTES = globals().get("_BYTES", {})


@triton.jit
def _expand(
    W,
    GRID,
    WS,
    WG,
    GS,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    LENGTH: tl.constexpr,
    START: tl.constexpr,
    CODEC: tl.constexpr,
    PER_ROW: tl.constexpr,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    if CODEC == "iq2_xxs":
        values = _decode(W, GRID, rows, START + cols, N, K)
    elif CODEC == "q8_0":
        values = _decode_q8(W, rows, START + cols, N, K)
    else:
        values = _decode_nvfp4(W, WS, WG, rows, START + cols, N, K, PER_ROW)
    scales = tl.load(GS + rows * tl.cdiv(LENGTH, 128) + tl.program_id(1), rows < N, 1.0)
    quant = lib.nearbyint(values * tl.div_rn(1.0, scales[:, None])).to(tl.int8)
    tl.store(
        OUT + rows[:, None] * LENGTH + cols[None, :],
        quant,
        (rows[:, None] < N) & (cols[None, :] < LENGTH),
    )


def cached_gemm(q, scales, packed, grid, groups, ws, wg, full_k, start, codec):
    """Return FP32 GEMM output, or None when this weight exceeds the cache budget.

    Only dense projections call this function. The original checkpoint and its
    existing derived scales stay alive; the cache adds one byte per weight.
    No eviction occurs while captured graphs can reference a cached tensor.
    """
    m, k = q.shape
    n = packed.shape[0]
    if m <= 64 and n <= 512:
        return None
    device = packed.device
    budget = int(float(os.environ.get("VLLM_BOLT_DENSE_CACHE_GIB", "0")) * 2**30)
    if budget <= 0:
        return None
    version = None if packed.is_inference() else packed._version
    key = (device, packed.data_ptr(), version, start, k, codec)
    entry = _CACHE.get(key)
    if entry is None:
        size = n * k
        if _BYTES.get(device, 0) + size > budget:
            return None
        # Populate during eager warmup, not from within a captured graph.
        if torch.cuda.is_current_stream_capturing():
            return None
        weight = torch.empty((n, k), device=device, dtype=torch.int8)
        _expand[(triton.cdiv(n, 16), triton.cdiv(k, 128))](
            packed,
            grid,
            ws,
            wg,
            groups,
            weight,
            n,
            full_k,
            k,
            start,
            codec,
            codec == "nvfp4" and wg.numel() != 1,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _CACHE[key] = (packed, groups, weight)
        _BYTES[device] = _BYTES.get(device, 0) + size
    else:
        weight = entry[2]
    bm, bn, warps = (
        (16, 64, 4) if m <= 16 else (64, 64, 4) if m <= 64 else (128, 128, 8)
    )
    if m <= 64 and n <= 2048:
        bm = 16
    config = {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": warps,
        "num_stages": 1,
        "cache_modifier": ".cg",
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": 2048,
        "enable_fp_fusion": False,
    }
    return gemm_a8w8_blockscale(
        q, weight, scales, groups[0], dtype=torch.float32, config=config
    )
