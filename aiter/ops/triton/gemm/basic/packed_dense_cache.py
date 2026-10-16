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
_RELEASED_BYTES = globals().get("_RELEASED_BYTES", {})


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


def _cache_weight(packed, grid, groups, ws, wg, full_k, start, k, codec):
    n = packed.shape[0]
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
        entry = (packed, groups, weight)
        _CACHE[key] = entry
        _BYTES[device] = _BYTES.get(device, 0) + size
    return key, entry


def prepare_dense_weight(
    packed, grid, length, *, start=0, codec="iq2_xxs", ws=None, wg=None
):
    """Transfer a budgeted dense projection to immutable INT8-only ownership.

    Call after checkpoint loading and before graph capture. On budget exhaustion
    return None without releasing the packed source. No cache entry retains the
    source on success; callers must also replace their parameter and delete the
    codec metadata. Expert weights never enter this path.
    """
    from .gemm_packed_g128 import _cached_weight_scales

    n = packed.shape[0]
    full_k = packed.shape[1] * (
        2 if codec == "nvfp4" else 256 if codec == "iq2_xxs" else 32
    )
    if codec == "nvfp4":
        if ws is None or wg is None:
            raise ValueError("NVFP4 requires original block and global scales")
        ws = ws.view(torch.uint8)
    else:
        ws = wg = grid
    groups = _cached_weight_scales(
        packed, grid, ws, wg, n, full_k, start, length, codec
    )
    result = _cache_weight(packed, grid, groups, ws, wg, full_k, start, length, codec)
    if result is None:
        return None
    key, (_, groups, weight) = result
    del _CACHE[key]
    # The old address may be reused as soon as the caller drops its parameter.
    # A distinct key prevents collisions with future packed allocations.
    _CACHE[(packed.device, "resident", weight.data_ptr())] = (None, groups, weight)
    from .gemm_packed_g128 import _weight_scale_cache

    scale_key = (packed.data_ptr(), start, length, codec, key[2])
    _weight_scale_cache.pop(scale_key, None)
    return weight, groups


def cached_gemm(q, scales, packed, grid, groups, ws, wg, full_k, start, codec):
    """Lazy cache with packed fallback for callers retaining original weights."""
    m, k = q.shape
    if m <= 64 and packed.shape[0] <= 512:
        return None
    result = _cache_weight(packed, grid, groups, ws, wg, full_k, start, k, codec)
    if result is None:
        return None
    return int8_gemm(q, scales, result[1][2], groups)


def int8_gemm(q, scales, weight, groups, *, relu2=False, quant_out=False):
    """GS128 GEMM using only immutable INT8 weights and FP32 group scales."""
    m, k = q.shape
    n = weight.shape[0]
    if weight.dtype != torch.int8 or weight.shape[1] != k:
        raise ValueError("Expected INT8 [N,K] weights matching the activations")
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
    result = gemm_a8w8_blockscale(
        q, weight, scales, groups[0], dtype=torch.float32, config=config
    )
    if not relu2 and not quant_out:
        return result
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_packed_g128 import (
        _split_reduce,
    )

    output = torch.empty(
        (m, n), device=q.device, dtype=torch.int8 if quant_out else torch.float32
    )
    output_scales = torch.empty(
        (m, triton.cdiv(n, 128)), device=q.device, dtype=torch.float32
    )
    _split_reduce[(m, triton.cdiv(n, 128))](
        result, output, output_scales, m, n, 1, relu2, quant_out
    )
    return (output, output_scales) if quant_out else output
