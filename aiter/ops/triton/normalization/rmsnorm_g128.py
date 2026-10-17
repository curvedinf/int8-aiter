# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dual-output native RMSNorm for large gfx908 GS128 consumers."""

import torch

from aiter.ops.triton._triton_kernels.normalization.rmsnorm_g128 import (
    _fused_add_rms_norm_g128,
    _rms_norm_g128,
)
from aiter.ops.triton.normalization.rmsnorm import num_programs


def rms_norm_g128(x, weight, epsilon, residual=None):
    """Return normal output, residual, INT8 values and FP32 GS128 scales.

    The native row loop and its default floating-point fusion are preserved.
    Quantization consumes the same output-dtype rounding as the stored norm.
    No-residual calls retain the original input as the residual, as the model
    decoder does. This inference entry does not replace native RMSNorm APIs.
    """
    if x.ndim != 2 or x.shape[0] < 256 or x.shape[1] != 4096:
        raise ValueError("Expected a prefill matrix with at least 256 rows and K4096")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Expected BF16 or FP32 norm input")
    if residual is not None and (
        residual.shape != x.shape
        or residual.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("Expected a matching BF16 or FP32 residual")
    x = x.contiguous()
    weight = weight.contiguous()
    residual = residual.contiguous() if residual is not None else None
    m, k = x.shape
    out = torch.empty_like(x)
    res_out = torch.empty_like(residual) if residual is not None else x
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scales = torch.empty((m, k // 128), device=x.device, dtype=torch.float32)
    rsigma = torch.empty((m,), device=x.device, dtype=torch.float32)
    programs = num_programs(x)
    if residual is None:
        _rms_norm_g128[(programs,)](
            x,
            out,
            weight,
            rsigma,
            q,
            scales,
            k,
            k,
            m,
            k,
            epsilon,
            k,
            programs,
            num_warps=4,
        )
    else:
        _fused_add_rms_norm_g128[(programs,)](
            x,
            out,
            residual,
            res_out,
            weight,
            rsigma,
            q,
            scales,
            k,
            k,
            m,
            k,
            epsilon,
            k,
            programs,
            num_warps=4,
        )
    return out, res_out, q, scales
