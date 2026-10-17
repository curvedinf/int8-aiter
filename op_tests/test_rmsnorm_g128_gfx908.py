# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dual-output norm parity with the original AITER norm and GS128 quantizer."""

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_packed_g128 import quantize
from aiter.ops.triton.normalization.rmsnorm import rms_norm, rmsnorm2d_fwd_with_add
from aiter.ops.triton.normalization.rmsnorm_g128 import rms_norm_g128

pytestmark = pytest.mark.skipif(
    not torch.version.hip
    or not torch.cuda.is_available()
    or "gfx908" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="gfx908 GPU required",
)


def assert_bits(a, b):
    dtype = torch.int32 if a.dtype == torch.float32 else torch.int16
    if a.dtype == torch.int8:
        dtype = torch.int8
    assert torch.equal(a.contiguous().view(dtype), b.contiguous().view(dtype))


def reference(x, weight, epsilon, residual):
    if residual is None:
        y = rms_norm(x.contiguous(), weight.contiguous(), epsilon)
        r = x
    else:
        y, r = torch.empty_like(x.contiguous()), torch.empty_like(residual.contiguous())
        rmsnorm2d_fwd_with_add(
            y, x.contiguous(), residual.contiguous(), r, weight.contiguous(), epsilon
        )
    q, s = quantize(y)
    return y, r, q, s


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rdtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("wdtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m", [256, 257, 4096, 4160])
@torch.inference_mode()
def test_native_norm_residual_quant_scale_bits(dtype, rdtype, wdtype, m):
    torch.manual_seed(381 + m)
    x = torch.randn((m, 4096), device="cuda", dtype=dtype)
    r = torch.randn_like(x, dtype=rdtype) if rdtype is not None else None
    w = torch.randn((4096,), device="cuda", dtype=wdtype) * 0.1 + 1
    saved = [v.clone() if v is not None else None for v in (x, r, w)]
    expected = reference(x, w, 1e-5, r)
    actual = rms_norm_g128(x, w, 1e-5, r)
    for a, b in zip(actual, expected):
        assert_bits(a, b)
    for a, b in zip((x, r, w), saved):
        if a is not None:
            assert_bits(a, b)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rdtype", [None, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_nonfinite_signed_zero_strides_and_graph(dtype, rdtype):
    x = torch.randn((257, 4096), device="cuda", dtype=dtype)
    x[0].zero_()
    x[1].fill_(-0.0)
    x[2].fill_(float("inf"))
    x[3].fill_(float("-inf"))
    x[4].fill_(float("nan"))
    x[5].fill_(torch.finfo(dtype).tiny)
    x[6].fill_(torch.finfo(dtype).max)
    x = x.T.contiguous().T
    r = torch.randn_like(x, dtype=rdtype) if rdtype is not None else None
    w = torch.ones((8192,), device="cuda")[::2]
    expected = reference(x, w, 1e-5, r)
    actual = rms_norm_g128(x, w, 1e-5, r)
    for a, b in zip(actual, expected):
        assert_bits(a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rms_norm_g128(x, w, 1e-5, r)
    x[7:].mul_(0.5)
    graph.replay()
    expected = reference(x, w, 1e-5, r)
    for a, b in zip(captured, expected):
        assert_bits(a, b)


@pytest.mark.parametrize("shape", [(64, 4096), (255, 4096), (256, 2048)])
def test_unsupported_geometry_is_explicit(shape):
    x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(shape[1], device="cuda")
    with pytest.raises(ValueError, match="prefill matrix"):
        rms_norm_g128(x, w, 1e-5)
