# SPDX-License-Identifier: Apache-2.0
"""Static dense ownership, packed arithmetic parity and graph replay."""

import gc
import math
import weakref

import pytest
import torch

from aiter.ops.triton.gemm.basic import gemm_packed_g128 as ops
from aiter.ops.triton.gemm.basic import packed_dense_cache as caches

pytestmark = pytest.mark.skipif(
    not torch.version.hip
    or not torch.cuda.is_available()
    or "gfx908" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="gfx908 GPU required",
)


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    monkeypatch.setenv("VLLM_BOLT_DENSE_CACHE_GIB", "0.25")
    yield
    caches._CACHE.clear()
    caches._BYTES.clear()
    caches._RELEASED_BYTES.clear()
    ops._weight_scale_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


def make_weights(codec, n, length, start=0):
    torch.manual_seed(19)
    kwargs = {"codec": codec, "start": start}
    if codec == "nvfp4":
        k = math.ceil((start + length) / 16) * 16
        packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
        ws = (torch.rand((n, k // 16), device="cuda") + 0.5).to(torch.float8_e4m3fn)
        wg = torch.full((n,), 0.125, device="cuda", dtype=torch.float32)
        kwargs.update(weight_scales=ws, global_scale=wg)
    else:
        block, size = (256, 66) if codec == "iq2_xxs" else (32, 34)
        packed = torch.randint(
            0,
            256,
            (n, math.ceil((start + length) / block), size),
            device="cuda",
            dtype=torch.uint8,
        )
        packed[..., 0] = 0
        packed[..., 1] = 0x30
    return packed, ops.make_grid("cuda"), kwargs


def prepare(packed, grid, length, kwargs):
    return caches.prepare_dense_weight(
        packed,
        grid,
        length,
        codec=kwargs["codec"],
        start=kwargs["start"],
        ws=kwargs.get("weight_scales"),
        wg=kwargs.get("global_scale"),
    )


@pytest.mark.parametrize("codec", ["nvfp4", "iq2_xxs", "q8_0"])
@pytest.mark.parametrize("m", [8, 64, 256])
def test_static_dense_matches_existing_cache(codec, m):
    length = 576 if codec == "iq2_xxs" else 1024
    start = 64 if codec == "iq2_xxs" else 0
    packed, grid, kwargs = make_weights(codec, 768, length, start)
    q, scales = ops.quantize(
        torch.randn((m, length), device="cuda", dtype=torch.bfloat16)
    )
    reference = ops.packed_gemm(q, scales, packed, grid, **kwargs)
    weight, groups = prepare(packed, grid, length, kwargs)
    actual = caches.int8_gemm(q, scales, weight, groups)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("m", [8, 56, 64, 256])
def test_small_lilicorr_nvfp4_projection_matches_packed(m):
    # This projection formerly bypassed the expanded cache at small M.
    packed, grid, kwargs = make_weights("nvfp4", 512, 1024)
    q, scales = ops.quantize(
        torch.randn((m, 1024), device="cuda", dtype=torch.bfloat16)
    )
    reference = ops.packed_gemm(q, scales, packed, grid, **kwargs)
    weight, groups = prepare(packed, grid, 1024, kwargs)
    actual = caches.int8_gemm(q, scales, weight, groups)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("codec", ["nvfp4", "iq2_xxs", "q8_0"])
def test_fused_relu2_and_quant_output_match(codec):
    packed, grid, kwargs = make_weights(codec, 768, 1024)
    q, scales = ops.quantize(
        torch.randn((8, 1024), device="cuda", dtype=torch.bfloat16)
    )
    expected, expected_scales = ops.packed_gemm(
        q, scales, packed, grid, relu2=True, quant_out=True, **kwargs
    )
    weight, groups = prepare(packed, grid, 1024, kwargs)
    actual, actual_scales = caches.int8_gemm(
        q, scales, weight, groups, relu2=True, quant_out=True
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_scales, expected_scales, atol=0, rtol=0)


def test_cache_does_not_retain_original_tensors_and_replays_graph():
    packed, grid, kwargs = make_weights("nvfp4", 768, 1024)
    refs = [
        weakref.ref(t)
        for t in (packed, grid, kwargs["weight_scales"], kwargs["global_scale"])
    ]
    weight, groups = prepare(packed, grid, 1024, kwargs)
    assert not ops._weight_scale_cache
    assert all(entry[0] is None for entry in caches._CACHE.values())
    del packed, grid, kwargs
    gc.collect()
    assert all(ref() is None for ref in refs)
    q, scales = ops.quantize(
        torch.randn((8, 1024), device="cuda", dtype=torch.bfloat16)
    )
    caches.int8_gemm(q, scales, weight, groups)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = caches.int8_gemm(q, scales, weight, groups)
    q.zero_()
    graph.replay()
    assert torch.count_nonzero(output).item() == 0
    assert sum(caches._BYTES.values()) == weight.numel()


def test_insufficient_budget_keeps_packed_fallback(monkeypatch):
    packed, grid, kwargs = make_weights("iq2_xxs", 768, 1024)
    monkeypatch.setenv("VLLM_BOLT_DENSE_CACHE_GIB", "0")
    assert prepare(packed, grid, 1024, kwargs) is None
    assert not caches._CACHE
    q, scales = ops.quantize(
        torch.randn((8, 1024), device="cuda", dtype=torch.bfloat16)
    )
    assert torch.isfinite(ops.packed_gemm(q, scales, packed, grid, **kwargs)).all()
