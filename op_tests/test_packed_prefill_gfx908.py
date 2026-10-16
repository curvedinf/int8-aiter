# SPDX-License-Identifier: Apache-2.0
"""Large routed batches against the independent packed Triton path."""

import gc

import pytest
import torch

from aiter.ops.triton.gemm.basic import gemm_packed_g128 as ops
from aiter.ops.triton.gemm.basic import packed_expert_cache as caches
from op_tests.test_packed_expert_cache_gfx908 import make_plan, make_weights

pytestmark = pytest.mark.skipif(
    not torch.version.hip
    or not torch.cuda.is_available()
    or "gfx908" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="gfx908 GPU required",
)


@pytest.fixture(autouse=True)
def clean_scales(monkeypatch):
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_GIB", "0.5")
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_SLOTS", "114")
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_REFRESH", "128")
    yield
    caches._CACHE.clear()
    caches._BYTES.clear()
    caches._LAYERS.clear()
    ops._weight_scale_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize(
    "m,intermediate,start,experts",
    [
        (128, 384, 0, 240),
        (512, 576, 64, 304),
        (257, 576, 128, 496),
        (129, 576, 192, 240),
        (4096, 192, 0, 240),
    ],
)
def test_routed_prefill_matches_packed(
    m, intermediate, start, experts, resident, monkeypatch
):
    up, down, grid = make_weights(intermediate, start, experts)
    torch.manual_seed(27)
    q, scales = ops.quantize(
        torch.randn((m, 1024), device="cuda", dtype=torch.bfloat16)
    )
    ids = torch.randint(experts, (m, 9), device="cuda", dtype=torch.int32)
    plan = make_plan(ids, experts)
    if resident:
        cache = caches.get_cache(up, down, grid, intermediate, start)
        assert cache is not None
        cache.step(ids[:8])
        assert caches.find_projection(up, 0, 1024, down=False) is cache.up
        assert caches.find_projection(down, start, intermediate, down=True) is cache.down
        assert caches.find_projection(down, start + 4, intermediate, down=True) is None
        hits = cache.mapping[ids.long()] >= 0
        assert hits.any() and (~hits).any()
    calls = []
    original = ops.gemm_packed_native.packed_experts

    def record_native(*args, **kwargs):
        calls.append(args[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(ops.gemm_packed_native, "packed_experts", record_native)
    # Without a routing plan the wrapper uses the independent Triton kernel.
    expected, expected_scales = ops.packed_experts(q, scales, up, grid, ids)
    actual, actual_scales = ops.packed_experts(q, scales, up, grid, ids, plan=plan)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)
    expected_down = ops.packed_experts(
        expected, expected_scales, down, grid, ids, start=start, down=True
    )
    actual_down = ops.packed_experts(
        actual, actual_scales, down, grid, ids, start=start, down=True, plan=plan
    )
    torch.testing.assert_close(actual_down, expected_down, rtol=0, atol=0)
    if resident:
        assert calls == [cache.up, cache.down]
    else:
        assert not calls
