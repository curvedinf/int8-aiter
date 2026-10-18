# SPDX-License-Identifier: Apache-2.0
"""Single-launch resident expert prefills match independent packed decoding."""

import gc

import pytest
import torch

from aiter.ops.triton.gemm.basic import gemm_packed_g128 as ops
from aiter.ops.triton.gemm.basic import packed_expert_cache as caches
from op_tests.test_packed_expert_cache_gfx908 import (
    make_plan,
    make_weights,
    pytestmark,  # noqa: F401
)


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_GIB", "0.5")
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_SLOTS", "4")
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_REFRESH", "128")
    yield
    caches._CACHE.clear()
    caches._BYTES.clear()
    caches._LAYERS.clear()
    ops._weight_scale_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


def varied_weights(width, start, experts):
    up, down, grid = make_weights(width, start, experts)
    for weight in (up, down):
        scales = (
            torch.linspace(-0.25, 0.375, weight[..., 0].numel(), device="cuda")
            .half()
            .reshape(weight.shape[:-1])
        )
        weight[..., :2] = scales.unsqueeze(-1).view(torch.uint8)
    return up, down, grid


def mlp(q, scales, up, down, grid, ids, start, plan=None, block=16):
    mid, ms = ops.packed_experts(q, scales, up, grid, ids, plan=plan, block_size=block)
    out = ops.packed_experts(
        mid,
        ms,
        down,
        grid,
        ids,
        start=start,
        down=True,
        plan=plan,
        block_size=block,
    )
    return mid, ms, out


@pytest.mark.parametrize("state", ["absent", "cold", "hot"])
@pytest.mark.parametrize(
    "m,width,start,experts,block",
    [
        (65, 384, 128, 8, 16),
        (129, 512, 0, 17, 64),
        (257, 576, 192, 17, 64),
        (4096, 640, 64, 240, 64),
        (96, 384, 0, 496, 16),
    ],
)
def test_prefill_hits_misses_and_no_admission(
    state, m, width, start, experts, block, monkeypatch
):
    up, down, grid = varied_weights(width, start, experts)
    torch.manual_seed(27)
    ids = torch.randint(experts, (m, 9), device="cuda", dtype=torch.int32)
    plan = make_plan(ids, experts, block)
    q, scales = ops.quantize(
        torch.randn((m, 1024), device="cuda", dtype=torch.bfloat16)
    )
    cache = None
    if state != "absent":
        cache = caches.get_cache(up, down, grid, width, start)
        assert cache is not None
        # Poison allocations while every mapping is a miss. A cold slot must
        # never contribute, even though the cache object already exists.
        cache.up.weight.fill_(127)
        cache.down.weight.fill_(-127)
        if state == "hot":
            cache.step(ids[:8])
            hits = cache.mapping[ids.long()] >= 0
            assert hits.any() and (~hits).any()
        else:
            assert (cache.mapping == -1).all()
        assert caches.find_projection(up, 0, 1024, down=False) is cache.up
        assert caches.find_projection(down, start, width, down=True) is cache.down
        assert caches.find_projection(down, start + 4, width, down=True) is None
        before = [
            v.clone()
            for v in (
                cache.counts,
                cache.phase,
                cache.mapping,
                cache.ids,
                cache.changed,
            )
        ]
    allocated = dict(caches._BYTES)
    expected = mlp(q, scales, up, down, grid, ids, start)
    calls = []
    kernel = ops._gemm

    class CountLaunches:
        def __getitem__(self, launch_grid):
            launch = kernel[launch_grid]

            def run(*args, **kwargs):
                calls.append(kwargs["EXPERT_CACHED"])
                return launch(*args, **kwargs)

            return run

    monkeypatch.setattr(ops, "_gemm", CountLaunches())
    actual = mlp(q, scales, up, down, grid, ids, start, plan, block)
    assert calls == [cache is not None, cache is not None]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert caches._BYTES == allocated
    if cache is not None:
        after = (cache.counts, cache.phase, cache.mapping, cache.ids, cache.changed)
        assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_prefill_graph_observes_refreshed_mapping(monkeypatch):
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_REFRESH", "1")
    up, down, grid = varied_weights(576, 192, 17)
    ids = torch.randint(17, (129, 9), device="cuda", dtype=torch.int32)
    decode_ids = torch.zeros((8, 9), device="cuda", dtype=torch.int32)
    plan = make_plan(ids, 17, 64)
    q, scales = ops.quantize(torch.randn((129, 1024), device="cuda"))
    cache = caches.get_cache(up, down, grid, 576, 192)
    assert cache is not None

    def apply():
        # Admission belongs to decode, followed by a read-only prefill.
        cache.step(decode_ids)
        return mlp(q, scales, up, down, grid, ids, 192, plan, 64)

    apply()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = apply()
    for expert in (9, 16, 4):
        cache.counts.zero_()
        decode_ids.fill_(expert)
        q.copy_(torch.randint(-127, 128, q.shape, device="cuda", dtype=torch.int8))
        graph.replay()
        assert cache.mapping[expert] >= 0
        expected = mlp(q, scales, up, down, grid, ids, 192)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
