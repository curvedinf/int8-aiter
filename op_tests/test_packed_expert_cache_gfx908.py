# SPDX-License-Identifier: Apache-2.0
"""Expert admission, partial TP slices and captured replay against packed IQ2."""

import gc
import math

import pytest
import torch
import triton

from aiter.ops.triton.gemm.basic import gemm_packed_g128 as ops
from aiter.ops.triton.gemm.basic import packed_expert_cache as caches

pytestmark = pytest.mark.skipif(
    not torch.version.hip
    or not torch.cuda.is_available()
    or "gfx908" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="gfx908 GPU required",
)


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_GIB", "0.5")
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_SLOTS", "64")
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_REFRESH", "4")
    yield
    caches._CACHE.clear()
    caches._BYTES.clear()
    caches._LAYERS.clear()
    ops._weight_scale_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


def make_weights(intermediate=576, start=0, experts=240, hidden=1024):
    torch.manual_seed(17)
    up = torch.randint(
        0,
        256,
        (experts, intermediate, hidden // 256, 66),
        device="cuda",
        dtype=torch.uint8,
    )
    down = torch.randint(
        0,
        256,
        (experts, hidden, math.ceil((start + intermediate) / 256), 66),
        device="cuda",
        dtype=torch.uint8,
    )
    # Valid finite FP16 block multipliers; remaining bytes exercise all codes
    # and signs, including overlapping IQ2 blocks at partial TP boundaries.
    for w in (up, down):
        w[..., 0] = 0
        w[..., 1] = 0x30
    return up, down, ops.make_grid("cuda")


def make_plan(ids, experts, block_size=16):
    flat = ids.cpu().flatten().tolist()
    routes = len(flat)
    sorted_ids, expert_ids = [], []
    for e in range(experts):
        selected = [i for i, value in enumerate(flat) if value == e]
        selected += [routes] * (-len(selected) % block_size)
        sorted_ids.extend(selected)
        expert_ids.extend([e] * (len(selected) // block_size))
    return (
        torch.tensor(sorted_ids, device="cuda", dtype=torch.int32),
        torch.tensor(expert_ids, device="cuda", dtype=torch.int32),
        torch.tensor([len(sorted_ids)], device="cuda", dtype=torch.int32),
    )


@pytest.mark.parametrize("actual", [240, 304, 496])
def test_rank_real_experts_and_preserve_surviving_slots(actual):
    padded, slots, period = triton.next_power_of_2(actual), 64, 4
    counts = torch.zeros(padded, device="cuda", dtype=torch.int32)
    phase = torch.zeros((), device="cuda", dtype=torch.int32)
    mapping = torch.full((padded,), -1, device="cuda", dtype=torch.int32)
    slot_ids = torch.full((slots,), -1, device="cuda", dtype=torch.int32)
    changed = torch.zeros(slots, device="cuda", dtype=torch.int32)
    reference = torch.zeros(padded, dtype=torch.int64)
    previous = [-1] * slots
    for step in range(13):
        # Zero-ID graph warmup originally exposed padded-expert admission.
        values = [0] * 576 if step == 0 else [128 + i % 64 for i in range(576)]
        ids = torch.tensor(values, device="cuda", dtype=torch.int32)
        reference += torch.bincount(torch.tensor(values), minlength=padded)
        caches._update_routes[(1,)](
            ids,
            counts,
            phase,
            mapping,
            slot_ids,
            changed,
            len(values),
            padded,
            actual,
            slots,
            triton.next_power_of_2(len(values)),
            period,
            num_warps=4,
        )
        if step % period == 0:
            expected = sorted(range(actual), key=lambda e: (-int(reference[e]), e))[
                :slots
            ]
            current = slot_ids.cpu().tolist()
            assert set(current) == set(expected)
            assert len(set(current)) == slots
            for slot, old in enumerate(previous):
                if old in expected:
                    assert current[slot] == old
            assert changed.cpu().tolist() == [
                int(a != b) for a, b in zip(current, previous)
            ]
            previous = current
            reference //= 2
        torch.testing.assert_close(counts.cpu().long(), reference, rtol=0, atol=0)
        lookup = mapping.cpu().tolist()
        assert all(value == -1 for value in lookup[actual:])
        assert all(lookup[e] == slot for slot, e in enumerate(previous))


@pytest.mark.parametrize(
    "intermediate,rank",
    [(384, 1), (512, 0), (576, 0), (576, 1), (576, 2), (576, 3), (640, 1)],
)
@pytest.mark.parametrize("m", [8, 64])
def test_cached_hits_misses_and_replacement_match_packed(intermediate, rank, m):
    start = (rank * intermediate) % 256
    up, down, grid = make_weights(intermediate, start)
    cache = caches.get_cache(up, down, grid, intermediate, start)
    assert cache is not None
    torch.manual_seed(21)
    x, xs = ops.quantize(torch.randn((m, 1024), device="cuda"))
    for step in range(5):
        if step == 0:
            ids = torch.zeros((m, 9), device="cuda", dtype=torch.int32)
        else:
            ids = torch.randint(128, 240, (m, 9), device="cuda", dtype=torch.int32)
        plan = make_plan(ids, 240)
        cache.step(ids)
        original, original_scales = ops.packed_experts(x, xs, up, grid, ids, plan=plan)
        cached, cached_scales = ops.packed_experts(
            x, xs, up, grid, ids, plan=plan, cache=cache.up
        )
        torch.testing.assert_close(cached, original, rtol=0, atol=0)
        torch.testing.assert_close(cached_scales, original_scales, rtol=0, atol=0)
        a = ops.packed_experts(
            original,
            original_scales,
            down,
            grid,
            ids,
            start=start,
            down=True,
            plan=plan,
        )
        b = ops.packed_experts(
            cached,
            cached_scales,
            down,
            grid,
            ids,
            start=start,
            down=True,
            plan=plan,
            cache=cache.down,
        )
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("slots", [64, 114])
def test_cached_graph_replay_refreshes_experts_without_reallocation(slots, monkeypatch):
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_SLOTS", str(slots))
    up, down, grid = make_weights(576, 64)
    cache = caches.get_cache(up, down, grid, 576, 64)
    ids = torch.zeros((8, 9), device="cuda", dtype=torch.int32)
    plan = make_plan(ids, 240)
    x, scales = ops.quantize(torch.randn((8, 1024), device="cuda"))
    cache.step(ids)
    # Warm every captured entrypoint before beginning capture.
    mid, mid_scales = ops.packed_experts(
        x, scales, up, grid, ids, plan=plan, cache=cache.up
    )
    ops.packed_experts(
        mid,
        mid_scales,
        down,
        grid,
        ids,
        start=64,
        down=True,
        plan=plan,
        cache=cache.down,
    )
    torch.cuda.synchronize()
    pointers = (
        cache.up.weight.data_ptr(),
        cache.down.weight.data_ptr(),
        cache.mapping.data_ptr(),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.step(ids)
        mid, mid_scales = ops.packed_experts(
            x, scales, up, grid, ids, plan=plan, cache=cache.up
        )
        output = ops.packed_experts(
            mid,
            mid_scales,
            down,
            grid,
            ids,
            start=64,
            down=True,
            plan=plan,
            cache=cache.down,
        )
    assert int(cache.mapping[239]) == -1
    for _ in range(8):
        ids.fill_(239)
        plan[1].fill_(239)
        graph.replay()
        a, a_scales = ops.packed_experts(x, scales, up, grid, ids, plan=plan)
        reference = ops.packed_experts(
            a, a_scales, down, grid, ids, start=64, down=True, plan=plan
        )
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.cuda.synchronize()
    assert int(cache.mapping[239]) >= 0
    assert all(e < 240 for e in cache.ids.cpu().tolist())
    assert pointers == (
        cache.up.weight.data_ptr(),
        cache.down.weight.data_ptr(),
        cache.mapping.data_ptr(),
    )


def test_budget_and_repeated_lookup(monkeypatch):
    up, down, grid = make_weights(384, 128)
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_GIB", "0.001")
    assert caches.get_cache(up, down, grid, 384, 128) is None
    assert not caches._BYTES
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_GIB", "0.5")
    cache = caches.get_cache(up, down, grid, 384, 128)
    before = dict(caches._BYTES)
    assert caches.get_cache(up, down, grid, 384, 128) is cache
    assert caches._BYTES == before
    assert cache.up.weight.numel() + cache.down.weight.numel() == 2 * 64 * 384 * 1024
    assert sum(before.values()) < 0.5 * 2**30


def test_auto_capacity_covers_all_layers_and_uses_budget(monkeypatch):
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_SLOTS", "auto")
    monkeypatch.setenv("VLLM_BOLT_EXPERT_CACHE_GIB", "0.375")
    layers = [make_weights(i, 0, experts=e) for i, e in [(384, 240), (640, 496)]]
    for (up, down, _), i in zip(layers, [384, 640]):
        caches.register_layer(up, down, i, 0)
    plan = caches.plan_slots(layers[0][0].device, int(0.375 * 2**30))
    entries = [
        caches.get_cache(up, down, grid, i, 0)
        for (up, down, grid), i in zip(layers, [384, 640])
    ]
    assert all(entry is not None and entry.slots > 64 for entry in entries)
    assert len(plan) == len(entries) == 2
    assert sum(caches._BYTES.values()) <= 0.375 * 2**30
    remaining = int(0.375 * 2**30) - sum(caches._BYTES.values())
    assert remaining < 2 * 384 * 1024 + 8
    # A non-power-of-two capacity must rank only real experts on zero-ID warmup.
    for entry in entries:
        entry.step(torch.zeros((8, 9), device="cuda", dtype=torch.int32))
        selected = entry.ids.cpu().tolist()
        assert len(set(selected)) == entry.slots
        assert all(0 <= e < entry.experts for e in selected)
    # A new layer cannot invalidate the addresses/quotas of captured caches.
    up, down, _ = make_weights(512)
    with pytest.raises(RuntimeError, match="before allocating"):
        caches.register_layer(up, down, 512, 0)
