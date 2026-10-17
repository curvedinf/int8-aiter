# SPDX-License-Identifier: Apache-2.0
"""Larger route tiles preserve the independent packed expert result."""

import pytest
import torch

from aiter.ops.triton.gemm.basic import gemm_packed_g128 as ops
from op_tests.test_packed_expert_cache_gfx908 import (
    clean_cache,  # noqa: F401
    make_plan,
    make_weights,
    pytestmark,  # noqa: F401
)


@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize(
    "m,width,start", [(1, 384, 128), (65, 512, 0), (129, 576, 192), (257, 640, 64)]
)
def test_larger_route_tiles_match_unsorted_oracle(block_size, m, width, start):
    up, down, grid = make_weights(width, start, experts=8)
    # Vary signs and exponents of finite multipliers; the byte decoder is shared
    # but each oracle route uses a separate original 16-row matrix program.
    for weight in (up, down):
        multipliers = (
            torch.linspace(-0.25, 0.375, weight[..., 0].numel(), device="cuda")
            .half()
            .reshape(weight.shape[:-1])
            .view(torch.uint8)
            .reshape(*weight.shape[:-1], 2)
        )
        weight[..., :2] = multipliers
    torch.manual_seed(101 + m)
    ids = torch.randint(7, (m, 9), device="cuda", dtype=torch.int32)
    ids[:, :3] = 0  # Skew plus an entirely empty final expert.
    plan = make_plan(ids, 8, block_size)
    q, scales = ops.quantize(torch.randn((m, 1024), device="cuda"))
    expected, expected_scales = ops.packed_experts(q, scales, up, grid, ids)
    actual, actual_scales = ops.packed_experts(
        q, scales, up, grid, ids, plan=plan, block_size=block_size
    )
    assert torch.equal(expected, actual)
    assert torch.equal(expected_scales, actual_scales)
    expected_down = ops.packed_experts(
        expected, expected_scales, down, grid, ids, start=start, down=True
    )
    actual_down = ops.packed_experts(
        actual,
        actual_scales,
        down,
        grid,
        ids,
        start=start,
        down=True,
        plan=plan,
        block_size=block_size,
    )
    assert torch.equal(expected_down, actual_down)


def test_larger_route_tile_graph_replay():
    up, down, grid = make_weights(576, 192, experts=8)
    ids = torch.randint(8, (65, 9), device="cuda", dtype=torch.int32)
    plan = make_plan(ids, 8, 32)
    q, scales = ops.quantize(torch.randn((65, 1024), device="cuda"))

    def apply():
        mid, ms = ops.packed_experts(q, scales, up, grid, ids, plan=plan, block_size=32)
        return ops.packed_experts(
            mid, ms, down, grid, ids, start=192, down=True, plan=plan, block_size=32
        )

    apply()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = apply()
    for _ in range(2):
        q.copy_(torch.randint(-127, 128, q.shape, device="cuda", dtype=torch.int8))
        graph.replay()
        assert torch.equal(output, apply())


@pytest.mark.parametrize("block_size", [0, 8, 128, 32, 64])
def test_route_tile_requires_supported_size_and_plan(block_size):
    with pytest.raises(ValueError, match="matching"):
        ops.packed_experts(None, None, None, None, None, block_size=block_size)
