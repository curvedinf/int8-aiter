# SPDX-License-Identifier: Apache-2.0
"""Route product/reduction fusion preserves ATen FP32 order and output casts."""

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_packed_g128 import combine_experts
from op_tests.test_packed_expert_cache_gfx908 import pytestmark  # noqa: F401


def original(y, weights, dtype):
    m, topk = weights.shape
    return (y.view(m, topk, -1) * weights[..., None]).sum(1).to(dtype)


def assert_bits(actual, expected):
    integer = torch.int32 if expected.dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(integer), expected.view(integer))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "m,n",
    [(1, 4), (8, 1024), (64, 1024), (65, 1024), (257, 132), (4096, 1024), (4160, 1024)],
)
def test_nine_routes_match_aten_bits(m, n, dtype):
    torch.manual_seed(703 + m + n)
    y = torch.randn((m * 9, n), device="cuda")
    weights = torch.randn((m, 9), device="cuda").softmax(-1)
    assert_bits(combine_experts(y, weights, dtype), original(y, weights, dtype))


def test_four_accumulator_cancellation_and_bf16_boundaries():
    m, n = 8, 1024
    values = torch.tensor(
        [2**24, 1, -(2**24), 3, 2**24, 5, -(2**24), 7, 11],
        device="cuda",
        dtype=torch.float32,
    )
    indices = (
        torch.arange(9, device="cuda")[:, None]
        + torch.arange(n, device="cuda")[None, :]
    ) % 9
    y = values[indices].repeat(m, 1)
    weights = torch.ones((m, 9), device="cuda")
    assert_bits(
        combine_experts(y, weights, torch.float32), original(y, weights, torch.float32)
    )
    tie = torch.tensor(1 + 1 / 256, device="cuda")
    boundary = torch.stack(
        (tie, torch.nextafter(tie, tie - 1), torch.nextafter(tie, tie + 1), -tie)
    )
    y.zero_()
    y.view(m, 9, n)[:, 0] = boundary[torch.arange(n, device="cuda") % 4]
    assert_bits(
        combine_experts(y, weights, torch.bfloat16),
        original(y, weights, torch.bfloat16),
    )
    y.fill_(-0.0)
    assert_bits(
        combine_experts(y, weights, torch.float32), original(y, weights, torch.float32)
    )


@pytest.mark.parametrize(
    "kind", ["topk", "width", "weights_dtype", "weights_stride", "y_stride"]
)
def test_unsupported_geometry_uses_original(kind):
    m, topk, n = 8, 8 if kind == "topk" else 9, 127 if kind == "width" else 128
    y = torch.randn((m * topk, n), device="cuda")
    weights = torch.randn((m, topk), device="cuda")
    if kind == "weights_dtype":
        weights = weights.to(torch.bfloat16)
    if kind == "weights_stride":
        weights = weights.T.contiguous().T
    if kind == "y_stride":
        y = torch.randn((m * topk, 2 * n), device="cuda")[:, ::2]
    assert_bits(
        combine_experts(y, weights, torch.bfloat16),
        original(y, weights, torch.bfloat16),
    )


@pytest.mark.parametrize("m", [8, 65, 4096])
def test_combine_graph_updates(m):
    n = 1024
    y = torch.randn((m * 9, n), device="cuda")
    weights = torch.randn((m, 9), device="cuda").softmax(-1)
    combine_experts(y, weights, torch.bfloat16)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = combine_experts(y, weights, torch.bfloat16)
    for zero in (False, True):
        y.copy_(torch.zeros_like(y) if zero else torch.randn_like(y))
        weights.copy_(torch.randn_like(weights).softmax(-1))
        graph.replay()
        assert_bits(actual, original(y, weights, torch.bfloat16))


@pytest.mark.parametrize("kind", ["rows", "y_rank", "weights_rank", "device"])
def test_bad_geometry_is_rejected(kind):
    y = torch.zeros((72, 128), device="cuda")
    weights = torch.ones((8, 9), device="cuda")
    if kind == "rows":
        y = y[:-1]
    if kind == "y_rank":
        y = y.reshape(8, 9, 128)
    if kind == "weights_rank":
        weights = weights.flatten()
    if kind == "device":
        weights = weights.cpu()
    with pytest.raises(ValueError, match="route|Route"):
        combine_experts(y, weights, torch.bfloat16)
