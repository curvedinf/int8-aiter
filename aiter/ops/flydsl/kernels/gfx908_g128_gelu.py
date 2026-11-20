# SPDX-License-Identifier: Apache-2.0
"""W8A8 vision projection with bias, exact GELU and G128 activation quantization."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gfx908_g128 import _minimum, _wave_product
from .gfx908_g128_swiglu import _quantize_pair


def _project(tile_type, waves, operands, bias, codes, scales, rows, width, groups):
    tid = fx.Int32(fx.gpu.thread_id("x"))
    wave, lane = tid // 64, tid % 64
    base = fx.Int32(fx.gpu.block_id("x")) * 32
    group = fx.Int32(fx.gpu.block_id("y")) * (waves // 2) + wave // 2
    matrix_tile = group * 2 + wave % 2
    row_in_tile = [8 * ((i % 16) // 4) + 4 * (lane // 32) + i % 4 for i in range(32)]
    actual = [_minimum(base + row, rows - 1) for row in row_in_tile]
    geometry = (
        lane,
        matrix_tile,
        _minimum(base + lane % 32, rows - 1),
        actual,
        width,
        groups,
    )
    product = _wave_product(False, operands, geometry, fx.Int32(0), groups)
    tile = fx.SharedAllocator().allocate(tile_type).peek().data
    group_start = (wave // 2) * 32 * 128
    for i in range(32):
        column = (wave % 2) * 64 + lane % 32 + (i // 16) * 32
        value = fx.Float32(
            fx.Float16(product[i] + fx.Float32(bias[group * 128 + column]))
        )
        # The original projection and GELU each produce FP16 before quantization.
        gelu = (value * fx.Float32(0.5)) * (
            fx.Float32(1.0)
            + fx.Float32(fx.math.erf(value * fx.Float32(0.7071067811865476)))
        )
        tile[group_start + row_in_tile[i] * 128 + column] = fx.Float16(gelu)
    fx.gpu.barrier()
    for i in range(16):
        row = (wave % 2) * 16 + i
        offset = group_start + row * 128 + lane
        q0, q1, scale = _quantize_pair(
            fx.Float32(tile[offset]), fx.Float32(tile[offset + 64]), lane
        )
        valid = scf.IfOp((base + row < rows).ir_value())
        with ir.InsertionPoint(valid.then_block):
            target = (base + row) * width + group * 128 + lane
            codes[target] = fx.Int8(q0)
            codes[target + 64] = fx.Int8(q1)
            first_lane = scf.IfOp((lane == 0).ir_value())
            with ir.InsertionPoint(first_lane.then_block):
                scales[(base + row) * (width // 128) + group] = scale
                scf.YieldOp([])
            scf.YieldOp([])


@lru_cache(maxsize=2)
def _kernel(waves):
    threads = waves * 64

    @fx.struct
    class Tile:
        data: fx.Array[fx.Float16, waves * 2048, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gelu(
        x: fx.Pointer,
        w: fx.Pointer,
        xs: fx.Pointer,
        ws: fx.Pointer,
        b: fx.Pointer,
        y: fx.Pointer,
        ys: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        groups: fx.Int32,
    ):
        _project(Tile, waves, (x, w, xs, ws, x, ws), b, y, ys, m, n, groups)

    @flyc.jit
    def launch(
        x: fx.Pointer,
        w: fx.Pointer,
        xs: fx.Pointer,
        ws: fx.Pointer,
        b: fx.Pointer,
        y: fx.Pointer,
        ys: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        groups: fx.Int32,
        stream: fx.Stream,
    ):
        gelu(x, w, xs, ws, b, y, ys, m, n, groups).launch(
            grid=((m + 31) // 32, n // (waves * 64), 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch


def _outputs(x, weight):
    m, n = x.shape[0], weight.shape[0]
    return x.new_empty((m, n)), x.new_empty((m, n // 128), dtype=torch.float32)


@torch.library.custom_op(
    "aiter::gfx908_g128_gelu", mutates_args=(), device_types="cuda"
)
def gfx908_g128_gelu(
    x: torch.Tensor,
    scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor,
    waves: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    n = weight.shape[0]
    if waves not in (2, 4) or n % (waves * 64) or k % 128 or bias.shape != (n,):
        raise ValueError(
            "Expected complete padded GELU output groups and input G128 weights"
        )
    result = _outputs(x, weight)
    if m:
        types = (
            fx.Int32,
            fx.Int32,
            fx.Float32,
            fx.Float16,
            fx.Float16,
            fx.Int8,
            fx.Float32,
        )
        tensors = x, weight, scales, weight_scales, bias, *result
        args = [
            flyc.from_c_void_p(t, v.data_ptr(), assumed_align=16)
            for t, v in zip(types, tensors)
        ]
        _run_compiled(
            _kernel(waves),
            *args,
            m,
            n,
            k // 128,
            fx.Stream(torch.cuda.current_stream(x.device)),
        )
    return result


@gfx908_g128_gelu.register_fake
def _fake_gelu(x, scales, weight, weight_scales, bias, waves):
    return _outputs(x, weight)
