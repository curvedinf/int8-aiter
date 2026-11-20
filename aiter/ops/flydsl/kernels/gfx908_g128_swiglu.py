# SPDX-License-Identifier: Apache-2.0
"""Fuse gate/up G128 GEMM, FP16 SwiGLU and row-group INT8 quantization on CDNA1."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, scf
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gfx908_g128 import _minimum, _wave_product


def _shuffle_xor(value, lane, distance):
    integer = isinstance(value, fx.Int32)
    word = value if integer else value.bitcast(fx.Int32)
    other = fx.Int32(
        fx.rocdl.ds_bpermute(T.i32, ((lane ^ distance) * 4).ir_value(), word.ir_value())
    )
    return other if integer else other.bitcast(fx.Float32)


def _maximum(a, b):
    return fx.Float32(arith.maximumf(a.ir_value(), b.ir_value()))


def _quantize_pair(low, high, lane):
    # Each wave holds one complete G128 row as two values per lane.
    peak = _maximum(fx.Float32(fx.math.absf(low)), fx.Float32(fx.math.absf(high)))
    for distance in (1, 2, 4, 8, 16, 32):
        peak = _maximum(peak, _shuffle_xor(peak, lane, distance))
    scale = _maximum(peak / fx.Float32(127.0), fx.Float32(1e-30))
    q0 = fx.Int32(fx.Float32(fx.math.roundeven(low / scale)))
    q1 = fx.Int32(fx.Float32(fx.math.roundeven(high / scale)))
    return q0, q1, scale


def _quantize_rows(shared, wave, lane, base, group, codes, scales, sums, m, width):
    for local in range(8):
        row = wave * 8 + local
        values = []
        for half in (0, 64):
            column = lane + half
            pos = row * 128 + column
            gate = fx.Float32(shared[pos])
            up = fx.Float32(shared[32 * 128 + pos])
            activated = fx.Float16(
                gate / (fx.Float32(1.0) + fx.Float32(fx.math.exp(-gate)))
            )
            values.append(fx.Float32(fx.Float16(fx.Float32(activated) * up)))
        q0, q1, scale = _quantize_pair(*values, lane)
        total = q0 + q1
        for step in (1, 2, 4, 8, 16, 32):
            total = total + _shuffle_xor(total, lane, step)
        valid = scf.IfOp((base + row < m).ir_value())
        with ir.InsertionPoint(valid.then_block):
            dest = (base + row) * width + group * 128 + lane
            codes[dest] = fx.Int8(q0)
            codes[dest + 64] = fx.Int8(q1)
            first_lane = scf.IfOp((lane == 0).ir_value())
            with ir.InsertionPoint(first_lane.then_block):
                at = (base + row) * (width // 128) + group
                scales[at] = scale
                sums[at] = total
                scf.YieldOp([])
            scf.YieldOp([])


def _store_and_quantize(
    Tile, wave, lane, base, group, local, product, y, ys, ysum, m, n
):
    shared = fx.SharedAllocator().allocate(Tile).peek().data
    for i in range(32):
        column = (wave % 2) * 64 + lane % 32 + (i // 16) * 32
        shared[((wave // 2) * 32 + local[i]) * 128 + column] = fx.Float16(product[i])
    fx.gpu.barrier()
    _quantize_rows(shared, wave, lane, base, group, y, ys, ysum, m, n // 2)


@cache
def _kernel(w4):
    @fx.struct
    class Tile:
        # Four waves write 32 x 128 gate and up values before reducing each row.
        data: fx.Array[fx.Float16, 8192, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def gate_up(
        x: fx.Pointer,
        weight: fx.Pointer,
        xs: fx.Pointer,
        ws: fx.Pointer,
        sums: fx.Pointer,
        zeros: fx.Pointer,
        y: fx.Pointer,
        ys: fx.Pointer,
        ysum: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        groups: fx.Int32,
    ):
        tid = fx.Int32(fx.gpu.thread_id("x"))
        wave, lane = tid // 64, tid % 64
        base = fx.Int32(fx.gpu.block_id("x")) * 32
        group = fx.Int32(fx.gpu.block_id("y"))
        tile = group * 2 + wave % 2 + (wave // 2) * (n // 128)
        local = [8 * ((i % 16) // 4) + 4 * (lane // 32) + i % 4 for i in range(32)]
        rows = [_minimum(base + row, m - 1) for row in local]
        geometry = lane, tile, _minimum(base + lane % 32, m - 1), rows, n, groups
        product = _wave_product(
            w4, (x, weight, xs, ws, sums, zeros), geometry, fx.Int32(0), groups
        )
        _store_and_quantize(
            Tile, wave, lane, base, group, local, product, y, ys, ysum, m, n
        )

    @flyc.jit
    def launch(
        x: fx.Pointer,
        w: fx.Pointer,
        xs: fx.Pointer,
        ws: fx.Pointer,
        sums: fx.Pointer,
        zw: fx.Pointer,
        y: fx.Pointer,
        ys: fx.Pointer,
        ysum: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        groups: fx.Int32,
        stream: fx.Stream,
    ):
        gate_up(x, w, xs, ws, sums, zw, y, ys, ysum, m, n, groups).launch(
            grid=((m + 31) // 32, n // 256, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def _outputs(x, weight):
    rows, width = x.shape[0], weight.shape[0] // 2
    codes = x.new_empty((rows, width))
    scales = x.new_empty((rows, width // 128), dtype=torch.float32)
    sums = x.new_empty((rows, width // 128), dtype=torch.int32)
    return codes, scales, sums


@torch.library.custom_op(
    "aiter::gfx908_g128_swiglu", mutates_args=(), device_types="cuda"
)
def gfx908_g128_swiglu(
    x: torch.Tensor,
    scales: torch.Tensor,
    sums: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    zeros: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = x.shape
    n = weight.shape[0]
    if n % 256 or k % 128 or (zeros is not None and sums is None):
        raise ValueError("Expected paired complete G128 gate and up projections")
    result = _outputs(x, weight)
    if m:
        tensors = (
            x,
            weight,
            scales,
            weight_scales,
            sums if sums is not None else x,
            zeros if zeros is not None else weight_scales,
            *result,
        )
        types = (
            fx.Int32,
            fx.Int32,
            fx.Float32,
            fx.Float16,
            fx.Int32,
            fx.Float16,
            fx.Int8,
            fx.Float32,
            fx.Int32,
        )
        ptrs = [
            flyc.from_c_void_p(ty, t.data_ptr(), assumed_align=16)
            for ty, t in zip(types, tensors)
        ]
        _run_compiled(
            _kernel(zeros is not None),
            *ptrs,
            m,
            n,
            k // 128,
            fx.Stream(torch.cuda.current_stream(x.device)),
        )
    return result


@gfx908_g128_swiglu.register_fake
def _fake_gate(x, scales, sums, weight, weight_scales, zeros):
    return _outputs(x, weight)
