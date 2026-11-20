# SPDX-License-Identifier: Apache-2.0
"""Normalize two strided DFly2 inputs directly into the following W8A8 projection."""

from functools import lru_cache, cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf

from .gfx908_g128_swiglu import _quantize_pair, _shuffle_xor
from .tensor_shim import _run_compiled


def _normalize_half(shared, x, weight, codes, scales, stride, offset, epsilon):
    tid = fx.Int32(fx.gpu.thread_id("x"))
    row = fx.Int32(fx.gpu.block_id("x"))
    stripe = fx.Int32(fx.gpu.block_id("z"))
    wave, lane = tid // 64, tid % 64
    total = fx.Float32(0)
    values = []
    for tile in range(3):
        index = (tile * 8 + wave) * 128 + lane
        low = fx.Float32(x[row * stride + index])
        high = fx.Float32(x[row * stride + index + 64])
        total = total + low * low
        total = total + high * high
        values.append((low, high))
    for distance in (1, 2, 4, 8, 16, 32):
        total = total + _shuffle_xor(total, lane, distance)
    first_lane = scf.IfOp((lane == 0).ir_value())
    with ir.InsertionPoint(first_lane.then_block):
        shared[wave] = total
        scf.YieldOp([])
    fx.gpu.barrier()
    variance_sum = fx.Float32(0)
    for i in range(8):
        variance_sum = variance_sum + fx.Float32(shared[i])
    inverse = fx.Float32(
        fx.math.rsqrt(variance_sum / fx.Float32(3072.0) + fx.Float32(epsilon))
    )
    # At draft batch sizes there are too few rows to fill the card. Launch
    # three independent CTAs per input row. Each writes eight distinct G128
    # groups after reducing the complete row, with no intermediate FP16 tensor.
    for tile, (low, high) in enumerate(values):
        selected = scf.IfOp((stripe == tile).ir_value())
        with ir.InsertionPoint(selected.then_block):
            group = tile * 8 + wave
            index = group * 128 + lane
            a = fx.Float32(fx.Float16(low * inverse * fx.Float32(weight[index])))
            b = fx.Float32(fx.Float16(high * inverse * fx.Float32(weight[index + 64])))
            q0, q1, scale = _quantize_pair(a, b, lane)
            dest = row * 6144 + offset + index
            codes[dest] = fx.Int8(q0)
            codes[dest + 64] = fx.Int8(q1)
            writer = scf.IfOp((lane == 0).ir_value())
            with ir.InsertionPoint(writer.then_block):
                scales[row * 48 + offset // 128 + group] = scale
                scf.YieldOp([])
            scf.YieldOp([])


@cache
def _kernel(epsilon):
    @fx.struct
    class Reduction:
        sums: fx.Array[fx.Float32, 8, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def dual_norm(
        x: fx.Pointer,
        y: fx.Pointer,
        gamma: fx.Pointer,
        beta: fx.Pointer,
        q: fx.Pointer,
        s: fx.Pointer,
        sx: fx.Int32,
        sy: fx.Int32,
    ):
        shared = fx.SharedAllocator().allocate(Reduction).peek().sums
        left = scf.IfOp((fx.Int32(fx.gpu.block_id("y")) == 0).ir_value(), has_else=True)
        with ir.InsertionPoint(left.then_block):
            _normalize_half(shared, x, gamma, q, s, sx, 0, epsilon)
            scf.YieldOp([])
        with ir.InsertionPoint(left.else_block):
            _normalize_half(shared, y, beta, q, s, sy, 3072, epsilon)
            scf.YieldOp([])

    @flyc.jit
    def launch(
        x: fx.Pointer,
        y: fx.Pointer,
        gamma: fx.Pointer,
        beta: fx.Pointer,
        q: fx.Pointer,
        s: fx.Pointer,
        sx: fx.Int32,
        sy: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        dual_norm(x, y, gamma, beta, q, s, sx, sy).launch(
            grid=(rows, 2, 3), block=(512, 1, 1), stream=stream
        )

    return launch


def _outputs(x):
    return (
        x.new_empty((x.shape[0], 6144), dtype=torch.int8),
        x.new_empty((x.shape[0], 48), dtype=torch.float32),
    )


@torch.library.custom_op(
    "aiter::gfx908_g128_dual_rms", mutates_args=(), device_types="cuda"
)
def gfx908_g128_dual_rms(
    x: torch.Tensor,
    y: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        x.ndim != 2
        or x.shape[1] != 3072
        or y.shape != x.shape
        or left_weight.shape != (3072,)
        or right_weight.shape != (3072,)
    ):
        raise ValueError(
            "Expected two [rows, 3072] inputs and two 3072-channel RMS weights"
        )
    tensors = (x, y, left_weight, right_weight)
    if any(
        t.dtype != torch.float16 or t.stride(-1) != 1 or t.device != x.device
        for t in tensors
    ):
        raise ValueError("Expected FP16 tensors with contiguous channels on one device")
    result = _outputs(x)
    if x.shape[0]:
        types = (fx.Float16, fx.Float16, fx.Float16, fx.Float16, fx.Int8, fx.Float32)
        pointers = [
            flyc.from_c_void_p(dtype, t.data_ptr(), assumed_align=t.element_size())
            for dtype, t in zip(types, (*tensors, *result))
        ]
        _run_compiled(
            _kernel(epsilon),
            *pointers,
            x.stride(0),
            y.stride(0),
            x.shape[0],
            fx.Stream(torch.cuda.current_stream(x.device)),
        )
    return result


@gfx908_g128_dual_rms.register_fake
def _fake_dual_rms(x, y, left_weight, right_weight, epsilon):
    return _outputs(x)
