# SPDX-License-Identifier: Apache-2.0
"""Four-head Engram RMS normalization and projection for small MI100 batches."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl, scf
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gfx908_g128 import _minimum
from .gfx908_mla_values import _wave_read

_vector = fx.Vector.from_elements


def pack_projection(weight: torch.Tensor) -> torch.Tensor:
    """Each wave reads four consecutive K elements for one of 16 output channels."""
    channels, features = weight.shape
    if features != 384 or channels % 128 or weight.dtype != torch.float16:
        raise ValueError("Expected FP16 projection with K=384 and N divisible by 128")
    return (
        weight.view(channels // 16, 16, 4, 6, 4, 4)
        .permute(0, 2, 3, 4, 1, 5)
        .contiguous()
    )


def _projection(
    rows,
    hashes,
    gamma,
    weight,
    bias,
    output,
    count,
    channels,
    row_stride,
    hash_stride,
    tiles,
    half_input,
):
    lane = fx.Int32(fx.gpu.thread_id("x"))
    first_row = fx.Int32(fx.gpu.block_id("x")) * 16
    first_tile = fx.Int32(fx.gpu.block_id("y")) * tiles
    row = _minimum(first_row + lane % 16, count - 1)
    zero = fx.Float32(0.0)
    row_type = T.f16 if half_input else T.f32
    alignment = 8 if half_input else 16

    def load(head, step):
        feature = head * 96 + step * 16 + (lane // 16) * 4
        data = fx.Vector(
            llvm.load(
                T.vec(4, row_type),
                (rows + row * row_stride + feature).llvm_ptr,
                alignment=alignment,
            )
        )
        scale = fx.Vector(
            llvm.load(T.vec(4, T.f16), (gamma + feature).llvm_ptr, alignment=8)
        )
        return _vector([fx.Float32(data[j]) for j in range(4)]), scale

    def squares(total, values):
        for j in range(4):
            total = total + values[j] * values[j]
        return total

    def normalize(total):
        for distance in (16, 32):
            total = total + _wave_read(total, lane ^ distance)
        return fx.Float32(
            fx.math.rsqrt(total / fx.Float32(96.0) + fx.Float32(2.0**-23))
        )

    data, scales = [], []
    total = zero
    for step in range(6):
        values, scale = load(0, step)
        data.append(values)
        scales.append(scale)
        total = squares(total, values)
    products = [_vector([zero] * 4) for _ in range(tiles)]
    for head in range(4):
        inv = normalize(total)
        valid = hashes[row * hash_stride + head] >= 0
        next_data, next_scales = [], []
        total = zero
        for step in range(6):
            normalized = _vector(
                [
                    valid.select(
                        fx.Float16(data[step][j] * inv * fx.Float32(scales[step][j])),
                        fx.Float16(0.0),
                    )
                    for j in range(4)
                ]
            )
            for tile in range(tiles):
                index = ((((first_tile + tile) * 4 + head) * 6 + step) * 64 + lane) * 4
                coefficients = fx.Vector(
                    llvm.load(T.vec(4, T.f16), (weight + index).llvm_ptr, alignment=8)
                )
                products[tile] = fx.Vector(
                    rocdl.mfma_f32_16x16x16f16_(
                        T.vec(4, T.f32),
                        normalized,
                        coefficients,
                        products[tile],
                        0,
                        0,
                        ir.Attribute.parse("#rocdl<mfma_perm_b none>"),
                    )
                )
            # This input and its RMS reduction are independent of the current MFMA.
            if head < 3:
                values, scale = load(head + 1, step)
                next_data.append(values)
                next_scales.append(scale)
                total = squares(total, values)
        data, scales = next_data, next_scales
    for tile, product in enumerate(products):
        channel = (first_tile + tile) * 16 + lane % 16
        offset = fx.Float32(bias[channel])
        for j in range(4):
            r = first_row + (lane // 16) * 4 + j
            valid_row = scf.IfOp((r < count).ir_value())
            with ir.InsertionPoint(valid_row.then_block):
                output[r * channels + channel] = fx.Float16(product[j] + offset)
                scf.YieldOp([])


@lru_cache(maxsize=12)
def _build(tiles, half_input, hash_dtype):
    # Hash width is part of the pointer ABI of the compiled launcher.
    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(
        x: fx.Pointer,
        ids: fx.Pointer,
        gamma: fx.Pointer,
        w: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        row_stride: fx.Int32,
        hash_stride: fx.Int32,
    ):
        _projection(
            x, ids, gamma, w, b, out, m, n, row_stride, hash_stride, tiles, half_input
        )

    @flyc.jit
    def launch(
        x: fx.Pointer,
        ids: fx.Pointer,
        gamma: fx.Pointer,
        w: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        row_stride: fx.Int32,
        hash_stride: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(x, ids, gamma, w, b, out, m, n, row_stride, hash_stride).launch(
            grid=((m + 15) // 16, n // (16 * tiles), 1), block=(64, 1, 1), stream=stream
        )

    return launch


@torch.library.custom_op(
    "aiter::gfx908_engram_projection", mutates_args=(), device_types="cuda"
)
def gfx908_engram_projection(
    rows: torch.Tensor,
    hashes: torch.Tensor,
    gamma: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    count = rows.shape[0]
    channels = bias.numel()
    if (
        rows.shape != (count, 4, 96)
        or rows.stride()[1:] != (96, 1)
        or rows.stride(0) % 4
        or rows.dtype not in (torch.float16, torch.float32)
        or hashes.shape != (count, 4)
        or hashes.stride(1) != 1
        or hashes.dtype not in (torch.int32, torch.int64)
        or gamma.shape != (4, 96)
        or weight.shape != (channels // 16, 4, 6, 4, 16, 4)
        or channels % 128
        or bias.shape != (channels,)
        or any(
            t.dtype != torch.float16 or not t.is_contiguous()
            for t in (gamma, weight, bias)
        )
    ):
        raise ValueError("Invalid four-head gfx908 Engram projection inputs")
    output = rows.new_empty((count, channels), dtype=torch.float16)
    if count:
        tiles = 1 if count <= 16 else (4 if count <= 64 else 8)
        half = rows.dtype == torch.float16
        types = (
            fx.Float16 if half else fx.Float32,
            fx.Int32 if hashes.dtype == torch.int32 else fx.Int64,
            fx.Float16,
            fx.Float16,
            fx.Float16,
            fx.Float16,
        )
        pointers = [
            flyc.from_c_void_p(dtype, t.data_ptr())
            for dtype, t in zip(types, (rows, hashes, gamma, weight, bias, output))
        ]
        _run_compiled(
            _build(tiles, half, hashes.dtype),
            *pointers,
            count,
            channels,
            rows.stride(0),
            hashes.stride(0),
            fx.Stream(torch.cuda.current_stream(rows.device)),
        )
    return output


@gfx908_engram_projection.register_fake
def _fake(rows, hashes, gamma, weight, bias):
    return rows.new_empty((rows.shape[0], bias.shape[0]), dtype=torch.float16)
