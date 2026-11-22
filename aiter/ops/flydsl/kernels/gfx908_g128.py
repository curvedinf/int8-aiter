# SPDX-License-Identifier: Apache-2.0
"""G128 W4A8 and W8A8 GEMM for a gfx908 wavefront.

MFMA computes 32 rows by 64 columns. While it accumulates one K128 group,
the vector ALU scales the 32 results per lane from the preceding group.
Weights have the 64-channel layout from pack_affine_weights(overlap=True).
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm, rocdl, scf
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


def _minimum(left, right):
    return fx.Int32(arith.minsi(left.ir_value(), right.ir_value()))


def _load_packet(pointer, offset, words):
    return fx.Vector(
        llvm.load(T.vec(words, T.i32), (pointer + offset).llvm_ptr, alignment=words * 4)
    )


def _scaled(
    dot, activation_scale, weight_scale, activation_sum, weight_zero, symmetric=False
):
    if symmetric:
        # The packed, signed nibble was shifted by four before the INT8 MFMA.
        dot = dot >> 4
    elif weight_zero is not None:
        # Each signed nibble was shifted to the high half of an INT8 byte.
        # Recover the original unsigned checkpoint dot before applying the
        # floating zero point, preserving the NeUQI affine operation order.
        unsigned_dot = (dot >> 4) + activation_sum * 8
        dot = fx.Float32(unsigned_dot) - fx.Float32(activation_sum) * weight_zero
    return (fx.Float32(dot) * activation_scale) * weight_scale


def _group(
    w4, operands, geometry, group, previous, output, independent=None, symmetric=False
):
    activation, weight, scales, weight_scales, sums, zeros = operands
    lane, tile, activation_row, rows, channels, groups = geometry
    packed_width = 2 if w4 else 4
    integer = fx.Vector.from_elements([fx.Int32(0)] * 32)
    draining = previous is not None
    if draining:
        prior = group - 1
        # Each half of the accumulator belongs to a different set of columns.
        offsets = [
            prior * channels + tile * 64 + lane % 32 + 32 * half for half in range(2)
        ]
        scale = [fx.Float32(weight_scales[offset]) for offset in offsets]
        zero = (
            [fx.Float32(zeros[offset]) for offset in offsets]
            if w4 and not symmetric
            else [None] * 2
        )
        updated = []
    for packet in range(8):
        a = _load_packet(
            activation, (activation_row * groups + group) * 32 + packet * 4, 4
        )
        base = (((tile * groups + group) * 8 + packet) * 64 + lane) * packed_width
        b = _load_packet(weight, base, packed_width)
        for word in range(4):
            value = b[word // 2] if w4 else b[word]
            if w4:
                if word % 2 == 0:
                    value = value << 4
                value = value & fx.Int32(-252645136)  # 0xf0f0f0f0
            integer = fx.Vector(
                rocdl.mfma_i32_32x32x4i8_(
                    T.vec(32, T.i32),
                    a[word].ir_value(),
                    value.ir_value(),
                    integer,
                    1,
                    0,
                    ir.Attribute.parse("#rocdl<mfma_perm_b none>"),
                )
            )
            if independent is not None:
                independent(packet, word)
            if draining:
                i = packet * 4 + word
                position = rows[i] * groups + prior
                updated.append(
                    output[i]
                    + _scaled(
                        previous[i],
                        scales[position],
                        scale[i // 16],
                        sums[position] if w4 and not symmetric else None,
                        zero[i // 16],
                        symmetric,
                    )
                )
    return integer, fx.Vector.from_elements(updated) if draining else output


def _wave_product(
    w4, operands, geometry, first, end, independent=None, symmetric=False
):
    """Return the 32 scaled output elements per lane, shared by both epilogues."""
    lane, tile, _, rows, n, groups = geometry
    fp = fx.Vector.from_elements([fx.Float32(0.0)] * 32)
    integer, _ = _group(w4, operands, geometry, first, None, fp, independent, symmetric)
    loop = scf.ForOp(
        (first + 1).ir_value(), end.ir_value(), fx.Int32(1).ir_value(), [integer, fp]
    )
    with ir.InsertionPoint(loop.body):
        group = fx.Int32(loop.induction_variable)
        previous, partial = [fx.Vector(v) for v in loop.inner_iter_args]
        current, partial = _group(
            w4, operands, geometry, group, previous, partial, symmetric=symmetric
        )
        scf.YieldOp([current, partial])
    previous, partial = [fx.Vector(v) for v in loop.results]
    _, _, scales, weight_scales, sums, zeros = operands
    result = []
    for i in range(32):
        col = tile * 64 + lane % 32 + (i // 16) * 32
        pos = rows[i] * groups + end - 1
        wpos = (end - 1) * n + col
        last = _scaled(
            previous[i],
            scales[pos],
            fx.Float32(weight_scales[wpos]),
            sums[pos] if w4 and not symmetric else None,
            fx.Float32(zeros[wpos]) if w4 and not symmetric else None,
            symmetric,
        )
        result.append(partial[i] + last)
    return result


def _tile(w4, split, operands, out, m, n, groups, symmetric):
    lane = fx.Int32(fx.gpu.thread_id("x"))
    base_row = fx.Int32(fx.gpu.block_id("x")) * 32
    tile = fx.Int32(fx.gpu.block_id("y"))
    stripe = fx.Int32(fx.gpu.block_id("z"))
    first, end = stripe * (groups // split), (stripe + 1) * (groups // split)
    actual = [
        base_row + 8 * ((i % 16) // 4) + 4 * (lane // 32) + i % 4 for i in range(32)
    ]
    rows = [_minimum(row, m - 1) for row in actual]
    geometry = lane, tile, _minimum(base_row + lane % 32, m - 1), rows, n, groups
    product = _wave_product(w4, operands, geometry, first, end, symmetric=symmetric)
    pitch = ((m + 31) // 32 * 32) * n
    for i in range(32):
        col = tile * 64 + lane % 32 + (i // 16) * 32
        out[stripe * pitch + actual[i] * n + col] = product[i]


@cache
def _make_kernel(w4, split, fp32_weight_scales=False, symmetric=False):
    # Weight-scale pointer precision is part of the compiled launch ABI.
    @flyc.kernel(known_block_size=[64, 1, 1])
    def g128_overlap(
        x: fx.Pointer,
        w: fx.Pointer,
        xs: fx.Pointer,
        ws: fx.Pointer,
        sums: fx.Pointer,
        zeros: fx.Pointer,
        y: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        groups: fx.Int32,
    ):
        _tile(w4, split, (x, w, xs, ws, sums, zeros), y, m, n, groups, symmetric)

    @flyc.jit
    def launch(
        x: fx.Pointer,
        w: fx.Pointer,
        xs: fx.Pointer,
        ws: fx.Pointer,
        sums: fx.Pointer,
        zeros: fx.Pointer,
        y: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        groups: fx.Int32,
        stream: fx.Stream,
    ):
        g128_overlap(x, w, xs, ws, sums, zeros, y, m, n, groups).launch(
            grid=((m + 31) // 32, n // 64, split), block=(64, 1, 1), stream=stream
        )

    return launch


@torch.library.custom_op(
    "aiter::gfx908_g128_overlap", mutates_args=(), device_types="cuda"
)
def gfx908_g128_overlap(
    x: torch.Tensor,
    scales: torch.Tensor,
    sums: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    zeros: torch.Tensor | None,
    split: int,
    symmetric: bool = False,
) -> torch.Tensor:
    """Compute the scaled FP32 product, with one FP32 reduction for split K."""
    m, k = x.shape
    n = weight.shape[0]
    w4 = zeros is not None
    if weight_scales.dtype not in (torch.float16, torch.float32) or (
        w4 and zeros.dtype != weight_scales.dtype
    ):
        raise ValueError(
            "Expected FP16 or FP32 weight scales with matching affine zeros"
        )
    fp32_weight_scales = weight_scales.dtype == torch.float32
    scale_type = fx.Float32 if fp32_weight_scales else fx.Float16
    if symmetric and not w4:
        raise ValueError("A fixed zero point requires packed W4 weights")
    if (
        split < 1
        or n % 64
        or k % (split * 128)
        or (w4 and not symmetric and sums is None)
    ):
        raise ValueError("Expected packed 64-channel tiles and complete K128 stripes")
    if not m:
        return scales.new_empty((0, n))
    height = (m + 31) // 32 * 32
    partials = scales.new_empty((split, height, n))
    # The W8 specialization eliminates the sums and zeros loads.
    tensors = (
        x,
        weight,
        scales,
        weight_scales,
        sums if w4 and not symmetric else x,
        zeros if w4 else weight_scales,
        partials,
    )
    types = (
        fx.Int32,
        fx.Int32,
        fx.Float32,
        scale_type,
        fx.Int32,
        scale_type,
        fx.Float32,
    )
    ptrs = [
        flyc.from_c_void_p(ty, tensor.data_ptr(), assumed_align=16)
        for tensor, ty in zip(tensors, types)
    ]
    stream = fx.Stream(torch.cuda.current_stream(x.device))
    _run_compiled(
        _make_kernel(w4, split, fp32_weight_scales, symmetric),
        *ptrs,
        m,
        n,
        k // 128,
        stream,
    )
    return (partials[0] if split == 1 else partials.sum(0))[:m]


@gfx908_g128_overlap.register_fake
def _fake_overlap(
    x, scales, sums, weight, weight_scales, zeros, split, symmetric=False
):
    return scales.new_empty((x.shape[0], weight.shape[0]))
