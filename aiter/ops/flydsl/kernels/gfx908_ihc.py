# SPDX-License-Identifier: Apache-2.0
"""Eight hyper-connection gate dots on MFMA, with residual norms on the VALU."""

from functools import lru_cache, cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl, scf
from flydsl.expr.typing import T

from .gfx908_g128 import _minimum
from .gfx908_mla_values import _wave_read
from .tensor_shim import _run_compiled


def pack_ihc_gate(weight):
    """Use the other eight MFMA columns for the FP32 weight's FP16 residual."""
    if weight.shape != (8, 12288) or weight.dtype != torch.float32:
        raise ValueError("Expected eight FP32 gate vectors over four 3072-wide streams")
    high = weight.half()
    low = (weight - high.float()).half()
    return (
        torch.cat((high, low), dim=0)
        .view(16, -1, 4, 4)
        .permute(1, 2, 0, 3)
        .contiguous()
    )


def _load_half4(pointer, offset):
    return fx.Vector(
        llvm.load(T.vec(4, T.f16), (pointer + offset).llvm_ptr, alignment=8)
    )


def _compute(inputs, outputs, rows, chunk, post):
    projection_input, residual, gates, weights = inputs
    partials, updated = outputs
    lane = fx.Int32(fx.gpu.thread_id("x"))
    base = fx.Int32(fx.gpu.block_id("x")) * 16
    stripe = fx.Int32(fx.gpu.block_id("y"))
    row = _minimum(base + lane % 16, rows - 1)
    valid = base + lane % 16 < rows
    zero = fx.Float32(0)
    accum = fx.Vector.from_elements([zero] * 4)
    loop = scf.ForOp(
        fx.Int32(0).ir_value(),
        fx.Int32(chunk // 64).ir_value(),
        fx.Int32(1).ir_value(),
        [accum, zero.ir_value()],
    )
    with ir.InsertionPoint(loop.body):
        index = fx.Int32(loop.induction_variable)
        accum = fx.Vector(loop.inner_iter_args[0])
        norm = fx.Float32(loop.inner_iter_args[1])
        for step in range(4):
            k16 = stripe * (chunk // 16) + index * 4 + step
            offset = k16 * 16 + 4 * (lane // 16)
            data = _load_half4(residual, row * 12288 + offset)
            if post:
                addend = _load_half4(projection_input, row * 3072 + offset % 3072)
                scale = fx.Float32(gates[row * 4 + offset // 3072])
                data = fx.Vector.from_elements(
                    [
                        fx.Float16(scale * fx.Float32(addend[i]) + fx.Float32(data[i]))
                        for i in range(4)
                    ]
                )
                store = scf.IfOp(valid.ir_value())
                with ir.InsertionPoint(store.then_block):
                    llvm.store(
                        data, (updated + row * 12288 + offset).llvm_ptr, alignment=8
                    )
                    scf.YieldOp([])
            weight = _load_half4(weights, (k16 * 64 + lane) * 4)
            accum = fx.Vector(
                rocdl.mfma_f32_16x16x16f16_(
                    T.vec(4, T.f32),
                    data,
                    weight,
                    accum,
                    0,
                    0,
                    ir.Attribute.parse("#rocdl<mfma_perm_b none>"),
                )
            )
            # Independent vector work on the same four fetched residual values.
            for i in range(4):
                value = fx.Float32(data[i])
                norm = norm + value * value
        scf.YieldOp([accum, norm.ir_value()])
    accum = fx.Vector(loop.results[0])
    norm = fx.Float32(loop.results[1])
    for distance in (16, 32):
        norm = norm + _wave_read(norm, lane ^ distance)
    stripes = 12288 // chunk
    store = scf.IfOp(((lane < 16) & valid).ir_value())
    with ir.InsertionPoint(store.then_block):
        partials[(row * stripes + stripe) * 9] = norm
        scf.YieldOp([])
    for i in range(4):
        dest = base + (lane // 16) * 4 + i
        dot = accum[i] + _wave_read(accum[i], lane ^ 8)
        store = scf.IfOp(((lane % 16 < 8) & (dest < rows)).ir_value())
        with ir.InsertionPoint(store.then_block):
            partials[(dest * stripes + stripe) * 9 + 1 + lane % 8] = dot
            scf.YieldOp([])


@cache
def _build(chunk, post):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def ihc(
        x: fx.Pointer,
        r: fx.Pointer,
        g: fx.Pointer,
        w: fx.Pointer,
        p: fx.Pointer,
        y: fx.Pointer,
        m: fx.Int32,
    ):
        _compute((x, r, g, w), (p, y), m, chunk, post)

    @flyc.jit
    def launch(
        x: fx.Pointer,
        r: fx.Pointer,
        g: fx.Pointer,
        w: fx.Pointer,
        p: fx.Pointer,
        y: fx.Pointer,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        ihc(x, r, g, w, p, y, m).launch(
            grid=((m + 15) // 16, 12288 // chunk, 1), block=(64, 1, 1), stream=stream
        )

    return launch


@torch.library.custom_op(
    "aiter::gfx908_ihc_partial",
    mutates_args=("partial", "updated"),
    device_types="cuda",
)
def gfx908_ihc_partial(
    residual: torch.Tensor,
    weight: torch.Tensor,
    chunk: int,
    partial: torch.Tensor,
    x: torch.Tensor | None = None,
    gates: torch.Tensor | None = None,
    updated: torch.Tensor | None = None,
) -> None:
    rows = residual.shape[0]
    post = x is not None
    if (
        residual.shape != (rows, 4, 3072)
        or residual.dtype != torch.float16
        or weight.shape != (768, 4, 16, 4)
        or weight.dtype != torch.float16
        or chunk not in (256, 512, 1024)
        or partial.shape != (rows, 12288 // chunk, 9)
        or partial.dtype != torch.float32
        or (gates is not None) != post
        or (updated is not None) != post
    ):
        raise ValueError("Invalid four-stream FP16 iHC tensors or partition size")
    if post and (
        x.shape != (rows, 3072)
        or x.dtype != residual.dtype
        or gates.shape != (rows, 4)
        or gates.dtype != torch.float32
        or updated.shape != residual.shape
        or updated.dtype != residual.dtype
    ):
        raise ValueError("Invalid iHC post-gate inputs")
    tensors = (
        x if post else residual,
        residual,
        gates if post else partial,
        weight,
        partial,
        updated if post else residual,
    )
    if any(t.device != residual.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Expected contiguous tensors on one GPU")
    if rows:
        types = (fx.Float16, fx.Float16, fx.Float32, fx.Float16, fx.Float32, fx.Float16)
        ptrs = [
            flyc.from_c_void_p(ty, t.data_ptr(), assumed_align=8)
            for ty, t in zip(types, tensors)
        ]
        _run_compiled(
            _build(chunk, post),
            *ptrs,
            rows,
            fx.Stream(torch.cuda.current_stream(residual.device)),
        )


@gfx908_ihc_partial.register_fake
def _fake(residual, weight, chunk, partial, x=None, gates=None, updated=None):
    return None
