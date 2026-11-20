# SPDX-License-Identifier: Apache-2.0
"""Pipeline the next selected-key softmax with the current FP16 value GEMM."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, rocdl, scf
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gfx908_g128 import _minimum


def _wave_read(value, lane):
    raw = fx.rocdl.ds_bpermute(
        T.i32, (lane * 4).ir_value(), value.bitcast(fx.Int32).ir_value()
    )
    return fx.Int32(raw).bitcast(fx.Float32)


def _maximum(a, b):
    return fx.Float32(arith.maximumf(a.ir_value(), b.ir_value()))


def _exp(x):
    return fx.Float32(fx.math.exp2(x * fx.Float32(1.4426950408889634)))


def _reduce(values, lane, fn):
    while len(values) > 1:
        values = [fn(a, b) for a, b in zip(values[::2], values[1::2])]
    value = values[0]
    for distance in (16, 32):
        value = fn(value, _wave_read(value, lane ^ distance))
    return value


def _compute(config, tensors, sizes, tile_peaks=None):
    heads, group, splits, value_tiles, k16 = config
    keys, scales, scores, addresses, sinks, output, maxima, normalizers, partials = (
        tensors
    )
    width, pitch = sizes
    row = fx.Int32(fx.gpu.block_id("x"))
    stripe = fx.Int32(fx.gpu.block_id("y"))
    value_start = fx.Int32(fx.gpu.block_id("z")) * value_tiles * 64
    lane = fx.Int32(fx.gpu.thread_id("x"))
    head_in = lane % 16
    span = ((width + splits * 32 - 1) // (splits * 32)) * 32
    first, end = stripe * span, (stripe + 1) * span
    negative = fx.Float32(-3.4028234663852886e38)
    zero = fx.Float32(0.0)
    vector = fx.Vector.from_elements
    perm = ir.Attribute.parse("#rocdl<mfma_perm_b none>")

    def load_scores(base):
        start = (row * heads + head_in % heads) * pitch
        col = base + (lane // 16) * 8
        return [
            (col + i < width).select(
                scores[start + _minimum(col + i, width - 1)], negative
            )
            for i in range(8)
        ]

    def tile_peak(base, values):
        if tile_peaks is None:
            return _reduce(values, lane, _maximum)
        count = (pitch + 31) // 32
        dest = (row * heads + head_in % heads) * count + _minimum(base // 32, count - 1)
        return (base < width).select(tile_peaks[dest], negative)

    def probabilities(values, peak):
        return [(v > negative).select(_exp(v - peak), zero) for v in values]

    def pack(values):
        return vector([fx.Float16(v) for v in values])

    def do_tile(base, probs, acc, retain, next_base, running_max, running_sum):
        # The native compiler folds the rescaled partial into MFMA's accumulator.
        # Starting MFMA at zero would change rounding across 32-key tiles.
        def prior(tile, index):
            head = (lane // 16) * 4 + index % 4
            return acc[tile * 16 + index] * _wave_read(retain, head)

        dots = [
            [vector([prior(t, block * 4 + j) for j in range(4)]) for block in range(4)]
            if k16
            else [vector([prior(t, j) for j in range(16)])]
            for t in range(value_tiles)
        ]
        new_probs = []
        words = probs.bitcast(fx.Int32)
        for step in range(8):
            if k16:
                src = head_in + 16 * ((step // 4) * 2 + (lane // 16) // 2)
                select = (lane // 16) % 2 == 0
                operand = [
                    select.select(
                        fx.Int32(
                            fx.rocdl.ds_bpermute(
                                T.i32, (src * 4).ir_value(), words[j].ir_value()
                            )
                        ),
                        fx.Int32(
                            fx.rocdl.ds_bpermute(
                                T.i32, (src * 4).ir_value(), words[2 + j].ir_value()
                            )
                        ),
                    )
                    for j in range(2)
                ]
                key = base + (step // 4) * 16 + (lane // 16) * 4
            else:
                src = head_in + (step // 2) * 16
                operand = [
                    fx.Int32(
                        fx.rocdl.ds_bpermute(
                            T.i32,
                            (src * 4).ir_value(),
                            words[(step % 2) * 2 + j].ir_value(),
                        )
                    )
                    for j in range(2)
                ]
                key = base + step * 4
            p = vector(operand).bitcast(fx.Float16)
            offsets = [
                addresses[row * pitch + _minimum(key + j, width - 1)] for j in range(4)
            ]
            for tile in range(value_tiles):
                block = step % 4 if k16 else 0
                col = (
                    value_start + 64 * tile + (lane % 16 + block * 16 if k16 else lane)
                )
                v = []
                for j, off in enumerate(offsets):
                    valid = (key + j < width) & (off >= 0)
                    address = valid.select(off, fx.Int64(0))
                    val = (
                        fx.Float16(keys[address + col])
                        * scales[address // 2 + 288 + col // group]
                    )
                    v.append(valid.select(val, fx.Float16(0.0)))
                if k16:
                    mat = rocdl.mfma_f32_16x16x16f16_(
                        T.vec(4, T.f32), p, vector(v), dots[tile][block], 0, 0, perm
                    )
                else:
                    mat = rocdl.mfma_f32_16x16x4f16_(
                        T.vec(16, T.f32), p, vector(v), dots[tile][0], 2, 0, perm
                    )
                dots[tile][block] = fx.Vector(mat)

            # These operations depend on the next score row and never on the
            # in-flight matrix result. Its normalization still follows tile order.
            if next_base is not None:
                if step == 0:
                    vals = load_scores(next_base)
                elif step == 1:
                    peak = _maximum(running_max, tile_peak(next_base, vals))
                    next_retain = _exp(running_max - peak)
                elif 2 <= step <= 5:
                    new_probs.extend(
                        probabilities(vals[(step - 2) * 2 : (step - 1) * 2], peak)
                    )
                elif step == 6:
                    total = running_sum * next_retain + _reduce(
                        new_probs, lane, lambda a, b: a + b
                    )
        updated = vector(
            [
                dots[t][j // 4][j % 4] if k16 else dots[t][0][j]
                for t in range(value_tiles)
                for j in range(16)
            ]
        )
        if next_base is None:
            return updated
        return updated, peak, total, next_retain, pack(new_probs)

    vals = load_scores(first)
    peak = tile_peak(first, vals)
    probs = probabilities(vals, peak)
    total = _reduce(probs, lane, lambda a, b: a + b)
    initial = [
        vector([zero] * (16 * value_tiles)),
        peak.ir_value(),
        total.ir_value(),
        _exp(negative - peak).ir_value(),
        pack(probs),
    ]
    loop = scf.ForOp(
        (first + 32).ir_value(), end.ir_value(), fx.Int32(32).ir_value(), initial
    )
    with ir.InsertionPoint(loop.body):
        acc, mx, sm, rt, p = loop.inner_iter_args
        base = fx.Int32(loop.induction_variable)
        acc, mx, sm, rt, p = do_tile(
            base - 32,
            fx.Vector(p),
            fx.Vector(acc),
            fx.Float32(rt),
            base,
            fx.Float32(mx),
            fx.Float32(sm),
        )
        scf.YieldOp([acc, mx.ir_value(), sm.ir_value(), rt.ir_value(), p])
    acc, mx, sm, rt, p = loop.results
    acc = do_tile(
        end - 32, fx.Vector(p), fx.Vector(acc), fx.Float32(rt), None, None, None
    )
    mx, sm = fx.Float32(mx), fx.Float32(sm)
    if splits > 1:
        store = scf.IfOp(((value_start == 0) & (lane < heads)).ir_value())
        with ir.InsertionPoint(store.then_block):
            i = (row * splits + stripe) * heads + lane
            maxima[i] = mx
            normalizers[i] = sm
            scf.YieldOp([])
    else:
        sink = sinks[head_in % heads]
        final = _maximum(sink, mx)
        rem = _exp(mx - final)
        factor = rem / (sm * rem + _exp(sink - final))
    for j in range(16):
        head = (lane // 16) * 4 + j % 4
        col = value_start + lane % 16 + (j // 4) * 16
        store = scf.IfOp((head < heads).ir_value())
        with ir.InsertionPoint(store.then_block):
            for tile in range(value_tiles):
                if splits == 1:
                    output[(row * heads + head) * 512 + col + tile * 64] = fx.Float16(
                        acc[tile * 16 + j] * _wave_read(factor, head)
                    )
                else:
                    partials[
                        ((row * splits + stripe) * heads + head) * 512 + col + tile * 64
                    ] = acc[tile * 16 + j]
            scf.YieldOp([])


@cache
def _kernel(config):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def values(
        k: fx.Pointer,
        ks: fx.Pointer,
        scores: fx.Pointer,
        ad: fx.Pointer,
        sinks: fx.Pointer,
        out: fx.Pointer,
        mx: fx.Pointer,
        sm: fx.Pointer,
        part: fx.Pointer,
        width: fx.Int32,
        pitch: fx.Int32,
    ):
        _compute(config, (k, ks, scores, ad, sinks, out, mx, sm, part), (width, pitch))

    @flyc.jit
    def launch(
        k: fx.Pointer,
        ks: fx.Pointer,
        scores: fx.Pointer,
        ad: fx.Pointer,
        sinks: fx.Pointer,
        out: fx.Pointer,
        mx: fx.Pointer,
        sm: fx.Pointer,
        part: fx.Pointer,
        width: fx.Int32,
        pitch: fx.Int32,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        values(k, ks, scores, ad, sinks, out, mx, sm, part, width, pitch).launch(
            grid=(m, config[2], 8 // config[3]), block=(64, 1, 1), stream=stream
        )

    return launch


@torch.library.custom_op(
    "aiter::gfx908_mla_values",
    mutates_args=("output", "maxima", "normalizers", "partials"),
    device_types="cuda",
)
def gfx908_mla_values(
    cache: torch.Tensor,
    scores: torch.Tensor,
    addresses: torch.Tensor,
    sinks: torch.Tensor,
    output: torch.Tensor,
    maxima: torch.Tensor | None,
    normalizers: torch.Tensor | None,
    partials: torch.Tensor | None,
    width: int,
    group: int,
    splits: int,
    value_tiles: int,
    k16: bool,
) -> None:
    rows, heads, pitch = scores.shape
    tensors = (
        cache,
        cache,
        scores,
        addresses,
        sinks,
        output,
        maxima,
        normalizers,
        partials,
    )
    types = (
        fx.Int8,
        fx.Float16,
        fx.Float32,
        fx.Int64,
        fx.Float32,
        fx.Float16,
        fx.Float32,
        fx.Float32,
        fx.Float32,
    )
    pointers = [
        flyc.from_c_void_p(t, (output if buf is None else buf).data_ptr())
        for t, buf in zip(types, tensors)
    ]
    _run_compiled(
        _kernel((heads, group, splits, value_tiles, k16)),
        *pointers,
        width,
        pitch,
        rows,
        fx.Stream(torch.cuda.current_stream(scores.device)),
    )


@gfx908_mla_values.register_fake
def _fake_values(
    cache,
    scores,
    addresses,
    sinks,
    output,
    maxima,
    normalizers,
    partials,
    width,
    group,
    splits,
    value_tiles,
    k16,
):
    return None
