# SPDX-License-Identifier: Apache-2.0
"""Selected INT8 MLA scores for gfx908 with separate latent and rotary scales."""

from functools import lru_cache, cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, rocdl, scf
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gfx908_g128 import _load_packet, _minimum


def _byte_offset(slot, page_size, page_stride, row_stride):
    safe = fx.Int32(arith.maxsi(slot.ir_value(), fx.Int32(0).ir_value()))
    return fx.Int64(safe // page_size) * fx.Int64(page_stride) + fx.Int64(
        (safe % page_size) * row_stride
    )


def _score_groups(
    query, qs, keys, ks, lane, row, heads, group, rope_group, offset, offsets
):
    """MFMA owns 16 heads by 64 keys. Scale group g-1 during the next MFMA."""
    latent_groups = 512 // group
    total_groups = latent_groups + 64 // rope_group
    q_offset = (row * heads + (lane % 16) % heads) * 144
    q_scales = [
        (row * heads + (4 * (lane // 16) + i) % heads) * total_groups for i in range(4)
    ]
    latent = [fx.Float32(0.0)] * 16
    rotary = [fx.Float32(0.0)] * 16
    previous = None

    def scaled(index, dot, current_group):
        a = fx.Float32(qs[q_scales[index % 4] + current_group])
        b = fx.Float32(ks[offsets[index // 4] // 2 + 288 + current_group])
        return (fx.Float32(dot[index]) * a) * b

    perm = ir.Attribute.parse("#rocdl<mfma_perm_b none>")
    for g in range(total_groups):
        start = (
            g * group if g < latent_groups else 512 + (g - latent_groups) * rope_group
        ) // 4
        steps = (group if g < latent_groups else rope_group) // 4
        sums = latent if g - 1 < latent_groups else rotary
        current = fx.Vector.from_elements([fx.Int32(0)] * 16)
        for step in range(steps):
            if step % 4 == 0:
                qa = _load_packet(query, q_offset + start + step, 4)
                kb = _load_packet(keys, offset // 4 + start + step, 4)
            current = fx.Vector(
                rocdl.mfma_i32_16x16x4i8_(
                    T.vec(16, T.i32),
                    qa[step % 4].ir_value(),
                    kb[step % 4].ir_value(),
                    current,
                    2,
                    0,
                    perm,
                )
            )
            if g:
                for index in range(step, 16, steps):
                    sums[index] = sums[index] + scaled(index, previous, g - 1)
        previous = current
    for i in range(16):
        rotary[i] = rotary[i] + scaled(i, previous, total_groups - 1)
    return [a + b for a, b in zip(latent, rotary)]


def _score_tile(config, sources, outputs, sizes):
    heads, group, rope_group, scale, window, save_addresses, skip_empty = config
    query, qs, keys, ks, ids, owners, table, starts, lengths = sources
    scores, addresses = outputs
    (
        width,
        pitch,
        page,
        page_stride,
        row_stride,
        table_stride,
        pages,
        id_stride,
        row_offset,
    ) = sizes
    lane = fx.Int32(fx.gpu.thread_id("x"))
    row = fx.Int32(fx.gpu.block_id("x"))
    first = fx.Int32(fx.gpu.block_id("y")) * 64
    col = first + lane
    owner = owners[row]
    if window:
        token = row + row_offset
        position = lengths[owner] - (starts[owner + 1] - token)
        live = (starts[owner] <= token) & (token < starts[owner + 1])
        logical = live.select(position - window + 1 + col, fx.Int32(-1))
        valid = (col < width) & (col < window)
    else:
        logical = ids[row * id_stride + _minimum(col, width - 1)]
        valid = col < width
    valid = valid & (logical >= 0) & (logical < pages * page)
    logical = valid.select(logical, fx.Int32(0))
    physical = table[owner * table_stride + logical // page]
    slot = (valid & (physical >= 0)).select(
        physical * page + logical % page, fx.Int32(-1)
    )
    offset = _byte_offset(slot, page, page_stride, row_stride)
    if save_addresses:
        branch = scf.IfOp((col < width).ir_value())
        with ir.InsertionPoint(branch.then_block):
            addresses[row * pitch + col] = (slot >= 0).select(offset, fx.Int64(-1))
            scf.YieldOp([])

    # Each lane's sixteen accumulators span four keys and four heads.
    slots = [
        fx.Int32(
            fx.rocdl.ds_bpermute(
                T.i32, ((lane % 16 + 16 * i) * 4).ir_value(), slot.ir_value()
            )
        )
        for i in range(4)
    ]
    offsets = [_byte_offset(s, page, page_stride, row_stride) for s in slots]

    def write(values):
        minimum = fx.Float32(-3.4028234663852886e38)
        for i in range(16):
            key = first + lane % 16 + (i // 4) * 16
            head = 4 * (lane // 16) + i % 4
            valid_output = scf.IfOp(((key < width) & (head < heads)).ir_value())
            with ir.InsertionPoint(valid_output.then_block):
                value = (
                    minimum
                    if values is None
                    else (slots[i // 4] >= 0).select(
                        values[i] * fx.Float32(scale), minimum
                    )
                )
                scores[(row * heads + head) * pitch + key] = value
                scf.YieldOp([])

    def compute():
        write(
            _score_groups(
                query,
                qs,
                keys,
                ks,
                lane,
                row,
                heads,
                group,
                rope_group,
                offset,
                offsets,
            )
        )

    if skip_empty:
        live = fx.Int64(fx.rocdl.ballot(T.i64, (slot >= 0).ir_value()))
        branch = scf.IfOp((live != 0).ir_value(), has_else=True)
        with ir.InsertionPoint(branch.then_block):
            compute()
            scf.YieldOp([])
        with ir.InsertionPoint(branch.else_block):
            write(None)
            scf.YieldOp([])
    else:
        compute()


@cache
def _make(config):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def score(
        q: fx.Pointer,
        qs: fx.Pointer,
        k: fx.Pointer,
        ks: fx.Pointer,
        ids: fx.Pointer,
        owners: fx.Pointer,
        table: fx.Pointer,
        starts: fx.Pointer,
        lengths: fx.Pointer,
        out: fx.Pointer,
        ad: fx.Pointer,
        width: fx.Int32,
        pitch: fx.Int32,
        pg: fx.Int32,
        ps: fx.Int32,
        rs: fx.Int32,
        ts: fx.Int32,
        pages: fx.Int32,
        ds: fx.Int32,
        row_offset: fx.Int32,
    ):
        _score_tile(
            config,
            (q, qs, k, ks, ids, owners, table, starts, lengths),
            (out, ad),
            (width, pitch, pg, ps, rs, ts, pages, ds, row_offset),
        )

    @flyc.jit
    def launch(
        q: fx.Pointer,
        qs: fx.Pointer,
        k: fx.Pointer,
        ks: fx.Pointer,
        ids: fx.Pointer,
        owners: fx.Pointer,
        table: fx.Pointer,
        starts: fx.Pointer,
        lengths: fx.Pointer,
        out: fx.Pointer,
        ad: fx.Pointer,
        width: fx.Int32,
        pitch: fx.Int32,
        pg: fx.Int32,
        ps: fx.Int32,
        rs: fx.Int32,
        ts: fx.Int32,
        pages: fx.Int32,
        ds: fx.Int32,
        row_offset: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        score(
            q,
            qs,
            k,
            ks,
            ids,
            owners,
            table,
            starts,
            lengths,
            out,
            ad,
            width,
            pitch,
            pg,
            ps,
            rs,
            ts,
            pages,
            ds,
            row_offset,
        ).launch(grid=(rows, (width + 63) // 64, 1), block=(64, 1, 1), stream=stream)

    return launch


def _outputs(query, pitch, save_addresses):
    rows, heads = query.shape[:2]
    return (
        query.new_empty((rows, heads, pitch), dtype=torch.float32),
        query.new_empty((rows, pitch) if save_addresses else (0,), dtype=torch.int64),
    )


@torch.library.custom_op(
    "aiter::gfx908_mla_scores", mutates_args=(), device_types="cuda"
)
def gfx908_mla_scores(
    query: torch.Tensor,
    scales: torch.Tensor,
    cache: torch.Tensor,
    ids: torch.Tensor,
    owners: torch.Tensor,
    table: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    group: int,
    rope_group: int,
    pitch: int,
    window: int,
    row_offset: int,
    save_addresses: bool,
    skip_empty: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, heads, depth = query.shape
    width = ids.shape[1]
    if (
        heads not in (8, 16)
        or depth != 576
        or group not in (32, 64, 128)
        or rope_group not in (32, 64)
        or pitch < width
        or cache.stride(1) % 16
    ):
        raise ValueError(
            "Expected aligned gfx908 MLA groups and an eight- or sixteen-head query"
        )
    result = _outputs(query, pitch, save_addresses)
    if rows and width:
        config = heads, group, rope_group, scale, window, save_addresses, skip_empty
        tensors = (
            query,
            scales,
            cache,
            cache,
            ids,
            owners,
            table,
            starts,
            lengths,
            *result,
        )
        types = (
            fx.Int32,
            fx.Float16,
            fx.Int32,
            fx.Float16,
            fx.Int32,
            fx.Int32,
            fx.Int32,
            fx.Int32,
            fx.Int32,
            fx.Float32,
            fx.Int64,
        )
        pointers = [
            flyc.from_c_void_p(ty, t.data_ptr()) for ty, t in zip(types, tensors)
        ]
        _run_compiled(
            _make(config),
            *pointers,
            width,
            pitch,
            cache.shape[1],
            cache.stride(0),
            cache.stride(1),
            table.stride(0),
            table.shape[1],
            ids.stride(0),
            row_offset,
            rows,
            fx.Stream(torch.cuda.current_stream(query.device)),
        )
    return result


@gfx908_mla_scores.register_fake
def _fake_scores(
    query,
    scales,
    cache,
    ids,
    owners,
    table,
    starts,
    lengths,
    scale,
    group,
    rope_group,
    pitch,
    window,
    row_offset,
    save_addresses,
    skip_empty,
):
    return _outputs(query, pitch, save_addresses)
