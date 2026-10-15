# SPDX-License-Identifier: Apache-2.0
"""Tile-local IQ2/Q8/NVFP4 decoding and GS128 INT8 MFMA for gfx908."""

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


@triton.jit
def _decode(W, GRID, rows, ks, N: tl.constexpr, K: tl.constexpr):
    base = rows[:, None].to(tl.int64) * (K // 256) * 66 + (ks[None, :] // 256) * 66
    valid = (rows[:, None] < N) & (ks[None, :] < K)
    lo = tl.load(W + base, valid, 0).to(tl.uint16)
    hi = tl.load(W + base + 1, valid, 0).to(tl.uint16)
    d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    sub = (ks[None, :] % 256) // 32
    lane = (ks[None, :] % 32) // 8
    j = ks[None, :] % 8
    code = tl.load(W + base + 2 + sub * 8 + lane, valid, 0).to(tl.int32)
    bits = tl.full(base.shape, 0, tl.uint32)
    for b in tl.static_range(4):
        bits |= tl.load(W + base + 2 + sub * 8 + 4 + b, valid, 0).to(tl.uint32) << (
            8 * b
        )
    sign7 = (bits >> (7 * lane)) & 127
    parity = sign7 ^ (sign7 >> 4)
    parity ^= parity >> 2
    parity ^= parity >> 1
    signs = sign7 | ((parity & 1) << 7)
    grid = tl.load(GRID + code * 8 + j).to(tl.float32)
    value = d * (0.5 + (bits >> 28).to(tl.float32)) * 0.25 * grid
    return tl.where(valid, tl.where(((signs >> j) & 1) != 0, -value, value), 0.0)


@triton.jit
def _decode_q8(W, rows, ks, N: tl.constexpr, K: tl.constexpr):
    base = rows[:, None].to(tl.int64) * (K // 32) * 34 + (ks[None, :] // 32) * 34
    valid = (rows[:, None] < N) & (ks[None, :] < K)
    lo = tl.load(W + base, valid, 0).to(tl.uint16)
    hi = tl.load(W + base + 1, valid, 0).to(tl.uint16)
    scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    q = tl.load(W + base + 2 + ks[None, :] % 32, valid, 0).to(tl.int8, bitcast=True)
    return q.to(tl.float32) * scale


@triton.jit
def _decode_nvfp4(
    W, WS, WG, rows, ks, N: tl.constexpr, K: tl.constexpr, WG_PER_ROW: tl.constexpr
):
    valid = (rows[:, None] < N) & (ks[None, :] < K)
    byte = tl.load(
        W + rows[:, None].to(tl.int64) * (K // 2) + ks[None, :] // 2, valid, 0
    ).to(tl.int32)
    nibble = (byte >> ((ks[None, :] % 2) * 4)) & 15
    mag = nibble & 7
    value = tl.where(
        mag < 2,
        mag.to(tl.float32) * 0.5,
        (((mag >> 1) + 126) << 23 | ((mag & 1) << 22)).to(tl.float32, bitcast=True),
    )
    value = tl.where((nibble & 8) != 0, -value, value)
    # E4M3 scales are read as bytes: gfx908 has no native FP8 load/compute.
    bits = tl.load(
        WS + rows[:, None].to(tl.int64) * (K // 16) + ks[None, :] // 16, valid, 0
    ).to(tl.int32)
    exp = (bits >> 3) & 15
    mantissa = bits & 7
    scale = tl.where(
        exp == 0,
        mantissa.to(tl.float32) * (2.0**-9),
        (((exp + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True),
    )
    scale = tl.where((bits & 128) != 0, -scale, scale)
    if WG_PER_ROW:
        global_scale = tl.load(WG + rows[:, None], rows[:, None] < N, 0)
    else:
        global_scale = tl.load(WG)
    return value * scale * global_scale.to(tl.float32)


@triton.jit
def _quant(X, Q, S, M: tl.constexpr, K: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    k = group * 128 + tl.arange(0, 128)
    v = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(v), 0), 127.0), 1.0e-30)
    q = tldevice.nearbyint(tl.div_rn(v, scale))
    tl.store(Q + row * K + k, q.to(tl.int8), k < K)
    tl.store(S + row * tl.cdiv(K, 128) + group, scale)


@triton.jit(do_not_specialize=["M"])
def _gemm(
    X,
    XS,
    W,
    GRID,
    WS,
    WG,
    Y,
    YS,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    START: tl.constexpr,
    LENGTH: tl.constexpr,
    RELU2: tl.constexpr,
    QUANT_OUT: tl.constexpr,
    BM: tl.constexpr,
    CODEC: tl.constexpr,
    IDS=None,
    TOPK: tl.constexpr = 1,
    ROUTED: tl.constexpr = False,
    DOWN: tl.constexpr = False,
    WSTRIDE: tl.constexpr = 0,
    SORTED=None,
    COUNT=None,
    GROUPED: tl.constexpr = False,
    WG_PER_ROW: tl.constexpr = False,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    if ROUTED:
        route = tl.program_id(0)
        if GROUPED:
            if route * BM >= tl.load(COUNT):
                return
            expert = tl.load(IDS + route).to(tl.int64)
            rm = tl.load(SORTED + route * BM + tl.arange(0, BM))
        else:
            expert = tl.load(IDS + route).to(tl.int64)
            rm = tl.where(tl.arange(0, BM) == 0, route, M)
        W += expert * WSTRIDE
    xm = rm
    if ROUTED and not DOWN:
        xm = rm // TOPK
    rn = tl.program_id(1) * 128 + tl.arange(0, 128)
    rk = tl.arange(0, 128)
    acc = tl.zeros((BM, 128), tl.float32)
    for group in range(tl.cdiv(LENGTH, 128)):
        local_k = group * 128 + rk
        if CODEC == "iq2_xxs":
            w = _decode(W, GRID, rn, START + local_k, N, K)
        elif CODEC == "q8_0":
            w = _decode_q8(W, rn, START + local_k, N, K)
        else:
            w = _decode_nvfp4(W, WS, WG, rn, START + local_k, N, K, WG_PER_ROW)
        w = tl.where(local_k[None, :] < LENGTH, w, 0.0)
        sw = tl.maximum(tl.div_rn(tl.max(tl.abs(w), 1), 127.0), 1.0e-30)
        wq = tldevice.nearbyint(w * tl.div_rn(1.0, sw)[:, None]).to(tl.int8)
        x = tl.load(
            X + xm[:, None] * LENGTH + local_k[None, :],
            (rm[:, None] < M) & (local_k[None, :] < LENGTH),
            0,
        )
        sx = tl.load(XS + xm * tl.cdiv(LENGTH, 128) + group, rm < M, 0)
        dot = tl.dot(x, tl.trans(wq), out_dtype=tl.int32)
        acc += dot.to(tl.float32) * sx[:, None] * sw[None, :]
    if RELU2:
        acc = tl.maximum(acc, 0.0)
        acc *= acc
    if QUANT_OUT:
        acc = tl.where(rn[None, :] < N, acc, 0.0)
        sy = tl.maximum(tl.div_rn(tl.max(tl.abs(acc), 1), 127.0), 1.0e-30)
        q = tldevice.nearbyint(tl.div_rn(acc, sy[:, None])).to(tl.int8)
        tl.store(
            Y + rm[:, None] * N + rn[None, :], q, (rm[:, None] < M) & (rn[None, :] < N)
        )
        tl.store(YS + rm * tl.cdiv(N, 128) + tl.program_id(1), sy, rm < M)
    else:
        tl.store(
            Y + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _embedding(
    W, IDS, OUT, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    ks = tl.program_id(1) * B + tl.arange(0, B)
    idx = tl.load(IDS + row)
    value = _decode_q8(W, tl.full((1,), 0, tl.int32) + idx, ks, N, K)
    tl.store(OUT + row * K + ks, tl.reshape(value, (B,)), ks < K)
