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
def _gated_fp16(gate, up, LIMIT: tl.constexpr):
    gate = gate.to(tl.float16).to(tl.float32)
    up = up.to(tl.float16).to(tl.float32)
    if LIMIT is not None:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    silu = tl.div_rn(gate, 1.0 + tldevice.exp(-gate)).to(tl.float16).to(tl.float32)
    return (silu * up).to(tl.float16).to(tl.float32)


@triton.jit
def _quant(
    X,
    Q,
    S,
    K: tl.constexpr,
    SUM=None,
    SWIGLU: tl.constexpr = False,
    LIMIT: tl.constexpr = None,
):
    row, group = tl.program_id(0), tl.program_id(1)
    k = group * 128 + tl.arange(0, 128)
    offset = row.to(tl.int64) * K + k
    if SWIGLU:
        base = row.to(tl.int64) * (2 * K) + k
        gate = tl.load(X + base, k < K, 0)
        up = tl.load(X + base + K, k < K, 0)
        v = _gated_fp16(gate, up, LIMIT)
    else:
        v = tl.load(X + offset, k < K, 0).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(v), 0), 127.0), 1.0e-30)
    q = tldevice.nearbyint(tl.div_rn(v, scale))
    tl.store(Q + offset, q.to(tl.int8), k < K)
    index = row.to(tl.int64) * tl.cdiv(K, 128) + group
    tl.store(S + index, scale)
    if SUM is not None:
        tl.store(SUM + index, tl.sum(q.to(tl.int32), 0))


@triton.jit
def _affine_codes(
    W, rn, rk, N: tl.constexpr, K: tl.constexpr, W4: tl.constexpr, WAVE: tl.constexpr
):
    if WAVE == 3:
        # 64 outputs per wave, one contiguous K16 vector per output.
        row = rn[:, None].to(tl.int64)
        col = rk[None, :]
        packet = (
            (row // 64 * (K // 128) + col // 128) * 8 + col % 128 // 16
        ) * 64 + row % 64
        if W4:
            byte = tl.load(W + packet * 8 + col % 16 // 8 * 4 + col % 4, row < N, 0)
            return (((byte >> ((col % 8 // 4) * 4)) & 15) ^ 8).to(tl.int8)
        return tl.load(W + packet * 16 + col % 16, row < N, 0).to(tl.int8)
    # Index the same N16/K16 lane packing as the gfx908 HIP kernel.
    if WAVE:
        if WAVE == 2:
            words = (
                (
                    ((rn[:, None].to(tl.int64) // 16) * (K // 128) + rk[None, :] // 128)
                    * 4
                    + (rk[None, :] % 16) // 4
                )
                * 16
                + rn[:, None] % 16
            ) * 8 + (rk[None, :] % 128) // 16
        else:
            words = (
                ((rn[:, None].to(tl.int64) // 16) * (K // 16) + rk[None, :] // 16) * 4
                + (rk[None, :] % 16) // 4
            ) * 16 + rn[:, None] % 16
        if W4:
            offset = words * 2 + rk[None, :] % 4 // 2
        else:
            offset = words * 4 + rk[None, :] % 4
    else:
        if W4:
            offset = rn[:, None].to(tl.int64) * (K // 2) + rk[None, :] // 2
        else:
            offset = rn[:, None].to(tl.int64) * K + rk[None, :]
    data = tl.load(W + offset, rn[:, None] < N, 0)
    if W4:
        data = (data >> ((rk[None, :] % 2) * 4)) & 15
    return data.to(tl.int8)


@triton.jit
def _affine_wave_tile(
    W,
    row0,
    group,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    W4: tl.constexpr,
    WAVE: tl.constexpr,
    PAIRED: tl.constexpr = False,
):
    if WAVE == 3 and BN < 64:
        return _affine_codes(
            W, row0 + tl.arange(0, BN), group * 128 + tl.arange(0, 128), N, K, W4, WAVE
        )
    elif WAVE == 3:
        width: tl.constexpr = 8 if W4 else 16
        size: tl.constexpr = 8 * 64 * width
        i = tl.arange(0, BN // 64 * size)
        tile = row0 // 64 + i // size
        if PAIRED:
            tile += tl.where(i // size >= BN // 128, (N - BN) // 128, 0)
        addr = (tile.to(tl.int64) * (K // 128) + group) * size + i % size
        data = tl.load(W + addr, tile * 64 < N, 0)
        if W4:
            pairs = (data ^ 136).reshape((BN // 64, 8, 64, 2, 4))
            codes = tl.join((pairs & 15).to(tl.int8), (pairs >> 4).to(tl.int8))
            return codes.permute(0, 2, 1, 3, 5, 4).reshape((BN, 128))
        else:
            return (
                data.reshape((BN // 64, 8, 64, 16))
                .permute(0, 2, 1, 3)
                .reshape((BN, 128))
            )
    else:
        # Load a whole contiguous G128 packet per 16 output rows. Reorder locally
        # into the same [N,K] dot operand used by the symmetric INT8 path.
        word: tl.constexpr = 2 if W4 else 4
        packet: tl.constexpr = 16 * 8 * 4 * word
        i = tl.arange(0, (BN // 16) * packet)
        tile = row0 // 16 + i // packet
        if PAIRED:
            tile += tl.where(i // packet >= BN // 32, (N - BN) // 32, 0)
        offset = (tile.to(tl.int64) * (K // 128) + group) * packet + i % packet
        data = tl.load(W + offset, tile * 16 < N, 0)
        if W4:
            codes = tl.join((data & 15).to(tl.int8), (data >> 4).to(tl.int8))
            if WAVE == 2:
                codes = codes.reshape((BN // 16, 4, 16, 8, word, 2)).permute(
                    0, 2, 3, 1, 4, 5
                )
            else:
                codes = codes.reshape((BN // 16, 8, 4, 16, word, 2)).permute(
                    0, 3, 1, 2, 4, 5
                )
        else:
            if WAVE == 2:
                codes = data.reshape((BN // 16, 4, 16, 8, word)).permute(0, 2, 3, 1, 4)
            else:
                codes = data.reshape((BN // 16, 8, 4, 16, word)).permute(0, 3, 1, 2, 4)
        return codes.reshape((BN,128))


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
    SPLIT_K: tl.constexpr = 1,
    WEIGHT_SCALES=None,
    CACHED_SCALES: tl.constexpr = False,
    EXPERT_WEIGHT=None,
    EXPERT_MAP=None,
    EXPERT_CACHED: tl.constexpr = False,
    ASUM=None,
    WZERO=None,
    AFFINE_WAVE: tl.constexpr = False,
    BN: tl.constexpr = 128,
    PACKET_LOAD: tl.constexpr = False,
    GATE_OUT: tl.constexpr = False,
    GATE_LIMIT: tl.constexpr = None,
    YSUM=None,
):
    slot = -1
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
        if WZERO is not None:
            WZERO += expert * N * (K // 128)
        if EXPERT_CACHED:
            slot = tl.load(EXPERT_MAP + expert)
            EXPERT_WEIGHT += tl.maximum(slot, 0).to(tl.int64) * N * LENGTH
        if CACHED_SCALES:
            WEIGHT_SCALES += expert * N * tl.cdiv(LENGTH, 128)
    xm = rm
    if ROUTED and not DOWN:
        xm = rm // TOPK
    if GATE_OUT:
        tl.static_assert(BN == 256 and SPLIT_K == 1 and PACKET_LOAD)
        columns = tl.arange(0, BN)
        out_col = tl.program_id(1) * 128
        rn = out_col + columns + tl.where(columns >= 128, (N - BN) // 2, 0)
    else:
        out_col = tl.program_id(1) * BN
        rn=out_col+tl.arange(0,BN)
    rk = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), tl.float32)
    groups: tl.constexpr = (LENGTH + 127) // 128
    span: tl.constexpr = (groups + SPLIT_K - 1) // SPLIT_K
    if SPLIT_K == 1:
        begin = 0
        end = groups
    else:
        begin = tl.program_id(2) * span
        end = tl.minimum((tl.program_id(2) + 1) * span, groups)
    for group in range(begin, end):
        local_k = group * 128 + rk
        if CODEC == "affine_w4" or CODEC == "signed_w8":
            if AFFINE_WAVE == 3:
                gi = group * N + rn
            elif AFFINE_WAVE:
                gi = ((rn // 16) * groups + group) * 16 + rn % 16
            else:
                gi = rn * groups + group
            if AFFINE_WAVE and PACKET_LOAD:
                wq = _affine_wave_tile(
                    W,
                    out_col,
                    group,
                    N,
                    K,
                    BN,
                    CODEC == "affine_w4",
                    AFFINE_WAVE,
                    GATE_OUT,
                )
            else:
                wq = _affine_codes(
                    W, rn, local_k, N, K, CODEC == "affine_w4", AFFINE_WAVE
                )
            sw = tl.load(WEIGHT_SCALES + gi, rn < N, 0).to(tl.float32)
        elif EXPERT_CACHED and slot >= 0:
            # One expert per program makes this branch uniform. Never load a
            # cold slot: its allocation can still contain unwritten values.
            wq = tl.load(
                EXPERT_WEIGHT + rn[:, None] * LENGTH + local_k[None, :],
                (rn[:, None] < N) & (local_k[None, :] < LENGTH),
                0,
            )
            sw = tl.load(WEIGHT_SCALES + rn * tl.cdiv(LENGTH, 128) + group, rn < N, 1.0)
        else:
            if CODEC == "iq2_xxs":
                w = _decode(W, GRID, rn, START + local_k, N, K)
            elif CODEC == "q8_0":
                w = _decode_q8(W, rn, START + local_k, N, K)
            else:
                w = _decode_nvfp4(W, WS, WG, rn, START + local_k, N, K, WG_PER_ROW)
            w = tl.where(local_k[None, :] < LENGTH, w, 0.0)
            if CACHED_SCALES:
                sw = tl.load(
                    WEIGHT_SCALES + rn * tl.cdiv(LENGTH, 128) + group, rn < N, 1.0
                )
            else:
                sw = tl.maximum(tl.div_rn(tl.max(tl.abs(w), 1), 127.0), 1.0e-30)
            wq = tldevice.nearbyint(w * tl.div_rn(1.0, sw)[:, None]).to(tl.int8)
        x = tl.load(
            X + xm[:, None] * LENGTH + local_k[None, :],
            (rm[:, None] < M) & (local_k[None, :] < LENGTH),
            0,
        )
        sx = tl.load(XS + xm * tl.cdiv(LENGTH, 128) + group, rm < M, 0)
        dot = tl.dot(x, tl.trans(wq), out_dtype=tl.int32)
        product = dot.to(tl.float32)
        if CODEC == "affine_w4":
            zx = tl.load(ASUM + xm * groups + group, rm < M, 0).to(tl.float32)
            zw = tl.load(WZERO + gi, rn < N, 0).to(tl.float32)
            product = product - zx[:, None] * zw[None, :]
        acc += product * sx[:, None] * sw[None, :]
    if SPLIT_K > 1:
        tl.store(
            Y + tl.program_id(2) * M * N + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )
        return
    if GATE_OUT:
        ga, up = tl.split(acc.reshape((BM, 2, 128)).permute(0, 2, 1))
        activation = _gated_fp16(ga, up, GATE_LIMIT)
        scale = tl.maximum(tl.div_rn(tl.max(tl.abs(activation), 1), 127.0), 1.0e-30)
        code = tldevice.nearbyint(tl.div_rn(activation, scale[:, None])).to(tl.int8)
        out_k: tl.constexpr = N // 2
        cg = tl.program_id(1)
        cols = out_col + tl.arange(0, 128)
        tl.store(
            Y + rm[:, None].to(tl.int64) * out_k + cols[None, :], code, rm[:, None] < M
        )
        tl.store(YS + rm * (out_k // 128) + cg, scale, rm < M)
        tl.store(YSUM + rm * (out_k // 128) + cg, tl.sum(code.to(tl.int32), 1), rm < M)
        return
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


@triton.jit
def _route_product(
    Y, WEIGHTS, token, cols, route: tl.constexpr, N: tl.constexpr, TOPK: tl.constexpr
):
    value = tl.load(Y + (token * TOPK + route) * N + cols, cols < N, 0)
    return value * tl.load(WEIGHTS + token * TOPK + route).to(tl.float32)


@triton.jit
def _combine_experts(Y, WEIGHTS, OUT, N: tl.constexpr, B: tl.constexpr, TOPK: tl.constexpr = 9):
    token = tl.program_id(0)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    # Match the four lane-local accumulators used by the native reduction.
    zero = tl.full((B,), 0.0, tl.float32)
    a0 = zero
    a1 = zero
    a2 = zero
    a3 = zero
    for i in tl.static_range(0, TOPK, 4):
        a0 += _route_product(Y, WEIGHTS, token, cols, i, N, TOPK)
        if i + 1 < TOPK:
            a1 += _route_product(Y, WEIGHTS, token, cols, i + 1, N, TOPK)
        if i + 2 < TOPK:
            a2 += _route_product(Y, WEIGHTS, token, cols, i + 2, N, TOPK)
        if i + 3 < TOPK:
            a3 += _route_product(Y, WEIGHTS, token, cols, i + 3, N, TOPK)
    result = ((a0+a1)+a2)+a3
    result = tl.where(result == 0.0, 0.0, result)
    tl.store(OUT + token * N + cols, result, cols < N)


@triton.jit
def _split_reduce(
    P,
    Y,
    YS,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    RELU2: tl.constexpr,
    QUANT_OUT: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.program_id(1) * 128 + tl.arange(0, 128)
    acc = tl.full((128,), 0, tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(P + s * M * N + row * N + n, n < N, 0)
    if RELU2:
        acc = tl.maximum(acc, 0.0)
        acc *= acc
    if QUANT_OUT:
        sy = tl.maximum(tl.div_rn(tl.max(tl.abs(acc), 0), 127.0), 1.0e-30)
        q = tldevice.nearbyint(tl.div_rn(acc, sy)).to(tl.int8)
        tl.store(Y + row * N + n, q, n < N)
        tl.store(YS + row * tl.cdiv(N, 128) + tl.program_id(1), sy)
    else:
        tl.store(Y + row * N + n, acc, n < N)


@triton.jit
def _prepare_weight_scales(
    W,
    GRID,
    WS,
    WG,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    START: tl.constexpr,
    LENGTH: tl.constexpr,
    CODEC: tl.constexpr,
    WSTRIDE: tl.constexpr,
    ROUTED: tl.constexpr,
    WG_PER_ROW: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    expert = tile // tl.cdiv(N, 16)
    rn = (tile % tl.cdiv(N, 16)) * 16 + tl.arange(0, 16)
    local_k = group * 128 + tl.arange(0, 128)
    if ROUTED:
        W += expert * WSTRIDE
    if CODEC == "iq2_xxs":
        w = _decode(W, GRID, rn, START + local_k, N, K)
    elif CODEC == "q8_0":
        w = _decode_q8(W, rn, START + local_k, N, K)
    else:
        w = _decode_nvfp4(W, WS, WG, rn, START + local_k, N, K, WG_PER_ROW)
    w = tl.where(local_k[None, :] < LENGTH, w, 0.0)
    sw = tl.maximum(tl.div_rn(tl.max(tl.abs(w), 1), 127.0), 1.0e-30)
    tl.store(OUT + (expert * N + rn) * tl.cdiv(LENGTH, 128) + group, sw, rn < N)
