# SPDX-License-Identifier: Apache-2.0
"""Bounded, frequency-ranked INT8 GS128 copies of routed IQ2 experts."""

import os
import weakref

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as lib

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_packed_g128 import _decode


@triton.jit
def _update_routes(
    IDS,
    COUNTS,
    PHASE,
    MAP,
    SLOTS,
    CHANGED,
    R: tl.constexpr,
    E: tl.constexpr,
    ACTUAL: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    PERIOD: tl.constexpr,
):
    i = tl.arange(0, B)
    ids = tl.load(IDS + i, i < R, 0)
    valid = (i < R) & (ids >= 0) & (ids < ACTUAL)
    hist = tl.histogram(tl.where(valid, ids, 0), E, mask=valid)
    e = tl.arange(0, E)
    counts = tl.load(COUNTS + e) + hist
    phase = tl.load(PHASE)
    if phase == 0:
        # Unique scores break ties toward lower real expert IDs. Padding must
        # never enter the cache (layers have 240..496 experts, not 512).
        score = tl.where(e < ACTUAL, (counts.to(tl.int64) + 1) * E + (E - 1 - e), -1)
        ranked = tl.sort(score, descending=True)
        cutoff = tl.sum(tl.where(e == S - 1, ranked, 0), 0)
        previous_map = tl.load(MAP + e)
        previous_ids = tl.load(SLOTS + e, e < S, -1)
        previous_scores = tl.gather(score, tl.maximum(previous_ids, 0), 0)
        keep = (e < S) & (previous_ids >= 0) & (previous_scores >= cutoff)

        # Retain surviving experts in their existing slots. Only newly admitted
        # experts need expansion; a change in popularity order copies no data.
        candidates = tl.sort(
            tl.where((score >= cutoff) & (previous_map < 0), score, -1),
            descending=True,
        )
        free_rank = tl.cumsum(((e < S) & ~keep).to(tl.int32), 0) - 1
        candidate = tl.gather(candidates, tl.maximum(free_rank, 0), 0)
        candidate_id = (E - 1 - candidate % E).to(tl.int32)
        new_ids = tl.where(keep, previous_ids, candidate_id)
        tl.store(CHANGED + e, (new_ids != previous_ids).to(tl.int32), e < S)
        tl.store(SLOTS + e, new_ids, e < S)
        tl.store(MAP + e, -1)
        tl.debug_barrier()
        tl.store(MAP + new_ids, e, e < S)
        counts = counts // 2
    tl.store(COUNTS + e, counts)
    tl.store(PHASE, (phase + 1) % PERIOD)


@triton.jit
def _refresh_weights(
    UP,
    DOWN,
    GRID,
    UP_GS,
    DOWN_GS,
    UP_OUT,
    DOWN_OUT,
    PHASE,
    SLOTS,
    CHANGED,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    DOWN_K: tl.constexpr,
    START: tl.constexpr,
    S: tl.constexpr,
    PERIOD: tl.constexpr,
):
    if tl.load(PHASE) == 1 % PERIOD:
        up_nr, up_ng = tl.cdiv(I, 16), tl.cdiv(H, 128)
        down_nr, down_ng = tl.cdiv(H, 16), tl.cdiv(I, 128)
        up_tiles = up_nr * up_ng
        per_slot = up_tiles + down_nr * down_ng
        for tile in range(tl.program_id(0), S * per_slot, tl.num_programs(0)):
            slot = tile // per_slot
            if tl.load(CHANGED + slot) != 0:
                expert = tl.load(SLOTS + slot)
                local_tile = tile % per_slot
                if local_tile < up_tiles:
                    group = local_tile % up_ng
                    rows = local_tile // up_ng * 16 + tl.arange(0, 16)
                    cols = group * 128 + tl.arange(0, 128)
                    values = _decode(UP, GRID, expert * I + rows, cols, E * I, H)
                    scale = tl.load(
                        UP_GS + (expert * I + rows) * up_ng + group, rows < I, 1.0
                    )
                    q = lib.nearbyint(values * tl.div_rn(1.0, scale[:, None])).to(
                        tl.int8
                    )
                    tl.store(
                        UP_OUT + (slot * I + rows[:, None]) * H + cols[None, :],
                        q,
                        (rows[:, None] < I) & (cols[None, :] < H),
                    )
                else:
                    local_tile -= up_tiles
                    group = local_tile % down_ng
                    rows = local_tile // down_ng * 16 + tl.arange(0, 16)
                    cols = group * 128 + tl.arange(0, 128)
                    values = _decode(
                        DOWN, GRID, expert * H + rows, START + cols, E * H, DOWN_K
                    )
                    scale = tl.load(
                        DOWN_GS + (expert * H + rows) * down_ng + group,
                        rows < H,
                        1.0,
                    )
                    q = lib.nearbyint(values * tl.div_rn(1.0, scale[:, None])).to(
                        tl.int8
                    )
                    tl.store(
                        DOWN_OUT + (slot * H + rows[:, None]) * I + cols[None, :],
                        q,
                        (rows[:, None] < H) & (cols[None, :] < I),
                    )


class CachedProjection:
    def __init__(self, weight, mapping):
        self.weight = weight
        self.mapping = mapping


class ExpertCache:
    def __init__(self, up, down, grid, intermediate, start, slots, period):
        from .gemm_packed_g128 import _cached_weight_scales

        self.source_up, self.source_down, self.grid = up, down, grid
        self.experts, self.intermediate = up.shape[:2]
        self.hidden = down.shape[1]
        self.start, self.slots, self.period = start, slots, period
        self.down_k = down.shape[2] * 256
        padded = triton.next_power_of_2(self.experts)
        device = up.device
        self.counts = torch.zeros(padded, device=device, dtype=torch.int32)
        self.phase = torch.zeros((), device=device, dtype=torch.int32)
        self.mapping = torch.full((padded,), -1, device=device, dtype=torch.int32)
        self.ids = torch.full((slots,), -1, device=device, dtype=torch.int32)
        self.changed = torch.zeros(slots, device=device, dtype=torch.int32)
        self.up = CachedProjection(
            torch.empty(
                (slots, intermediate, self.hidden), device=device, dtype=torch.int8
            ),
            self.mapping,
        )
        self.down = CachedProjection(
            torch.empty(
                (slots, self.hidden, intermediate), device=device, dtype=torch.int8
            ),
            self.mapping,
        )
        self.up_groups = _cached_weight_scales(
            up,
            grid,
            grid,
            grid,
            intermediate,
            self.hidden,
            0,
            self.hidden,
            "iq2_xxs",
            routed=True,
        )
        self.down_groups = _cached_weight_scales(
            down,
            grid,
            grid,
            grid,
            self.hidden,
            self.down_k,
            start,
            intermediate,
            "iq2_xxs",
            routed=True,
        )

    def step(self, ids):
        ids = ids.contiguous()
        _update_routes[(1,)](
            ids,
            self.counts,
            self.phase,
            self.mapping,
            self.ids,
            self.changed,
            ids.numel(),
            self.counts.numel(),
            self.experts,
            self.slots,
            triton.next_power_of_2(ids.numel()),
            self.period,
            num_warps=4,
        )
        _refresh_weights[(120,)](
            self.source_up,
            self.source_down,
            self.grid,
            self.up_groups,
            self.down_groups,
            self.up.weight,
            self.down.weight,
            self.phase,
            self.ids,
            self.changed,
            self.experts,
            self.hidden,
            self.intermediate,
            self.down_k,
            self.start,
            self.slots,
            self.period,
            num_warps=4,
            enable_fp_fusion=False,
        )


# Keep allocations alive across diagnostic module reloads and graph replays.
_CACHE = globals().get("_CACHE", {})
_BYTES = globals().get("_BYTES", {})
_LAYERS = globals().get("_LAYERS", {})


def register_layer(up, down, intermediate, start):
    """Register geometry before warmup so auto capacity covers every layer."""
    key = (up.device, up.data_ptr(), down.data_ptr(), start, intermediate)
    if key not in _LAYERS:
        if any(k[0] == up.device for k in _CACHE):
            raise RuntimeError(
                "Register all expert layers before allocating auto caches"
            )
        _LAYERS[key] = (weakref.ref(up), up.shape[0], intermediate, down.shape[1])


def plan_slots(device, budget):
    """Equalize layer coverage, then fill spare bytes with the smallest pairs."""
    layers = {k: v for k, v in _LAYERS.items() if k[0] == device and v[0]() is not None}
    if not layers:
        return {}

    def size(v, slots):
        _, experts, intermediate, hidden = v
        if not slots:
            return 0
        return (
            2 * slots * intermediate * hidden
            + (2 * triton.next_power_of_2(experts) + 2 * slots + 1) * 4
        )

    lo, hi = 0, max(v[1] for v in layers.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(size(v, min(mid, v[1])) for v in layers.values()) <= budget:
            lo = mid
        else:
            hi = mid - 1
    slots = {k: min(lo, v[1]) for k, v in layers.items()}
    remaining = budget - sum(size(layers[k], s) for k, s in slots.items())
    # Registration order is model order on every TP rank; stable sorting keeps
    # allocation deterministic even for layers with identical geometry.
    for k, v in sorted(layers.items(), key=lambda item: 2 * item[1][2] * item[1][3]):
        current = slots[k]
        step = 2 * v[2] * v[3] + 8
        if not current:
            first = size(v, 1)
            if remaining < first:
                continue
            slots[k] = current = 1
            remaining -= first
        extra = min(v[1] - current, remaining // step)
        slots[k] += extra
        remaining -= extra * step
    return slots


def get_cache(up, down, grid, intermediate, start):
    budget = int(float(os.environ.get("VLLM_BOLT_EXPERT_CACHE_GIB", "0")) * 2**30)
    if budget <= 0:
        return None
    key = (up.device, up.data_ptr(), down.data_ptr(), start, intermediate)
    if key in _CACHE:
        return _CACHE[key]
    setting = os.environ.get("VLLM_BOLT_EXPERT_CACHE_SLOTS", "64")
    if setting == "auto":
        if key not in _LAYERS:
            register_layer(up, down, intermediate, start)
        slots = plan_slots(up.device, budget).get(key, 0)
        if slots == 0:
            return None
    else:
        slots = min(int(setting), up.shape[0])
    period = int(os.environ.get("VLLM_BOLT_EXPERT_CACHE_REFRESH", "128"))
    if slots < 1 or period < 1 or period > 1024:
        raise ValueError("Expert cache requires positive slots and refresh in 1..1024")
    if (
        up.ndim != 4
        or down.ndim != 4
        or up.shape[0] != down.shape[0]
        or up.shape[-1] != 66
        or down.shape[-1] != 66
        or up.shape[1] != intermediate
        or up.shape[2] * 256 != down.shape[1]
        or intermediate <= 0
        or start < 0
        or intermediate % 4
        or start % 4
        or start + intermediate > down.shape[2] * 256
        or not up.is_contiguous()
        or not down.is_contiguous()
        or up.device != down.device
    ):
        raise ValueError("Unsupported IQ2 expert cache geometry")
    # Weight payloads plus the shared frequency/mapping metadata; scales are
    # already allocated by the ordinary GS128 path and are not duplicated.
    size = 2 * slots * intermediate * down.shape[1]
    size += (2 * triton.next_power_of_2(up.shape[0]) + 2 * slots + 1) * 4
    if (
        _BYTES.get(up.device, 0) + size > budget
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    cache = ExpertCache(up, down, grid, intermediate, start, slots, period)
    _CACHE[key] = cache
    _BYTES[up.device] = _BYTES.get(up.device, 0) + size
    return cache


def find_projection(packed, start, length, *, down):
    """Find a resident projection without allocating or changing admission."""
    for entry in _CACHE.values():
        if down:
            if (
                packed is entry.source_down
                and start == entry.start
                and length == entry.intermediate
            ):
                return entry.down
        elif packed is entry.source_up and start == 0 and length == entry.hidden:
            return entry.up
    return None
