# SPDX-License-Identifier: Apache-2.0
"""Walsh-Hadamard rotation for quantized KV caches (llama.cpp #21038).

Rotate Q and K with the same orthonormal block H before the score, and V
before the quantized cache write; un-rotate the attention output afterwards
(V's rotation is undone because H is symmetric and HH = I):

    (Hq) . (Hk) = q.k        out <- H (H (P V)) = P V

H is the normalized Sylvester matrix over 64-wide blocks of the head dim.
Block size 64 is deliberate: upstream's follow-up found smaller matrices
quantize better than one head-sized transform.

Enabled by default for int8 per-token-head caches with head_dim % 64 == 0;
``VLLM_GFX908_KV_ROTATE=0`` disables it. The flag must not change while a
cache is live: rotated and unrotated entries cannot share a cache.

Scope: int8 only (the fork's other KV modes have their own mechanisms).
"""

import os

import torch

BLOCK = 64
_H64: dict = {}


def enabled(head_dim: int) -> bool:
    if os.environ.get("VLLM_GFX908_KV_ROTATE", "1") == "0":
        return False
    return head_dim % BLOCK == 0


def _h64(device) -> torch.Tensor:
    key = str(device)
    h = _H64.get(key)
    if h is None:
        h = torch.ones((1, 1), dtype=torch.float32, device=device)
        while h.shape[0] < BLOCK:
            h = torch.cat(
                [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
            )
        _H64[key] = h
    return h


def _torch_rotate(x: torch.Tensor) -> torch.Tensor:
    lead = x.shape[:-1]
    y = x.reshape(-1, BLOCK).float() @ _h64(x.device)
    y = y * (1.0 / BLOCK)
    return y.reshape(*lead, x.shape[-1]).to(x.dtype)


def rotate(x: torch.Tensor) -> torch.Tensor:
    """Forward (and inverse) normalized H; returns x.dtype."""
    from aiter.ops.int8_native import head_hadamard

    y = head_hadamard(x)
    return _torch_rotate(x) if y is None else y


def inverse(x: torch.Tensor) -> torch.Tensor:
    return rotate(x)


def inverse_(x: torch.Tensor) -> None:
    from aiter.ops.int8_native import head_hadamard

    if head_hadamard(x, out=x) is None:
        x.copy_(_torch_rotate(x))
