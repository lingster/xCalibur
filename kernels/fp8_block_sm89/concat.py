"""Concat groups: GEMMs that read the same input become one GEMM over their concatenated FP8 weights.

q|k|v (12288 + 1024 + 1024 = 14336), gate|up (34816) and in_proj_qkv|z (16384) all have N a multiple of 128, so
the 128x128 weight scale blocks concatenate along N with no block straddling two projections. The activation
quantisation is per row and per K group, independent of N, so every output element sees the same quantised
inputs, the same scales and the same K loop: concatenation is exact (bitwise for the same tile config). It cuts
launches (3 -> 1 for attention, 2 -> 1 for MLP and DeltaNet) and enlarges the grid for skinny M.

Works with any GEMM of the stock contract ``gemm(x, w, ws) -> y``: the stock hub kernel included, so this lever needs
no new kernel at all.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from space import QGROUP


@dataclass(frozen=True)
class Concat:
  w: torch.Tensor                     # [sum N, K] fp8
  ws: torch.Tensor                    # [sum N / 128, K / 128] fp32
  splits: tuple[int, ...]             # each member's N


def concat_weights(members: list[tuple[torch.Tensor, torch.Tensor]]) -> Concat:
  for w, _ in members:
    if w.shape[0] % QGROUP:
      raise ValueError(f"N={w.shape[0]} is not a multiple of the {QGROUP}-row scale block: concat would straddle")
  return Concat(torch.cat([w for w, _ in members]).contiguous(), torch.cat([s for _, s in members]).contiguous(),
                tuple(w.shape[0] for w, _ in members))


def forward(x: torch.Tensor, cat: Concat, gemm: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
  """One GEMM, then per-member views of its output (no copy)."""
  return list(torch.split(gemm(x, cat.w, cat.ws), cat.splits, dim=-1))
