"""The W8A8 linear: y = x W^T with W int8 per output channel and x int8 per token, both symmetric.

  >>> wt = W8A8Weight.from_float(linear.weight)          # once: int8 [N, K] + fp32 scales [N]
  >>> y = w8a8_linear(x, wt)                             # every call: quantise x per token, int8 GEMM, bf16 out

Backends: "triton" (gemm.int8_gemm_dequant: dequant fused into the GEMM epilogue), "intmm" (torch._int_mm + a separate
dequant), "auto" (the default: triton below AUTO_INTMM_MIN_M rows, intmm from there) and "reference" (torch, float64 on
the exact int sum: an oracle that also runs on the CPU). Why "auto": on the RTX 3090, cuBLASLt's IMMA kernels win at
M >= 4,096 on five of Kev-9B's six shapes (by up to 31%), while at M = 512 the two split evenly and the fused epilogue
needs no int32 buffer (run 20261105T114150Z-c2de74, README). All backends give the same int32 sums, so the switch
moves only the time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from . import grouped, quant
from .gemm import int8_gemm_dequant, int_mm_dequant

Backend = Literal["auto", "triton", "intmm", "reference"]
AUTO_INTMM_MIN_M = 1024


def resolve_backend(backend: Backend, m: int) -> Backend:
  """"auto" -> "triton" for m < AUTO_INTMM_MIN_M rows, else "intmm"; any other backend as given."""
  if backend != "auto":
    return backend
  return "intmm" if m >= AUTO_INTMM_MIN_M else "triton"


@dataclass
class W8A8Weight:
  """A weight [N, K] as int8 codes and fp32 per-output-channel scales: W ≈ scale[:, None] * q."""
  q: torch.Tensor
  scale: torch.Tensor

  @classmethod
  def from_float(cls, w: torch.Tensor) -> "W8A8Weight":
    return cls(*quant.quant_weight(w.detach()))

  @property
  def out_features(self) -> int:
    return self.q.shape[0]

  @property
  def in_features(self) -> int:
    return self.q.shape[1]


def gemm_quantised(xq: torch.Tensor, xs: torch.Tensor, w: W8A8Weight, backend: Backend = "auto",
                   group: int | None = None) -> torch.Tensor:
  """The GEMM + dequant on an already quantised 2-D activation (for producers that quantise in their own epilogue,
  e.g. quant.swiglu_quant). group: xs holds per-(token, group) scales [M, K / group] (grouped.py: Triton at every M)."""
  if group is not None:
    if backend == "reference":
      return grouped.gemm_groups_reference(xq, xs, w.q, w.scale, group)
    return grouped.gemm_groups(xq, xs, w.q, w.scale, group)
  backend = resolve_backend(backend, xq.shape[0])
  if backend == "triton":
    return int8_gemm_dequant(xq, xs, w.q, w.scale)
  if backend == "intmm":
    return int_mm_dequant(xq, xs, w.q, w.scale)
  if backend == "reference":
    return quant.w8a8_reference_from_quantised(xq, xs, w.q, w.scale)
  raise ValueError(f"unknown backend {backend!r}")


def quantise_activation(x2: torch.Tensor, backend: Backend = "auto", col_scale: torch.Tensor | None = None,
                        group: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-token (group None) or per-(token, group) int8 codes and fp32 scales of x2 [M, K]."""
  if group is not None:
    if backend == "reference":
      return grouped.quant_groups_reference(x2, group, col_scale)
    return grouped.quant_groups(x2, group, col_scale)
  if backend == "reference":
    return quant.quant_rows_reference(x2, col_scale)
  return quant.quant_rows(x2, col_scale)


def w8a8_linear(x: torch.Tensor, w: W8A8Weight, backend: Backend = "auto", act_scale: torch.Tensor | None = None,
                group: int | None = None) -> torch.Tensor:
  """x [..., K] -> bf16 [..., N]: per-token quantisation, int8 GEMM with int32 accumulate, dequant to bf16.
  act_scale: fp32 [K] multiplied into x inside the quantiser (SmoothQuant's 1/s where it cannot be folded), or None.
  group: one activation scale per token and `group` channels instead of per token (grouped.py), or None."""
  lead = x.shape[:-1]
  x2 = x.reshape(-1, x.shape[-1])
  xq, xs = quantise_activation(x2, backend, act_scale, group)
  return gemm_quantised(xq, xs, w, backend, group).reshape(*lead, w.out_features)


class W8A8Linear(nn.Module):
  """A drop-in for a bias-free nn.Linear whose weight is held as int8 + scales (buffers: they follow .to()).
  act_scale: an optional fp32 [K] the quantiser multiplies into the input (SmoothQuant's 1/s, kev_smooth.py).
  group: per-(token, group) activation scales (grouped.py), or None for per token."""

  def __init__(self, w: W8A8Weight, backend: Backend = "auto", act_scale: torch.Tensor | None = None,
               group: int | None = None):
    super().__init__()
    self.register_buffer("q", w.q)
    self.register_buffer("scale", w.scale)
    self.register_buffer("act_scale", None if act_scale is None else act_scale.float().contiguous())
    self.backend: Backend = backend
    self.group = group
    self.in_features, self.out_features = w.in_features, w.out_features

  @classmethod
  def from_linear(cls, linear: nn.Module, backend: Backend = "auto") -> "W8A8Linear":
    if getattr(linear, "bias", None) is not None:
      raise ValueError("W8A8Linear covers bias-free linears only")
    return cls(W8A8Weight.from_float(linear.weight), backend)

  @property
  def weight8(self) -> W8A8Weight:
    return W8A8Weight(self.q, self.scale)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return w8a8_linear(x, self.weight8, self.backend, self.act_scale, self.group)

  def extra_repr(self) -> str:
    extra = (", act_scale" if self.act_scale is not None else "") + (f", group={self.group}" if self.group else "")
    return f"in_features={self.in_features}, out_features={self.out_features}, int8 W8A8, backend={self.backend}{extra}"
