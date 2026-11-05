"""The block-FP8 GEMM's math in plain torch (CPU or GPU): the contract both the hub kernel and ours implement.

  activations  per row, per 128-wide K group g:  s[m, g] = max|x[m, g*128:(g+1)*128]| / 448,
               q = e4m3(x / max(s, 1e-12))      (sm_89: truncate to fp16, then round to nearest even; |q| <= 448)
  weights      per 128 x 128 block (i, j):  W[n, k] = e4m3 code * Ws[n // 128, k // 128]   (Ws = "weight_scale_inv")
  output       C[m, n] = sum_g s[m, g] * Ws[n // 128, g] * sum_{k in g} q[m, k] * code[n, k]   (fp32), cast to bf16

Within a group the products of two e4m3 values are exact in fp32, so a kernel may differ from this reference only
by fp32 summation order and the final bf16 rounding.
"""
from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
GROUP = 128


def to_e4m3_sm89(y: torch.Tensor) -> torch.Tensor:
  """fp32 -> e4m3 as Triton lowers it on sm_89: truncate to fp16 (RZ), then round to nearest even. Differs from a
  single rounding only when y sits just above a tie by less than an fp16 ulp."""
  b = y.contiguous().view(torch.int32) & 0x7FFFFFFF
  t = torch.where(b >= 0x38800000, (b & -8192).view(torch.float32), torch.trunc(y.abs() * 2.0 ** 24) / 2.0 ** 24)
  return torch.where(y < 0, -t, t).to(FP8)


def quantise_activation(x: torch.Tensor, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
  """(q fp8 [M, K], s fp32 [M, K // group]): the hub's dynamic per-token-group quantisation, op for op."""
  xf = x.float().reshape(*x.shape[:-1], x.shape[-1] // group, group)
  s = xf.abs().amax(-1) / FP8_MAX
  q = to_e4m3_sm89(xf / s.clamp_min(1e-12).unsqueeze(-1))
  return q.reshape(x.shape), s


def quantise_weight(w: torch.Tensor, block: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
  """(codes fp8 [N, K], scales fp32 [N / block, K / block]) as transformers' FineGrainedFP8 quantiser stores them."""
  N, K = w.shape
  wb = w.float().reshape(N // block, block, K // block, block)
  s = wb.abs().amax(dim=(1, 3)) / FP8_MAX
  q = (wb / s.clamp_min(1e-12)[:, None, :, None]).to(FP8)
  return q.reshape(N, K), s


def dequantise(q: torch.Tensor, s: torch.Tensor, block_rows: int, block_cols: int = GROUP) -> torch.Tensor:
  """fp32 values from codes and scales; block_rows = 1 for activations, 128 for weights."""
  return q.float() * s.repeat_interleave(block_rows, 0)[: q.shape[0]].repeat_interleave(block_cols, 1)[:, : q.shape[1]]


def block_fp8_matmul(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor,
                     x_scale: torch.Tensor | None = None) -> torch.Tensor:
  """fp32 reference of C = quant(x) @ W^T; x is bf16 (quantised here) or fp8 with ``x_scale``."""
  q, s = (x, x_scale) if x_scale is not None else quantise_activation(x)
  return dequantise(q.reshape(-1, q.shape[-1]), s.reshape(-1, s.shape[-1]), 1) @ dequantise(w, ws, GROUP).T


def random_problem(M: int, N: int, K: int, device: str = "cpu", seed: int = 0):
  """bf16 activations with uneven group ranges and a block-quantised weight, like a real projection."""
  g = torch.Generator().manual_seed(seed)
  x = (torch.randn(M, K, generator=g) * torch.linspace(0.1, 4.0, K)).to(torch.bfloat16)
  w, ws = quantise_weight(torch.randn(N, K, generator=g) * 0.02)
  return x.to(device), w.to(device), ws.to(device)
