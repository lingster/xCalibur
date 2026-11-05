"""Symmetric INT8 quantisers for W8A8, as Triton kernels, plus their torch reference.

Weights: per output channel (row of W [N, K]), offline.   Activations: per token (row of X [M, K]), every call.

$$ s_r = \\frac{\\max(\\lVert x_r \\rVert_\\infty,\\ \\varepsilon)}{127}, \\qquad
   q_{rj} = \\operatorname{round}\\!\\Big(\\frac{127\\, x_{rj}}{\\max(\\lVert x_r \\rVert_\\infty,\\ \\varepsilon)}\\Big) \\in [-127, 127] $$

round = half away from zero (CUDA ``roundf``), and both divisions are IEEE round-to-nearest, so the Triton kernels and
the torch reference agree on every scale and (up to a tie) every code. The GEMM then accumulates
$\\sum_j q^x_{mj} q^w_{nj}$ exactly in int32, and $y_{mn} = s^x_m\\, s^w_n \\cdot \\text{acc}_{mn}$ (gemm.py).

``swiglu_quant`` fuses the MLP's SiLU(gate)·up with the quantiser of the down projection's input: one read of gate and
up, one int8 write, instead of a bf16 round trip.

``col_scale`` (optional, fp32 [K]) multiplies each column before the absmax: SmoothQuant's $X \\operatorname{diag}(1/s)$
where the 1/s cannot be folded into the op that produces X (smooth.py, kev_smooth.py). One multiply per element.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

EPS = 1e-8


# ------------------------------------------------------------------ torch reference (any device)

def _round_away(v: torch.Tensor) -> torch.Tensor:
  return torch.where(v >= 0, torch.floor(v + 0.5), torch.ceil(v - 0.5))


def quant_rows_reference(x: torch.Tensor, col_scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  """x [M, K] float -> (int8 codes [M, K], fp32 scales [M]); the oracle for quant_rows and quant_weight. col_scale:
  fp32 [K], multiplied into x (in fp32) before the absmax."""
  xf = x.float() if col_scale is None else x.float() * col_scale.float()
  amax = xf.abs().amax(-1).clamp_min(EPS)
  c127 = torch.full_like(amax, 127.0)     # tensor / tensor: IEEE division (a CUDA tensor / python scalar multiplies by
  q = _round_away(xf * (c127 / amax)[:, None]).clamp_(-127, 127).to(torch.int8)   # the reciprocal, 1 ulp off div_rn)
  return q, amax / c127


def quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """W [N, K] -> (int8 [N, K], fp32 per-output-channel scales [N]). Offline, so the torch path is enough; done in
  row chunks to bound the fp32 temporary on big weights."""
  qs, ss = [], []
  for chunk in w.split(4096, 0):
    q, s = quant_rows_reference(chunk)
    qs.append(q); ss.append(s)
  return torch.cat(qs).contiguous(), torch.cat(ss).contiguous()


def w8a8_reference_from_quantised(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
  """The exact int sum in float64 (|acc| <= K * 127^2 << 2^53), scaled, rounded once to bf16."""
  acc = xq.double() @ wq.double().T
  return (acc * xs.double()[:, None] * ws.double()[None, :]).to(torch.bfloat16)


def w8a8_reference(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
  xq, xs = quant_rows_reference(x)
  return w8a8_reference_from_quantised(xq, xs, wq, ws)


# ------------------------------------------------------------------ Triton

@triton.jit
def _quantise(v, mask, q_row, s_ptr, row, offs):
  """Shared tail of both kernels: absmax of the fp32 row v, codes to q_row, scale to s_ptr[row]."""
  amax = tl.maximum(tl.max(tl.where(mask, tl.abs(v), 0.0), 0), 1e-8)
  q = libdevice.round(v * tl.math.div_rn(127.0, amax))
  tl.store(q_row + offs, q.to(tl.int8), mask=mask)
  tl.store(s_ptr + row, tl.math.div_rn(amax, 127.0))


@triton.jit
def _quant_rows_kernel(x_ptr, c_ptr, q_ptr, s_ptr, K, stride_x, stride_q, BLOCK: tl.constexpr, HAS_COL: tl.constexpr):
  row = tl.program_id(0).to(tl.int64)
  offs = tl.arange(0, BLOCK)
  mask = offs < K
  v = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
  if HAS_COL:
    v = v * tl.load(c_ptr + offs, mask=mask, other=0.0)
  _quantise(v, mask, q_ptr + row * stride_q, s_ptr, row, offs)


@triton.jit
def _swiglu_quant_kernel(g_ptr, u_ptr, q_ptr, s_ptr, K, stride_g, stride_u, stride_q, BLOCK: tl.constexpr):
  row = tl.program_id(0).to(tl.int64)
  offs = tl.arange(0, BLOCK)
  mask = offs < K
  g = tl.load(g_ptr + row * stride_g + offs, mask=mask, other=0.0).to(tl.float32)
  u = tl.load(u_ptr + row * stride_u + offs, mask=mask, other=0.0).to(tl.float32)
  _quantise(g * tl.sigmoid(g) * u, mask, q_ptr + row * stride_q, s_ptr, row, offs)


def _launch_shape(k: int) -> tuple[int, int]:
  block = triton.next_power_of_2(k)
  return block, max(4, min(16, block // 1024))


def _rows(x: torch.Tensor) -> torch.Tensor:
  return x if x.stride(-1) == 1 else x.contiguous()


def quant_rows(x: torch.Tensor, col_scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-token dynamic quantisation on the GPU: x [M, K] (bf16/fp16/fp32, unit column stride or copied) ->
  (int8 [M, K], fp32 [M]). One program per row; K up to 32768. col_scale: fp32 [K] multiplied in first, or None."""
  x = _rows(x)
  col = None if col_scale is None else col_scale.float().contiguous()
  m, k = x.shape
  q = torch.empty((m, k), dtype=torch.int8, device=x.device)
  s = torch.empty((m,), dtype=torch.float32, device=x.device)
  if m:
    block, warps = _launch_shape(k)
    _quant_rows_kernel[(m,)](x, x if col is None else col, q, s, k, x.stride(0), q.stride(0), BLOCK=block,
                             HAS_COL=col is not None, num_warps=warps)
  return q, s


def swiglu_quant(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """quant_rows(SiLU(gate) * up) in one pass, in fp32 (no bf16 rounding of the product). gate, up [M, K], e.g. the two
  strided halves of a fused gate|up output."""
  gate, up = _rows(gate), _rows(up)
  m, k = gate.shape
  q = torch.empty((m, k), dtype=torch.int8, device=gate.device)
  s = torch.empty((m,), dtype=torch.float32, device=gate.device)
  if m:
    block, warps = _launch_shape(k)
    _swiglu_quant_kernel[(m,)](gate, up, q, s, k, gate.stride(0), up.stride(0), q.stride(0), BLOCK=block, num_warps=warps)
  return q, s


@triton.jit
def _dequant_kernel(acc_ptr, sa_ptr, sb_ptr, c_ptr, M, N, stride_acc, stride_c, BM: tl.constexpr, BN: tl.constexpr):
  rm = tl.program_id(0) * BM + tl.arange(0, BM)
  rn = tl.program_id(1) * BN + tl.arange(0, BN)
  mask = (rm[:, None] < M) & (rn[None, :] < N)
  rm64 = rm.to(tl.int64)
  acc = tl.load(acc_ptr + rm64[:, None] * stride_acc + rn[None, :], mask=mask, other=0).to(tl.float32)
  sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
  sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
  tl.store(c_ptr + rm64[:, None] * stride_c + rn[None, :], (acc * sa[:, None] * sb[None, :]).to(tl.bfloat16), mask=mask)


def dequant(acc: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
  """int32 acc [M, N] -> bf16 sa[m] * sb[n] * acc: the separate epilogue for the torch._int_mm path."""
  m, n = acc.shape
  c = torch.empty((m, n), dtype=torch.bfloat16, device=acc.device)
  grid = (triton.cdiv(m, 32), triton.cdiv(n, 256))
  _dequant_kernel[grid](acc, sa, sb, c, m, n, acc.stride(0), c.stride(0), BM=32, BN=256, num_warps=8)
  return c
