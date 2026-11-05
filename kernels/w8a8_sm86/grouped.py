"""W8A8 with per-token, per-group activation scales: one scale per token and per G input channels (G = 128).

$$ s^x_{m,g} = \\frac{\\max_{k \\in g} |X_{mk}|}{127}, \\qquad
   Y_{mn} = s^w_n \\sum_g s^x_{m,g} \\sum_{k \\in g} q^x_{mk}\\, q^w_{nk} $$

Per-token scales leave most channels of a token a few INT8 levels when one channel of that token is large; at Kev-9B's
down_proj and out_proj inputs that happens token by token, not in fixed channels, so SmoothQuant cannot remove it
(diagnose.py: relative error 0.06 per token, 0.02 per 128-group). The GEMM takes K blocks of exactly G channels: each
block's int32 partial sum is exact, then is scaled by s^x[m, g] and added to an fp32 accumulator. torch._int_mm cannot
do that, so this path is Triton at every M. The weight stays per output channel (linear.W8A8Weight).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .quant import EPS, _round_away, _rows

GROUP = 128


# ------------------------------------------------------------------ torch reference (any device)

def quant_groups_reference(x: torch.Tensor, group: int = GROUP, col_scale: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
  """x [M, K] -> (int8 [M, K], fp32 scales [M, K // group]); quant.quant_rows_reference per (token, group)."""
  m, k = x.shape
  if k % group:
    raise ValueError(f"group {group} must divide K = {k}")
  xf = (x.float() if col_scale is None else x.float() * col_scale.float()).view(m, k // group, group)
  amax = xf.abs().amax(-1).clamp_min(EPS)
  c127 = torch.full_like(amax, 127.0)
  q = _round_away(xf * (c127 / amax)[..., None]).clamp_(-127, 127).to(torch.int8)
  return q.view(m, k), amax / c127


def gemm_groups_reference(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor,
                          group: int = GROUP) -> torch.Tensor:
  """The exact per-group int sums in float64, scaled per (token, group) and per output channel, rounded once to bf16."""
  m, k = xq.shape
  xg = xq.double().view(m, k // group, group)
  wg = wq.double().view(wq.shape[0], k // group, group)
  acc = torch.einsum("mgk,ngk->mng", xg, wg)
  return ((acc * xs.double()[:, None, :]).sum(-1) * ws.double()[None, :]).to(torch.bfloat16)


# ------------------------------------------------------------------ Triton

@triton.jit
def _quantise_groups(v, offs, mask, q_row, s_row, NG, G: tl.constexpr, NGB: tl.constexpr):
  v2 = tl.reshape(v, (NGB, G))
  amax = tl.maximum(tl.max(tl.abs(v2), 1), 1e-8)
  q = libdevice.round(v2 * tl.math.div_rn(127.0, amax)[:, None])
  tl.store(q_row + offs, tl.reshape(q, (NGB * G,)).to(tl.int8), mask=mask)
  g = tl.arange(0, NGB)
  tl.store(s_row + g, tl.math.div_rn(amax, 127.0), mask=g < NG)


@triton.jit
def _quant_groups_kernel(x_ptr, c_ptr, q_ptr, s_ptr, K, NG, stride_x, stride_q, G: tl.constexpr, NGB: tl.constexpr,
                         HAS_COL: tl.constexpr):
  row = tl.program_id(0).to(tl.int64)
  offs = tl.arange(0, NGB * G)
  mask = offs < K
  v = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
  if HAS_COL:
    v = v * tl.load(c_ptr + offs, mask=mask, other=0.0)
  _quantise_groups(v, offs, mask, q_ptr + row * stride_q, s_ptr + row * NG, NG, G, NGB)


@triton.jit
def _swiglu_quant_groups_kernel(g_ptr, u_ptr, q_ptr, s_ptr, K, NG, stride_g, stride_u, stride_q, G: tl.constexpr,
                                NGB: tl.constexpr):
  row = tl.program_id(0).to(tl.int64)
  offs = tl.arange(0, NGB * G)
  mask = offs < K
  a = tl.load(g_ptr + row * stride_g + offs, mask=mask, other=0.0).to(tl.float32)
  u = tl.load(u_ptr + row * stride_u + offs, mask=mask, other=0.0).to(tl.float32)
  _quantise_groups(a * tl.sigmoid(a) * u, offs, mask, q_ptr + row * stride_q, s_ptr + row * NG, NG, G, NGB)


def _launch(k: int, group: int) -> tuple[int, int, int]:
  if k % group:
    raise ValueError(f"group {group} must divide K = {k}")
  ng = k // group
  ngb = triton.next_power_of_2(ng)
  return ng, ngb, max(4, min(16, ngb * group // 1024))


def quant_groups(x: torch.Tensor, group: int = GROUP, col_scale: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
  """quant_groups_reference on the GPU, one program per row."""
  x = _rows(x)
  m, k = x.shape
  ng, ngb, warps = _launch(k, group)
  q = torch.empty((m, k), dtype=torch.int8, device=x.device)
  s = torch.empty((m, ng), dtype=torch.float32, device=x.device)
  col = None if col_scale is None else col_scale.float().contiguous()
  if m:
    _quant_groups_kernel[(m,)](x, x if col is None else col, q, s, k, ng, x.stride(0), q.stride(0), G=group, NGB=ngb,
                               HAS_COL=col is not None, num_warps=warps)
  return q, s


def swiglu_quant_groups(gate: torch.Tensor, up: torch.Tensor, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
  """quant_groups(SiLU(gate) * up) in one pass (fp32 product, no bf16 intermediate)."""
  gate, up = _rows(gate), _rows(up)
  m, k = gate.shape
  ng, ngb, warps = _launch(k, group)
  q = torch.empty((m, k), dtype=torch.int8, device=gate.device)
  s = torch.empty((m, ng), dtype=torch.float32, device=gate.device)
  if m:
    _swiglu_quant_groups_kernel[(m,)](gate, up, q, s, k, ng, gate.stride(0), up.stride(0), q.stride(0), G=group, NGB=ngb,
                                      num_warps=warps)
  return q, s


def _configs() -> list[triton.Config]:
  """BK is the group (128); tiles that fit sm_86's 99 KB of shared memory."""
  c = lambda bm, bn, s, w: triton.Config({"BM": bm, "BN": bn, "GROUP_M": 8}, num_stages=s, num_warps=w)
  return [c(128, 128, 3, 8), c(128, 256, 2, 8), c(256, 128, 2, 8), c(64, 128, 4, 4), c(128, 64, 4, 4), c(64, 64, 4, 4),
          c(32, 64, 4, 4), c(16, 128, 4, 4)]


@triton.autotune(configs=_configs(), key=["M_CLASS", "N", "K"])
@triton.jit
def _w8a8_groups_kernel(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, M, N, K, NG, M_CLASS, stride_am, stride_bn, stride_cm,
                        BK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
  pid = tl.program_id(0)
  num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
  width = GROUP_M * num_n                               # L2 swizzle, as gemm._w8a8_kernel
  first_m = (pid // width) * GROUP_M
  size_m = min(num_m - first_m, GROUP_M)
  pid_m = first_m + (pid % width) % size_m
  pid_n = (pid % width) // size_m

  rm = pid_m * BM + tl.arange(0, BM)
  rn = pid_n * BN + tl.arange(0, BN)
  ram = tl.max_contiguous(tl.multiple_of(rm % M, BM), BM).to(tl.int64)
  rbn = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN).to(tl.int64)
  rk = tl.arange(0, BK)
  a = a_ptr + ram[:, None] * stride_am + rk[None, :]
  b = b_ptr + rbn[None, :] * stride_bn + rk[:, None]
  sa = sa_ptr + ram * NG
  acc = tl.zeros((BM, BN), dtype=tl.float32)
  for g in range(0, NG):                                # K = NG * BK exactly: no K mask
    part = tl.dot(tl.load(a), tl.load(b), out_dtype=tl.int32)
    acc += part.to(tl.float32) * tl.load(sa + g)[:, None]
    a += BK
    b += BK

  sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
  c = c_ptr + rm.to(tl.int64)[:, None] * stride_cm + rn[None, :]
  tl.store(c, (acc * sb[None, :]).to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemm_groups(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, group: int = GROUP) -> torch.Tensor:
  """bf16 [M, N] = ws[n] * sum_g xs[m, g] * (xq[m, g] @ wq[n, g]^T), int8 rows unit-stride, K a multiple of group."""
  from .gemm import m_class
  m, k = xq.shape
  n = wq.shape[0]
  assert wq.shape[1] == k and k % group == 0 and xs.shape == (m, k // group), "grouped operands do not match"
  c = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
  if m:
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    _w8a8_groups_kernel[grid](xq, wq, c, xs.contiguous(), ws, m, n, k, k // group, m_class(m), xq.stride(0), wq.stride(0),
                              c.stride(0), BK=group)
  return c
