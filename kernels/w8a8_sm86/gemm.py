"""The INT8 GEMM of W8A8 on Ampere (sm_86): int8 x int8 -> int32 on the tensor cores, dequantised in the epilogue.

$$ Y_{mn} = s^x_m\\, s^w_n \\sum_k q^x_{mk}\\, q^w_{nk} \\quad (\\text{int32 accumulate, one bf16 rounding}) $$

Two ways to run it:

* ``int8_gemm_dequant``: a Triton kernel (``mma.sync`` m16n8k32 s8 on sm_86) that keeps the int32 tile in registers and
  writes bf16 directly. No int32 [M, N] round trip through memory: at M = 28,672 and N = 24,576 that buffer alone is
  2.8 GB, which a 24 GB card running Kev-9B does not have to spare.
* ``int_mm_dequant``: ``torch._int_mm`` (cuBLASLt, IMMA) into an int32 buffer, then quant.dequant. cuBLASLt wants
  M > 16 and K, N multiples of 8; short inputs are zero-padded to 32 rows.

Both are exact up to the final rounding (integer sums do not depend on the tiling or the split), so the backend and the
autotuned tile change the speed, never the numbers.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .quant import dequant


def _configs() -> list[triton.Config]:
  """Tiles that fit sm_86's 99 KB of shared memory per block (int8 tiles: BM*BK + BN*BK bytes per stage)."""
  c = lambda bm, bn, bk, s, w: triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8}, num_stages=s, num_warps=w)
  return [c(128, 256, 64, 3, 8), c(256, 128, 64, 3, 8), c(128, 128, 128, 3, 8), c(128, 128, 64, 4, 4),
      c(128, 128, 64, 4, 8), c(64, 128, 128, 4, 4), c(128, 64, 128, 4, 4), c(64, 64, 128, 4, 4), c(32, 64, 128, 4, 4),
      c(16, 128, 128, 4, 4)]


def m_class(m: int) -> int:
  """The autotuning bucket of the row count: one tuning per bucket, not per M (Kev's question rows vary M per call)."""
  return min(max(triton.next_power_of_2(m), 16), 32768)


@triton.autotune(configs=_configs(), key=["M_CLASS", "N", "K"])
@triton.heuristics({"EVEN_K": lambda a: a["K"] % a["BK"] == 0})
@triton.jit
def _w8a8_kernel(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, M, N, K, M_CLASS, stride_am, stride_bn, stride_cm,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
  pid = tl.program_id(0)
  num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
  group = GROUP_M * num_n                               # L2 swizzle: GROUP_M row tiles walk the columns together
  first_m = (pid // group) * GROUP_M
  size_m = min(num_m - first_m, GROUP_M)
  pid_m = first_m + (pid % group) % size_m
  pid_n = (pid % group) // size_m

  rm = pid_m * BM + tl.arange(0, BM)
  rn = pid_n * BN + tl.arange(0, BN)
  ram = tl.max_contiguous(tl.multiple_of(rm % M, BM), BM).to(tl.int64)       # wrapped rows: loads need no M/N mask
  rbn = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN).to(tl.int64)
  rk = tl.arange(0, BK)
  a = a_ptr + ram[:, None] * stride_am + rk[None, :]
  b = b_ptr + rbn[None, :] * stride_bn + rk[:, None]                         # W is [N, K]: a K-major [BK, BN] tile
  acc = tl.zeros((BM, BN), dtype=tl.int32)
  for k in range(0, tl.cdiv(K, BK)):
    if EVEN_K:
      ta, tb = tl.load(a), tl.load(b)
    else:
      left = K - k * BK
      ta = tl.load(a, mask=rk[None, :] < left, other=0)
      tb = tl.load(b, mask=rk[:, None] < left, other=0)
    acc = tl.dot(ta, tb, acc, out_dtype=tl.int32)
    a += BK
    b += BK

  sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
  sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
  y = (acc.to(tl.float32) * sa[:, None] * sb[None, :]).to(tl.bfloat16)
  c = c_ptr + rm.to(tl.int64)[:, None] * stride_cm + rn[None, :]
  tl.store(c, y, mask=(rm[:, None] < M) & (rn[None, :] < N))


def int8_gemm_dequant(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor,
                      out: torch.Tensor | None = None) -> torch.Tensor:
  """bf16 [M, N] = xs[m] * ws[n] * (xq [M, K] int8 @ wq [N, K]^T int8). Rows of xq and wq must be unit-stride."""
  m, k = xq.shape
  n = wq.shape[0]
  assert wq.shape[1] == k and xq.stride(1) == 1 and wq.stride(1) == 1, "int8 operands need unit-stride rows of equal K"
  c = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device) if out is None else out
  if m:
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    _w8a8_kernel[grid](xq, wq, c, xs, ws, m, n, k, m_class(m), xq.stride(0), wq.stride(0), c.stride(0))
  return c


def int_mm_dequant(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
  """The cuBLASLt path: torch._int_mm into int32, then the separate dequant kernel."""
  m = xq.shape[0]
  if m <= 16:                                            # cuBLASLt int8 needs M > 16
    xq = torch.cat([xq, xq.new_zeros(32 - m, xq.shape[1])])
  acc = torch._int_mm(xq.contiguous(), wq.t())
  return dequant(acc[:m], xs, ws)
