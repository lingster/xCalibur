# Portions adapted from kernels-community/finegrained-fp8 @ 8c178950e8710e26a2210e4e909cd02dc16c8715
# (build/torch-cuda/matmul.py: w8a8_block_dynamic_fp8_matmul_kernel, _swizzle_offsets; utils.py: fp8_act_quant_inline),
# Copyright 2025-2026 The HuggingFace Inc. team, licensed under the Apache License, Version 2.0.
# Modified by xArmory: exposed tile knobs, a pre-quantised activation path, split-K, stand-in dot types.
"""Block-FP8 GEMM for sm_89: C = quant(A) @ W^T with 128x128 block weight scales, fp32 accumulate, bf16 out.

The hub kernel's math, op for op (see reference.py), with its tiling opened up:

  BLOCK_M / BLOCK_N / BLOCK_K / GROUP_M / num_warps / num_stages   free (the hub pins N = K = 128 and M by rule)
  quant = "fused"   A is bf16 and quantised per K tile in the loop (the hub's contract; needs BLOCK_K = 128)
  quant = "once"    A arrives fp8 with per-row, per-128-group fp32 scales (``act_quant``), shared across GEMMs
  SPLIT_K           K tiles split over a third grid axis; fp32 partials summed in a fixed order (deterministic)

``dot`` picks the matrix instruction: "fp8" (e4m3 mma.sync, sm_89), or a stand-in that runs the same quantised
values elsewhere: "bf16" on sm_86 (e4m3 values are exact in bf16, products exact in the fp32 accumulator) and
"fp32" (ieee) under Triton's CPU interpreter. Only fp32 summation order separates the three.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import triton
import triton.language as tl

from space import QGROUP, TileConfig

DOTS = {"fp8": 0, "bf16": 1, "fp32": 2}


@triton.jit
def _round_e4m3(y):
  """fp32 -> fp32 on the e4m3 grid exactly as sm_89 casts it (sass.py): truncate to fp16 (F2FP.F16.F32...RZ), then
  round to nearest even with saturation at 448 (F2FP.SATFINITE.E4M3.F16)."""
  m0 = y.to(tl.int32, bitcast=True) & 0x7FFFFFFF
  t16 = tl.where(m0 >= 0x38800000, (m0 & -8192).to(tl.float32, bitcast=True),         # fp16 normals: 10 bits
                 (tl.minimum(tl.abs(y), 1.0) * 16777216.0).to(tl.int32).to(tl.float32) / 16777216.0)  # 2^-24
  y = tl.where(y < 0, -t16, t16)
  mag = y.to(tl.int32, bitcast=True) & 0x7FFFFFFF
  nrm = ((mag + 0x7FFFF + ((mag >> 20) & 1)) & -1048576).to(tl.float32, bitcast=True)   # keep 3 mantissa bits
  v = tl.abs(y) * 512.0                                                                 # subnormals: ulp 2^-9
  sub = ((v + 8388608.0) - 8388608.0) / 512.0
  r = tl.minimum(tl.where(mag >= 0x3C800000, nrm, sub), 448.0)                         # 0x3C800000 = 2^-6
  return tl.where(y < 0, -r, r)


@triton.jit
def _encode_e4m3(r):
  """fp32 already on the e4m3 grid -> its uint8 code (sign | exponent + 7 | 3 mantissa bits)."""
  bits = r.to(tl.int32, bitcast=True)
  mag = bits & 0x7FFFFFFF
  nrm = (((mag >> 23) - 120) << 3) | ((mag >> 20) & 7)
  sub = (tl.abs(r) * 512.0).to(tl.int32)
  return (tl.where(mag >= 0x3C800000, nrm, sub) | ((bits >> 24) & 128)).to(tl.uint8)


@triton.jit
def _decode_e4m3(code):
  """uint8 e4m3 code -> exact fp32 value."""
  c = code.to(tl.int32)
  e = (c >> 3) & 15
  m = c & 7
  nrm = (((c & 128) << 24) | ((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
  sub = m.to(tl.float32) * 0.001953125
  return tl.where(e == 0, tl.where((c & 128) != 0, -sub, sub), nrm)


@triton.jit
def _quant(a_raw):
  """Per-row amax over the tile -> fp32 scale amax / 448, values scaled into e4m3 range (fp8_act_quant_inline)."""
  a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0
  return a_raw / tl.maximum(a_s[:, None], 1e-12), a_s


@triton.jit
def _dot(a, b, DOT: tl.constexpr):
  """DOT 0: native e4m3 operands. Stand-ins (operands are exact e4m3 values in fp32): 1 bf16 mma, 2 ieee fp32."""
  if DOT == 0:
    c = tl.dot(a, b)
  elif DOT == 1:
    c = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
  else:
    c = tl.dot(a, b, input_precision="ieee")
  return c


@triton.jit
def _act_quant_kernel(X, Y, S, M, stride_xm, stride_ym, stride_sm, ROWS: tl.constexpr, QG: tl.constexpr,
                      CODES: tl.constexpr):
  rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
  cols = tl.program_id(1) * QG + tl.arange(0, QG)
  mask = rows[:, None] < M
  x = tl.load(X + rows[:, None] * stride_xm + cols[None, :], mask=mask, other=0.0).to(tl.float32)
  y, s = _quant(x)
  if CODES:                                       # stand-in: sm_86 has no fp8e4nv type, store the codes as uint8
    q = _encode_e4m3(_round_e4m3(y))
  else:
    q = y.to(tl.float8e4nv)
  tl.store(Y + rows[:, None] * stride_ym + cols[None, :], q, mask=mask)
  tl.store(S + rows * stride_sm + tl.program_id(1), s, mask=rows < M)


@triton.jit
def _gemm_kernel(A, As, B, Bs, C, M, N, K,
                 stride_am, stride_ak, stride_asm, stride_ask, stride_bn, stride_bk, stride_bsn, stride_bsk,
                 stride_cs, stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                 SPLIT_K: tl.constexpr, FUSED: tl.constexpr, DOT: tl.constexpr, QG: tl.constexpr):
  # Tile schedule as the hub's _swizzle_offsets: swizzled (pid_m, pid_n) for L2 reuse of B, %-wrapped offsets.
  num_pid_m = tl.cdiv(M, BLOCK_M)
  num_pid_n = tl.cdiv(N, BLOCK_N)
  pid_m, pid_n = tl.swizzle2d(tl.program_id(0), tl.program_id(1), num_pid_m, num_pid_n, GROUP_M)
  offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
  offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
  offs_k = tl.arange(0, BLOCK_K)
  k_tiles = tl.cdiv(K, BLOCK_K)
  per_split = tl.cdiv(k_tiles, SPLIT_K)
  k_lo = tl.program_id(2) * per_split
  k_hi = tl.minimum(k_lo + per_split, k_tiles)
  bs_ptrs = Bs + (offs_n // QG) * stride_bsn

  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for kt in range(k_lo, k_hi):
    k0 = kt * BLOCK_K
    k_ok = offs_k < K - k0
    a = tl.load(A + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak, mask=k_ok[None, :], other=0.0)
    if FUSED:
      y, a_s = _quant(a.to(tl.float32))
      if DOT == 0:
        a = y.to(tl.float8e4nv)
      else:
        a = _round_e4m3(y)
    else:
      a_s = tl.load(As + offs_m * stride_asm + (k0 // QG) * stride_ask)
      if DOT != 0:
        a = _decode_e4m3(a)
    b = tl.load(B + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=k_ok[:, None], other=0.0)
    if DOT != 0:
      b = _decode_e4m3(b)
    b_s = tl.load(bs_ptrs + (k0 // QG) * stride_bsk)
    acc += _dot(a, b, DOT) * a_s[:, None] * b_s[None, :]

  offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  c_ptrs = C + tl.program_id(2) * stride_cs + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
  tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def interpreting() -> bool:
  """True when Triton runs kernels on the CPU interpreter (TRITON_INTERPRET=1 before import)."""
  return type(_gemm_kernel).__name__ == "InterpretedFunction"


def _codes(t: torch.Tensor, dot: str) -> torch.Tensor:
  """fp8 operands as Triton sees them: native e4m3 for the fp8 dot, raw uint8 codes for the stand-ins."""
  return t if dot == "fp8" or t.dtype != torch.float8_e4m3fn else t.view(torch.uint8)


def default_dot(device: torch.device | str) -> str:
  if interpreting() or torch.device(device).type != "cuda":
    return "fp32"
  return "fp8" if torch.cuda.get_device_capability(device) >= (8, 9) else "bf16"


def act_quant(x: torch.Tensor, rows: int = 16, dot: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantise activations once: (fp8 [..., K], fp32 scales [..., K // 128])."""
  x2 = x.reshape(-1, x.shape[-1])
  M, K = x2.shape
  if K % QGROUP:
    raise ValueError(f"K={K} is not a multiple of the quant group {QGROUP}")
  q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
  s = torch.empty((M, K // QGROUP), dtype=torch.float32, device=x.device)
  dot = dot or default_dot(x.device)
  _act_quant_kernel[(triton.cdiv(M, rows), K // QGROUP)](x2, _codes(q, dot), s, M, x2.stride(0), q.stride(0),
                                                        s.stride(0), ROWS=rows, QG=QGROUP, CODES=dot != "fp8")
  return q.reshape(x.shape), s.reshape(*x.shape[:-1], K // QGROUP)


def _check(cfg: TileConfig) -> None:
  if cfg.quant == "fused" and cfg.BLOCK_K != QGROUP:
    raise ValueError("fused quant takes the group amax over one K tile: BLOCK_K must be 128")
  if QGROUP % cfg.BLOCK_K:
    raise ValueError(f"BLOCK_K={cfg.BLOCK_K} must divide the scale group {QGROUP}")


def launch(a: torch.Tensor, a_scale: torch.Tensor | None, w: torch.Tensor, ws: torch.Tensor, cfg: TileConfig,
           out_dtype: torch.dtype = torch.bfloat16, dot: str | None = None):
  """Run one GEMM on 2-D operands; returns (C [M, N], the compiled kernel handle or None)."""
  _check(cfg)
  M, K = a.shape
  N = w.shape[0]
  split = cfg.SPLIT_K > 1
  c = torch.empty((cfg.SPLIT_K, M, N) if split else (1, M, N), dtype=torch.float32 if split else out_dtype,
                  device=a.device)
  s = a_scale if a_scale is not None else ws                       # unused by the fused path; any valid pointer
  dot = dot or default_dot(a.device)
  grid = (triton.cdiv(M, cfg.BLOCK_M), triton.cdiv(N, cfg.BLOCK_N), cfg.SPLIT_K)
  ck = _gemm_kernel[grid](
    _codes(a, dot), s, _codes(w, dot), ws, c, M, N, K, a.stride(0), a.stride(1), s.stride(0), s.stride(1), w.stride(0), w.stride(1),
    ws.stride(0), ws.stride(1), c.stride(0), c.stride(1), c.stride(2),
    BLOCK_M=cfg.BLOCK_M, BLOCK_N=cfg.BLOCK_N, BLOCK_K=cfg.BLOCK_K, GROUP_M=cfg.GROUP_M, SPLIT_K=cfg.SPLIT_K,
    FUSED=cfg.quant == "fused", DOT=DOTS[dot], QG=QGROUP,
    num_warps=cfg.num_warps, num_stages=cfg.num_stages)
  out = c.sum(0).to(out_dtype) if split else c[0]
  return out, ck


def block_fp8_matmul(a: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, cfg: TileConfig, *,
                     a_scale: torch.Tensor | None = None, out_dtype: torch.dtype = torch.bfloat16,
                     dot: str | None = None) -> torch.Tensor:
  """C[..., N] = quant(a) @ w^T. ``a`` is bf16 ([..., K]); with cfg.quant == "once" it may instead be fp8 with
  ``a_scale`` from ``act_quant`` (otherwise it is quantised here, once for this call)."""
  lead = a.shape[:-1]
  if cfg.quant == "once" and a_scale is None:
    a, a_scale = act_quant(a, dot=dot)
  a2 = a.reshape(-1, a.shape[-1])
  s2 = a_scale.reshape(-1, a_scale.shape[-1]) if cfg.quant == "once" else None
  out, _ = launch(a2, s2, w, ws, cfg, out_dtype, dot)
  return out.reshape(*lead, w.shape[0])


def linear_group(x: torch.Tensor, members: list[tuple[torch.Tensor, torch.Tensor, TileConfig]],
                 dot: str | None = None, on_quant=None) -> list[torch.Tensor]:
  """GEMMs that read the same input (q/k/v, gate/up, in_proj_qkv/z): quantise x once for every "once" member."""
  q = s = None
  outs = []
  for w, ws, cfg in members:
    if cfg.quant == "once" and q is None:
      q, s = act_quant(x, dot=dot)
      if on_quant:
        on_quant()
    outs.append(block_fp8_matmul(q if cfg.quant == "once" else x, w, ws, cfg, a_scale=s if cfg.quant == "once"
                                 else None, dot=dot))
  return outs


def source_sha() -> str:
  """Identity of the measured kernel source: a tuned table is only valid for the source it was measured on."""
  return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


KERNEL_SHA = source_sha()
