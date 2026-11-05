"""The design space of the block-FP8 GEMM on sm_89, and its exact legality rules (no GPU needed).

A design is a dict of compile-time knobs. Two activation paths share one kernel:

  fused   A arrives bf16 and is quantised inside the GEMM, per K tile (the hub kernel's contract). The quant group
          must then be the K tile, so BLOCK_K = 128. Each of the N / BLOCK_N column programs re-quantises its A tiles.
  once    A arrives already quantised (fp8 + one fp32 scale per row per 128-group). BLOCK_K may be 32, 64 or 128
          (any divisor of the group, >= the k32 of Ada's m16n8k32 e4m3 mma.sync), and one quantisation is shared by
          every GEMM that reads the same input (q/k/v, gate/up, in_proj_qkv/z).

Rules (sm_89: no wgmma or TMA; 99 KB of shared memory per block, opt-in; 255 registers per thread):

  fused_needs_group_tile  fused quant computes the group amax over one K tile: BLOCK_K must be 128
  mma_tile                every warp needs at least one 16x8 mma tile: BLOCK_M * BLOCK_N >= 128 * num_warps
  acc_regs                the fp32 accumulator alone must fit in 128 registers per thread (above that Triton spills)
  smem                    (num_stages - 1) * BLOCK_K * (a_bytes * BLOCK_M + BLOCK_N), plus BLOCK_M * BLOCK_K for the
                          fused path's quantised A tile, <= 99 KB. Exact for the tiles we compiled for sm_89; a lower
                          bound otherwise (the epilogue can need more), so ``aot.compile_sm89`` is the final judge.
                          Consequence: the stock fused 128x128x128 tile fits only num_stages = 2 on Ada.

``STOCK`` is the hub kernel itself (its BLOCK_M rule, 128 x 128 tiles, GROUP_M = 8, and its own autotuned warps and
stages); 0 means "the stock kernel's choice". It is the reference every design is timed against.
"""
from __future__ import annotations

import functools
import itertools
import math

import xcalibur as xc
from dataclasses import asdict, dataclass

SMEM_LIMIT_SM89 = xc.ARCHS["rtx6000ada"].smem_max_cta      # 101376 B (99 KB) opt-in per block, as sm_86
ACC_REG_CAP = 128
QGROUP = 128

SPACE: dict[str, tuple] = {
  "quant": ("fused", "once"),
  "BLOCK_M": (16, 32, 64, 128, 256),
  "BLOCK_N": (32, 64, 128, 256),
  "BLOCK_K": (32, 64, 128),
  "GROUP_M": (1, 4, 8, 16, 32, 64),                # 32, 64: values in vLLM's L40S tables (public.py)
  "SPLIT_K": (1, 2, 4, 8),
  "num_warps": (2, 4, 8),
  "num_stages": (2, 3, 4, 5),
}
FIELDS = tuple(SPACE)
STOCK = {"quant": "fused", "BLOCK_M": 0, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "SPLIT_K": 1,
         "num_warps": 0, "num_stages": 0}
# Stock's own autotune grid (hub utils.get_accelerator_autotuning_configs, block-scaled path): warps x stages.
STOCK_GRID = tuple(itertools.product((2, 4, 8, 16), (2, 3, 4)))


@dataclass(frozen=True)
class TileConfig:
  """A resolved build: the launch knobs of ``kernel.block_fp8_matmul``."""
  quant: str = "fused"
  BLOCK_M: int = 64
  BLOCK_N: int = 128
  BLOCK_K: int = 128
  GROUP_M: int = 8
  SPLIT_K: int = 1
  num_warps: int = 4
  num_stages: int = 3

  def as_dict(self) -> dict:
    return asdict(self)


def hub_block_m(M: int) -> int:
  """The hub kernel's M tile: min(max(next_pow2(M), 16), 128)."""
  return min(max(1 << max(0, (M - 1).bit_length()), 16), 128)


def is_stock(d: dict) -> bool:
  return all(d.get(k) == v for k, v in STOCK.items())


def to_config(d: dict, M: int, warps: int = 4, stages: int = 3) -> TileConfig:
  """Resolve a design for size M; 0-valued knobs take the hub rule (BLOCK_M) or the given stock choice."""
  return TileConfig(d["quant"], d["BLOCK_M"] or hub_block_m(M), d["BLOCK_N"], d["BLOCK_K"], d["GROUP_M"],
                    d["SPLIT_K"], d["num_warps"] or warps, d["num_stages"] or stages)


def a_bytes(d: dict, standin_bytes: int = 1) -> int:
  """Bytes per staged A element: bf16 for fused quant; fp8 (or the stand-in's storage) for quantise-once."""
  return 2 if d["quant"] == "fused" else standin_bytes


def smem_bytes(d: dict, standin_bytes: int = 1) -> int:
  """Triton's software pipeline on sm_8x keeps num_stages - 1 operand buffers; fused quant stages bf16 A and adds
  the quantised fp8 A tile (calibrated against sm_89 compiles, see tests)."""
  pipe = (d["num_stages"] - 1) * d["BLOCK_K"] * (a_bytes(d, standin_bytes) * d["BLOCK_M"] + d["BLOCK_N"])
  return pipe + (d["BLOCK_M"] * d["BLOCK_K"] if d["quant"] == "fused" else 0)


def why_illegal(d: dict, limit: int = SMEM_LIMIT_SM89, standin_bytes: int = 1) -> str | None:
  """The first rule a design breaks, or None when it is legal on sm_89."""
  if d["quant"] == "fused" and d["BLOCK_K"] != QGROUP:
    return "fused_needs_group_tile"
  if d["BLOCK_M"] * d["BLOCK_N"] < 128 * d["num_warps"]:
    return "mma_tile"
  if d["BLOCK_M"] * d["BLOCK_N"] > ACC_REG_CAP * 32 * d["num_warps"]:
    return "acc_regs"
  if smem_bytes(d, standin_bytes) > limit:
    return "smem"
  return None


@functools.lru_cache(maxsize=4)
def _designs(limit: int, standin_bytes: int) -> tuple[tuple, ...]:
  out = []
  for v in itertools.product(*SPACE.values()):
    d = dict(zip(FIELDS, v))
    if why_illegal(d, limit, standin_bytes) is None:
      out.append(v)
  return tuple(out)


def designs(limit: int = SMEM_LIMIT_SM89, standin_bytes: int = 1) -> list[dict]:
  """Every legal design (fresh dicts; the stock design is not in the space, it is the reference)."""
  return [dict(zip(FIELDS, v)) for v in _designs(limit, standin_bytes)]


def bitwise_safe(d: dict) -> bool:
  """BLOCK_K = 128 and no split-K: every output element sums the same products in stock's order, so the build is
  bitwise equal to stock whatever its M / N tiles, GROUP_M, warps, stages or quant path (the kernel tests prove it).
  Other builds differ by fp32 summation order (<= 1 bf16 ulp), which FP8 can amplify downstream (docs/writeups/WRITEUP_SM86_SM89.md, ADA2)."""
  return d["BLOCK_K"] == 128 and d["SPLIT_K"] == 1


def applicable(d: dict, M: int, K: int | None = None) -> bool:
  """Worth timing at size M: no M tile beyond the next power of two (pure padding); split-K only for skinny M
  (<= 2k rows) or, at most 2-way, for long K (>= 16k: down_proj). Stock is always applicable."""
  if is_stock(d):
    return True
  if d["BLOCK_M"] > max(16, 1 << max(0, (M - 1).bit_length())):
    return False
  return d["SPLIT_K"] == 1 or M <= 2048 or (K is not None and K >= 16384 and d["SPLIT_K"] <= 2)


def key(d: dict) -> str:
  return repr(sorted(d.items()))


def frontier(history: list[dict], limit: int = SMEM_LIMIT_SM89, standin_bytes: int = 1) -> list[dict]:
  seen = {key(h["design"]) for h in history}
  return [d for d in designs(limit, standin_bytes) if key(d) not in seen]


def featurize(d: dict, compiled: dict | None) -> list[float]:
  """Physical features: log2 knobs, the quant path, accumulator registers per thread, staged bytes, waves of
  K tiles per split; plus the compiled kernel's registers, spills and shared memory when known (-1 otherwise)."""
  r = to_config(d, 128)
  lg = lambda v: math.log2(v)                                                         # noqa: E731
  acc = r.BLOCK_M * r.BLOCK_N / (32 * r.num_warps)
  c = compiled if compiled and compiled.get("ok") else {}
  return [float(d["quant"] == "once"), float(is_stock(d)), lg(r.BLOCK_M), lg(r.BLOCK_N), lg(r.BLOCK_K),
          lg(r.GROUP_M), lg(r.SPLIT_K), lg(r.num_warps), float(r.num_stages), acc,
          smem_bytes(r.as_dict()) / 1024.0, lg(r.BLOCK_M * r.BLOCK_N),
          float(c.get("n_regs", -1)), float(c.get("n_spills", -1)), float(c.get("shared", -1)) / 1024.0]
