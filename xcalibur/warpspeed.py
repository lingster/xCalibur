"""warpspeed_check: an exact, typed check of a WarpSpeed CTA plan.

WarpSpeed (NVIDIA CCCL, ``cub/detail/warpspeed``) describes a warp-specialized kernel as squads (contiguous warp
ranges) that pass shared-memory resources (``stages x stride_bytes`` rings) through 2..4 phases. Each phase owns
one mbarrier per stage whose arrival count is the owner squads' thread count; a TMA phase adds expected transaction
bytes. Acquiring phase p waits on phase p-1, so the phases of a resource form a ring.

This checker follows those semantics and adds per-architecture hardware facts (register caps, setmaxnreg rules,
shared-memory and residency limits, and the Hopper-only features wgmma, TMA and setmaxnreg; ``ARCHS`` holds the H200
and sm_86 / sm_89 boards, with roofline facts). It returns either a pass with every derived quantity, or the first
violation as a typed counterexample. It is accounting, not a compiler or a timer: allocated registers, spills, admission and speed still
need compilation and measurement.

  >>> plan = plan_of([("mma", 16), ("w_load", 1), ("x_load", 1)],
  ...                [("w_ring", 3, 36864, [("w_load", 36864), ("mma", None)]),
  ...                 ("x_ring", 3, 32768, [("x_load", 32768), ("mma", None)])])
  >>> r = warpspeed_check(**plan)
  >>> r["pass"], r["registers"]["launch_cap"], r["smem"]["total_bytes"]
  (True, 96, 208992)
"""
from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
  """Per-board facts. Defaults are Hopper (compute capability 9.0), so H200 checks are unchanged."""
  name: str
  sms: int
  regs_per_sm: int = 65536
  subpartitions: int = 4
  max_threads_sm: int = 2048
  max_threads_cta: int = 1024
  max_ctas_sm: int = 32
  smem_per_sm: int = 233472
  smem_reserved_per_cta: int = 1024
  smem_max_cta: int = 232448
  max_resources: int = 10      # WarpSpeed SyncHandler
  max_phases: int = 4
  wgmma: bool = True           # warpgroup MMA (sm_90a)
  tma: bool = True             # TMA bulk copies with mbarrier expect-tx (sm_90)
  setmaxnreg: bool = True      # per-warpgroup register reallocation (sm_90a)
  l2_bytes: int | None = None
  bf16_tflops_dense: float | None = None   # tensor cores, fp32 accumulate, no sparsity, at boost clock
  fp8_tflops_dense: float | None = None    # None: no FP8 tensor cores
  int8_tops_dense: float | None = None
  dram_gbs: float | None = None


# Limits per compute capability: CUDA C++ Programming Guide, "Compute Capabilities", tables of SM and memory
# information (8.6 / 8.9: 1,536 threads, 64K registers, 100 KB shared per SM, 99 KB per block, 16 / 24 blocks per SM;
# 9.0: 2,048 threads, 228 / 227 KB, 32 blocks). 1 KB reserved per CTA: the Ampere, Ada and Hopper tuning guides.
# Throughput: NVIDIA datasheets and architecture whitepapers, dense rates at boost clock.
# H200 product page: bf16 1,979 and FP8 / INT8 3,958 TFLOPS with sparsity, 4.8 TB/s. L2 50 MB is the Hopper tuning
# guide's H100 figure (same GH100 die); unverified for H200.
H200 = Arch("H200 SXM (sm_90a)", sms=132, l2_bytes=50 << 20, bf16_tflops_dense=989.5, fp8_tflops_dense=1979,
            int8_tops_dense=1979, dram_gbs=4800)

_SM8X = dict(max_threads_sm=1536, smem_per_sm=102400, smem_max_cta=101376, wgmma=False, tma=False, setmaxnreg=False)
_SM86, _SM89 = dict(_SM8X, max_ctas_sm=16), dict(_SM8X, max_ctas_sm=24)
# GA102 whitepaper v2, Table 9: on GeForce, fp16/bf16 tensor math with fp32 accumulate runs at half rate (142 -> 71).
RTX3090 = Arch("RTX 3090 (sm_86)", sms=82, l2_bytes=6 << 20, bf16_tflops_dense=71, int8_tops_dense=284,
               dram_gbs=936, **_SM86)
# GA102 whitepaper v2, Table 3: professional board, full rate with fp32 accumulate.
A6000 = Arch("RTX A6000 (sm_86)", sms=84, l2_bytes=6 << 20, bf16_tflops_dense=154.8, int8_tops_dense=309.7,
             dram_gbs=768, **_SM86)
# RTX 6000 Ada datasheet: 18,176 cores (142 SMs), 1,457 FP8 TFLOPS sparse, 960 GB/s. bf16 = FP8 / 2 and INT8 = FP8 are
# derived (full rate, as on L40S); L2 96 MB is AD102's. Unverified on the datasheet.
RTX6000ADA = Arch("RTX 6000 Ada (sm_89)", sms=142, l2_bytes=96 << 20, bf16_tflops_dense=364.2, fp8_tflops_dense=728.5,
                  int8_tops_dense=728.5, dram_gbs=960, **_SM89)
# L40S datasheet (secondary sources; unverified here). Not the L40, which the Ada whitepaper lists at half rate (181).
L40S = Arch("L40S (sm_89)", sms=142, l2_bytes=96 << 20, bf16_tflops_dense=362.05, fp8_tflops_dense=733,
            int8_tops_dense=733, dram_gbs=864, **_SM89)
# Ada whitepaper v2.02, Appendix A: GeForce, half rate with fp32 accumulate (fp16 accumulate: 330.3 bf16 / 660.6 FP8).
RTX4090 = Arch("RTX 4090 (sm_89)", sms=128, l2_bytes=72 << 20, bf16_tflops_dense=165.2, fp8_tflops_dense=330.3,
               int8_tops_dense=660.6, dram_gbs=1008, **_SM89)
ARCHS = {"h200": H200, "rtx3090": RTX3090, "a6000": A6000, "rtx6000ada": RTX6000ADA, "l40s": L40S, "rtx4090": RTX4090}


def launch_register_cap(arch: Arch, warps: int, ctas_per_sm: int = 1) -> int:
  """Registers per thread at launch: regs / (128 x ceil(warps / 4) x CTAs per SM), floored to 8, at most 255."""
  _integer(warps, "warps", 1)
  _integer(ctas_per_sm, "ctas_per_sm", 1)
  cap = arch.regs_per_sm // (ctas_per_sm * 128 * math.ceil(warps / arch.subpartitions)) // 8 * 8
  return min(255, cap)


def _integer(value, label, minimum):
  if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < minimum:
    raise ValueError(f"{label} must be an integer >= {minimum}")


def _validate_plan(squads, resources, ctas_per_sm, extra_smem_bytes):
  """Validate the input domain before resource arithmetic can make an invalid plan pass."""
  _integer(ctas_per_sm, "ctas_per_sm", 1)
  _integer(extra_smem_bytes, "extra_smem_bytes", 0)
  for collection, kind in ((squads, "squad"), (resources, "resource")):
    names = [item["name"] for item in collection]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
      raise ValueError(f"{kind} names must be nonempty, unique strings")
  for s in squads:
    _integer(s["warps"], "squad warps", 1)
    if s["warps"] > 32:
      raise ValueError("a squad cannot exceed 32 warps")
    if "regs" in s:
      _integer(s["regs"], "setmaxnreg target", 24)
      if s["regs"] > 256:
        raise ValueError("setmaxnreg target must be <= 256")
    if "regs_needed" in s:
      _integer(s["regs_needed"], "regs_needed", 1)
    if "wgmma" in s and not isinstance(s["wgmma"], bool):
      raise ValueError("squad wgmma must be a boolean")
  for r in resources:
    for key in ("stages", "stride_bytes"):
      _integer(r[key], key, 1)
    align = r.get("align", 16)
    _integer(align, "align", 1)
    if align & (align - 1):
      raise ValueError(f"resource {r['name']}: align must be a power of two")
    for phase in r["phases"]:
      owners = phase["owners"]
      if not isinstance(owners, (list, tuple)) or not owners or any(not isinstance(o, str) for o in owners):
        raise ValueError("phase owners must be a nonempty sequence of squad names")
      if len(set(owners)) != len(owners):
        raise ValueError("phase owners must be unique")
      if "tx_bytes" in phase:
        _integer(phase["tx_bytes"], "tx_bytes", 0)


def plan_of(squads, resources):
  """Build a plan from tuples: squads [(name, warps)], resources [(name, stages, stride, [(owner, tx_bytes|None)])]."""
  return {"squads": [{"name": n, "warps": w} for n, w in squads],
      "resources": [{"name": n, "stages": st, "stride_bytes": b,
                           "phases": [{"owners": [o], **({"tx_bytes": tx} if tx is not None else {})} for o, tx in ph]}
                          for n, st, b, ph in resources]}


def warpspeed_check(squads, resources, ctas_per_sm=1, extra_smem_bytes=0, arch: Arch | str = H200):
  _validate_plan(squads, resources, ctas_per_sm, extra_smem_bytes)
  if isinstance(arch, str):
    if arch not in ARCHS:
      raise ValueError(f"unknown arch {arch!r}; known: {sorted(ARCHS)}")
    arch = ARCHS[arch]
  violations = []

  def fail(code, message, **where):
    violations.append(dict(code=code, message=message, **where))

  names = [s["name"] for s in squads]
  if not names:
    raise ValueError("a CTA plan needs at least one squad")
  if len(set(names)) != len(names):
    raise ValueError("squad names must be unique")
  warps = sum(s["warps"] for s in squads)
  threads = 32 * warps
  if threads > arch.max_threads_cta:
    fail("cta_threads", f"{threads} threads exceed {arch.max_threads_cta} per CTA")
  if threads * ctas_per_sm > arch.max_threads_sm:
    fail("sm_threads", f"{ctas_per_sm} CTAs x {threads} threads exceed {arch.max_threads_sm} per SM")
  if ctas_per_sm > arch.max_ctas_sm:
    fail("sm_ctas", f"{ctas_per_sm} CTAs exceed {arch.max_ctas_sm} resident per SM")
  cap = launch_register_cap(arch, warps, ctas_per_sm)
  pool = threads * cap

  rows, begin, requested = [], 0, 0
  wants = any("regs" in s for s in squads)
  if wants and not arch.setmaxnreg:
    fail("setmaxnreg_arch", f"{arch.name} has no setmaxnreg; register budgets are fixed at launch")
  for s in squads:                                  # squads take contiguous warp ranges in declaration order
    w = s["warps"]
    per_sp = [sum(1 for k in range(begin, begin + w) if k % arch.subpartitions == p) for p in range(arch.subpartitions)]
    row = dict(name=s["name"], warp_begin=begin, warp_end=begin + w, threads=32 * w,
                   warpgroup_aligned=begin % 4 == 0 and w % 4 == 0, warps_per_subpartition=per_sp, register_cap=cap)
    if "regs" in s:
      if s["regs"] % 8:
        fail("setmaxnreg_range", f"squad {s['name']}: setmaxnreg target {s['regs']} is not a multiple of 8", squad=s["name"])
      row["register_cap"] = s["regs"]
      requested += 32 * w * s["regs"]
    else:
      requested += 32 * w * cap
    if "regs_needed" in s:
      row["regs_needed"] = s["regs_needed"]
      row["register_fit"] = s["regs_needed"] <= row["register_cap"]
      if not row["register_fit"]:
        fail("register_fit", f"squad {s['name']} needs ~{s['regs_needed']} registers/thread above its cap "
                     f"{row['register_cap']}", squad=s["name"])
    if s.get("wgmma"):
      if not arch.wgmma:
        fail("wgmma_arch", f"squad {s['name']}: {arch.name} has no wgmma (use mma.sync per warp)", squad=s["name"])
      elif not row["warpgroup_aligned"]:
        fail("wgmma_alignment", f"squad {s['name']}: wgmma needs whole warpgroups, has warps {begin}-{begin + w - 1}",
             squad=s["name"])
    rows.append(row)
    begin += w
  if wants:                                         # setmaxnreg: one value per whole warpgroup
    per_warp = [s.get("regs") for s in squads for _ in range(s["warps"])]
    for g in range(0, warps, 4):
      group = per_warp[g:g + 4]
      if any(v is not None for v in group) and (len(group) < 4 or len(set(group)) != 1):
        fail("setmaxnreg_alignment", f"warpgroup {g // 4} mixes setmaxnreg targets {group}")
    if requested > pool:
      fail("setmaxnreg_pool", f"setmaxnreg requests {requested} registers above the launch pool {pool}")

  if len(resources) > arch.max_resources:
    fail("resource_count", f"{len(resources)} resources exceed {arch.max_resources}")
  offset, max_align, res_rows = 0, 16, []
  for r in resources:                               # buffers first, in declaration order
    align = r.get("align", 16)
    if align & (align - 1):
      raise ValueError(f"resource {r['name']}: align must be a power of two")
    max_align = max(max_align, align)
    offset = -(-offset // align) * align
    res_rows.append(dict(name=r["name"], stages=r["stages"], stride_bytes=r["stride_bytes"], offset=offset,
                             bytes=r["stages"] * r["stride_bytes"], phases=[]))
    offset += r["stages"] * r["stride_bytes"]
  mbarriers = 0
  for r, row in zip(resources, res_rows):           # then one barrier array per phase (8 B per stage)
    n = len(r["phases"])
    if n > arch.max_phases:
      fail("phase_count", f"resource {r['name']} has {n} phases; at most {arch.max_phases}", resource=r["name"])
    if n < 2:
      fail("phase_ring", f"resource {r['name']} has {n} phase; a release/acquire ring needs at least 2", resource=r["name"])
    for i, ph in enumerate(r["phases"]):
      unknown = [o for o in ph["owners"] if o not in names]
      if unknown:
        fail("phase_owner", f"resource {r['name']} phase {i}: unknown squads {unknown}", resource=r["name"])
      arrivals = sum(32 * s["warps"] for s in squads if s["name"] in ph["owners"])
      offset = -(-offset // 8) * 8
      prow = dict(index=i, owners=ph["owners"], arrival_count=arrivals, acquire_waits_on_phase=(i - 1) % max(n, 1),
            barrier_offset=offset)
      if "tx_bytes" in ph:
        prow["tx_bytes"] = ph["tx_bytes"]
        if not arch.tma:
          fail("tma_arch", f"resource {r['name']} phase {i}: {arch.name} has no TMA or expect-tx "
                           f"(produce with cp.async and arrive)", resource=r["name"])
        if ph["tx_bytes"] > r["stride_bytes"]:
          fail("tx_bytes", f"resource {r['name']} phase {i}: tx {ph['tx_bytes']} B exceeds the stage stride", resource=r["name"])
      row["phases"].append(prow)
      offset += 8 * r["stages"]
      mbarriers += r["stages"]
  for s in rows:
    s["owned_phases"] = [f"{r['name']}:{p['index']}" for r in res_rows for p in r["phases"] if s["name"] in p["owners"]]
  total = offset + extra_smem_bytes + (max_align - 16 if max_align > 16 else 0)
  if total > arch.smem_max_cta:
    fail("smem_cta", f"{total} B of shared memory exceed {arch.smem_max_cta} B per CTA")
  if ctas_per_sm * (total + arch.smem_reserved_per_cta) > arch.smem_per_sm:
    fail("smem_sm", f"{ctas_per_sm} CTAs x ({total} + {arch.smem_reserved_per_cta}) B exceed {arch.smem_per_sm} B per SM")
  return dict(
    pass_=not violations, board=arch.name,
    cta=dict(warps=warps, threads=threads, warpgroups=math.ceil(warps / 4), ctas_per_sm=ctas_per_sm),
    registers=dict(launch_cap=cap, launch_pool=pool, requested_total=requested, setmaxnreg_requested=wants),
    squads=rows, resources=res_rows, mbarriers=mbarriers,
    smem=dict(resource_and_barrier_bytes=offset, extra_bytes=extra_smem_bytes, base_alignment=max_align,
                  total_bytes=total, cta_limit=arch.smem_max_cta, ctas_by_smem=arch.smem_per_sm // (total + arch.smem_reserved_per_cta)),
    limits=dict(resources=len(resources), max_resources=arch.max_resources, max_phases=arch.max_phases),
    violations=violations, counterexample=violations[0] if violations else None) | {"pass": not violations}


# A JSON tool schema, so a language model can call the checker directly (function calling).
SCHEMA = {"type": "function", "function": {
  "name": "warpspeed_check",
  "description": ("Check a WarpSpeed CTA plan: squads in warp order, staged shared-memory resources with 2..4 phases "
          "(each phase = owner squads; mbarrier arrival = owner threads; optional TMA tx bytes), launch and "
          "setmaxnreg register caps, sub-partition placement, shared-memory layout and limits, and per-arch features "
          "(wgmma squads, TMA tx bytes, setmaxnreg are Hopper-only). Returns pass "
          "plus every derived quantity, or the first violation as a counterexample. Accounting only."),
  "parameters": {"type": "object", "additionalProperties": False, "required": ["squads", "resources"], "properties": {
    "squads": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["name", "warps"], "properties": {
      "name": {"type": "string"}, "warps": {"type": "integer", "minimum": 1, "maximum": 32},
      "regs": {"type": "integer", "minimum": 24, "maximum": 256}, "regs_needed": {"type": "integer", "minimum": 1},
      "wgmma": {"type": "boolean"}}}},
    "resources": {"type": "array", "items": {"type": "object", "required": ["name", "stages", "stride_bytes", "phases"],
      "properties": {"name": {"type": "string"}, "stages": {"type": "integer", "minimum": 1},
                           "stride_bytes": {"type": "integer", "minimum": 1}, "align": {"type": "integer", "minimum": 1},
                           "phases": {"type": "array", "items": {"type": "object", "required": ["owners"], "properties": {
                               "owners": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string"}}, "tx_bytes": {"type": "integer", "minimum": 0}}}}}}},
    "ctas_per_sm": {"type": "integer", "minimum": 1}, "extra_smem_bytes": {"type": "integer", "minimum": 0},
    "arch": {"type": "string", "enum": sorted(ARCHS)}}}}}


def execute(arguments):
  """Tool-call entry point: real results or real errors, never an invented pass."""
  try:
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(args, dict):
      raise ValueError("tool arguments must be an object")
    r = warpspeed_check(**args)
    r.pop("pass_", None)
    return {"ok": True, "result": r}
  except Exception as e:
    return {"ok": False, "error": {"type": type(e).__name__, "message": str(e)[:400]}}
