"""warpspeed_check: an exact, typed check of a WarpSpeed CTA plan.

WarpSpeed (NVIDIA CCCL, ``cub/detail/warpspeed``) describes a warp-specialized kernel as squads (contiguous warp
ranges) that pass shared-memory resources (``stages x stride_bytes`` rings) through 2..4 phases. Each phase owns
one mbarrier per stage whose arrival count is the owner squads' thread count; a TMA phase adds expected transaction
bytes. Acquiring phase p waits on phase p-1, so the phases of a resource form a ring.

This checker follows those semantics and adds per-architecture hardware facts (register caps, setmaxnreg rules,
shared-memory limits). It returns either a pass with every derived quantity, or the first violation as a typed
counterexample. It is accounting, not a compiler or a timer: allocated registers, spills, admission and speed still
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
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
  name: str
  sms: int
  regs_per_sm: int = 65536
  subpartitions: int = 4
  max_threads_sm: int = 2048
  max_threads_cta: int = 1024
  smem_per_sm: int = 233472
  smem_reserved_per_cta: int = 1024
  smem_max_cta: int = 232448
  max_resources: int = 10      # WarpSpeed SyncHandler
  max_phases: int = 4


H200 = Arch("H200 SXM (sm_90a)", sms=132)
ARCHS = {"h200": H200}


def launch_register_cap(arch: Arch, warps: int, ctas_per_sm: int = 1) -> int:
  """Registers per thread at launch: regs / (128 x ceil(warps / 4) x CTAs per SM), floored to 8, at most 255."""
  cap = arch.regs_per_sm // (ctas_per_sm * 128 * math.ceil(warps / arch.subpartitions)) // 8 * 8
  return min(255, cap)


def plan_of(squads, resources):
  """Build a plan from tuples: squads [(name, warps)], resources [(name, stages, stride, [(owner, tx_bytes|None)])]."""
  return {"squads": [{"name": n, "warps": w} for n, w in squads],
      "resources": [{"name": n, "stages": st, "stride_bytes": b,
                           "phases": [{"owners": [o], **({"tx_bytes": tx} if tx else {})} for o, tx in ph]}
                          for n, st, b, ph in resources]}


def warpspeed_check(squads, resources, ctas_per_sm=1, extra_smem_bytes=0, arch: Arch | str = H200):
  arch = ARCHS[arch] if isinstance(arch, str) else arch
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
  cap = launch_register_cap(arch, warps, ctas_per_sm)
  pool = threads * cap

  rows, begin, requested = [], 0, 0
  wants = any("regs" in s for s in squads)
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
          "setmaxnreg register caps, sub-partition placement, shared-memory layout and limits. Returns pass "
          "plus every derived quantity, or the first violation as a counterexample. Accounting only."),
  "parameters": {"type": "object", "additionalProperties": False, "required": ["squads", "resources"], "properties": {
    "squads": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["name", "warps"], "properties": {
      "name": {"type": "string"}, "warps": {"type": "integer", "minimum": 1, "maximum": 32},
      "regs": {"type": "integer", "minimum": 24, "maximum": 256}, "regs_needed": {"type": "integer", "minimum": 1}}}},
    "resources": {"type": "array", "items": {"type": "object", "required": ["name", "stages", "stride_bytes", "phases"],
      "properties": {"name": {"type": "string"}, "stages": {"type": "integer", "minimum": 1},
                           "stride_bytes": {"type": "integer", "minimum": 1}, "align": {"type": "integer", "minimum": 1},
                           "phases": {"type": "array", "items": {"type": "object", "required": ["owners"], "properties": {
                               "owners": {"type": "array", "items": {"type": "string"}}, "tx_bytes": {"type": "integer", "minimum": 0}}}}}}},
    "ctas_per_sm": {"type": "integer", "minimum": 1}, "extra_smem_bytes": {"type": "integer", "minimum": 0}}}}}


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
