"""Check a warp-specialized grouped-GEMM CTA plan on an H200, then break it on purpose and read the counterexample."""

import json

import xcalibur as xc

plan = xc.plan_of(
  squads=[("mma", 16), ("w_load", 1), ("x_load", 1)],                      # 4 MMA warpgroups + two producer warps
  resources=[("w_ring", 3, 36864, [("w_load", 36864), ("mma", None)]),     # TMA-fed weight ring: produce -> consume
             ("x_ring", 3, 32768, [("x_load", 32768), ("mma", None)])])    # activation ring

r = xc.warpspeed_check(**plan)
print("pass:", r["pass"])
print("warps / threads / warpgroups:", r["cta"]["warps"], r["cta"]["threads"], r["cta"]["warpgroups"])
print("launch register cap:", r["registers"]["launch_cap"], "| mbarriers:", r["mbarriers"])
print("shared memory:", r["smem"]["total_bytes"], "of", r["smem"]["cta_limit"], "B per CTA")
for s in r["squads"]:
  print(f"  squad {s['name']:7s} warps {s['warp_begin']:2d}-{s['warp_end'] - 1:2d}  owns {s['owned_phases']}")

plan["resources"][0]["stages"] = 6                                          # deepen the weight ring
bad = xc.warpspeed_check(**plan)
print("\n6-stage weight ring -> pass:", bad["pass"])
print("counterexample:", json.dumps(bad["counterexample"]))
