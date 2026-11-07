#!/usr/bin/env python3
"""W8A8 vs bf16 F.linear at Kev-9B's projection shapes, paired (xcalibur.bench.Bench.paired: CUDA graphs, cold L2,
ABBA pairs, null replay subtracted). Ratio > 1 means W8A8 is faster.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 flock <gpu-lock> \\
      python kernels/w8a8_sm86/microbench.py --M 512 4096 28672 --pairs 20 --csv data/w8a8_sm86_microbench.csv

Per (projection, M) it times, each against F.linear(x, w) in its own paired run:
  triton  quant_rows(x) + Triton int8 GEMM with the dequant in its epilogue
  intmm   quant_rows(x) + torch._int_mm (cuBLASLt) + a separate dequant kernel
  gemm    the Triton int8 GEMM alone on a pre-quantised x (to read TOPS without the quantiser)
  auto    what kev_patch runs: triton below 1,024 rows, intmm from there (linear.resolve_backend)
  grouped per-(token, 128-channel) activation scales: grouped.quant_groups + the grouped Triton GEMM (kev_smooth's
          W8A8_GROUP for down_proj / out_proj / o_proj)
The screen picked "auto" from triton vs intmm; its speedup is then re-measured alone in a fresh run (--candidates auto),
because the best of several noisy candidates reads high.
TOPS = 2·M·N·K / time. Peaks (GA102 whitepaper, RTX 3090 at its 1,695 MHz boost): 284 dense INT8 TOPS,
71 bf16 TFLOPS with fp32 accumulate.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import secrets
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))              # kernels/ (the w8a8_sm86 package)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))              # the repo (xcalibur)
from w8a8_sm86.gemm import int8_gemm_dequant, int_mm_dequant  # noqa: E402
from w8a8_sm86.linear import W8A8Weight, w8a8_linear  # noqa: E402
from w8a8_sm86.quant import quant_rows  # noqa: E402
from xcalibur.bench import Bench  # noqa: E402

PEAK_INT8_TOPS, PEAK_BF16_TFLOPS = 284.0, 71.0
SHAPES = {                         # Kev-9B (Qwen3.5-9B): hidden 4096, MLP 12288, DeltaNet 16 k / 32 v heads of 128
  "deltanet.in_proj[qkv+z]": (12288, 4096),
  "deltanet.out_proj": (4096, 4096),
  "attn.qkv[q+gate,k,v]": (10240, 4096),
  "attn.o_proj": (4096, 4096),
  "mlp.gate_up": (24576, 4096),
  "mlp.down_proj": (4096, 12288),
}
CALLS_PER_PASS = {"deltanet.in_proj[qkv+z]": 24, "deltanet.out_proj": 24, "attn.qkv[q+gate,k,v]": 8, "attn.o_proj": 8,
                  "mlp.gate_up": 32, "mlp.down_proj": 32}


def run_id() -> str:
  return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(3)


def bench_shape(bench: Bench, name: str, m: int, n: int, k: int, pairs: int, which: list[str]) -> list[dict]:
  g = torch.Generator(device="cuda").manual_seed(0)
  x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
  w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
  wt = W8A8Weight.from_float(w)
  xq, xs = quant_rows(x)
  ref = F.linear(x, w)
  err = ((w8a8_linear(x, wt, "triton").float() - ref.float()).norm() / ref.float().norm()).item()
  del ref
  every = {"triton": lambda: w8a8_linear(x, wt, "triton"), "gemm": lambda: int8_gemm_dequant(xq, xs, wt.q, wt.scale),
       "intmm": lambda: int_mm_dequant(*quant_rows(x), wt.q, wt.scale), "auto": lambda: w8a8_linear(x, wt, "auto"),
       "grouped": lambda: w8a8_linear(x, wt, "auto", group=128)}
  cands = {c: every[c] for c in which}
  rows, flops = [], 2.0 * m * n * k
  for cand, fn in cands.items():
    r = bench.paired(fn, lambda: F.linear(x, w), pairs=pairs)
    rows.append({"projection": name, "M": m, "N": n, "K": k, "candidate": cand, "pairs": pairs,
           "cand_us": round(r["candidate_us"], 2), "bf16_us": round(r["reference_us"], 2),
           "ratio": round(r["ratio"], 4), "ci_lo": round(r["ci"][0], 4), "ci_hi": round(r["ci"][1], 4),
           "cand_tops": round(flops / r["candidate_us"] / 1e6, 1), "bf16_tflops": round(flops / r["reference_us"] / 1e6, 1),
           "rel_err": round(err, 5)})
    torch.cuda.empty_cache()
  return rows


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--M", type=int, nargs="+", default=[512, 4096, 28672])
  p.add_argument("--shapes", nargs="+", default=list(SHAPES))
  p.add_argument("--pairs", type=int, default=20)
  p.add_argument("--candidates", nargs="+", default=["triton", "gemm", "intmm"], choices=["triton", "gemm", "intmm", "auto", "grouped"])
  p.add_argument("--csv", help="append rows to this CSV")
  a = p.parse_args()
  rid, bench = run_id(), Bench()
  dev = torch.cuda.get_device_properties(0)
  print(f"run {rid} · {dev.name} sm_{dev.major}{dev.minor} · torch {torch.__version__} · triton {triton.__version__} · "
        f"null replay {bench.null_us():.1f} us")
  rows = []
  for m in a.M:
    for name in a.shapes:
      n, k = SHAPES[name]
      for r in bench_shape(bench, name, m, n, k, a.pairs, a.candidates):
        r["run_id"] = rid
        rows.append(r)
        print(f"{name:26s} M={m:6d} {r['candidate']:6s} {r['cand_us']:10.1f} us vs bf16 {r['bf16_us']:10.1f} us  "
              f"{r['ratio']:.3f}x [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  {r['cand_tops']:6.1f} TOPS "
              f"({100 * r['cand_tops'] / PEAK_INT8_TOPS:4.1f}%)  bf16 {r['bf16_tflops']:5.1f} TFLOPS "
              f"({100 * r['bf16_tflops'] / PEAK_BF16_TFLOPS:4.1f}%)  rel err {r['rel_err']:.4f}", flush=True)
  x, w = _aa_operands()
  aa = bench.aa(lambda: F.linear(x, w), pairs=a.pairs)
  del x, w
  _AA.clear()
  print(f"A/A (bf16 mlp.gate_up at M=4096 against itself): {aa['ratio']:.4f} [{aa['ci'][0]:.4f}, {aa['ci'][1]:.4f}]")
  for m in a.M:                                    # one Kev-9B state pass worth of projection GEMMs at this M
    pick = "auto" if "auto" in a.candidates else a.candidates[0]
    sel = [r for r in rows if r["M"] == m and r["candidate"] == pick]
    t8 = sum(CALLS_PER_PASS[r["projection"]] * r["cand_us"] for r in sel)
    t16 = sum(CALLS_PER_PASS[r["projection"]] * r["bf16_us"] for r in sel)
    if sel:
      print(f"M={m}: all projections of one pass (call-weighted, {pick}) {t16 / 1e3:.1f} ms bf16 -> {t8 / 1e3:.1f} ms W8A8, {t16 / t8:.3f}x")
  if a.csv:
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
      wr = csv.DictWriter(f, fieldnames=["run_id"] + [c for c in rows[0] if c != "run_id"])
      if new:
        wr.writeheader()
      wr.writerows(rows)


_AA = {}


def _aa_operands() -> tuple[torch.Tensor, torch.Tensor]:
  if not _AA:
    g = torch.Generator(device="cuda").manual_seed(1)
    _AA["x"] = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    _AA["w"] = torch.randn(24576, 4096, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
  return _AA["x"], _AA["w"]


if __name__ == "__main__":
  main()
