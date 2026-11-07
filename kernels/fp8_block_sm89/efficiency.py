"""How far is the FP8 GEMM from the card's peak, per Kev-27B shape, and where does the time go? Run this first.

Per (N, K) and token count M, each with two clocks (timing.py): eager = the Ada probe's method (host dispatch
included), graph = CUDA-graph replay (device time only).

  stock     the reference (stock.py): the hub kernel on Ada, its twin elsewhere
  twin      our kernel at the hub's tiling (on Ada: should match stock's device time; see parity.py for numbers)
  quant     act_quant alone (the cost quantise-once pays once per input)
  tuned     the table's build for this (N, K, M), if --table is given and the bucket is tuned
  cublas    torch._scaled_mm, per-tensor scales, no activation quant: the practical FP8 ceiling (sm_89 only)
  bf16      F.linear

TFLOPS = 2MNK / graph time. Efficiency is reported against the datasheet (--peak, 728.5 dense FP8 with fp32
accumulate) and against cuBLAS at the same shape (the ~210 TFLOPS the card sustains at its 300 W cap).
--groups adds the concat-group lever: each input group as separate stock calls vs one concatenated stock GEMM.

  python efficiency.py --out eff.json [--table tuned.json] [--groups]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F

import xcalibur as xc

import concat
import kernel
import reference
import session
import shapes
import stock
import timing
from table import TuneTable

DATASHEET_TFLOPS = xc.ARCHS["rtx6000ada"].fp8_tflops_dense   # 728.5: dense FP8, fp32 accumulate (1457 sparse)


def both(fn, min_ms: float) -> dict:
  return {"eager_ms": timing.eager_ms(fn, min_ms), "graph_ms": timing.graph_ms(fn, min_ms)}


def rotated(call, weights: list, min_ms: float) -> dict:
  """both() on the first copy, plus rot_ms over all of them (cold weights, as production reads them)."""
  w, ws = weights[0]
  return both(lambda: call(w, ws), min_ms) | {"rot_ms": timing.rot_ms(call, weights, min_ms), "copies": len(weights)}


def weight_copies(M: int, N: int, K: int, w, ws) -> list:
  """[(w, ws)] copies covering 2 x L2 (session.weight_copies); the first is the original."""
  return [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(session.weight_copies(M, N, K, steady=True) - 1)]


def shape_row(M: int, N: int, K: int, ref: stock.Reference, a, tab: TuneTable | None, fp8: bool) -> dict:
  x, w, ws = reference.random_problem(M, N, K, "cuda", seed=M)
  r: dict = {"M": M, "N": N, "K": K, "gemms": shapes.gemm_names(N, K)}
  copies = weight_copies(M, N, K, w, ws)
  r["stock"] = rotated(lambda w_, s_: ref.fn(x, w_, s_), copies, a.min_ms)
  r["stock_choice"] = ref.choice(N, K, M)
  twin = stock.Twin(a.dot)
  r["twin"] = both(lambda: twin(x, w, ws), a.min_ms) | {"choice": twin.choice(N, K, M)}
  r["quant"] = both(lambda: kernel.act_quant(x, dot=a.dot), a.min_ms)
  cfg = tab.config_for(N, K, M) if tab else None
  if cfg:
    r["tuned"] = rotated(lambda w_, s_: kernel.block_fp8_matmul(x, w_, s_, cfg, dot=a.dot), copies, a.min_ms) | {
      "config": cfg.as_dict()}
  if fp8:
    xa, one = x.to(torch.float8_e4m3fn), torch.ones((), device="cuda")
    r["cublas"] = both(lambda: torch._scaled_mm(xa, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16), a.min_ms)
  wb = (w.float() * 0.01).to(torch.bfloat16)
  r["bf16"] = both(lambda: F.linear(x, wb), a.min_ms)
  for k in ("stock", "twin", "quant", "tuned", "cublas", "bf16"):
    if k in r and k != "quant":
      r[k]["tflops"] = timing.tflops(M, N, K, r[k]["graph_ms"])
  st = r["stock"]
  st["pct_datasheet"] = 100 * st["tflops"] / a.peak
  if "cublas" in r:
    st["pct_cublas"] = 100 * st["tflops"] / r["cublas"]["tflops"]
  st["host_over_device"] = st["eager_ms"] / st["graph_ms"]
  return r


def group_row(M: int, name: str, ref: stock.Reference, a, tab: TuneTable | None) -> dict:
  members = shapes.input_groups()[name]
  K = members[0].K
  x = reference.random_problem(M, 128, K, "cuda", seed=M)[0]
  ws_ = [reference.random_problem(1, g.N, K, "cuda", seed=i)[1:] for i, g in enumerate(members)]
  cat = concat.concat_weights(ws_)

  def separate():
    for w, s in ws_:
      ref.fn(x, w, s)

  r = {"M": M, "group": name, "members": [g.name for g in members], "N_cat": int(cat.w.shape[0]), "K": K,
       "stock_separate": both(separate, a.min_ms), "stock_concat": both(lambda: concat.forward(x, cat, ref.fn), a.min_ms)}
  cfg = tab.config_for(int(cat.w.shape[0]), K, M) if tab else None
  if cfg:
    r["tuned_concat"] = both(lambda: concat.forward(x, cat, lambda q, w, s: kernel.block_fp8_matmul(q, w, s, cfg, dot=a.dot)),
                             a.min_ms) | {"config": cfg.as_dict()}
  for k in ("eager_ms", "graph_ms"):
    r[f"concat_gain_{k[:-3]}"] = r["stock_separate"][k] / r["stock_concat"][k]
  return r


def per_request(rows: list[dict]) -> dict:
  """Σ calls x time per workload, at each workload's M (only Ms that were timed)."""
  out = {}
  for wname, wl in shapes.WORKLOADS.items():
    calls = shapes.request_calls(wl)
    tot: dict = {"calls_covered": 0, "calls_total": sum(calls.values())}       # a partial sum must say so
    for (gname, M), n in calls.items():
      g = next(g for g in shapes.GEMMS if g.name == gname)
      row = next((r for r in rows if (r["M"], r["N"], r["K"]) == (shapes.bucket_of(M).M, g.N, g.K)), None)
      if row is None:
        continue
      tot["calls_covered"] += n
      for k in ("stock", "tuned", "cublas", "bf16"):
        src = row.get(k) or (row["stock"] if k == "tuned" else None)
        if src:
          for clock in ("eager_ms", "graph_ms"):
            tot[f"{k}_{clock}"] = tot.get(f"{k}_{clock}", 0.0) + n * src[clock]
    out[wname] = {k: round(v, 1) for k, v in tot.items()}
  return out


def fmt(r: dict) -> str:
  s = r["stock"]
  parts = [f"M={r['M']:>5} N={r['N']:>5} K={r['K']:>5}",
           f"stock eager {s['eager_ms']:.3f} / device {s['graph_ms']:.3f} ms ({s['tflops']:.0f} TF, "
           f"{s['pct_datasheet']:.0f}% datasheet" + (f", {s['pct_cublas']:.0f}% cuBLAS)" if "pct_cublas" in s else ")"),
           f"twin {r['twin']['tflops']:.0f}", f"quant {r['quant']['graph_ms']:.3f} ms"]
  parts += [f"{k} {r[k]['tflops']:.0f}" for k in ("tuned", "cublas", "bf16") if k in r]
  return " | ".join(parts)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--shapes", default="all")
  ap.add_argument("--tokens", default="64,512,4096,28672")
  ap.add_argument("--table", help="tuned table(s) from search.py, comma-separated: adds the tuned columns")
  ap.add_argument("--groups", action="store_true", help="also time the concat groups (q|k|v, gate|up, qkv|z)")
  ap.add_argument("--reference", default="auto", choices=("auto", "hub", "twin"))
  ap.add_argument("--hub-revision", default=None)
  ap.add_argument("--dot", default=None, choices=(None, "fp8", "bf16", "fp32"))
  ap.add_argument("--peak", type=float, default=DATASHEET_TFLOPS)
  ap.add_argument("--min-ms", type=float, default=200.0)
  ap.add_argument("--out", default="efficiency.json")
  a = ap.parse_args(argv)
  a.dot = a.dot or kernel.default_dot("cuda")
  fp8 = torch.cuda.get_device_capability() >= (8, 9)
  ref = stock.load_reference(a.reference, a.dot, a.hub_revision)
  tab = TuneTable.load_many(a.table.split(",")) if a.table else None
  cc = torch.cuda.get_device_capability()
  meta = {"gpu": torch.cuda.get_device_name(), "cc": f"sm_{cc[0]}{cc[1]}", "torch": torch.__version__,
          "triton": __import__("triton").__version__, "dot": a.dot, "reference": {"name": ref.name, **ref.provenance},
          "kernel_sha": kernel.KERNEL_SHA, "peak": a.peak, "table": a.table,
          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
  print(json.dumps(meta), flush=True)
  rows, groups = [], []
  for M in map(int, a.tokens.split(",")):
    for N, K in shapes.parse_shapes(a.shapes):
      rows.append(shape_row(M, N, K, ref, a, tab, fp8))
      print(fmt(rows[-1]), flush=True)
      torch.cuda.empty_cache()
    if a.groups:
      for name in shapes.input_groups():
        groups.append(group_row(M, name, ref, a, tab))
        g = groups[-1]
        print(f"M={M:>5} group {name:<11} separate {g['stock_separate']['eager_ms']:.3f}/{g['stock_separate']['graph_ms']:.3f}"
              f" ms vs concat {g['stock_concat']['eager_ms']:.3f}/{g['stock_concat']['graph_ms']:.3f} ms (eager/device):"
              f" x{g['concat_gain_eager']:.2f} eager, x{g['concat_gain_graph']:.2f} device", flush=True)
        torch.cuda.empty_cache()
  req = per_request(rows)
  print("per-request GEMM ms (M = each workload's bucket representative):", json.dumps(req), flush=True)
  json.dump({"meta": meta, "rows": rows, "groups": groups, "per_request": req}, open(a.out, "w"), indent=1)
  print("wrote", a.out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
