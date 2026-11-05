"""Parity: a tuned build must compute the stock kernel's numbers (same math, different tiling).

Same per-128-group e4m3 activation quantisation, same 128 x 128 weight block scales, fp32 accumulation: a retiling
with the same K tile and no split is bitwise equal; a different K tile or split-K changes only the fp32 summation
order, which after the bf16 cast moves an element by at most one bf16 ulp (or by fp32 noise next to cancellation).
That bound is the rule: anything beyond it changed the math and is rejected.

  python parity.py --table tuned.json                     # every tuned entry, at its bucket's representative M
  python parity.py --shapes 5120x17408 --tokens 64,28672 --config '{"quant": "once", "BLOCK_M": 128, ...}'
  python parity.py --twin-check --reference hub          # on Ada: our kernel at stock tiling vs the hub, bitwise?
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

import kernel
import reference
import shapes
import stock
from space import TileConfig
from table import TuneTable


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
  """The bf16 spacing at |x| (7 stored mantissa bits); the subnormal floor for tiny values."""
  e = torch.floor(torch.log2(x.abs().float().clamp_min(2.0 ** -126)))
  return torch.exp2(e - 7)


def compare(out: torch.Tensor, ref: torch.Tensor, atol_rel: float = 1e-4) -> dict:
  """Elementwise out vs stock: bitwise flag, max |diff|, max diff in bf16 ulps of the larger magnitude, and ok =
  every element within one ulp or within ``atol_rel * max|stock|`` (fp32 order noise next to cancellation)."""
  a, b = out.float(), ref.float()
  d = (a - b).abs()
  ulps = d / bf16_ulp(torch.maximum(a.abs(), b.abs()))
  floor = atol_rel * b.abs().max().clamp_min(1e-30)
  return {"bitwise": bool(torch.equal(out.view(torch.int16) if out.dtype == torch.bfloat16 else out,
                                      ref.view(torch.int16) if ref.dtype == torch.bfloat16 else ref)),
          "max_abs": float(d.max()), "max_ulp": float(ulps.max()), "mismatch_frac": float((d > 0).float().mean()),
          "ok": bool(((ulps <= 1.0) | (d <= floor)).all())}


def check(M: int, N: int, K: int, cfg: TileConfig, ref: stock.Reference, dot: str | None = None, seed: int = 0) -> dict:
  """Tuned vs stock at one size, plus each one's max |error| against the fp32 reference math."""
  x, w, ws = reference.random_problem(M, N, K, "cuda", seed)
  tuned = kernel.block_fp8_matmul(x, w, ws, cfg, dot=dot)
  base = ref.fn(x, w, ws)
  exact = reference.block_fp8_matmul(x, w, ws)
  out = compare(tuned, base)
  out.update(M=M, N=N, K=K, config=cfg.as_dict(), reference=ref.name,
             tuned_vs_fp32=float((tuned.float() - exact).abs().max()), stock_vs_fp32=float((base.float() - exact).abs().max()))
  return out


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--table", help="tuned table JSON from search.py")
  ap.add_argument("--config", help="one TileConfig as JSON (with --shapes / --tokens)")
  ap.add_argument("--shapes", default="all")
  ap.add_argument("--tokens", default="64,512,4096,28672")
  ap.add_argument("--reference", default="auto", choices=("auto", "hub", "twin"))
  ap.add_argument("--hub-revision", default=None)
  ap.add_argument("--dot", default=None, choices=(None, "fp8", "bf16", "fp32"))
  ap.add_argument("--twin-check", action="store_true", help="our kernel at the hub's tiling vs the reference")
  ap.add_argument("--out", default="parity.json")
  a = ap.parse_args(argv)
  ref = stock.load_reference(a.reference, a.dot, a.hub_revision)
  jobs = []
  if a.table:
    for e in TuneTable.load(a.table).entries:
      if e["config"]:
        jobs.append((e["M"], e["N"], e["K"], TileConfig(**e["config"])))
  if a.config:
    cfg = TileConfig(**json.loads(a.config))
    jobs += [(M, N, K, cfg) for N, K in shapes.parse_shapes(a.shapes) for M in map(int, a.tokens.split(","))]
  twin = stock.Twin(a.dot) if a.twin_check else None
  if twin:
    jobs += [(M, N, K, None) for N, K in shapes.parse_shapes(a.shapes) for M in map(int, a.tokens.split(","))]
  if not jobs:
    ap.error("nothing to check: give --table with tuned entries, --config, or --twin-check")
  rows = []
  for M, N, K, cfg in jobs:
    if cfg is None:                                     # twin check: resolve the twin's autotuned stock config
      x, w, ws = reference.random_problem(M, N, K, "cuda", 0)
      cfg = twin.config(x, w, ws)
    r = check(M, N, K, cfg, ref, a.dot)
    rows.append(r)
    print(f"{'OK ' if r['ok'] else 'BAD'} M={M} N={N} K={K} bitwise={r['bitwise']} max_abs={r['max_abs']:.3g} "
          f"max_ulp={r['max_ulp']:.2f} tuned_vs_fp32={r['tuned_vs_fp32']:.3g} stock_vs_fp32={r['stock_vs_fp32']:.3g} "
          f"{json.dumps(r['config'])}", flush=True)
  json.dump({"reference": ref.provenance | {"name": ref.name}, "kernel_sha": kernel.KERNEL_SHA, "rows": rows},
            open(a.out, "w"), indent=1)
  print(f"wrote {a.out}: {sum(r['ok'] for r in rows)}/{len(rows)} ok")
  return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
  sys.exit(main())
