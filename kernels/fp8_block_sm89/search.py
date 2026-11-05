"""Per-shape search for the block-FP8 GEMM: xCalibur's measured game, one build per (N, K, M bucket).

For each (N, K): loads are M buckets; the incumbent is the stock hub kernel; every timing is a paired ABBA run
against stock in the same session (``xcalibur.Bench.paired``: CUDA graphs, cold L2, null replay subtracted).

  1. A/A: stock against itself per bucket; the binding margin is max(--margin, |A/A - 1| + CI half-width).
  2. Seed screen (stock, anchors, random legal designs), confirm the seed leaders next to stock.
  3. Rounds of the game: surrogate (LightGBM, or numpy ridge without it) ranks, sm_89 compiles decide legality and
     give exact features (aot.py), screens nominate, confirms decide, evidence binds (>= 2 samples, mean gain above
     the margin, no negative sample).
  4. Winner's curse: each bound build is re-timed in --fresh-runs fresh confirm sessions next to stock (A/A first)
     and must bind again on that fresh evidence alone (``plan.fresh_confirm``); only then is it a winner.
  5. Parity: each winner is checked against stock (parity.py); a build that changed the math is dropped.
  6. The table: per (N, K, bucket) the winning build, or null = keep stock.

Seeds include vLLM's tuned L40S (AD102, sm_89) block-FP8 configs nearest each shape (public.py).

  python search.py --shapes 5120x17408 --buckets m28k,m4k,m512,m64 --out tuned.json         # on Ada
  python search.py --dry-run --out dry.json                                                  # plumbing on sm_86
  python search.py --confirm-table tuned.json --out confirmed.json      # winner's curse, again, in a fresh process
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time

import torch

import xcalibur as xc

import aot
import kernel
import parity
import reference
import shapes
import space
import stock
import public
from plan import aa_margin, fresh_confirm, make_game, play, seed_designs
from table import TuneTable

DRY = dict(shapes="1024x5120", buckets="m64,m4k", rounds=1, seed_n=20, screen_pairs=4, confirm_pairs=8, fresh_runs=2,
           compile_n=12, screen_n=6, per_load=4, reference="twin")


def log(msg: str) -> None:
  print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Session:
  """The hardware half of one shape's game: problems per bucket, the reference, and the measure/compile callbacks."""

  def __init__(self, N: int, K: int, buckets: list[shapes.Bucket], ref: stock.Reference, a, bench: xc.Bench):
    self.N, self.K, self.ref, self.a, self.bench = N, K, ref, a, bench
    self.loads = {b.name: b.M for b in buckets}
    _, self.w, self.ws = reference.random_problem(1, N, K, "cuda", seed=1)
    self.x = {b.name: reference.random_problem(b.M, 128, K, "cuda", seed=b.M)[0] for b in buckets}
    self.q = {L: kernel.act_quant(x, dot=a.dot) for L, x in self.x.items()} if a.quant_cost == "shared" else {}
    self.raw: list[dict] = []

  def _ref_fn(self, L):
    x = self.x[L]
    return lambda: self.ref.fn(x, self.w, self.ws)

  def _call(self, L, cfg: space.TileConfig):
    x, w, ws, dot = self.x[L], self.w, self.ws, self.a.dot
    if cfg.quant == "once" and L in self.q:                         # quant shared with the input group: not charged
      q, s = self.q[L]
      return lambda: kernel.block_fp8_matmul(q, w, ws, cfg, a_scale=s, dot=dot)
    return lambda: kernel.block_fp8_matmul(x, w, ws, cfg, dot=dot)   # "once" pays its own quant kernel here

  def measure(self, designs: list[dict], grade: str) -> list[dict]:
    pairs = self.a.screen_pairs if grade == "screen" else self.a.confirm_pairs
    rows = []
    for d in designs:
      row = {}
      for L, M in self.loads.items():
        if not space.applicable(d, M, self.K):
          continue
        ref_fn = self._ref_fn(L)
        try:
          fn = ref_fn if space.is_stock(d) else self._call(L, space.to_config(d, M))
          r = self.bench.paired(fn, ref_fn, pairs)
        except Exception as e:                                      # noqa: BLE001  (e.g. out of resources on sm_86)
          log(f"  measure failed {L} {d}: {type(e).__name__}: {str(e)[:120]}")
          continue
        row[L] = r["ratio"]
        self.raw.append({"load": L, "grade": grade, "design": d, "ratio": r["ratio"], "ci": r["ci"],
                         "candidate_us": r["candidate_us"], "reference_us": r["reference_us"]})
      rows.append(row)
    return rows

  def compile(self, designs: list[dict]) -> list[dict]:
    """Exact sm_89 compile of the fp8 build: legality (Triton, ptxas, <= 99 KB smem) and resource features."""
    return [aot.compile_sm89(space.to_config(d, 128)) for d in designs]

  def times(self, d: dict, L: str) -> dict:
    rs = [r for r in self.raw if r["load"] == L and r["design"] == d and r["grade"] == "confirm"]
    if not rs:
      return {}
    return {"tuned_us": statistics.median(r["candidate_us"] for r in rs),
            "stock_us": statistics.median(r["reference_us"] for r in rs),
            "ratio": statistics.median(r["ratio"] for r in rs), "confirm_runs": len(rs)}


def run_shape(N: int, K: int, buckets: list[shapes.Bucket], ref: stock.Reference, a, bench: xc.Bench,
              tab: TuneTable) -> None:
  log(f"shape N={N} K={K} ({', '.join(shapes.gemm_names(N, K)) or 'custom'}) buckets {[b.name for b in buckets]}")
  ses = Session(N, K, buckets, ref, a, bench)
  aa = {L: bench.aa(ses._ref_fn(L), pairs=a.confirm_pairs) for L in ses.loads}
  margin = max(aa_margin(r, a.margin) for r in aa.values())
  for L, r in aa.items():
    log(f"  A/A {L}: ratio {r['ratio']:.4f} CI {r['ci'][0]:.4f}-{r['ci'][1]:.4f}  stock {r['reference_us']:.1f} us")
  log(f"  binding margin {margin * 100:.2f}%")
  keep = space.bitwise_safe if a.bitwise_only else None
  game = make_game(ses.loads, K, ses.measure, ses.compile, surrogate_backend=a.surrogate, margin=margin, keep=keep,
                   min_samples=a.min_samples, per_load=a.per_load, compile_n=a.compile_n, screen_n=a.screen_n)
  pub = [d for b in buckets for d in public.seeds(N, K, b.M)[:2]]
  seeds = [d for d in seed_designs(a.seed_n, random.Random(a.seed), extra=pub) if keep is None or keep(d)]
  bound = play(game, seeds, a.rounds, log=lambda m: log("  " + m))
  log(f"  search bound {sorted(bound)}; fresh confirm x{a.fresh_runs} next to stock (winner's curse)")
  fresh = fresh_confirm(ses.measure, bound, a.fresh_runs, margin, a.min_samples) if bound else {}
  for L in bound:
    log(f"  {L}: search +{bound[L]['gain'] * 100:.2f}% -> fresh " +
        (f"+{fresh[L]['gain'] * 100:.2f}% ({fresh[L]['samples']} runs)" if L in fresh else "did not bind: dropped"))
  bound = {L: dict(fresh[L], search_gain=bound[L]["gain"]) for L in fresh}
  for b in buckets:
    hit = bound.get(b.name)
    extra = {"aa": {"ratio": aa[b.name]["ratio"], "ci": aa[b.name]["ci"]}, "margin": margin,
             "stock_choice": ref.choice(N, K, b.M)}
    if hit:
      cfg = space.to_config(hit["design"], b.M)
      par = parity.check(b.M, N, K, cfg, ref, a.dot)
      extra.update(ses.times(hit["design"], b.name), parity={k: par[k] for k in ("ok", "bitwise", "max_abs", "max_ulp")})
      if not par["ok"]:
        log(f"  {b.name}: bound build FAILED parity ({par['max_ulp']:.1f} ulp), keeping stock")
        hit, cfg = None, None
    tab.add(N, K, b, cfg if hit else None, gain=hit["gain"] if hit else 0.0, samples=hit["samples"] if hit else 0,
            search_gain=hit["search_gain"] if hit else None, **extra)
    log(f"  {b.name}: " + (f"+{hit['gain'] * 100:.2f}% over stock ({hit['samples']} fresh runs) {cfg}" if hit
                           else "stock kept"))


def confirm_table(path: str, ref: stock.Reference, a, bench: xc.Bench, tab: TuneTable) -> None:
  """Re-time every tuned entry of a finished table in this (fresh) process, A/A first, and keep only entries that
  bind again on this session's evidence alone; the others revert to stock (config null)."""
  old = TuneTable.load(path)
  tab.entries = [dict(e) for e in old.entries]
  for N, K in dict.fromkeys((e["N"], e["K"]) for e in old.entries if e["config"]):
    rows = [e for e in tab.entries if (e["N"], e["K"]) == (N, K) and e["config"]]
    ses = Session(N, K, [shapes.bucket_named(e["bucket"]) for e in rows], ref, a, bench)
    margin = max(aa_margin(bench.aa(ses._ref_fn(L), pairs=a.confirm_pairs), a.margin) for L in ses.loads)
    fresh = fresh_confirm(ses.measure, {e["bucket"]: {"design": e["config"]} for e in rows}, a.fresh_runs, margin,
                          a.min_samples)
    for e in rows:
      f = fresh.get(e["bucket"])
      e["confirm"] = {"gain": f["gain"] if f else None, "samples": f["samples"] if f else 0, "margin": margin}
      log(f"N={N} K={K} {e['bucket']}: table +{e['gain'] * 100:.2f}% -> fresh " +
          (f"+{f['gain'] * 100:.2f}% ({f['samples']} runs): kept" if f else "did not bind: reverted to stock"))
      if not f:
        e["config"] = None
    tab.save(a.out)


def parse(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--shapes", default="all", help="'all' or NxK,NxK (concat groups: 14336x5120, 34816x5120, 16384x5120)")
  ap.add_argument("--buckets", default=",".join(shapes.DEFAULT_BUCKETS))
  ap.add_argument("--rounds", type=int, default=4)
  ap.add_argument("--seed-n", type=int, default=32)
  ap.add_argument("--screen-pairs", type=int, default=10)
  ap.add_argument("--confirm-pairs", type=int, default=30)
  ap.add_argument("--compile-n", type=int, default=32)
  ap.add_argument("--screen-n", type=int, default=12)
  ap.add_argument("--per-load", type=int, default=6)
  ap.add_argument("--margin", type=float, default=0.005)
  ap.add_argument("--min-samples", type=int, default=2)
  ap.add_argument("--fresh-runs", type=int, default=3, help="fresh confirm sessions per winner (winner's curse)")
  ap.add_argument("--bitwise-only", action="store_true",
                  help="search only builds bitwise equal to stock (BLOCK_K 128, no split-K): no answer drift possible")
  ap.add_argument("--quant-cost", default="call", choices=("call", "shared"),
                  help="'call': quantise-once designs pay their quant kernel per GEMM (conservative); 'shared': not "
                       "charged (for GEMMs whose input group shares one quantisation)")
  ap.add_argument("--reference", default="auto", choices=("auto", "hub", "twin"))
  ap.add_argument("--hub-revision", default=None, help=f"pin the hub kernel (production: {stock.HUB_REVISION})")
  ap.add_argument("--dot", default=None, choices=(None, "fp8", "bf16", "fp32"), help="default: fp8 on sm_89, bf16 below")
  ap.add_argument("--surrogate", default="auto", choices=("auto", "lightgbm", "ridge"))
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--dry-run", action="store_true", help="small budgets, twin reference: prove the plumbing on sm_86")
  ap.add_argument("--confirm-table", help="instead of searching: fresh confirm of every tuned entry in this table")
  ap.add_argument("--out", default="tuned.json")
  a = ap.parse_args(argv)
  if a.dry_run:
    for k, v in DRY.items():
      if getattr(a, k) == ap.get_default(k):
        setattr(a, k, v)
  return a


def main(argv=None) -> int:
  a = parse(argv)
  a.dot = a.dot or kernel.default_dot("cuda")
  ref = stock.load_reference(a.reference, a.dot, a.hub_revision)
  cc = torch.cuda.get_device_capability()
  tab = TuneTable(meta={"gpu": torch.cuda.get_device_name(), "cc": f"sm_{cc[0]}{cc[1]}", "torch": torch.__version__,
                        "triton": __import__("triton").__version__, "dot": a.dot, "kernel_sha": kernel.KERNEL_SHA,
                        "reference": {"name": ref.name, **ref.provenance}, "args": vars(a),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dry_run": a.dry_run})
  log(f"{tab.meta['gpu']} {tab.meta['cc']} dot={a.dot} reference={ref.name} kernel={kernel.KERNEL_SHA}")
  bench = xc.Bench()
  if a.confirm_table:
    confirm_table(a.confirm_table, ref, a, bench, tab)
    log(f"wrote {a.out}: {sum(1 for e in tab.entries if e['config'])} tuned entries survived the fresh confirm")
    return 0
  buckets = [shapes.bucket_named(b) for b in a.buckets.split(",")]
  for N, K in shapes.parse_shapes(a.shapes):
    run_shape(N, K, buckets, ref, a, bench, tab)
    tab.save(a.out)                                                 # after every shape: a long run keeps its work
  log(f"wrote {a.out}: {sum(1 for e in tab.entries if e['config'])}/{len(tab.entries)} buckets tuned")
  return 0


if __name__ == "__main__":
  sys.exit(main())
