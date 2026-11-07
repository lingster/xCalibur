"""Per-shape search for the block-FP8 GEMM: xCalibur's measured game, one build per (N, K, M bucket).

For each (N, K): loads are M buckets; the incumbent is the stock hub kernel; every timing is a paired ABBA run
against stock in the same session (``safebench.SafeBench``: CUDA graphs, steady-state ABBA blocks by default).

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
import sys
import time

import torch

import xcalibur as xc

import kernel
import parity
import safebench
import shapes
import space
import stock
import public
from combine import kernel_shas
from plan import aa_margin, fresh_confirm, make_game, play, seed_designs
from session import Session, log
from ship import parity_gate
from table import TuneTable

DRY = dict(shapes="1024x5120", buckets="m64,m4k", rounds=1, seed_n=20, screen_pairs=4, confirm_pairs=8, fresh_runs=2,
           compile_n=12, screen_n=6, per_load=4, reference="twin")


def run_shape(N: int, K: int, buckets: list[shapes.Bucket], ref: stock.Reference, a, bench: xc.Bench,
              tab: TuneTable) -> None:
  log(f"shape N={N} K={K} ({', '.join(shapes.gemm_names(N, K)) or 'custom'}) buckets {[b.name for b in buckets]}")
  ses = Session(N, K, buckets, ref, a, bench, a.drift_tol)
  aa = {L: ses.aa(L, a.confirm_pairs) for L in ses.loads}
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
  ses.restrict(bound)
  fresh = fresh_confirm(ses.measure, bound, a.fresh_runs, margin, a.min_samples) if bound else {}
  ses.only = None
  if ses.illegal or ses.dropped:
    log(f"  {len(ses.illegal)} designs illegal on this card, {ses.dropped} samples dropped (stock drifted from its A/A)")
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
      if not parity_gate(par, a.bitwise_only):
        log(f"  {b.name}: bound build FAILED parity (bitwise={par['bitwise']}, {par['max_ulp']:.1f} ulp), keeping stock")
        hit, cfg = None, None
    tab.add(N, K, b, cfg if hit else None, gain=hit["gain"] if hit else 0.0, samples=hit["samples"] if hit else 0,
            search_gain=hit["search_gain"] if hit else None, **extra)
    log(f"  {b.name}: " + (f"+{hit['gain'] * 100:.2f}% over stock ({hit['samples']} fresh runs) {cfg}" if hit
                           else "stock kept"))


def confirm_table(path: str, ref: stock.Reference, a, bench, tab: TuneTable) -> None:
  """Re-time every tuned entry of a finished table in this (fresh) process, A/A first, and keep only entries that
  bind again on this session's evidence alone; the others revert to stock (config null). Every entry starts as
  pending (config null, the build kept under ``candidate``) and the table is saved after each shape, so a crash can
  never leave an unconfirmed build in the output (ce74c95 left gate/up and down_proj that way)."""
  old = TuneTable.load(path)
  searched = kernel_shas(old.meta)
  if not searched or not all(kernel.fp8_path_compatible(s) for s in searched):
    raise ValueError(f"{path}: searched on kernel {searched}, not this fp8 path ({kernel.KERNEL_SHA}); search again")
  tab.meta["searched_kernel_sha"] = searched                       # meta kernel_sha is this (confirming) process's
  tab.entries = [dict(e, config=None, candidate=e["config"], confirm={"status": "pending"}) if e["config"] else dict(e)
                 for e in old.entries]
  tab.save(a.out)
  for N, K in dict.fromkeys((e["N"], e["K"]) for e in old.entries if e["config"]):
    rows = [e for e in tab.entries if (e["N"], e["K"]) == (N, K) and e.get("candidate")]
    ses = Session(N, K, [shapes.bucket_named(e["bucket"]) for e in rows], ref, a, bench, a.drift_tol)
    margin = max(aa_margin(ses.aa(L, a.confirm_pairs), a.margin) for L in ses.loads)
    bound = {e["bucket"]: {"design": e["candidate"]} for e in rows}
    ses.restrict(bound)
    fresh = fresh_confirm(ses.measure, bound, a.fresh_runs, margin, a.min_samples)
    for e in rows:
      f = fresh.get(e["bucket"])
      e["confirm"] = {"gain": f["gain"] if f else None, "samples": f["samples"] if f else 0, "margin": margin,
                      "status": "bound" if f else "reverted", **ses.times(e["candidate"], e["bucket"])}
      log(f"N={N} K={K} {e['bucket']}: table +{e['gain'] * 100:.2f}% -> fresh " +
          (f"+{f['gain'] * 100:.2f}% ({f['samples']} runs): kept" if f else "did not bind: reverted to stock"))
      if f:
        e["config"] = e["candidate"]
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
  ap.add_argument("--timing", default="steady", choices=("steady", "cold"),
                  help="steady: blocks of back-to-back replays at the power cap, weights rotated past L2 (default); "
                       "cold: xcalibur.Bench's one cold replay per sample (inflated gains on the 300 W Ada, ce74c95)")
  ap.add_argument("--drift-tol", type=float, default=0.10,
                  help="drop a paired sample whose stock time is this far from the session's A/A stock time")
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
  bench = safebench.SafeBench(mode=a.timing)
  try:
    if a.confirm_table:
      confirm_table(a.confirm_table, ref, a, bench, tab)
      log(f"wrote {a.out}: {sum(1 for e in tab.entries if e['config'])} tuned entries survived the fresh confirm")
      return 0
    buckets = [shapes.bucket_named(b) for b in a.buckets.split(",")]
    for N, K in shapes.parse_shapes(a.shapes):
      run_shape(N, K, buckets, ref, a, bench, tab)
      tab.save(a.out)                                               # after every shape: a long run keeps its work
  except safebench.CudaPoisoned as e:
    tab.save(a.out)
    log(f"STOPPED: {e}. Saved {a.out} (finished shapes only; rerun the rest in a new process)")
    return 2
  log(f"wrote {a.out}: {sum(1 for e in tab.entries if e['config'])}/{len(tab.entries)} buckets tuned")
  return 0


if __name__ == "__main__":
  sys.exit(main())
