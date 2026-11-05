"""What may ship: the rules that turn search / confirm / parity / efficiency outputs into a production table.

Under FP8 any non-bitwise change brings back the full answer drift (kev ledger FP8D), so production takes only
entries that are

  bitwise    parity against the hub kernel itself: bitwise equal (not merely "within 1 ulp")
  confirmed  bound again in a separate ``search.py --confirm-table`` process (>= 2 fresh samples, mean gain above that
             session's A/A margin). An entry without a ``confirm`` record was never confirmed: the ce74c95 confirm
             crashed before gate/up and down_proj, and their configs were still in the file.
  steady     (optional) faster than stock in efficiency.py's steady-state device clock by more than the margin:
             max(--steady-margin, the entry's confirm-session A/A margin), so a tie on noise does not ship (winner's
             curse). The clock is rot_ms (weights rotated past L2, as production reads them) when the eff file has it,
             else the warm-L2 graph_ms, which overstates small-M gains; the entry records which clock decided.
             ce74c95's search timed single cold replays, which on the power-capped Ada inflated gains about 2x.
  kernel     every source table was measured on this kernel's fp8 path (kernel.fp8_path_compatible).
  per-Linear (optional) a shape a single FP8Linear has; concat-group shapes (q|k|v, gate|up, qkv|z) need the
             weights concatenated first, which production does not do.

  python ship.py --confirmed confirmed_bitwise.json --parity parity_bitwise.json --eff eff_tuned.json \
      --out tables/ada_rtx6000_bitwise_r1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import shapes
from combine import kernel_shas
from table import TuneTable

STEADY_MARGIN = 0.01


def parity_gate(par: dict | None, bitwise_only: bool) -> bool:
  """The search's parity rule: bitwise searches need bitwise equality (on Ada an order-only build can be many ulps
  off, see the kernel tests); full searches need parity.compare's ``ok`` (the answer gate decides later)."""
  if not par:
    return False
  return bool(par.get("bitwise")) if bitwise_only else bool(par.get("ok"))


def confirmed(e: dict) -> bool:
  c = e.get("confirm") or {}
  return (bool(e.get("config")) and c.get("gain") is not None and c.get("samples", 0) >= 2
          and c["gain"] > c.get("margin", 0.0))


def _same(cfg_a: dict, cfg_b: dict) -> bool:
  return cfg_a is not None and cfg_b is not None and dict(cfg_a) == dict(cfg_b)


def bitwise_on_hardware(e: dict, parity_rows: list[dict]) -> bool:
  """A parity.py row for this (M, N, K, config) against the hub kernel that is bitwise; else the entry's own record."""
  rows = [r for r in parity_rows if (r["M"], r["N"], r["K"]) == (e["M"], e["N"], e["K"]) and _same(r["config"], e["config"])]
  if rows:
    return all(r.get("bitwise") and r.get("reference", "hub") == "hub" for r in rows)
  return bool((e.get("parity") or {}).get("bitwise"))


def steady(e: dict, eff_rows: list[dict]) -> tuple[float | None, str | None]:
  """(stock / tuned device time, clock) from efficiency.py at this (M, N, K) when it timed this very config.
  rot_ms (cold weights) when both sides have it, else graph_ms (warm L2: an upper bound for small M)."""
  for r in eff_rows:
    t = r.get("tuned")
    if (r["M"], r["N"], r["K"]) == (e["M"], e["N"], e["K"]) and t and _same(t.get("config"), e["config"]):
      clock = "rot_ms" if "rot_ms" in t and "rot_ms" in r["stock"] else "graph_ms"
      return r["stock"][clock] / t[clock], clock
  return None, None


def steady_ratio(e: dict, eff_rows: list[dict]) -> float | None:
  return steady(e, eff_rows)[0]


def steady_floor(e: dict, margin: float = STEADY_MARGIN) -> float:
  """The ratio a steady win must beat: 1 + max(margin, the confirm session's A/A margin)."""
  return 1.0 + max(margin, (e.get("confirm") or {}).get("margin") or 0.0)


def select(entries: list[dict], parity_rows: list[dict], eff_rows: list[dict] | None = None,
           per_linear_only: bool = True, margin: float = STEADY_MARGIN) -> tuple[list[dict], list[tuple[dict, str]]]:
  """(kept entries, [(dropped entry, reason)]). Entries with config null are stock already: dropped silently."""
  linear = set(shapes.unique_shapes())
  kept, dropped = [], []
  for e in entries:
    if not e.get("config"):
      continue
    if per_linear_only and (e["N"], e["K"]) not in linear:
      dropped.append((e, "concat-group shape (production does not concatenate FP8 weights)"))
    elif not confirmed(e):
      dropped.append((e, "no fresh confirm"))
    elif not bitwise_on_hardware(e, parity_rows):
      dropped.append((e, "not bitwise vs the hub kernel"))
    else:
      ratio, clock = steady(e, eff_rows) if eff_rows is not None else (None, None)
      floor = steady_floor(e, margin)
      if eff_rows is not None and (ratio is None or ratio <= floor):
        dropped.append((e, f"not faster in steady state ({'untimed' if ratio is None else f'x{ratio:.3f}'} "
                           f"<= x{floor:.3f})"))
        continue
      kept.append(dict(e, steady_ratio=ratio, steady_clock=clock))
  return kept, dropped


def _names(paths: str) -> list[str]:
  return [Path(p).name for p in filter(None, paths.split(","))]


def main(argv=None, compatible=None) -> int:
  """``compatible(sha)``: kernel.fp8_path_compatible unless given (tests)."""
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--confirmed", required=True, help="confirmed table(s) from search.py --confirm-table, comma-separated")
  ap.add_argument("--parity", default="", help="parity.py --table output(s) against the hub, comma-separated")
  ap.add_argument("--eff", help="efficiency.py output with the tuned columns: keep only entries faster in steady state")
  ap.add_argument("--with-concat", action="store_true", help="keep concat-group shapes too")
  ap.add_argument("--steady-margin", type=float, default=STEADY_MARGIN, help="minimum steady gain (default 0.01)")
  ap.add_argument("--out", required=True)
  a = ap.parse_args(argv)
  src = TuneTable.load_many(a.confirmed.split(","))
  shas = sorted({s for m in src.meta.get("merged", []) for s in kernel_shas(m)})
  if compatible is None:
    import kernel  # needs triton; imported here so the selection rules above stay CPU-testable without it
    compatible = kernel.fp8_path_compatible
  if not shas or not all(compatible(s) for s in shas):
    print(f"REFUSED: sources measured on kernel {shas}, not this fp8 path")
    return 1
  par = [r for p in filter(None, a.parity.split(",")) for r in json.loads(Path(p).read_text())["rows"]]
  eff = json.loads(Path(a.eff).read_text())["rows"] if a.eff else None
  kept, dropped = select(src.entries, par, eff, per_linear_only=not a.with_concat, margin=a.steady_margin)
  for e in kept:
    print(f"KEEP N={e['N']} K={e['K']} {e['bucket']}: confirm +{e['confirm']['gain'] * 100:.0f}% (cold), steady "
          f"{'x%.3f' % e['steady_ratio'] if e.get('steady_ratio') else 'n/a'} {e['config']}")
  for e, why in dropped:
    print(f"DROP N={e['N']} K={e['K']} {e['bucket']}: {why}")
  out = TuneTable(meta={"kind": "shippable", "bitwise_only": True, "kernel_sha": shas, "sources": _names(a.confirmed),
                        "parity": _names(a.parity), "eff": _names(a.eff or ""), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "rule": "confirmed AND bitwise vs hub" + (f" AND steady > 1+max({a.steady_margin}, A/A)" if eff else "")},
                  entries=kept)
  out.save(a.out)
  print(f"wrote {a.out}: {len(kept)} entries, {len(dropped)} dropped")
  return 0


if __name__ == "__main__":
  sys.exit(main())
