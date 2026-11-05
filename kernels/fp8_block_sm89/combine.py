"""Combine ship.py tables from separate rounds into one production table.

Each round ships its own table (r1: attention and small projections; r2: the MLP shapes). Production loads one
table, so the rounds are combined here under the rules kev_patch.py enforces at load time, checked up front:

  shippable   every input is a ship.py table with ``bitwise_only`` set
  bitwise     every entry's config is in the bitwise class (space.bitwise_safe)
  kernel      every input was measured on this kernel's fp8 path (kernel.fp8_path_compatible)

For the same (N, K, bucket), later tables win, so pass the newest round last.

  python combine.py --tables tables/ada_rtx6000_bitwise_r1.json,tables/ada_rtx6000_bitwise_r2.json \
      --out tables/ada_rtx6000_bitwise_r1r2.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import space
from table import TuneTable


def _key(e: dict) -> tuple:
  return e["N"], e["K"], e["bucket"]


def kernel_shas(meta: dict) -> list[str]:
  """A table's kernel hashes; ship.py writes a list, older tables a single string."""
  sha = meta.get("kernel_sha")
  return [sha] if isinstance(sha, str) else list(sha or [])


def check(path: str, t: TuneTable, compatible: Callable[[str], bool]) -> None:
  """Raise ValueError when kev_patch.py would refuse this table."""
  if t.meta.get("kind") != "shippable" or not t.meta.get("bitwise_only"):
    raise ValueError(f"{path}: not a ship.py bitwise table (kind={t.meta.get('kind')!r})")
  shas = kernel_shas(t.meta)
  if not shas or not all(compatible(s) for s in shas):
    raise ValueError(f"{path}: measured on kernel {shas}, not this fp8 path")
  for e in t.entries:
    if not e.get("config") or not space.bitwise_safe(e["config"]):
      raise ValueError(f"{path}: entry N={e['N']} K={e['K']} {e['bucket']} is not in the bitwise class")


def combine(paths: list[str], compatible: Callable[[str], bool]) -> tuple[TuneTable, list[tuple]]:
  """(combined table, [(N, K, bucket) replaced by a later table]). Later paths win on the same key."""
  by_key, replaced, shas, rules = {}, [], set(), []
  for p in paths:
    t = TuneTable.load(p)
    check(p, t, compatible)
    shas.update(kernel_shas(t.meta))
    rules.append(t.meta.get("rule"))
    for e in t.entries:
      if _key(e) in by_key:
        replaced.append(_key(e))
      by_key[_key(e)] = dict(e, source=Path(p).name)
  meta = {"kind": "shippable", "bitwise_only": True, "kernel_sha": sorted(shas), "sources": [Path(p).name for p in paths],
          "combined": True, "rule": " | ".join(sorted(set(filter(None, rules)))),
          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
  return TuneTable(meta=meta, entries=sorted(by_key.values(), key=_key)), replaced


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--tables", required=True, help="ship.py tables, comma-separated, oldest first")
  ap.add_argument("--out", required=True)
  a = ap.parse_args(argv)
  import kernel  # needs triton; imported here so the pure functions above stay CPU-testable without it
  out, replaced = combine(a.tables.split(","), kernel.fp8_path_compatible)
  for k in replaced:
    print(f"REPLACED N={k[0]} K={k[1]} {k[2]}: a later table wins")
  out.save(a.out)
  print(f"wrote {a.out}: {len(out.entries)} entries from {len(out.meta['sources'])} tables, kernel {out.meta['kernel_sha']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
