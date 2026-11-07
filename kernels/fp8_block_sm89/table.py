"""The tuned config table: one build per (N, K, M bucket), ("each problem size its own build", as in kernels/moe_w4a16_h200).

An entry with ``config: null`` keeps the stock hub kernel for that bucket (nothing bound, or parity failed), so a
lookup miss and a "stock" entry mean the same thing: run the stock kernel.

  {"meta": {...},
   "entries": [{"N": 5120, "K": 6144, "bucket": "rows40", "lo": 1, "hi": 64, "M": 40,
                "config": {"quant": "once", "BLOCK_M": 64, ...} | null, "gain": 0.081, "samples": 3, ...}]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from shapes import Bucket
from space import TileConfig


@dataclass
class TuneTable:
  meta: dict = field(default_factory=dict)
  entries: list[dict] = field(default_factory=list)

  def add(self, N: int, K: int, bucket: Bucket, config: TileConfig | None, **extra) -> dict:
    self.entries = [e for e in self.entries if (e["N"], e["K"], e["bucket"]) != (N, K, bucket.name)]
    e = {"N": N, "K": K, "bucket": bucket.name, "lo": bucket.lo, "hi": bucket.hi, "M": bucket.M,
         "config": config.as_dict() if config else None, **extra}
    self.entries.append(e)
    return e

  def config_for(self, N: int, K: int, M: int) -> TileConfig | None:
    """The tuned build for this problem size, or None for the stock kernel."""
    for e in self.entries:
      if e["N"] == N and e["K"] == K and e["lo"] <= M <= e["hi"]:
        return TileConfig(**e["config"]) if e["config"] else None
    return None

  def save(self, path: str | Path) -> None:
    Path(path).write_text(json.dumps({"meta": self.meta, "entries": self.entries}, indent=1, default=str) + "\n")

  @classmethod
  def load(cls, path: str | Path) -> "TuneTable":
    raw = json.loads(Path(path).read_text())
    return cls(raw.get("meta", {}), raw.get("entries", []))

  @classmethod
  def load_many(cls, paths) -> "TuneTable":
    """Several tables as one (e.g. per-shape and concat-group runs); for the same (N, K, bucket) later files win."""
    out = cls(meta={"merged": []})
    for p in paths:
      t = cls.load(p)
      out.meta["merged"].append(t.meta)
      for e in t.entries:
        out.entries = [x for x in out.entries if (x["N"], x["K"], x["bucket"]) != (e["N"], e["K"], e["bucket"])]
        out.entries.append(e)
    return out
