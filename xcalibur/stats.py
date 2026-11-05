"""Statistics for noisy kernel comparisons: intervals, pooling by build identity, and binding.

When real gains (0.5-2%) are the same size as run-to-run noise, three rules matter more than any model:

* **screens nominate, confirms decide**: a cheap screen only proposes; a design takes a title only after a larger
  paired run next to the incumbent beats it by a margin;
* **pool what is identical**: designs that are the same kernel for a load (equal ``ReachMap.identity``) are repeated
  measurements of one thing, so their samples count together;
* **beware the winner's curse**: the best of many noisy reads reads high; re-measure before believing it.
"""
from __future__ import annotations

import math
import random


def geomean(xs) -> float:
  xs = [x for x in xs if x and x > 0]
  return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def bootstrap_ci(ratios, level: float = 0.95, n: int = 2000, seed: int = 0):
  """Percentile bootstrap interval of the median of paired ratios."""
  if not ratios:
    return (float("nan"), float("nan"))
  rng = random.Random(seed)
  meds = sorted(sorted(rng.choices(ratios, k=len(ratios)))[len(ratios) // 2] for _ in range(n))
  lo, hi = meds[int((1 - level) / 2 * n)], meds[int((1 + level) / 2 * n) - 1]
  return (lo, hi)


class Evidence:
  """Per (load, identity): gains of confirm-grade runs, each measured next to the incumbent in the same session."""

  def __init__(self):
    self.rows = {}

  def add(self, load: str, identity: str, gain: float, design=None):
    e = self.rows.setdefault(load, {}).setdefault(identity, {"design": design, "gains": []})
    e["gains"].append(gain)

  def bind(self, min_samples: int = 2, margin: float = 0.005):
    """The best identity per load with >= min_samples, mean gain > margin and no negative sample;
    plus single-sample leaders above the margin, to re-measure next."""
    bound, again = {}, []
    for load, ids in self.rows.items():
      for ident, e in ids.items():
        g = e["gains"]
        m = sum(g) / len(g)
        if len(g) >= min_samples and m > margin and min(g) > 0:
          if m > bound.get(load, {}).get("gain", -1):
            bound[load] = {"identity": ident, "design": e["design"], "gain": m, "samples": len(g)}
        elif len(g) < min_samples and m > margin:
          again.append((load, ident, e["design"]))
    return bound, again
