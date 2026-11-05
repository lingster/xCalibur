"""Statistics for noisy kernel comparisons: intervals, pooling by build identity, and binding.

When real gains (0.5-2%) are the same size as run-to-run noise, three rules matter more than any model:

* **screens nominate, confirms decide**: a cheap screen only proposes; a design takes a title only after a larger
  paired run next to the incumbent beats it by a margin;
* **pool what is identical**: validate the declared ``ReachMap.identity`` against the lowering, then pool independent
  observations of that build under a fixed reference and protocol;
* **beware the winner's curse**: the best of many noisy reads reads high; re-measure before believing it.
"""
from __future__ import annotations

import math
import random
import copy
import statistics


def _positive_values(values):
  values = [float(x) for x in values]
  if not values or any(not math.isfinite(x) or x <= 0 for x in values):
    raise ValueError("expected a nonempty sequence of finite positive ratios")
  return values


def geomean(xs) -> float:
  """Geometric mean over every supplied ratio; invalid loads are never discarded."""
  xs = _positive_values(xs)
  return math.exp(math.fsum(map(math.log, xs)) / len(xs))


def bootstrap_ci(ratios, level: float = 0.95, n: int = 2000, seed: int = 0):
  """Percentile bootstrap interval of the median of paired ratios."""
  ratios = _positive_values(ratios)
  if not 0 < level < 1 or not isinstance(n, int) or isinstance(n, bool) or n < 1:
    raise ValueError("level must lie in (0, 1) and n must be a positive integer")
  rng = random.Random(seed)
  meds = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(n))
  def percentile(p):
    at = (n - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return meds[lo] + (meds[hi] - meds[lo]) * (at - lo)
  return (percentile((1 - level) / 2), percentile((1 + level) / 2))


class Evidence:
  """Confirmation gains against one fixed reference and measurement protocol.

  Each add must be an independent confirmation observation. The caller owns build
  identity validation and starts new evidence when the reference or protocol changes.
  """

  def __init__(self):
    self.rows = {}

  def add(self, load: str, identity: str, gain: float, design=None):
    gain = float(gain)
    if not math.isfinite(gain) or gain <= -1:
      raise ValueError("gain must be finite and greater than -1 (a positive timing ratio)")
    e = self.rows.setdefault(load, {}).setdefault(identity, {"design": copy.deepcopy(design), "gains": []})
    e["gains"].append(gain)

  def bind(self, min_samples: int = 2, margin: float = 0.005):
    """A repeat-and-margin heuristic, not a confidence-interval significance test.

    Return the best identity per load with >= min_samples, mean gain > margin and
    every gain positive, plus promising identities needing more observations.
    """
    if not isinstance(min_samples, int) or isinstance(min_samples, bool) or min_samples < 1:
      raise ValueError("min_samples must be a positive integer")
    if not math.isfinite(margin) or margin < 0:
      raise ValueError("margin must be finite and nonnegative")
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
