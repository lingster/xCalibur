"""Paired, judge-faithful timing of a candidate kernel against a reference.

The method copies what GPU kernel judges typically do, and each piece removes a known source of error:

* **CUDA graphs.** Each callable is captured once and replayed, so Python and launch overhead is the same for both.
* **Cold L2 by a read.** Before every replay a large buffer (default 256 MiB) is *read*. A write flush leaves L2 full
  of dirty lines whose write-back is then charged to the timed kernel; we measured that error at 5-11 us per replay.
* **ABBA pairs.** Candidate and reference alternate (A B B A ...), so drift (clocks, temperature) cancels in the ratio.
* **A null replay.** The fixed cost of replaying a graph (about 16-19 us on an H200 in our setup) is measured and
  subtracted, otherwise it dilutes every ratio of microsecond-scale kernels.
* **A/A.** Timing the reference against itself in the same session gives the noise floor to judge a gain against.

  >>> import torch, torch.nn.functional as F
  >>> x = torch.randn(16, 8192, device="cuda", dtype=torch.bfloat16)
  >>> w = torch.randn(2304, 8192, device="cuda", dtype=torch.bfloat16)
  >>> r = paired(lambda: x @ w.T, lambda: F.linear(x, w), pairs=20)          # doctest: +SKIP
  >>> r["ratio"], r["ci"]                                                    # doctest: +SKIP
"""
from __future__ import annotations

import statistics

from .stats import bootstrap_ci


class Bench:
  def __init__(self, flush_mib: int = 256, warmup: int = 3):
    import torch                              # noqa: PLC0415
    self.torch = torch
    self.flush = torch.empty(flush_mib * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    self.warmup = warmup
    self._null = None

  def graph(self, fn):
    """Capture fn in a CUDA graph (after warm-up on a side stream, as graph capture requires)."""
    t = self.torch
    s = t.cuda.Stream()
    s.wait_stream(t.cuda.current_stream())
    with t.cuda.stream(s):
      for _ in range(self.warmup):
        fn()
    t.cuda.current_stream().wait_stream(s)
    g = t.cuda.CUDAGraph()
    with t.cuda.graph(g):
      fn()
    return g

  def replay_us(self, g) -> float:
    """One cold replay: read-flush L2, then time the replay alone with events."""
    t = self.torch
    self.flush.sum()
    a, b = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    t.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0

  def null_us(self, n: int = 40) -> float:
    if self._null is None:
      tiny = self.torch.zeros(1, device="cuda")
      g = self.graph(lambda: tiny.add_(0))
      self._null = statistics.median(self.replay_us(g) for _ in range(n))
    return self._null

  def paired(self, candidate, reference, pairs: int = 30, subtract_null: bool = True) -> dict:
    """ABBA-paired times of candidate vs reference; ratio > 1 means the candidate is faster."""
    gc, gr = self.graph(candidate), self.graph(reference)
    tc, tr = [], []
    for i in range(pairs):
      if i % 2 == 0:
        tr.append(self.replay_us(gr)); tc.append(self.replay_us(gc))
      else:
        tc.append(self.replay_us(gc)); tr.append(self.replay_us(gr))
    z = self.null_us() if subtract_null else 0.0
    tc, tr = [x - z for x in tc], [x - z for x in tr]
    ratios = [r / c for r, c in zip(tr, tc)]
    return {"candidate_us": statistics.median(tc), "reference_us": statistics.median(tr),
        "ratio": statistics.median(ratios), "ci": bootstrap_ci(ratios), "null_us": z, "pairs": pairs}

  def aa(self, reference, pairs: int = 30) -> dict:
    """The reference against itself: the session's noise floor."""
    return self.paired(reference, reference, pairs)


def paired(candidate, reference, pairs: int = 30, **kw) -> dict:
  return Bench(**kw).paired(candidate, reference, pairs)
