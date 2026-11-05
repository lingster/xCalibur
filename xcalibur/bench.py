"""Paired CUDA-graph replay timing, with raw observations retained.

Each callable is captured once, with a read flush before every timed replay.
Alternating reference/candidate order reduces order bias; an A/A run estimates
repeatability. Match this protocol to the workload and benchmark being studied.

No timing correction is applied by default. Optional ``subtract_null=True``
subtracts a separately timed tiny-add graph, which includes real kernel work and
is not a universal launch-overhead estimate. It needs independent calibration.
Both raw and corrected samples remain available for inspection.

  >>> import torch, torch.nn.functional as F
  >>> x = torch.randn(16, 8192, device="cuda", dtype=torch.bfloat16)
  >>> w = torch.randn(2304, 8192, device="cuda", dtype=torch.bfloat16)
  >>> r = paired(lambda: x @ w.T, lambda: F.linear(x, w), pairs=20)          # doctest: +SKIP
  >>> r["ratio"], r["ci"]                                                    # doctest: +SKIP
"""
from __future__ import annotations

import statistics
import math

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
    """Cached tiny-add graph timing for optional, explicitly calibrated subtraction."""
    if self._null is None:
      tiny = self.torch.zeros(1, device="cuda")
      g = self.graph(lambda: tiny.add_(0))
      self._null = statistics.median(self.replay_us(g) for _ in range(n))
    return self._null

  def paired(self, candidate, reference, pairs: int = 30, subtract_null: bool = False) -> dict:
    """ABBA-paired times of candidate vs reference; ratio > 1 means the candidate is faster."""
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 1:
      raise ValueError("pairs must be a positive integer")
    gc, gr = self.graph(candidate), self.graph(reference)
    tc, tr = [], []
    for i in range(pairs):
      if i % 2 == 0:
        tr.append(self.replay_us(gr)); tc.append(self.replay_us(gc))
      else:
        tc.append(self.replay_us(gc)); tr.append(self.replay_us(gr))
    z = self.null_us() if subtract_null else 0.0
    raw_tc, raw_tr = tc, tr
    if not math.isfinite(z) or z < 0:
      raise ValueError("calibration must be finite and nonnegative")
    if any(not math.isfinite(t) or t <= 0 for t in raw_tc + raw_tr):
      raise ValueError("raw timing samples must be finite and positive")
    tc, tr = [x - z for x in raw_tc], [x - z for x in raw_tr]
    if any(t <= 0 for t in tc + tr):
      raise ValueError("calibration subtraction must leave positive timing samples")
    ratios = [r / c for r, c in zip(tr, tc)]
    return {"candidate_us": statistics.median(tc), "reference_us": statistics.median(tr),
        "ratio": statistics.median(ratios), "ci": bootstrap_ci(ratios), "null_us": z, "pairs": pairs,
        "raw_candidate_samples_us": raw_tc, "raw_reference_samples_us": raw_tr,
        "candidate_samples_us": tc, "reference_samples_us": tr, "ratios": ratios,
        "subtract_null": subtract_null, "protocol": "cuda_graph_read_flush_abba"}

  def aa(self, reference, pairs: int = 30, subtract_null: bool = False) -> dict:
    """The reference against itself: the session's noise floor."""
    return self.paired(reference, reference, pairs, subtract_null=subtract_null)


def paired(candidate, reference, pairs: int = 30, subtract_null: bool = False, **kw) -> dict:
  return Bench(**kw).paired(candidate, reference, pairs, subtract_null=subtract_null)
