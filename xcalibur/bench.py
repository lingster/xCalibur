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

``LongBench`` is for calls of milliseconds to seconds (a whole model pass), where graphs, the flush and the null
replay don't apply. It keeps ABBA pairs and adds what long calls need: CUDA events and wall time around each call, a
warm-up policy (``small_first``: Triton autotunes on its first call, and tuning on a full-size input can run a 24 GB
card out of memory), and nvidia-smi snapshots per block of pairs, because consumer boards sit at their power cap and
drop SM clocks mid-run.

  >>> warm = small_first({"candidate": lambda: new(short), "reference": lambda: old(short)})  # doctest: +SKIP
  >>> lb = LongBench(warmup=warm)                                            # doctest: +SKIP
  >>> r = lb.paired(lambda: new(long), lambda: old(long), pairs=8)          # doctest: +SKIP
  >>> r["ratio"], r["ci"], [s["clocks.sm"] for s in r["snapshots"]]        # doctest: +SKIP
"""
from __future__ import annotations

import os
import statistics
import math
import subprocess
import time

from .stats import bootstrap_ci


def _pairs(pairs) -> int:
  if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 1:
    raise ValueError("pairs must be a positive integer")
  return pairs


def _abba(time_c, time_r, pairs: int, on_pair=None):
  """Per-pair times in ABBA order: (reference, candidate), (candidate, reference), ..."""
  tc, tr = [], []
  for i in range(pairs):
    if on_pair:
      on_pair(i)
    if i % 2 == 0:
      tr.append(time_r()); tc.append(time_c())
    else:
      tc.append(time_c()); tr.append(time_r())
  return tc, tr


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
    _pairs(pairs)
    gc, gr = self.graph(candidate), self.graph(reference)
    tc, tr = _abba(lambda: self.replay_us(gc), lambda: self.replay_us(gr), pairs)
    z =self.null_us() if subtract_null else 0.0
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


SMI_QUERY = ("clocks.sm", "clocks.mem", "clocks.max.sm", "power.draw", "power.limit", "temperature.gpu",
             "clocks_throttle_reasons.active", "memory.used", "pstate")


def smi_snapshot(gpu=None) -> dict:
  """One nvidia-smi reading of one GPU (physical index or PCI bus id; default: the single CUDA_VISIBLE_DEVICES index
  under CUDA_DEVICE_ORDER=PCI_BUS_ID). {"error": ...} when unavailable; never raises."""
  if gpu is None:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpu = v if v.isdigit() and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" else None
  if gpu is None:
    return {"error": "which GPU? pass gpu= (or pin one with CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER=PCI_BUS_ID)"}
  cmd = ["nvidia-smi", f"--id={gpu}", f"--query-gpu={','.join(SMI_QUERY)}", "--format=csv,noheader,nounits"]
  try:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
  except (OSError, subprocess.SubprocessError) as e:
    return {"error": f"{type(e).__name__}: {e}"}
  values = [v.strip() for v in out.split(",")]
  return dict(zip(SMI_QUERY, values)) if len(values) == len(SMI_QUERY) else {"raw": out}


def wall_clock(fn):
  """(ms, wall ms) of one call by the host clock alone."""
  t = time.perf_counter()
  fn()
  ms = (time.perf_counter() - t) * 1000.0
  return ms, ms


def cuda_clock(torch):
  """A clock: CUDA events around the call, and the wall time between the synchronizes either side."""
  def clock(fn):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    a.record()
    fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b), (time.perf_counter() - t) * 1000.0
  return clock


def warm_repeat(n: int = 1):
  """Warm-up policy: call each side n times at full size."""
  def policy(role, fn):
    for _ in range(n):
      fn()
  return policy


def small_first(small: dict, then: int = 1):
  """Warm-up policy: small[role]() first (autotuning happens on a short input), then the full-size call `then` times."""
  def policy(role, fn):
    small[role]()
    warm_repeat(then)(role, fn)
  return policy


class LongBench:
  """ABBA-paired timing of long calls (ms to s); no graphs, flush or null replay. ratio > 1: the candidate is faster."""

  def __init__(self, warmup=None, clock=None, snapshot=None, gpu=None):
    if clock is None:
      import torch                            # noqa: PLC0415
      clock = cuda_clock(torch) if torch.cuda.is_available() else wall_clock
    self.clock, self.warmup = clock, warmup or warm_repeat(1)
    self.snapshot = snapshot if snapshot is not None else (lambda: smi_snapshot(gpu))

  def paired(self, candidate, reference, pairs: int = 8, block: int = 4) -> dict:
    """Warm both sides, then ABBA pairs with a GPU snapshot before every `block` pairs and after the last.

    Pool several inputs with bootstrap_ci(ratios, clusters=input ids), not by resampling single ratios.
    """
    _pairs(pairs)
    _pairs(block)
    self.warmup("candidate", candidate)
    self.warmup("reference", reference)
    snaps = []
    snap = lambda i: snaps.append({"pair": i, **self.snapshot()})   # noqa: E731
    tc, tr = _abba(lambda: self.clock(candidate), lambda: self.clock(reference), pairs,
                   on_pair=lambda i: i % block or snap(i))
    snap(pairs)
    gc, wc = [t for t, _ in tc], [w for _, w in tc]
    gr, wr = [t for t, _ in tr], [w for _, w in tr]
    if any(not math.isfinite(t) or t <= 0 for t in gc + gr):
      raise ValueError("timing samples must be finite and positive")
    ratios = [r / c for r, c in zip(gr, gc)]
    return {"candidate_ms": statistics.median(gc), "reference_ms": statistics.median(gr),
        "ratio": statistics.median(ratios), "ci": bootstrap_ci(ratios),
        "candidate_wall_ms": statistics.median(wc), "reference_wall_ms": statistics.median(wr),
        "candidate_samples_ms": gc, "reference_samples_ms": gr, "ratios": ratios, "pairs": pairs,
        "snapshots": snaps, "protocol": "long_call_events_abba"}

  def aa(self, reference, pairs: int = 8, block: int = 4) -> dict:
    """The reference against itself: the session's noise floor for long calls."""
    return self.paired(reference, reference, pairs, block)
