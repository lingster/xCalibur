"""Capture-safe, power-steady paired timing for the FP8 search: a wrapper around ``xcalibur.Bench``.

The Ada run of ce74c95 (2026-11-05) hit two problems in ``xcalibur.Bench`` that this module fixes inside the kit
(the library-level change is described in the README, "Changes for xcalibur.Bench"):

1. **Capture robustness.** A design whose capture was invalidated (``cudaErrorStreamCaptureInvalidated``) raised out
   of ``torch.cuda.graph``; ``Session.measure`` caught it, but nothing released the dead capture's memory pool, and a
   later capture (the next shape's A/A, outside any ``try``) died with "operation failed due to a previous error
   during capture", which killed ``--confirm-table`` half way. Here every graph is (a) run once eagerly and
   synchronised before capture, so a launch or compile failure surfaces outside any capture; (b) captured after
   ``gc.collect()`` + ``torch.cuda.empty_cache()``, so the caching allocator never has to ``cudaFree`` cached
   blocks (an illegal call) inside a capture when the largest bucket needs fresh memory; (c) captured in
   ``thread_local`` mode, so no other thread's CUDA call can invalidate it. A failure raises ``CaptureFailed``
   after ``recover()`` has synchronised, released the memory and proven the context healthy with a tiny kernel;
   if even that fails it raises ``CudaPoisoned`` and the caller saves its work and stops.

2. **Steady-state timing.** ``Bench.paired`` times ONE cold replay per sample after a synchronise and an L2 flush.
   On the 300 W RTX 6000 Ada (in power violation 78% of the busy time, SM clock 0.6-2.8 GHz) that regime gave
   physically impossible numbers: tuned builds at 240-300 TFLOPS (cuBLAS: ~200), stock 37% slower next to a
   candidate than in its own A/A. ``mode="steady"`` times a block of back-to-back replays after one untimed warm
   replay (no synchronise inside the block), the regime production runs in (layer after layer at the power cap);
   the caller rotates weight copies when the weights would otherwise sit in L2 (``search.Session``).
   ``mode="cold"`` keeps the library's protocol.
"""
from __future__ import annotations

import gc
import statistics

import torch

import xcalibur as xc


class CaptureFailed(RuntimeError):
  """A design could not be launched (``kind == "launch"``) or captured (``"capture"``); CUDA is healthy again."""

  def __init__(self, kind: str, cause: BaseException):
    super().__init__(f"{kind} failed: {type(cause).__name__}: {str(cause).splitlines()[0][:200] if str(cause) else ''}")
    self.kind, self.cause = kind, cause


class CudaPoisoned(RuntimeError):
  """The CUDA context did not recover: stop, save what was measured, start a new process."""


def health_check() -> None:
  """A tiny kernel and a read-back: raises CudaPoisoned when the context is unusable."""
  try:
    t = torch.ones(8, device="cuda")
    if float((t + 1).sum().item()) != 16.0:
      raise RuntimeError("wrong result")
    torch.cuda.synchronize()
  except Exception as e:                                                      # noqa: BLE001
    raise CudaPoisoned(f"CUDA context unusable after a failed capture: {type(e).__name__}: {e}") from e


def release() -> None:
  """Free cached blocks outside any capture (dead graph pools included)."""
  gc.collect()
  torch.cuda.empty_cache()


def recover() -> None:
  """After a failed launch or capture: end of story for that design, not for the session."""
  try:
    torch.cuda.synchronize()
  except Exception:                                                           # noqa: BLE001  (the error is reported)
    pass
  release()
  health_check()


def summary(tc: list[float], tr: list[float]) -> dict:
  """Per-pair ratios reference / candidate (> 1: candidate faster), medians and the bootstrap CI."""
  ratios = [r / c for r, c in zip(tr, tc)]
  return {"candidate_us": statistics.median(tc), "reference_us": statistics.median(tr),
          "ratio": statistics.median(ratios), "ci": xc.bootstrap_ci(ratios)}


def abba(time_c, time_r, pairs: int) -> tuple[list[float], list[float]]:
  """Samples in ABBA order: (reference, candidate), (candidate, reference), ..."""
  tc, tr = [], []
  for i in range(pairs):
    if i % 2 == 0:
      tr.append(time_r())
      tc.append(time_c())
    else:
      tc.append(time_c())
      tr.append(time_r())
  return tc, tr


class SafeBench(xc.Bench):
  """``xcalibur.Bench`` with validated, recoverable captures and a steady-state timing mode (the default)."""

  def __init__(self, flush_mib: int = 256, warmup: int = 3, mode: str = "steady", block_ms: float = 25.0,
               max_reps: int = 200):
    if mode not in ("steady", "cold"):
      raise ValueError(f"mode {mode!r}: 'steady' or 'cold'")
    super().__init__(flush_mib, warmup)
    self.mode, self.block_ms, self.max_reps = mode, block_ms, max_reps

  def graph(self, fn):
    try:
      fn()                                                                    # preflight: compile + launch, eagerly
      torch.cuda.synchronize()
    except Exception as e:                                                    # noqa: BLE001
      recover()
      raise CaptureFailed("launch", e) from e
    release()
    try:
      return self._capture(fn)
    except Exception as e:                                                    # noqa: BLE001
      recover()
      raise CaptureFailed("capture", e) from e

  def _capture(self, fn):
    t = self.torch
    s = t.cuda.Stream()
    s.wait_stream(t.cuda.current_stream())
    with t.cuda.stream(s):
      for _ in range(self.warmup):
        fn()
    t.cuda.current_stream().wait_stream(s)
    g = t.cuda.CUDAGraph()
    with t.cuda.graph(g, capture_error_mode="thread_local"):
      fn()
    return g

  def block_us(self, g, reps: int) -> float:
    """Device time per replay over ``reps`` back-to-back replays, after one untimed warm replay."""
    t = self.torch
    a, b = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
    g.replay()
    a.record()
    for _ in range(reps):
      g.replay()
    b.record()
    t.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0 / reps

  def reps_for(self, *graphs) -> int:
    """Replays per block so the longer of the graphs fills ~block_ms (at least 1, at most max_reps)."""
    longest = max(self.block_us(g, 1) for g in graphs)
    return max(1, min(self.max_reps, int(self.block_ms * 1000.0 / max(longest, 1.0))))

  def _library_paired(self, candidate, reference, pairs: int, subtract_null: bool) -> dict:
    """The library's cold protocol, null replay subtraction passed explicitly: upstream v2 changed its default to
    False, and an older Bench may not take the keyword at all (then it always subtracts)."""
    try:
      return super().paired(candidate, reference, pairs, subtract_null=subtract_null)
    except TypeError as e:
      if "subtract_null" not in str(e):
        raise
      return super().paired(candidate, reference, pairs)

  def paired(self, candidate, reference, pairs: int = 30, subtract_null: bool = True) -> dict:
    if self.mode == "cold":
      return self._library_paired(candidate, reference, pairs, subtract_null)
    gc_, gr = self.graph(candidate), self.graph(reference)
    reps = self.reps_for(gc_, gr)
    tc, tr = abba(lambda: self.block_us(gc_, reps), lambda: self.block_us(gr, reps), pairs)
    del gc_, gr
    return {**summary(tc, tr), "null_us": 0.0, "pairs": pairs, "reps": reps, "mode": "steady"}
