"""Two clocks for one call, so host overhead and device time are never confused.

  eager_ms   CUDA events around a plain Python call, warm L2, median of enough calls to fill ~200 ms: the method of
             the first Ada probe. It includes host dispatch: while Python prepares
             the launch the GPU idles between the events, so a launch-bound call reads as its host cost.
  graph_ms   the call captured once in a CUDA graph and replayed back-to-back between events: DEVICE time per call,
             no Python, no launch overhead (warm L2, comparable to eager_ms).
  rot_ms     graph_ms over one call per weight copy, divided by the copies: device time per call when every call reads
             its weight from DRAM, as in production where each layer has its own weights. Warm-L2 graph_ms overstates
             small-M gains (the weight of a skinny GEMM fits in L2 and stays there).

eager_ms >> graph_ms means the call is host-bound: fix launches (concat groups, CUDA graphs), not tiles.
For decisions (search, A/A) use ``safebench.SafeBench.paired``: ABBA pairs, validated captures, steady state by default.
"""
from __future__ import annotations

import torch


def eager_ms(fn, min_ms: float = 200.0, warmup: int = 3) -> float:
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  s.record(); fn(); e.record(); torch.cuda.synchronize()                       # noqa: E702
  reps = max(3, min(200, int(min_ms / max(s.elapsed_time(e), 1e-3))))
  out = []
  for _ in range(reps):
    s.record(); fn(); e.record(); torch.cuda.synchronize()                     # noqa: E702
    out.append(s.elapsed_time(e))
  return sorted(out)[len(out) // 2]


def capture(fn, warmup: int = 3) -> torch.cuda.CUDAGraph:
  st = torch.cuda.Stream()
  st.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(st):
    for _ in range(warmup):
      fn()
  torch.cuda.current_stream().wait_stream(st)
  g = torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
    fn()
  return g


def graph_ms(fn, min_ms: float = 200.0, rounds: int = 5) -> float:
  """Median over ``rounds`` of (back-to-back replays / count): device time per call."""
  g = capture(fn)
  g.replay(); torch.cuda.synchronize()                                         # noqa: E702
  s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  s.record(); g.replay(); e.record(); torch.cuda.synchronize()                 # noqa: E702
  n = max(3, min(500, int(min_ms / rounds / max(s.elapsed_time(e), 1e-3))))
  out = []
  for _ in range(rounds):
    s.record()
    for _ in range(n):
      g.replay()
    e.record(); torch.cuda.synchronize()                                       # noqa: E702
    out.append(s.elapsed_time(e) / n)
  return sorted(out)[len(out) // 2]


def rot_ms(call, weights: list, min_ms: float = 200.0, rounds: int = 5) -> float:
  """Device time per call(w, ws) when the calls rotate through ``weights`` ([(w, ws)], enough copies to overflow L2)."""
  def all_copies():
    for w, ws in weights:
      call(w, ws)
  return graph_ms(all_copies, min_ms, rounds) / len(weights)


def tflops(M: int, N: int, K: int, ms: float | None) -> float | None:
  return None if not ms else 2.0 * M * N * K / ms / 1e9
