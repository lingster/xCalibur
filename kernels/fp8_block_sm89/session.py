"""The hardware half of one shape's game: problems per M bucket, the reference, and the measure/compile callbacks.

Robustness added after the Ada run of ce74c95 (README, "Round 2"):

* A design that fails to launch or capture (``safebench.CaptureFailed``) is marked illegal for the session and never
  timed again; CUDA is proven healthy before the search moves on. ``safebench.CudaPoisoned`` propagates: the caller
  saves its table and stops.
* ``only``: during a fresh confirm each design is timed only at the loads it was bound for (ce74c95 timed an m64
  winner at m28k too, which is where its capture died), and stock only where a design needs its control.
* Reference drift guard: every paired sample's stock time is compared with the same session's A/A stock time. A pair
  in which stock reads more than ``drift_tol`` away from its own A/A is not evidence about the candidate (ce74c95: stock
  read 37% slow next to a candidate at m28k) and is dropped.
* Steady-state timing (safebench's default) rotates ``copies`` of the weight so that, as in production where every
  layer reads new weights, back-to-back replays do not find the weight in L2.
"""
from __future__ import annotations

import math
import statistics
import time

import aot
import kernel
import reference
import safebench
import shapes
import space

L2_BYTES = 96 << 20                                   # RTX 6000 Ada


def log(msg: str) -> None:
  print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def weight_copies(M: int, N: int, K: int, steady: bool, l2_bytes: int = L2_BYTES, cap: int = 32) -> int:
  """Weight copies to rotate so each call reads its weight from DRAM: enough to cover 2 x L2. One for the cold
  protocol (it flushes L2) and for large M (the activations alone overflow L2)."""
  if not steady or M * K * 2 > l2_bytes:
    return 1
  return max(1, min(cap, math.ceil(2 * l2_bytes / (N * K))))


class Session:
  def __init__(self, N: int, K: int, buckets: list[shapes.Bucket], ref, a, bench, drift_tol: float = 0.10):
    self.N, self.K, self.ref, self.a, self.bench, self.drift_tol = N, K, ref, a, bench, drift_tol
    self.loads = {b.name: b.M for b in buckets}
    steady = getattr(bench, "mode", "cold") == "steady"
    n_max = max(weight_copies(M, N, K, steady) for M in self.loads.values())
    self.w, self.ws = [], []
    for i in range(n_max):
      _, w, ws = reference.random_problem(1, N, K, "cuda", seed=1 + i)
      self.w.append(w)
      self.ws.append(ws)
    self.copies = {L: weight_copies(M, N, K, steady) for L, M in self.loads.items()}
    self.x = {b.name: reference.random_problem(b.M, 128, K, "cuda", seed=b.M)[0] for b in buckets}
    self.q = {L: kernel.act_quant(x, dot=a.dot) for L, x in self.x.items()} if a.quant_cost == "shared" else {}
    self.raw: list[dict] = []
    self.aa_ref_us: dict[str, float] = {}
    self.illegal: dict[str, str] = {}                 # design key -> why (launch / capture failure)
    self.only: dict[str, set[str]] | None = None      # design key -> loads to time (fresh confirm)
    self.dropped = 0

  def _rotate(self, L: str, call):
    """One callable over this load's weight copies, in order (call(w, ws) -> output)."""
    pairs = list(zip(self.w, self.ws))[: self.copies[L]]

    def fn():
      for w, ws in pairs:
        call(w, ws)
    return fn

  def _ref_fn(self, L):
    x = self.x[L]
    return self._rotate(L, lambda w, ws: self.ref.fn(x, w, ws))

  def _call(self, L, cfg: space.TileConfig):
    x, dot = self.x[L], self.a.dot
    if cfg.quant == "once" and L in self.q:                       # quant shared with the input group: not charged
      q, s = self.q[L]
      return self._rotate(L, lambda w, ws: kernel.block_fp8_matmul(q, w, ws, cfg, a_scale=s, dot=dot))
    return self._rotate(L, lambda w, ws: kernel.block_fp8_matmul(x, w, ws, cfg, dot=dot))

  def aa(self, L: str, pairs: int) -> dict:
    """Stock against itself; times in the returned dict are per GEMM call (a sample spans every weight copy)."""
    r = self.bench.aa(self._ref_fn(L), pairs=pairs)
    self.aa_ref_us[L] = r["reference_us"]
    n = self.copies[L]
    return dict(r, reference_us=r["reference_us"] / n, candidate_us=r["candidate_us"] / n)

  def restrict(self, bound: dict[str, dict]) -> None:
    """Time each design only at the loads it is bound for; stock at the union (its A/A control in fresh_confirm)."""
    only: dict[str, set[str]] = {}
    for L, b in bound.items():
      only.setdefault(space.key(b["design"]), set()).add(L)
    only[space.key(space.STOCK)] = set(bound)
    self.only = only

  def wanted(self, d: dict, L: str, M: int) -> bool:
    k = space.key(d)
    if k in self.illegal or not space.applicable(d, M, self.K):
      return False
    return self.only is None or L in self.only.get(k, ())

  def drifted(self, L: str, ref_us: float) -> bool:
    base = self.aa_ref_us.get(L)
    return base is not None and abs(ref_us / base - 1.0) > self.drift_tol

  def measure(self, designs: list[dict], grade: str) -> list[dict]:
    pairs = self.a.screen_pairs if grade == "screen" else self.a.confirm_pairs
    rows = []
    for d in designs:
      row = {}
      for L, M in self.loads.items():
        if not self.wanted(d, L, M):
          continue
        ref_fn = self._ref_fn(L)
        try:
          fn = ref_fn if space.is_stock(d) else self._call(L, space.to_config(d, M))
          r = self.bench.paired(fn, ref_fn, pairs)
        except safebench.CaptureFailed as e:
          self.illegal[space.key(d)] = f"{L}: {e}"
          log(f"  illegal for this session ({L}) {d}: {e}")
          break
        except safebench.CudaPoisoned:
          raise
        except Exception as e:                                    # noqa: BLE001  (e.g. out of resources on sm_86)
          log(f"  measure failed {L} {d}: {type(e).__name__}: {str(e)[:120]}")
          continue
        n = self.copies[L]
        if self.drifted(L, r["reference_us"]):
          self.dropped += 1
          log(f"  dropped {L} {grade}: stock {r['reference_us'] / n:.1f} us vs its A/A {self.aa_ref_us[L] / n:.1f} us")
          continue
        row[L] = r["ratio"]
        self.raw.append({"load": L, "grade": grade, "design": d, "ratio": r["ratio"], "ci": r["ci"],
                         "candidate_us": r["candidate_us"] / n, "reference_us": r["reference_us"] / n})
      rows.append(row)
    return rows

  def compile(self, designs: list[dict]) -> list[dict]:
    """Exact sm_89 compile of the fp8 build: legality (Triton, ptxas, <= 99 KB smem) and resource features."""
    return [aot.compile_sm89(space.to_config(d, 128)) for d in designs]

  def times(self, d: dict, L: str) -> dict:
    rs = [r for r in self.raw if r["load"] == L and r["design"] == d and r["grade"] == "confirm"]
    if not rs:
      return {}
    return {"tuned_us": statistics.median(r["candidate_us"] for r in rs),
            "stock_us": statistics.median(r["reference_us"] for r in rs),
            "ratio": statistics.median(r["ratio"] for r in rs), "confirm_runs": len(rs)}
