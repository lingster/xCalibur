"""Round-2 robustness of the search (the Ada run of ce74c95): capture failures, steady timing, partial confirms.

CPU tests use stubs for the session and the bench; the GPU tests (any CUDA card, no FP8 needed) poison a real CUDA
graph capture and check that the bench recovers. Triton runs in interpreter mode here, as in the kernel tests.
"""

import os
import sys

if os.environ.get("FP8_KERNEL_TEST_DEVICE", "cpu") == "cpu":
  os.environ.setdefault("TRITON_INTERPRET", "1")           # must precede the first @triton.jit

import pytest                                              # noqa: E402
import torch                                               # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kernels", "fp8_block_sm89"))

import safebench                                           # noqa: E402
import search                                              # noqa: E402
import session                                             # noqa: E402
import space                                               # noqa: E402
import xcalibur as xc                                      # noqa: E402
from table import TuneTable                                # noqa: E402

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
M64 = dict(space.STOCK, quant="once", BLOCK_M=64, BLOCK_N=32, GROUP_M=16, num_warps=4, num_stages=5)
M28K = dict(space.STOCK, quant="once", BLOCK_M=64, num_warps=4, num_stages=3)


# ---- session logic (no GPU) -----------------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,steady,copies", [
  (64, 1024, 5120, True, 32),                               # 5 MB weight: capped at 32 copies
  (64, 17408, 5120, True, 3),                               # 89 MB weight: 3 copies cover 2 x 96 MB
  (4096, 10240, 5120, True, 4),
  (28672, 10240, 5120, True, 1),                            # activations alone overflow L2
  (64, 1024, 5120, False, 1),                               # the cold protocol flushes L2 itself
])
def test_weight_copies_cover_twice_the_l2(M, N, K, steady, copies):
  assert session.weight_copies(M, N, K, steady) == copies


def _stub_session(loads=None):
  s = object.__new__(session.Session)
  s.K, s.loads, s.illegal, s.only, s.aa_ref_us, s.drift_tol = 5120, loads or {"m28k": 28672, "m64": 64}, {}, None, {}, 0.1
  return s


def test_restrict_times_each_design_only_where_it_was_bound():
  """ce74c95's --confirm-table timed the m64 winner at m28k too; its capture died there and crashed the process."""
  s = _stub_session()
  s.restrict({"m64": {"design": M64}, "m28k": {"design": M28K}})
  assert s.wanted(M64, "m64", 64) and not s.wanted(M64, "m28k", 28672)
  assert s.wanted(M28K, "m28k", 28672) and not s.wanted(M28K, "m64", 64)
  assert s.wanted(space.STOCK, "m64", 64) and s.wanted(space.STOCK, "m28k", 28672)
  s.only = None
  assert s.wanted(M64, "m28k", 28672)


def test_illegal_designs_are_never_timed_again():
  s = _stub_session()
  s.illegal[space.key(M64)] = "m28k: capture failed"
  assert not s.wanted(M64, "m64", 64) and s.wanted(M28K, "m64", 64)


def test_drift_guard_drops_pairs_where_stock_left_its_own_aa():
  """ce74c95, in_proj_qkv m28k: A/A stock 18,518 us, stock next to the candidate 25,394 us (+37%)."""
  s = _stub_session()
  assert not s.drifted("m28k", 25394.0)                     # no A/A yet: nothing to compare with
  s.aa_ref_us["m28k"] = 18518.0
  assert s.drifted("m28k", 25394.0) and not s.drifted("m28k", 18900.0)


class _FakeSession:
  calls = 0

  def __init__(self, N, K, buckets, ref, a, bench, drift_tol):
    self.N, self.loads = N, {b.name: b.M for b in buckets}

  def aa(self, L, pairs):
    return {"ratio": 1.0, "ci": (0.999, 1.001), "reference_us": 100.0}

  def restrict(self, bound):
    pass

  def measure(self, designs, grade):
    return []

  def times(self, d, L):
    return {}


def test_confirm_table_never_leaves_an_unconfirmed_build_after_a_crash(tmp_path, monkeypatch):
  src = tmp_path / "tuned.json"
  TuneTable(meta={"kernel_sha": search.kernel.KERNEL_SHA}, entries=[
    {"N": 10240, "K": 5120, "bucket": "m28k", "lo": 8193, "hi": 1 << 31, "M": 28672, "config": M28K, "gain": 1.0},
    {"N": 17408, "K": 5120, "bucket": "m28k", "lo": 8193, "hi": 1 << 31, "M": 28672, "config": M28K, "gain": 1.0},
    {"N": 1024, "K": 5120, "bucket": "m28k", "lo": 8193, "hi": 1 << 31, "M": 28672, "config": None, "gain": 0.0},
  ]).save(src)

  def fresh(measure, bound, runs, margin, min_samples):
    _FakeSession.calls += 1
    if _FakeSession.calls == 2:                              # the second shape kills CUDA, as on Ada
      raise safebench.CudaPoisoned("capture invalidated")
    return {L: {"gain": 0.07, "samples": 3, "design": b["design"]} for L, b in bound.items()}

  monkeypatch.setattr(search, "Session", _FakeSession)
  monkeypatch.setattr(search, "fresh_confirm", fresh)
  out = tmp_path / "confirmed.json"
  a = search.parse(["--confirm-table", str(src), "--out", str(out)])
  tab = TuneTable()
  with pytest.raises(safebench.CudaPoisoned):
    search.confirm_table(str(src), None, a, None, tab)
  t = TuneTable.load(out)
  by = {e["N"]: e for e in t.entries}
  assert by[10240]["config"] == M28K and by[10240]["confirm"]["status"] == "bound"
  assert by[17408]["config"] is None and by[17408]["confirm"]["status"] == "pending"
  assert by[17408]["candidate"] == M28K and by[1024]["config"] is None


def test_search_defaults_to_steady_timing_with_a_drift_guard():
  a = search.parse([])
  assert a.timing == "steady" and a.drift_tol == pytest.approx(0.10)


def test_hub_source_check_compares_the_metadata_digests(tmp_path, monkeypatch):
  import base64                                                               # noqa: PLC0415
  import hashlib                                                              # noqa: PLC0415
  import stock                                                                # noqa: PLC0415
  for f in ("matmul.py", "utils.py"):
    (tmp_path / f).write_text(f"# {f}\n")
  digests = {f: base64.b64encode(hashlib.sha256((tmp_path / f).read_bytes()).digest()).decode()
             for f in ("matmul.py", "utils.py")}
  monkeypatch.setattr(stock, "HUB_SOURCE_DIGESTS", digests)
  assert stock.hub_source_matches(str(tmp_path / "matmul.py")) is True
  (tmp_path / "utils.py").write_text("# changed\n")
  assert stock.hub_source_matches(str(tmp_path / "matmul.py")) is False
  assert stock.hub_source_matches(None) is None


def test_hub_digests_are_the_ones_ada_reported():
  """hub_metadata.json from the Ada run (snapshot 03ae3193, the v4 head) lists these digests for matmul/utils."""
  import stock                                                                # noqa: PLC0415
  assert stock.HUB_SOURCE_DIGESTS == {"matmul.py": "D2aLXZ4OfbBnXqRRgFK5Mggw9k8kaPLsvK59CHYzaAc=",
                                      "utils.py": "LLh/mKs4XkHJX98eG+x3eiXI4LsAOjV6csB6U8K3tdA="}


# ---- the bench ------------------------------------------------------------------------------------------------

def test_summary_and_abba_order():
  order = []
  tc, tr = safebench.abba(lambda: order.append("C") or 2.0, lambda: order.append("R") or 4.0, 4)
  assert "".join(order) == "RCCRRCCR"
  r = safebench.summary(tc, tr)
  assert r["ratio"] == 2.0 and r["candidate_us"] == 2.0 and r["reference_us"] == 4.0


@pytest.mark.parametrize("takes_kw", [True, False])
def test_cold_mode_passes_subtract_null_explicitly_or_falls_back(monkeypatch, takes_kw):
  """Upstream v2 flipped Bench.paired's subtract_null default to False; older Benches don't take the keyword."""
  seen = {}
  if takes_kw:
    def paired(self, c, r, pairs=30, subtract_null=False):
      seen["subtract_null"] = subtract_null
      return {"ratio": 1.0}
  else:
    def paired(self, c, r, pairs=30):
      seen["subtract_null"] = "library default"
      return {"ratio": 1.0}
  monkeypatch.setattr(xc.Bench, "paired", paired)
  b = object.__new__(safebench.SafeBench)
  b.mode = "cold"
  assert b.paired(None, None, 4) == {"ratio": 1.0}
  assert seen["subtract_null"] == (True if takes_kw else "library default")


def test_unknown_mode_is_rejected():
  with pytest.raises(ValueError):
    safebench.SafeBench.__init__(object.__new__(safebench.SafeBench), mode="hot")


@GPU
def test_a_poisoned_capture_is_reported_and_the_next_capture_works():
  bench = safebench.SafeBench(flush_mib=16, mode="steady", block_ms=2.0)
  x = torch.randn(256, 256, device="cuda")

  def bad():
    y = x @ x
    if torch.cuda.is_current_stream_capturing():
      y.sum().item()                                         # a sync inside the capture invalidates it
    return y

  with pytest.raises(safebench.CaptureFailed) as e:
    bench.paired(bad, lambda: x @ x, 2)
  assert e.value.kind == "capture" and not torch.cuda.is_current_stream_capturing()
  r = bench.paired(lambda: x @ x, lambda: x @ x, 4)
  assert r["mode"] == "steady" and r["reps"] >= 1 and 0.5 < r["ratio"] < 2.0
  safebench.health_check()


@GPU
def test_a_design_that_cannot_launch_fails_before_any_capture():
  bench = safebench.SafeBench(flush_mib=16)

  def broken():
    raise RuntimeError("out of resources: shared memory")

  with pytest.raises(safebench.CaptureFailed) as e:
    bench.graph(broken)
  assert e.value.kind == "launch"
  safebench.health_check()


@GPU
def test_steady_aa_ratio_is_near_one():
  bench = safebench.SafeBench(flush_mib=16, block_ms=5.0)
  x = torch.randn(1024, 1024, device="cuda")
  r = bench.aa(lambda: x @ x, pairs=6)
  assert abs(r["ratio"] - 1.0) < 0.05 and r["candidate_us"] > 0
