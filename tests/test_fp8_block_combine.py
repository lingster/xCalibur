"""CPU tests for combine.py: one production table from several ship.py rounds (no GPU, no Triton)."""

import json
import os
import sys

import pytest

HERE = os.path.join(os.path.dirname(__file__), "..", "kernels", "fp8_block_sm89")
sys.path.insert(0, HERE)

import combine                                             # noqa: E402

CFG = {"quant": "once", "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "SPLIT_K": 1, "num_warps": 4,
       "num_stages": 3}
TABLES = os.path.join(HERE, "tables")
R1, R2 = (os.path.join(TABLES, f"ada_rtx6000_bitwise_{r}.json") for r in ("r1", "r2"))
R1_SHA, R2_SHA = "b3b9c062de1d4ffb", "aa029a2e417a5070"


def _ok(sha):
  return sha in {R1_SHA, R2_SHA, "sha-a", "sha-b"}


def _table(tmp_path, name, entries, kind="shippable", bitwise_only=True, sha=("sha-a",)):
  p = tmp_path / name
  meta = {"kind": kind, "bitwise_only": bitwise_only, "kernel_sha": list(sha), "rule": "r"}
  p.write_text(json.dumps({"meta": meta, "entries": entries}))
  return str(p)


def _entry(N=1024, K=5120, bucket="m64", config=CFG, gain=0.1):
  return {"N": N, "K": K, "bucket": bucket, "lo": 1, "hi": 64, "M": 64, "config": config, "gain": gain}


def test_later_table_wins_on_the_same_key(tmp_path):
  a = _table(tmp_path, "a.json", [_entry(gain=0.1), _entry(N=6144, bucket="m28k")])
  b = _table(tmp_path, "b.json", [_entry(gain=0.2)], sha=("sha-b",))
  out, replaced = combine.combine([a, b], _ok)
  assert replaced == [(1024, 5120, "m64")]
  assert len(out.entries) == 2
  hit = next(e for e in out.entries if e["N"] == 1024)
  assert hit["gain"] == 0.2 and hit["source"] == "b.json"
  assert out.meta["kernel_sha"] == ["sha-a", "sha-b"] and out.meta["kind"] == "shippable" and out.meta["bitwise_only"]


@pytest.mark.parametrize("kw,why", [
  ({"kind": "confirmed"}, "not a ship.py bitwise table"),
  ({"bitwise_only": False}, "not a ship.py bitwise table"),
  ({"sha": ("unknown",)}, "not this fp8 path"),
  ({"sha": ()}, "not this fp8 path"),
])
def test_refuses_what_kev_patch_would_refuse(tmp_path, kw, why):
  bad = _table(tmp_path, "bad.json", [_entry()], **kw)
  with pytest.raises(ValueError, match=why):
    combine.combine([bad], _ok)


def test_refuses_an_entry_outside_the_bitwise_class(tmp_path):
  split = dict(CFG, SPLIT_K=2)                             # split-K reorders the fp32 sum: not bitwise
  bad = _table(tmp_path, "bad.json", [_entry(config=split)])
  with pytest.raises(ValueError, match="not in the bitwise class"):
    combine.combine([bad], _ok)


def test_kernel_shas_reads_list_and_string():
  assert combine.kernel_shas({"kernel_sha": ["x", "y"]}) == ["x", "y"]
  assert combine.kernel_shas({"kernel_sha": "x"}) == ["x"]
  assert combine.kernel_shas({}) == []


def test_shipped_rounds_combine_to_21_entries():
  out, replaced = combine.combine([R1, R2], _ok)
  assert len(out.entries) == 21                            # 17 + 6, minus N=1024 K=5120 at m64 and m512
  assert sorted(replaced) == [(1024, 5120, "m512"), (1024, 5120, "m64")]
  assert out.meta["kernel_sha"] == sorted([R1_SHA, R2_SHA])
  assert {e["source"] for e in out.entries if e["N"] == 1024 and e["bucket"] in ("m64", "m512")} == {
    "ada_rtx6000_bitwise_r2.json"}


def test_confirm_table_refuses_a_table_searched_on_another_kernel(tmp_path):
  pytest.importorskip("triton")
  import search
  bad = _table(tmp_path, "old.json", [_entry()], kind="search", sha=("deadbeefdeadbeef",))
  with pytest.raises(ValueError, match="search again"):
    search.confirm_table(bad, None, None, None, None)


def test_rot_ms_is_device_time_per_copy(monkeypatch):
  pytest.importorskip("torch")
  import timing
  calls = []
  monkeypatch.setattr(timing, "graph_ms", lambda fn, min_ms, rounds: (fn(), 12.0)[1])
  assert timing.rot_ms(lambda w, ws: calls.append(w), [(1, 0), (2, 0), (3, 0)]) == 4.0
  assert calls == [1, 2, 3]                                # one call per copy, in order
