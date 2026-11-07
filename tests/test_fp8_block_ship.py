"""CPU tests for what may ship (ship.py) and the round-2 table shipped in the kit (no GPU, no Triton)."""

import json
import os
import sys

import pytest

HERE = os.path.join(os.path.dirname(__file__), "..", "kernels", "fp8_block_sm89")
sys.path.insert(0, HERE)

import ship                                                # noqa: E402
import space                                               # noqa: E402
from table import TuneTable                                # noqa: E402

CFG = {"quant": "once", "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "SPLIT_K": 1, "num_warps": 4,
       "num_stages": 3}
SHIPPED = os.path.join(HERE, "tables", "ada_rtx6000_bitwise_r1.json")


def OK_SHA(sha):
  return sha == "b3b9c062de1d4ffb"


def _entry(N=10240, K=5120, bucket="m28k", M=28672, config=CFG, confirm=True, bitwise=True):
  e = {"N": N, "K": K, "bucket": bucket, "lo": 8193, "hi": 1 << 31, "M": M, "config": config,
       "parity": {"ok": True, "bitwise": bitwise, "max_ulp": 0.0}}
  if confirm:
    e["confirm"] = {"gain": 0.08, "samples": 3, "margin": 0.01}
  return e


def _eff(stock_ms, tuned_ms, N=10240, K=5120, M=28672, config=CFG):
  return {"M": M, "N": N, "K": K, "stock": {"graph_ms": stock_ms}, "tuned": {"graph_ms": tuned_ms, "config": config}}


@pytest.mark.parametrize("par,bitwise_only,ok", [
  ({"ok": True, "bitwise": True}, True, True),
  ({"ok": True, "bitwise": False}, True, False),           # within the ulp rule but not bitwise: not for FP8
  ({"ok": True, "bitwise": False}, False, True),
  ({"ok": False, "bitwise": False}, False, False),
  (None, False, False),
])
def test_parity_gate_needs_bitwise_for_bitwise_searches(par, bitwise_only, ok):
  assert ship.parity_gate(par, bitwise_only) is ok


@pytest.mark.parametrize("confirm,ok", [
  (None, False),                                            # ce74c95: gate/up/down kept a config, never confirmed
  ({"gain": None, "samples": 0, "margin": 0.01}, False),
  ({"gain": 0.05, "samples": 1, "margin": 0.01}, False),
  ({"gain": 0.005, "samples": 3, "margin": 0.01}, False),
  ({"gain": 0.05, "samples": 3, "margin": 0.01}, True),
])
def test_confirmed_needs_a_bound_fresh_confirm(confirm, ok):
  e = _entry(confirm=False)
  if confirm is not None:
    e["confirm"] = confirm
  assert ship.confirmed(e) is ok


def test_select_keeps_only_confirmed_bitwise_per_linear_steady_winners():
  entries = [_entry(), _entry(bucket="m4k", M=4096, confirm=False), _entry(N=14336, K=5120),
             _entry(N=6144), _entry(N=12288), _entry(N=5120, K=6144, config=None)]
  parity_rows = [{"M": 28672, "N": 6144, "K": 5120, "config": CFG, "bitwise": False, "reference": "hub"}]
  eff = [_eff(18.6, 17.3), _eff(11.2, 10.5, N=6144), _eff(22.5, 23.0, N=12288)]
  kept, dropped = ship.select(entries, parity_rows, eff)
  assert [(e["N"], e["bucket"]) for e in kept] == [(10240, "m28k")]
  assert kept[0]["steady_ratio"] == pytest.approx(18.6 / 17.3)
  why = {(e["N"], e["bucket"]): r for e, r in dropped}
  assert why[(10240, "m4k")] == "no fresh confirm"
  assert why[(14336, "m28k")].startswith("concat-group")
  assert why[(6144, "m28k")] == "not bitwise vs the hub kernel"           # the hardware parity row wins
  assert why[(12288, "m28k")].startswith("not faster in steady state")
  assert (5120, "m28k") not in why                                        # stock already: nothing to drop


def test_select_without_eff_does_not_require_steady_timing():
  kept, _ = ship.select([_entry()], [], None)
  assert len(kept) == 1 and kept[0]["steady_ratio"] is None


def test_cli_writes_a_shippable_table(tmp_path):
  src = tmp_path / "confirmed.json"
  TuneTable(meta={"kernel_sha": "b3b9c062de1d4ffb"}, entries=[_entry(), _entry(bucket="m4k", confirm=False)]).save(src)
  out = tmp_path / "ship.json"
  assert ship.main(["--confirmed", str(src), "--out", str(out)], compatible=OK_SHA) == 0
  t = TuneTable.load(out)
  assert t.meta["kind"] == "shippable" and t.meta["bitwise_only"] and t.meta["kernel_sha"] == ["b3b9c062de1d4ffb"]
  assert len(t.entries) == 1 and t.config_for(10240, 5120, 20000).BLOCK_M == 64
  assert t.config_for(10240, 5120, 4096) is None                          # unconfirmed bucket: stock


def test_steady_win_must_beat_the_margin_not_just_one():
  e = _entry()                                                            # confirm margin 0.01
  kept, dropped = ship.select([e], [], [_eff(10.08, 10.0)])               # x1.008: a tie on noise
  assert not kept and "<= x1.010" in dropped[0][1]
  kept, _ = ship.select([e], [], [_eff(10.08, 10.0)], margin=0.005)       # A/A margin still binds at 0.01
  assert not kept
  e["confirm"]["margin"] = 0.002
  kept, _ = ship.select([e], [], [_eff(10.08, 10.0)], margin=0.005)
  assert len(kept) == 1


def test_steady_prefers_cold_weights_over_warm_l2():
  row = _eff(10.0, 5.0)                                                   # warm L2: x2.0
  row["stock"]["rot_ms"], row["tuned"]["rot_ms"] = 10.0, 9.5             # cold weights: x1.053
  kept, _ = ship.select([_entry()], [], [row])
  assert kept[0]["steady_ratio"] == pytest.approx(10.0 / 9.5) and kept[0]["steady_clock"] == "rot_ms"
  kept, _ = ship.select([_entry()], [], [_eff(10.0, 5.0)])               # older eff file: warm clock, recorded
  assert kept[0]["steady_clock"] == "graph_ms"


def test_cli_refuses_sources_from_another_kernel(tmp_path):
  src = tmp_path / "confirmed.json"
  TuneTable(meta={"kernel_sha": "deadbeefdeadbeef"}, entries=[_entry()]).save(src)
  assert ship.main(["--confirmed", str(src), "--out", str(tmp_path / "o.json")], compatible=OK_SHA) == 1
  assert not (tmp_path / "o.json").exists()


def test_shipped_round2_table_is_confirmed_bitwise_and_per_linear():
  t = TuneTable.load(SHIPPED)
  assert t.meta["kind"] == "shippable" and t.meta["bitwise_only"]
  assert len(t.entries) == 17
  linear = {(N, K) for N, K in __import__("shapes").unique_shapes()}
  for e in t.entries:
    assert space.bitwise_safe(e["config"]) and ship.confirmed(e) and e["parity"]["bitwise"]
    assert (e["N"], e["K"]) in linear and e["steady_ratio"] > 1.0
  shapes_in = {(e["N"], e["K"]) for e in t.entries}
  assert (17408, 5120) not in shapes_in and (5120, 17408) not in shapes_in  # gate/up/down: confirm never ran
  assert json.loads(open(SHIPPED).read())["meta"]["kernel_sha"] == ["b3b9c062de1d4ffb"]
