"""CPU tests for the sm_89 block-FP8 tuning kit: shape inventory, design space, tune table, search plumbing.

No GPU and no Triton launch: the kernel itself is tested in test_fp8_block_kernel.py (Triton's CPU interpreter).
"""

import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kernels", "fp8_block_sm89"))

import shapes                                              # noqa: E402
import space                                               # noqa: E402
import table                                               # noqa: E402
from plan import PerShapeReach, aa_margin, fresh_confirm, make_game, play, seed_designs       # noqa: E402

import xcalibur as xc                                      # noqa: E402


# ---- shapes ---------------------------------------------------------------------------------------------------

def test_inventory_matches_the_27b_config():
  by = {g.name: g for g in shapes.GEMMS}
  assert len(shapes.GEMMS) == 10
  assert (by["deltanet.in_proj_qkv"].N, by["deltanet.in_proj_qkv"].K) == (10240, 5120)   # 2*16*128 + 48*128
  assert (by["attn.q_proj"].N, by["attn.k_proj"].N, by["attn.o_proj"].K) == (12288, 1024, 6144)
  assert (by["mlp.down_proj"].N, by["mlp.down_proj"].K) == (5120, 17408)
  assert sum(g.per_pass for g in shapes.GEMMS if g.name.startswith("mlp.")) == 3 * 64
  assert all(g.per_pass == 48 for g in shapes.GEMMS if g.name.startswith("deltanet."))
  assert all(g.per_pass == 16 for g in shapes.GEMMS if g.name.startswith("attn."))


def test_unique_shapes_and_input_groups():
  assert len(shapes.unique_shapes()) == 7                       # o_proj == out_proj, k == v, gate == up
  groups = shapes.input_groups()
  assert set(groups) == {"deltanet_in", "attn_in", "mlp_in"}
  assert [g.name for g in groups["attn_in"]] == ["attn.q_proj", "attn.k_proj", "attn.v_proj"]


def test_concat_shapes_are_exact_and_cut_launches():
  cs = shapes.concat_shapes()
  assert cs == {"deltanet_in": (16384, 5120), "attn_in": (14336, 5120), "mlp_in": (34816, 5120)}
  assert all(N % shapes.SCALE_BLOCK == 0 for N, _ in cs.values())
  assert shapes.launches_per_pass(concat=False) - shapes.launches_per_pass(concat=True) == 48 + 2 * 16 + 64


def test_flops_per_token_counts_every_fp8_linear():
  params = sum(g.N * g.K * g.per_pass for g in shapes.GEMMS)
  assert params == 48 * (10240 + 6144) * 5120 + 48 * 5120 * 6144 + 16 * (12288 + 2048) * 5120 + 16 * 5120 * 6144 \
                    + 64 * (2 * 17408 * 5120 + 5120 * 17408)
  assert shapes.gemm_flops_per_token() == 2 * params
  assert 48e9 < shapes.gemm_flops_per_token() < 49e9


def test_request_call_counts_weight_state_and_rows():
  w = shapes.request_calls(shapes.WORKLOADS["unpacked_rows"])
  assert w[("mlp.down_proj", 28672)] == 64 and w[("mlp.down_proj", 40)] == 64 * 8
  packed = shapes.request_calls(shapes.WORKLOADS["packed_rows"])
  assert packed[("attn.k_proj", 320)] == 16


@pytest.mark.parametrize("M,bucket", [(1, "m64"), (40, "m64"), (64, "m64"), (65, "m256"),
                                      (600, "m512"), (1025, "m4k"), (28000, "m28k"), (10**6, "m28k")])
def test_bucket_of(M, bucket):
  assert shapes.bucket_of(M).name == bucket


@pytest.mark.parametrize("M,bm", [(1, 16), (16, 16), (17, 32), (40, 64), (100, 128), (28672, 128)])
def test_hub_block_m_rule(M, bm):
  assert space.hub_block_m(M) == bm


# ---- design space ---------------------------------------------------------------------------------------------

def test_stock_design_resolves_to_the_hub_tiling():
  c = space.to_config(space.STOCK, 40)
  assert (c.quant, c.BLOCK_M, c.BLOCK_N, c.BLOCK_K, c.GROUP_M, c.SPLIT_K) == ("fused", 64, 128, 128, 8, 1)
  assert space.is_stock(space.STOCK) and not space.is_stock(dict(space.STOCK, BLOCK_N=64))


@pytest.mark.parametrize("change,reason", [
  ({"quant": "fused", "BLOCK_K": 64}, "fused_needs_group_tile"),
  ({"BLOCK_M": 256, "BLOCK_N": 256, "num_warps": 4}, "acc_regs"),
  ({"BLOCK_M": 16, "BLOCK_N": 32, "num_warps": 8}, "mma_tile"),
  ({"quant": "fused", "BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "num_warps": 8, "num_stages": 3}, "smem"),
])
def test_illegal_designs_name_their_rule(change, reason):
  d = dict(quant="once", BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, SPLIT_K=1, num_warps=4, num_stages=3)
  d.update(change)
  assert space.why_illegal(d) == reason


@pytest.mark.parametrize("quant,bm,bn,bk,stages,shared", [         # shared bytes of real sm_89 compiles (aot.py)
  ("once", 128, 128, 128, 2, 32768), ("once", 128, 128, 128, 4, 98304), ("fused", 128, 128, 128, 2, 65536),
  ("fused", 128, 128, 128, 3, 114688), ("fused", 64, 128, 128, 3, 73728), ("fused", 16, 128, 128, 3, 43008)])
def test_smem_model_matches_sm89_compiles(quant, bm, bn, bk, stages, shared):
  d = dict(quant=quant, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, SPLIT_K=1, num_warps=8, num_stages=stages)
  assert space.smem_bytes(d) == shared


def test_stock_tile_at_m128_only_fits_two_stages_on_ada():
  d = dict(quant="fused", BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, SPLIT_K=1, num_warps=8, num_stages=2)
  assert space.why_illegal(d) is None and space.why_illegal(dict(d, num_stages=3)) == "smem"
  assert space.why_illegal(dict(d, quant="once", num_stages=4)) is None


def test_space_is_legal_and_excludes_stock():
  ds = space.designs()
  assert 1000 < len(ds) < 20000
  assert all(space.why_illegal(d) is None for d in ds)
  assert space.STOCK not in ds


def test_applicable_prunes_wasted_tiles_per_size():
  big = dict(quant="once", BLOCK_M=256, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, SPLIT_K=1, num_warps=8, num_stages=3)
  assert not space.applicable(big, 40) and space.applicable(big, 28672)
  sk = dict(big, BLOCK_M=32, SPLIT_K=4)
  assert space.applicable(sk, 40) and not space.applicable(sk, 28672)
  sk2 = dict(big, SPLIT_K=2)                               # long-K (down_proj) may split at any M
  assert space.applicable(sk2, 28672, K=17408) and not space.applicable(sk2, 28672, K=5120)
  assert space.applicable(space.STOCK, 40)


def test_bitwise_safe_subspace_keeps_the_main_levers():
  """BLOCK_K = 128 and no split-K: the per-element fp32 sum order is stock's, so outputs are bitwise equal (the kernel
  tests prove it); the subspace still holds quantise-once, deeper pipelines and every M / N tile and GROUP_M."""
  safe = [d for d in space.designs() if space.bitwise_safe(d)]
  assert all(d["BLOCK_K"] == 128 and d["SPLIT_K"] == 1 for d in safe)
  assert any(d["quant"] == "once" and d["num_stages"] == 4 and d["BLOCK_M"] == 128 for d in safe)
  assert space.bitwise_safe(space.STOCK)
  game = make_game({"m64": 64}, 5120, measure=None, compile=None, keep=space.bitwise_safe)
  assert game.frontier([]) and all(space.bitwise_safe(d) for d in game.frontier([]))


def test_featurize_is_fixed_length_with_and_without_compile_record():
  d = space.designs()[0]
  a = space.featurize(d, None)
  b = space.featurize(d, {"ok": True, "n_regs": 128, "n_spills": 0, "shared": 65536})
  assert len(a) == len(b) == len(space.featurize(space.STOCK, None)) and a != b


# ---- public seeds (vLLM L40S, sm_89) --------------------------------------------------------------------------

def test_public_seed_for_our_exact_shape_is_vllms_pick():
  import public                                            # noqa: PLC0415
  s = public.seeds(10240, 5120, 4096)
  assert s[0] == dict(quant="once", BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=1, SPLIT_K=1, num_warps=8,
                      num_stages=4)
  assert public.provenance()["commit"].startswith("bdfb026")


@pytest.mark.parametrize("N,K", shapes.unique_shapes())
@pytest.mark.parametrize("M", [64, 512, 4096, 28672])
def test_public_seeds_exist_and_are_legal_for_every_shape(N, K, M):
  import public                                            # noqa: PLC0415
  s = public.seeds(N, K, M)
  assert s and all(space.why_illegal(d) is None for d in s)


# ---- tune table -----------------------------------------------------------------------------------------------

def test_table_round_trip_and_lookup(tmp_path):
  cfg = space.TileConfig("once", 64, 64, 128, 4, 2, 4, 4)
  t = table.TuneTable(meta={"gpu": "test"})
  t.add(5120, 6144, shapes.bucket_of(40), cfg, gain=0.08, samples=3)
  t.add(5120, 6144, shapes.bucket_of(28672), None, gain=0.0, samples=0)
  p = tmp_path / "t.json"
  t.save(p)
  u = table.TuneTable.load(p)
  assert u.config_for(5120, 6144, 50) == cfg
  assert u.config_for(5120, 6144, 20000) is None          # stock kept for that bucket
  assert u.config_for(1024, 5120, 50) is None              # untuned shape -> stock
  assert json.loads(p.read_text())["meta"]["gpu"] == "test"


def test_tables_merge_later_files_win(tmp_path):
  a, b = table.TuneTable(meta={"run": "a"}), table.TuneTable(meta={"run": "b"})
  a.add(5120, 6144, shapes.bucket_of(40), space.TileConfig("once", 64, 64, 128, 4, 1, 4, 4))
  b.add(5120, 6144, shapes.bucket_of(40), None)
  b.add(14336, 5120, shapes.bucket_of(40), space.TileConfig("once", 32, 128, 128, 8, 1, 4, 4))
  a.save(tmp_path / "a.json")
  b.save(tmp_path / "b.json")
  m = table.TuneTable.load_many([tmp_path / "a.json", tmp_path / "b.json"])
  assert m.config_for(5120, 6144, 40) is None and m.config_for(14336, 5120, 40).BLOCK_M == 32
  assert [x["run"] for x in m.meta["merged"]] == ["a", "b"]


# ---- surrogate fallback and the measured game -----------------------------------------------------------------

def test_surrogate_ridge_backend_ranks_a_linear_truth():
  rng = random.Random(0)
  X = [[rng.random() for _ in range(5)] for _ in range(60)]
  y = [2 * x[0] - x[3] for x in X]
  s = xc.Surrogate(backend="ridge").fit(X, y)
  mu, sd = s.predict([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]])
  assert mu[0] > mu[1] and (sd > 0).all()


def test_aa_margin_is_xcaliburs_rule_over_the_aa_ratio_and_its_ci():
  """xcalibur.stats.aa_margin: threshold 1 + 2 * max|r - 1| over the A/A median and CI ends; never below the floor."""
  assert aa_margin({"ratio": 1.0, "ci": (0.99, 1.01)}, floor=0.005) == pytest.approx(0.02)
  assert aa_margin({"ratio": 1.0, "ci": (0.999, 1.001)}, floor=0.005) == 0.005
  assert aa_margin({"ratio": 1.0, "ci": (0.99, 1.01)}) == pytest.approx(xc.aa_margin([1.0, 0.99, 1.01]) - 1)


def _fake_measure(truth, rng, sd=0.003, loads=(("m64", 64),)):
  def measure(designs, grade):
    return [{L: truth(d, L) * math.exp(rng.gauss(0, sd)) for L, M in loads if space.applicable(d, M)} for d in designs]
  return measure


def test_fresh_confirm_keeps_a_real_winner():
  w = dict(quant="once", BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, GROUP_M=8, SPLIT_K=2, num_warps=4, num_stages=4)
  truth = lambda d, L: 1.0 if space.is_stock(d) else 1.06                                       # noqa: E731
  out = fresh_confirm(_fake_measure(truth, random.Random(0)), {"m64": {"design": w, "gain": 0.06}}, runs=3,
                      margin=0.01)
  assert out["m64"]["design"] == w and out["m64"]["samples"] == 3 and out["m64"]["gain"] > 0.04


def test_fresh_confirm_rejects_the_winners_curse():
  """A search winner that read +5% only because it was the luckiest of many: fresh runs see no gain, so no title."""
  w = dict(quant="once", BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, GROUP_M=8, SPLIT_K=2, num_warps=4, num_stages=4)
  out = fresh_confirm(_fake_measure(lambda d, L: 1.0, random.Random(0)), {"m64": {"design": w, "gain": 0.05}},
                      runs=3, margin=0.01)
  assert out == {}


def test_reach_respects_applicability():
  r = PerShapeReach({"m64": 64, "m28k": 28672}, K=5120)
  big = dict(quant="once", BLOCK_M=256, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, SPLIT_K=1, num_warps=8, num_stages=3)
  assert not r.reaches("m64", big, space.STOCK) and r.reaches("m28k", big, space.STOCK)
  assert r.identity("m64", big) == r.identity("m28k", big)


def test_game_binds_a_planted_winner_with_fake_measurements():
  """Seed screen -> confirm -> rounds -> bind over the real design space, with a synthetic timing model: designs in
  one region (quantise-once, split-K 2, 64-wide N tiles) are 10% faster than stock for question rows only."""
  rng = random.Random(1)
  loads = {"m64": 64, "m28k": 28672}
  winner = lambda d: d["quant"] == "once" and d["SPLIT_K"] == 2 and d["BLOCK_N"] == 64          # noqa: E731

  def truth(d, load):
    if space.is_stock(d):
      return 1.0
    return 1.10 if (load == "m64" and winner(d)) else 0.97

  def measure(designs, grade):
    sd = 0.01 if grade == "screen" else 0.003
    return [{L: truth(d, L) * math.exp(rng.gauss(0, sd)) for L, M in loads.items() if space.applicable(d, M)}
            for d in designs]

  game = make_game(loads, 5120, measure, compile=lambda ds: [{"ok": True} for _ in ds], surrogate_backend="ridge",
                   margin=0.01, min_samples=2, screen_n=12, compile_n=24, per_load=6)
  seeds = seed_designs(30, rng)
  assert space.is_stock(seeds[0]) and all(space.why_illegal(d) is None for d in seeds[1:])
  bound = play(game, seeds, rounds=4, log=lambda *_: None)
  assert winner(bound["m64"]["design"]) and bound["m64"]["gain"] > 0.08
  assert "m28k" not in bound
