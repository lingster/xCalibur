"""kev_patch: the opt-in production hook (KEV_FP8_TUNED=1) that runs FP8Linear on the confirmed bitwise table.

Runs the real kernel under Triton's CPU interpreter (exact fp32 dot as the e4m3 stand-in), so the tests show the
dispatch, the refusals, the hygiene of the import and the launcher, and that a hit is bitwise equal to the stock
tiling. The hardware bitwise claim per entry is parity.py's on Ada (tables/ada_rtx6000_bitwise_r1.json).
"""

import json
import os
import sys

if os.environ.get("FP8_KERNEL_TEST_DEVICE", "cpu") == "cpu":
  os.environ.setdefault("TRITON_INTERPRET", "1")           # must precede the first @triton.jit

import pytest                                              # noqa: E402
import torch                                               # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kernels", "fp8_block_sm89"))
sys.path.insert(0, HERE)

import kev_patch                                           # noqa: E402
import reference                                           # noqa: E402

FP8Linear = pytest.importorskip("transformers.integrations.finegrained_fp8").FP8Linear
CFG = {"quant": "once", "BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 1, "SPLIT_K": 1, "num_warps": 4,
       "num_stages": 5}


@pytest.fixture(scope="module")
def kit():
  k = kev_patch.import_kit()
  if not k["kernel"].interpreting():
    pytest.skip("triton was imported before TRITON_INTERPRET was set (run this file on its own)")
  return k


def _table(tmp_path, entries=None, **meta):
  m = {"kind": "shippable", "bitwise_only": True, "kernel_sha": ["b3b9c062de1d4ffb"]} | meta
  e = entries if entries is not None else [{"N": 256, "K": 384, "bucket": "m64", "lo": 1, "hi": 64, "M": 64,
                                            "config": CFG}]
  p = tmp_path / "t.json"
  p.write_text(json.dumps({"meta": m, "entries": e}))
  return p


def _tuned(path, kit):
  t = kev_patch.TunedLinear(path, kit)
  t.require_cuda, t.dot = False, "fp32"
  return t


def _linear(N=256, K=384, bias=False, seed=0):
  _, w, ws = reference.random_problem(1, N, K, "cpu", seed)
  mod = FP8Linear(K, N, block_size=(128, 128), has_bias=bias)
  mod.weight.data, mod.weight_scale_inv.data = w, ws
  if bias:
    mod.bias.data = torch.randn(N).to(torch.bfloat16)
  return mod


# ---- import hygiene and refusals ------------------------------------------------------------------------------

def test_import_kit_leaves_no_generic_module_names_behind(monkeypatch):
  sentinel = type(sys)("table")
  monkeypatch.setitem(sys.modules, "table", sentinel)                      # production's own `table`, say
  before = list(sys.path)
  k = kev_patch.import_kit()
  assert sys.modules["table"] is sentinel and sys.path == before
  assert hasattr(k["kernel"], "block_fp8_matmul") and hasattr(k["table"], "TuneTable")


@pytest.mark.parametrize("meta,entries,match", [
  ({"kind": "search"}, None, "not a ship.py"),
  ({"bitwise_only": False}, None, "not a ship.py"),
  ({"kernel_sha": ["0123456789abcdef"]}, None, "measured on kernel"),
  ({}, [{"N": 256, "K": 384, "bucket": "m64", "lo": 1, "hi": 64, "M": 64, "config": dict(CFG, BLOCK_K=64)}],
   "bitwise class"),
  ({}, [{"N": 256, "K": 384, "bucket": "m64", "lo": 1, "hi": 64, "M": 64, "config": dict(CFG, SPLIT_K=2)}],
   "bitwise class"),
])
def test_tables_outside_the_bitwise_contract_are_refused(tmp_path, kit, meta, entries, match):
  with pytest.raises(ValueError, match=match):
    kev_patch.TunedLinear(_table(tmp_path, entries, **meta), kit)


def test_the_shipped_table_loads_and_routes_by_m_bucket(kit):
  t = kev_patch.TunedLinear(None, kit)
  assert t.path == kev_patch.DEFAULT_TABLE and len(t.table.entries) == 17
  assert t.config(1024, 5120, 40).BLOCK_M == 32                             # m64 entry: question rows
  assert t.config(1024, 5120, 100) is None                                  # m256 has no entry: stock
  assert t.config(10240, 5120, 28000).BLOCK_M == 64 and t.config(17408, 5120, 28000) is None   # gate/up: stock


# ---- dispatch -------------------------------------------------------------------------------------------------

def test_a_hit_is_bitwise_equal_to_the_stock_tiling_and_adds_bias(tmp_path, kit):
  t = _tuned(_table(tmp_path), kit)
  mod = _linear(bias=True)
  x = reference.random_problem(40, 128, 384, "cpu", seed=3)[0]
  out = t(mod, x, original=lambda _: pytest.fail("a hit must not call the stock forward"))
  stock_cfg = kit["space"].TileConfig("fused", 64, 128, 128, 8, 1, 4, 2)    # the hub's tiling at M = 40
  stock = kit["kernel"].block_fp8_matmul(x, mod.weight, mod.weight_scale_inv, stock_cfg, dot="fp32")
  assert torch.equal(out, stock.add_(mod.bias)) and out.dtype == torch.bfloat16
  assert sum(t.hits.values()) == 1 and not t.misses


def test_verify_mode_returns_stock_and_counts_mismatches(tmp_path, kit, monkeypatch):
  monkeypatch.setenv(kev_patch.ENV_VERIFY, "1")
  t = _tuned(_table(tmp_path), kit)
  mod = _linear()
  x = reference.random_problem(40, 128, 384, "cpu", seed=3)[0]
  stock_cfg = kit["space"].TileConfig("fused", 64, 128, 128, 8, 1, 4, 2)
  stock = kit["kernel"].block_fp8_matmul(x, mod.weight, mod.weight_scale_inv, stock_cfg, dot="fp32")
  assert torch.equal(t(mod, x, original=lambda _: stock.clone()), stock) and t.mismatches == 0
  wrong = stock.clone()
  wrong[0, 0] += 1
  assert torch.equal(t(mod, x, original=lambda _: wrong), wrong) and t.mismatches == 1
  assert "verify mismatches 1" in t.report()


@pytest.mark.parametrize("case", ["m_outside", "fp32_input", "non_contiguous", "other_shape", "static_scale"])
def test_misses_and_ineligible_calls_run_the_original_forward(tmp_path, kit, case):
  t = _tuned(_table(tmp_path), kit)
  mod = _linear(N=128 if case == "other_shape" else 256)
  x = reference.random_problem(100 if case == "m_outside" else 40, 128, 384, "cpu", seed=3)[0]
  if case == "fp32_input":
    x = x.float()
  if case == "non_contiguous":
    x = torch.cat([x, x], 1)[:, ::2]
  if case == "static_scale":
    mod.activation_scale = torch.nn.Parameter(torch.tensor(1.0))
  assert t(mod, x, original=lambda _: "stock") == "stock"
  assert not t.hits and sum(t.misses.values()) == 1


def test_cuda_is_required_by_default(tmp_path, kit):
  t = kev_patch.TunedLinear(_table(tmp_path), kit)
  x = reference.random_problem(40, 128, 384, "cpu", seed=3)[0]
  assert t(_linear(), x, original=lambda _: "stock") == "stock"


# ---- switches: nothing changes without the flag -----------------------------------------------------------------

def test_without_the_flag_nothing_is_patched(monkeypatch):
  monkeypatch.delenv(kev_patch.ENV_FLAG, raising=False)
  before = FP8Linear.forward
  assert kev_patch.install() is False and FP8Linear.forward is before
  model = torch.nn.Sequential(_linear())
  assert kev_patch.patch(model) == 0 and "forward" not in vars(model[0])


def test_install_and_uninstall_swap_the_class_forward(monkeypatch, tmp_path):
  monkeypatch.setenv(kev_patch.ENV_FLAG, "1")
  before = FP8Linear.forward
  try:
    assert kev_patch.install(_table(tmp_path), check_platform=False) is True
    assert FP8Linear.forward is not before
    assert kev_patch.install(_table(tmp_path), check_platform=False) is True   # idempotent
  finally:
    kev_patch.uninstall()
  assert FP8Linear.forward is before


def test_patch_hook_patches_only_this_models_plain_fp8_linears(monkeypatch, tmp_path):
  monkeypatch.setenv(kev_patch.ENV_FLAG, "1")

  class Emulated(FP8Linear):                                                # kev.fp8_emulation-style subclass
    pass

  other = Emulated(384, 256, block_size=(128, 128))
  model = torch.nn.Sequential(_linear(), _linear(seed=1), other)
  assert kev_patch.patch(model, _table(tmp_path), check_platform=False) == 2
  assert "forward" in vars(model[0]) and "forward" in vars(model[1]) and "forward" not in vars(other)
  assert "forward" not in vars(_linear())                                   # other models untouched


def test_launcher_runs_the_script_without_kit_names_on_the_path(monkeypatch, tmp_path):
  monkeypatch.delenv(kev_patch.ENV_FLAG, raising=False)
  out = tmp_path / "seen.json"
  script = tmp_path / "serve.py"
  script.write_text("import json, sys\n"
                    f"json.dump({{'argv': sys.argv, 'kit_on_path': {HERE!r} in sys.path, 'first': sys.path[0]}}, "
                    f"open({str(out)!r}, 'w'))\n")
  monkeypatch.setattr(sys, "path", list(sys.path))
  monkeypatch.setattr(sys, "argv", list(sys.argv))
  kev_patch.main([str(script), "--port", "8010"])
  seen = json.loads(out.read_text())
  assert seen["argv"] == [str(script), "--port", "8010"]
  assert seen["kit_on_path"] is False and seen["first"] == str(tmp_path.resolve())
