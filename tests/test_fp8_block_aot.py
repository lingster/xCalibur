"""The real FP8 path, compiled for sm_89 on this machine (no Ada, no GPU): Triton cross-compiles to an sm_89 cubin.

The only test of the native e4m3 path we can run without FP8 hardware: the fp8 kernel compiles, its K loop issues
Ada's native e4m3 MMA (QMMA.16832.F32.E4M3.E4M3 in SASS, not converts + FP16 HMMA), the smem rule agrees with the
compiler, and quantise-once removes the in-loop quantisation the stock kernel pays per K tile. The compiles run in a
subprocess (aot.py's CLI) because the kernel tests may have put this process's Triton into interpreter mode.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.join(os.path.dirname(__file__), "..", "kernels", "fp8_block_sm89")
sys.path.insert(0, HERE)
if importlib.util.find_spec("triton") is None:              # find, never import: importing triton here would fix
  pytest.skip("needs triton", allow_module_level=True)      # its interpreter mode before the kernel tests choose it

import space                                               # noqa: E402
from space import TileConfig                               # noqa: E402

CONFIGS = [
  TileConfig("fused", 64, 128, 128, 8, 1, 4, 3),           # the stock tiling at M <= 64
  TileConfig("fused", 128, 128, 128, 8, 1, 8, 2),          # the stock tiling at M > 64 (only 2 stages fit)
  TileConfig("once", 128, 128, 128, 8, 1, 8, 4),
  TileConfig("once", 32, 64, 64, 4, 4, 2, 3),              # split-K, fp32 workspace output
  TileConfig("fused", 128, 128, 128, 8, 1, 8, 3),          # the stock tile with 3 stages: out of shared memory
]
SAMPLE = 6


@pytest.fixture(scope="module")
def records():
  env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"} | {"CUDA_VISIBLE_DEVICES": ""}
  out = subprocess.run([sys.executable, os.path.join(HERE, "aot.py"), "--configs",
                        json.dumps([c.as_dict() for c in CONFIGS]), "--sample", str(SAMPLE), "--hub-grid"],
                       capture_output=True, text=True, env=env, timeout=1800)
  assert out.returncode == 0, out.stderr[-2000:]
  rows = json.loads(out.stdout.strip().splitlines()[-1])
  return {"ours": rows[:len(CONFIGS)], "sample": rows[len(CONFIGS):len(CONFIGS) + SAMPLE],
          "hub": rows[len(CONFIGS) + SAMPLE:]}


@pytest.mark.parametrize("i", range(4), ids=lambda i: f"{CONFIGS[i].quant}-{CONFIGS[i].BLOCK_M}x{CONFIGS[i].BLOCK_N}"
                         f"x{CONFIGS[i].BLOCK_K}-s{CONFIGS[i].num_stages}-sk{CONFIGS[i].SPLIT_K}")
def test_fp8_build_compiles_for_sm89_with_native_e4m3_mma(records, i):
  r = records["ours"][i]
  assert r["ok"], r
  assert r["mma"] > 0 and r["native_fp8_mma"] and r["loop_mma"] > 0 and 0 < r["n_regs"] <= 255
  est = space.smem_bytes(CONFIGS[i].as_dict())
  assert est <= r["shared"] <= est + 1024                   # the rule is a lower bound; small tiles add scratch


def test_stock_tile_with_three_stages_is_out_of_shared_memory(records):
  r = records["ours"][4]
  assert not r["ok"] and r["shared"] > space.SMEM_LIMIT_SM89


def test_quantise_once_drops_the_in_loop_conversions(records):
  fused, once = records["ours"][1], records["ours"][2]       # same 128 x 128 x 128 tile
  assert fused["loop_convert"] > 0 and once["loop_convert"] == 0
  assert once["loop_instructions"] < fused["loop_instructions"]


def test_hub_kernel_on_ada_native_mma_and_only_two_stages_at_m128(records):
  hub = {(h["hub"]["BLOCK_SIZE_M"], h["hub"]["num_warps"], h["hub"]["num_stages"]): h for h in records["hub"]}
  assert len(hub) == 4 * 12
  assert all(h["native_fp8_mma"] for h in hub.values() if h["ok"])
  assert {s for (bm, w, s), h in hub.items() if bm == 128 and h["ok"]} == {2}
  ours = records["ours"][1]
  assert hub[(128, 8, 2)]["shared"] == ours["shared"]       # our stock-tiling build stages the same bytes


def test_legal_designs_compile_on_a_sample(records):
  """The cheap legality rules rarely admit a design the compiler rejects; the search's compile step catches the rest."""
  bad = [r for r in records["sample"] if not r["ok"]]
  assert len(bad) <= 1, bad


def _signature(kernel_file: str, configs: list[TileConfig]) -> list[str]:
  env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"} | {"CUDA_VISIBLE_DEVICES": ""}
  out = subprocess.run([sys.executable, os.path.join(HERE, "aot.py"), "--signature", kernel_file, "--configs",
                        json.dumps([c.as_dict() for c in configs])], capture_output=True, text=True, env=env,
                       timeout=1800)
  assert out.returncode == 0, out.stderr[-2000:]
  return json.loads(out.stdout.strip().splitlines()[-1])


def _history(path: str) -> dict[str, bytes]:
  """{sha256[:16]: source} of every committed revision of a kit file (empty without git)."""
  import hashlib                                                              # noqa: PLC0415
  root = os.path.join(HERE, "..", "..")
  log = subprocess.run(["git", "-C", root, "log", "--format=%H", "--", path], capture_output=True, text=True)
  out: dict[str, bytes] = {}
  for rev in log.stdout.split() if log.returncode == 0 else []:
    blob = subprocess.run(["git", "-C", root, "show", f"{rev}:{path}"], capture_output=True).stdout
    out.setdefault(hashlib.sha256(blob).hexdigest()[:16], blob)
  return out


def test_fp8_path_of_equivalent_sources_is_the_same_machine_code(tmp_path):
  """Tables measured on an older kernel.py stay valid only if its fp8 path compiles to the same SASS: every sha in
  kernel.FP8_EQUIVALENT_SHAS is found in git history and compiles (GEMM at the shipped tiles, act_quant) to the same
  instruction stream as the current source."""
  import re                                                                   # noqa: PLC0415
  src = open(os.path.join(HERE, "kernel.py")).read()
  body = re.search(r"FP8_EQUIVALENT_SHAS = frozenset\(\{([^}]*)\}\)", src).group(1)
  shas = set(re.findall(r"[0-9a-f]{16}", body))
  history = _history("kernels/fp8_block_sm89/kernel.py")
  if not history:
    pytest.skip("no git history (tarball)")
  assert shas <= set(history), f"not in history: {shas - set(history)}"
  shipped = [TileConfig("once", 64, 128, 128, 8, 1, 4, 3), TileConfig("once", 32, 64, 128, 1, 1, 4, 5),
             TileConfig("once", 128, 128, 128, 1, 1, 8, 3), TileConfig("fused", 128, 128, 128, 8, 1, 8, 2)]
  current = _signature("current", shipped)
  mutant = tmp_path / "kernel_mutant.py"                                      # the signature does see a math change
  mutant.write_text(src.replace("axis=1) / 448.0", "axis=1) / 447.0"))
  changed = [a != b for a, b in zip(_signature(str(mutant), shipped), current)]
  assert changed == [False, False, False, True, True]                         # the fused GEMM and act_quant quantise
  for sha in shas:
    old = tmp_path / f"kernel_{sha}.py"
    old.write_bytes(history[sha])
    assert _signature(str(old), shipped) == current, sha
