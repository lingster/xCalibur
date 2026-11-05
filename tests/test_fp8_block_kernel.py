"""The block-FP8 GEMM against a CPU reference of the block-scaled math (dequantise -> fp32 matmul).

Runs the real Triton kernel without FP8 hardware: by default under Triton's CPU interpreter (TRITON_INTERPRET=1)
with the exact fp32 dot as the stand-in for the e4m3 mma (products of e4m3 values are exact in fp32), so the
quantised values, scales, tiling, masks and split-K are the kernel's own. FP8_KERNEL_TEST_DEVICE=cuda runs the same
tests on a GPU instead (bf16 stand-in dot on sm_86, the real fp8 dot on sm_89).
"""

import os
import sys

DEVICE = os.environ.get("FP8_KERNEL_TEST_DEVICE", "cpu")
if DEVICE == "cpu":
  os.environ["TRITON_INTERPRET"] = "1"                     # must precede the first @triton.jit

import pytest                                              # noqa: E402
torch = pytest.importorskip("torch")                         # noqa: E402  (CPU CI installs no torch)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kernels", "fp8_block_sm89"))

import kernel                                              # noqa: E402
import reference                                           # noqa: E402
from space import TileConfig                               # noqa: E402

if DEVICE == "cpu" and not kernel.interpreting():
  pytest.skip("triton was imported before TRITON_INTERPRET was set (run this file on its own)", allow_module_level=True)


def _problem(M, N, K, seed=0):
  return reference.random_problem(M, N, K, DEVICE, seed)


def _close_to_reference(out, x, w, ws):
  """The GEMM against the reference math. On a GPU the kernel divides with div.full.f32 (as the hub kernel does), so
  a rare activation code sits one e4m3 step from the IEEE reference (test_act_quant_matches_reference bounds that);
  the GEMM is then checked on the kernel's own quantised activations, which the fused path shares bit for bit."""
  if DEVICE == "cpu":
    ref = reference.block_fp8_matmul(x.cpu(), w.cpu(), ws.cpu())
  else:
    q, s = kernel.act_quant(x)
    ref = reference.block_fp8_matmul(q.cpu(), w.cpu(), ws.cpu(), x_scale=s.cpu())
  diff = (out.float().cpu() - ref).abs()
  # bf16 output: at most one bf16 rounding (2^-8 relative) plus fp32 summation-order noise.
  assert (diff <= ref.abs() * 2.0 ** -7 + 1e-4 * ref.abs().max()).all(), f"max abs diff {diff.max().item()}"


def test_act_quant_matches_reference():
  """Bitwise under the interpreter (IEEE division). On a GPU, Triton lowers fp32 '/' to div.full.f32 (<= 2 ulp, in
  the hub kernel too, so ours still equals stock bit for bit): scales agree to 2 ulp and a rare code sitting on a
  rounding boundary may move one e4m3 step."""
  x, _, _ = _problem(37, 128, 384)
  q, s = kernel.act_quant(x)
  rq, rs = reference.quantise_activation(x.cpu())
  assert q.dtype == torch.float8_e4m3fn and s.shape == (37, 3)
  if DEVICE == "cpu":
    assert torch.equal(s.cpu(), rs)
    assert torch.equal(q.cpu().view(torch.uint8), rq.view(torch.uint8))
  else:
    assert torch.allclose(s.cpu(), rs, rtol=2.0 ** -21, atol=0)
    steps = (q.cpu().view(torch.uint8).int() - rq.view(torch.uint8).int()).abs()
    assert steps.max() <= 1 and (steps > 0).float().mean() < 1e-2           # measured 0.13% on sm_86


def test_e4m3_rounding_edges_match_torch():
  """A row with amax 448 has scale exactly 1, so the codes are e4m3(x): carries into the next binade, ties to even,
  subnormals (ulp 2^-9), the 2^-6 boundary and signed zero, against the reference and torch's own cast."""
  edges = [448.0, 7.9, 7.99, 15.5, 126.4, 124.0, 0.12395, 0.0019, 0.00293, 2.0 ** -10, 2.0 ** -6 * 0.999,
           0.015625, 3.25, 3.75, 9.0, 11.0, -0.0, 0.0, -7.99, -0.0019, 1.0625, 1.1875, 240.0, 232.0]
  x = torch.zeros(1, 128)
  x[0, : len(edges)] = torch.tensor(edges)
  q, s = kernel.act_quant(x.to(DEVICE))
  assert s.item() == 1.0
  assert torch.equal(q.cpu().view(torch.uint8), reference.to_e4m3_sm89(x).view(torch.uint8))
  assert torch.equal(q.cpu().float(), x.to(torch.float8_e4m3fn).float())   # no value here sits just above a tie


def test_quant_follows_ada_double_rounding():
  """On sm_89 Triton lowers fp32 -> e4m3 as F2FP.F16.F32.PACK_AB.RZ then F2FP.SATFINITE.E4M3.F16 (sass.py): truncate
  to fp16, then round to nearest even. 8.5 + 2^-12 truncates to the tie 8.5 and rounds to 8 (one rounding gives 9)."""
  x = torch.zeros(1, 128)
  x[0, :4] = torch.tensor([448.0, 8.5 + 2.0 ** -12, -(8.5 + 2.0 ** -12), 8.5 + 2.0 ** -6])
  q, s = kernel.act_quant(x.to(DEVICE))
  rq, _ = reference.quantise_activation(x)
  assert q.cpu().float()[0, :4].tolist() == [448.0, 8.0, -8.0, 9.0]
  assert torch.equal(rq.view(torch.uint8), q.cpu().view(torch.uint8))


def test_reference_quantisation_roundtrip_is_tight():
  x = torch.randn(8, 256) * 3
  q, s = reference.quantise_activation(x)
  back = reference.dequantise(q, s, 1, 128)
  assert ((back - x).abs() <= x.abs() * 2.0 ** -4 + s.repeat_interleave(128, 1) * 2.0 ** -9).all()
  assert (q.float().abs().amax(1) <= 448).all()


CONFIGS = [
  TileConfig("fused", 64, 128, 128, 8, 1, 4, 3),           # the hub tiling at M <= 64
  TileConfig("fused", 16, 256, 128, 1, 1, 4, 2),
  TileConfig("once", 64, 128, 128, 8, 1, 4, 3),
  TileConfig("once", 32, 64, 64, 4, 1, 2, 3),              # BLOCK_K below the quant group
  TileConfig("once", 16, 32, 32, 1, 1, 2, 2),
  TileConfig("once", 32, 128, 128, 8, 2, 4, 3),            # split-K
  TileConfig("once", 16, 64, 64, 4, 4, 2, 2),              # split-K with sub-group K tiles (uneven split of 6)
]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.quant}-{c.BLOCK_M}x{c.BLOCK_N}x{c.BLOCK_K}-sk{c.SPLIT_K}")
@pytest.mark.parametrize("M", [1, 40])
def test_matmul_matches_reference(cfg, M):
  x, w, ws = _problem(M, 384, 384, seed=M)
  out = kernel.block_fp8_matmul(x, w, ws, cfg)
  assert out.shape == (M, 384) and out.dtype == torch.bfloat16
  _close_to_reference(out, x, w, ws)


def test_fused_and_quantise_once_are_bitwise_equal_on_the_hub_tiling():
  x, w, ws = _problem(40, 256, 384)
  a = kernel.block_fp8_matmul(x, w, ws, TileConfig("fused", 64, 128, 128, 8, 1, 4, 3))
  q, s = kernel.act_quant(x)
  b = kernel.block_fp8_matmul(q, w, ws, TileConfig("once", 64, 128, 128, 8, 1, 4, 3), a_scale=s)
  assert torch.equal(a, b)


def test_tiling_without_k_change_is_bitwise_equal():
  """Same K tile and no split: the per-element sum order is the tile loop's, whatever BLOCK_M/N/GROUP_M are."""
  x, w, ws = _problem(40, 256, 384)
  a = kernel.block_fp8_matmul(x, w, ws, TileConfig("once", 64, 128, 128, 8, 1, 4, 3))
  b = kernel.block_fp8_matmul(x, w, ws, TileConfig("once", 16, 64, 128, 1, 1, 2, 2))
  assert torch.equal(a, b)


def test_split_k_is_deterministic():
  x, w, ws = _problem(40, 256, 512)
  cfg = TileConfig("once", 32, 64, 64, 4, 4, 2, 2)
  assert torch.equal(kernel.block_fp8_matmul(x, w, ws, cfg), kernel.block_fp8_matmul(x, w, ws, cfg))


def test_leading_dims_are_flattened_and_restored():
  x, w, ws = _problem(6, 128, 256)
  out = kernel.block_fp8_matmul(x.view(2, 3, 256), w, ws, TileConfig("once", 16, 64, 128, 1, 1, 2, 2))
  assert out.shape == (2, 3, 128)
  _close_to_reference(out.view(6, 128), x, w, ws)


def test_linear_group_quantises_once_and_matches_single_calls():
  x, w1, s1 = _problem(40, 256, 384, seed=1)
  _, w2, s2 = _problem(40, 128, 384, seed=2)
  c1, c2 = TileConfig("once", 64, 128, 128, 8, 1, 4, 3), TileConfig("fused", 64, 128, 128, 8, 1, 4, 3)
  calls = []
  outs = kernel.linear_group(x, [(w1, s1, c1), (w2, s2, c2), (w1, s1, c1)], on_quant=lambda: calls.append(1))
  assert len(calls) == 1                                   # one activation quantisation for the whole group
  assert torch.equal(outs[0], kernel.block_fp8_matmul(x, w1, s1, c1))
  assert torch.equal(outs[1], kernel.block_fp8_matmul(x, w2, s2, c2))
  assert torch.equal(outs[0], outs[2])


def test_concat_group_is_one_gemm_bitwise_equal_to_separate_gemms():
  """q|k|v-style concatenation along N (block scales concatenated too) is exact: one launch, same outputs."""
  import concat                                            # noqa: PLC0415
  x, w1, s1 = _problem(40, 256, 384, seed=1)
  _, w2, s2 = _problem(40, 128, 384, seed=2)
  cat = concat.concat_weights([(w1, s1), (w2, s2), (w2, s2)])
  assert cat.w.shape == (512, 384) and cat.ws.shape == (4, 3) and cat.splits == (256, 128, 128)
  for cfg in (TileConfig("fused", 64, 128, 128, 8, 1, 4, 3), TileConfig("once", 32, 64, 64, 4, 1, 2, 2)):
    gemm = lambda a, w, ws, c=cfg: kernel.block_fp8_matmul(a, w, ws, c)              # noqa: E731
    outs = concat.forward(x, cat, gemm)
    assert [o.shape[-1] for o in outs] == [256, 128, 128]
    assert torch.equal(outs[0], gemm(x, w1, s1)) and torch.equal(outs[1], gemm(x, w2, s2))


def test_concat_rejects_n_off_the_scale_block():
  import concat                                            # noqa: PLC0415
  with pytest.raises(ValueError):
    concat.concat_weights([(torch.zeros(100, 128), torch.zeros(1, 1))])


def test_parity_passes_retilings_and_split_k_and_flags_a_real_change():
  import parity                                            # noqa: PLC0415
  x, w, ws = _problem(40, 256, 1024)
  stock = kernel.block_fp8_matmul(x, w, ws, TileConfig("fused", 64, 128, 128, 8, 1, 4, 3))
  same = parity.compare(kernel.block_fp8_matmul(x, w, ws, TileConfig("once", 16, 64, 128, 1, 1, 2, 2)), stock)
  assert same["bitwise"] and same["ok"] and same["max_abs"] == 0.0
  order = parity.compare(kernel.block_fp8_matmul(x, w, ws, TileConfig("once", 32, 64, 32, 4, 4, 2, 2)), stock)
  assert order["ok"] and order["max_ulp"] <= 1                       # summation order: at most one bf16 ulp
  wrong = parity.compare(kernel.block_fp8_matmul(x, w, ws * 1.01, TileConfig("fused", 64, 128, 128, 8, 1, 4, 3)),
                         stock)
  assert not wrong["ok"]


@pytest.mark.parametrize("bad", [
  dict(cfg=TileConfig("fused", 64, 128, 64, 8, 1, 4, 3)),                 # fused quant needs BLOCK_K = 128
  dict(cfg=TileConfig("once", 64, 128, 256, 8, 1, 4, 3)),                 # K tile must divide the group
])
def test_wrapper_rejects_configs_that_change_the_math(bad):
  x, w, ws = _problem(4, 128, 256)
  with pytest.raises(ValueError):
    kernel.block_fp8_matmul(x, w, ws, bad["cfg"])


def test_kernel_source_hash_is_stable():
  assert len(kernel.KERNEL_SHA) == 16 and kernel.KERNEL_SHA == kernel.source_sha()
