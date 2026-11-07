"""W8A8 linear (kernels/w8a8_sm86): quantisers, the int8 GEMM's dequant epilogue and Kev's projection swap.

The CPU tests check the torch reference and the patch's structure; the GPU tests check the Triton kernels and
torch._int_mm against that reference. INT8 products accumulate exactly in int32, so every backend must agree with the
float64 reference up to the final bf16 rounding.
"""
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
from w8a8_sm86 import kev_patch, quant  # noqa: E402
from w8a8_sm86.linear import W8A8Linear, W8A8Weight, w8a8_linear  # noqa: E402

nn, F = torch.nn, torch.nn.functional
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _x(m, k, device="cpu", seed=0):
  g = torch.Generator(device="cpu").manual_seed(seed)
  return torch.randn(m, k, generator=g).to(device, torch.bfloat16)


# ------------------------------------------------------------------ torch reference (CPU)

def test_quant_rows_reference_is_symmetric_per_row_absmax():
  x = torch.tensor([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0], [127.0, 3.0, -127.0]], dtype=torch.bfloat16)
  q, s = quant.quant_rows_reference(x)
  assert q.dtype == torch.int8 and s.dtype == torch.float32 and s.shape == (3,)
  assert q[0].tolist() == [64, -127, 32] and s[0].item() == pytest.approx(2.0 / 127)
  assert q[1].tolist() == [0, 0, 0] and s[1].item() > 0            # an all-zero row stays zero, no NaN
  assert q[2].tolist() == [127, 3, -127]


def test_quant_weight_is_per_output_channel():
  w = torch.tensor([[0.5, -1.0], [4.0, 2.0]], dtype=torch.bfloat16)
  q, s = quant.quant_weight(w)
  assert q.tolist() == [[64, -127], [127, 64]] and s.tolist() == pytest.approx([1.0 / 127, 4.0 / 127])


def test_reference_w8a8_matches_float_matmul_within_quantisation_error():
  x, w = _x(64, 256), _x(96, 256, seed=1) * 0.05
  wt = W8A8Weight.from_float(w)
  y = quant.w8a8_reference(x, wt.q, wt.scale)
  ref = F.linear(x.float(), w.float())
  assert y.dtype == torch.bfloat16 and y.shape == (64, 96)
  assert (y.float() - ref).norm() / ref.norm() < 0.02


@pytest.mark.parametrize("splits,quantised", [((8192, 4096, 32, 32), 2), ((8192, 1024, 1024), 3), ((12288, 12288), 2)])
def test_quantised_prefix_keeps_small_trailing_parts_bf16(splits, quantised):
  assert kev_patch.quantised_prefix(splits, min_out=256) == quantised


def test_quantised_prefix_refuses_a_small_part_before_a_large_one():
  with pytest.raises(ValueError):
    kev_patch.quantised_prefix((32, 4096), min_out=256)


class _Concat:                                   # stands in for kev.fused_proj.ConcatProjection
  def __init__(self, linears):
    self.weight = torch.cat([l.weight for l in linears], 0)
    self.splits = [l.out_features for l in linears]

  def __call__(self, x):
    return F.linear(x, self.weight).split(self.splits, -1)


def _fake_lm(hidden=256, inter=512):
  lin = lambda i, o: nn.Linear(i, o, bias=False).to(torch.bfloat16)
  delta = types.SimpleNamespace(block_type="linear_attention", linear_attn=types.SimpleNamespace(
      in_proj=_Concat([lin(hidden, 512), lin(hidden, 256), lin(hidden, 4), lin(hidden, 4)]), out_proj=lin(256, hidden)))
  attn = types.SimpleNamespace(block_type="full_attention", self_attn=types.SimpleNamespace(
      qkv=_Concat([lin(hidden, 512), lin(hidden, 256), lin(hidden, 256)]), o_proj=lin(256, hidden)))
  for layer in (delta, attn):
    layer.mlp = types.SimpleNamespace(gate_up=_Concat([lin(hidden, inter), lin(hidden, inter)]), down_proj=lin(inter, hidden))
  return types.SimpleNamespace(lm=types.SimpleNamespace(layers=[delta, attn]))


def test_patch_swaps_every_backbone_projection_and_keeps_b_a_in_bf16():
  model = _fake_lm()
  report = kev_patch.patch(model, backend="reference")
  delta, attn = model.lm.layers
  assert isinstance(delta.linear_attn.in_proj, kev_patch.W8A8Projection)
  assert delta.linear_attn.in_proj.tail.dtype == torch.bfloat16 and delta.linear_attn.in_proj.tail.shape[0] == 8
  assert isinstance(delta.linear_attn.out_proj, W8A8Linear) and isinstance(attn.self_attn.o_proj, W8A8Linear)
  assert attn.self_attn.qkv.tail is None and isinstance(attn.mlp.down_proj, W8A8Linear)
  assert report["groups"] == 4 and report["linears"] == 4


def test_patched_layers_compute_close_to_bf16():
  model = _fake_lm()
  x = _x(32, 256, seed=3) * 0.1
  want_in, want_mlp = model.lm.layers[0].linear_attn.in_proj(x), model.lm.layers[1].mlp.gate_up(x)   # bf16, before the swap
  kev_patch.patch(model, backend="reference")
  got_in = model.lm.layers[0].linear_attn.in_proj(x)
  assert [t.shape for t in got_in] == [t.shape for t in want_in]
  assert torch.equal(got_in[2], want_in[2]) and torch.equal(got_in[3], want_in[3])     # b, a: still bf16, bit-identical
  for g, w in zip(got_in[:2], want_in[:2]):
    assert (g.float() - w.float()).norm() / w.float().norm() < 0.03
  got_mlp = model.lm.layers[1].mlp.gate_up(x)
  for g, w in zip(got_mlp, want_mlp):
    assert (g.float() - w.float()).norm() / w.float().norm() < 0.03


def test_patched_mlp_forward_runs_swiglu_then_down():
  model = _fake_lm()
  mlp = model.lm.layers[0].mlp
  x = _x(16, 256, seed=4) * 0.1
  gate, up = mlp.gate_up(x)
  want = F.linear(F.silu(gate.float()) * up.float(), mlp.down_proj.weight.float())
  kev_patch.patch(model, backend="reference")
  got = kev_patch.mlp_forward(mlp, x)
  assert (got.float() - want).norm() / want.norm() < 0.05


# ------------------------------------------------------------------ GPU kernels

@gpu
@pytest.mark.parametrize("m,k", [(1, 4096), (17, 4096), (300, 12288), (64, 200)])
def test_triton_quant_rows_matches_reference(m, k):
  x = _x(m, k, "cuda") * 3
  q, s = quant.quant_rows(x)
  qr, sr = quant.quant_rows_reference(x)
  assert torch.equal(s, sr) and (q.int() - qr.int()).abs().max() <= 1          # a tie may round the other way


@gpu
def test_triton_swiglu_quant_matches_reference():
  gu = _x(40, 2 * 1024, "cuda")
  gate, up = gu.split(1024, -1)                       # strided views, as kev's gate_up split produces
  q, s = quant.swiglu_quant(gate, up)
  qr, sr = quant.quant_rows_reference(F.silu(gate.float()) * up.float())
  assert torch.allclose(s, sr, rtol=1e-5) and (q.int() - qr.int()).abs().max() <= 1


@gpu
@pytest.mark.parametrize("backend", ["triton", "intmm"])
@pytest.mark.parametrize("m,n,k", [(1, 256, 4096), (16, 1024, 4096), (200, 8192, 4096), (513, 4096, 12288), (40, 264, 256)])
def test_w8a8_linear_backends_match_reference(backend, m, n, k):
  x, w = _x(m, k, "cuda"), _x(n, k, "cuda", seed=2) * 0.02
  wt = W8A8Weight.from_float(w)
  y = w8a8_linear(x, wt, backend=backend)
  xq, xs = quant.quant_rows(x)
  ref = quant.w8a8_reference_from_quantised(xq, xs, wt.q, wt.scale)
  assert y.dtype == torch.bfloat16 and y.shape == (m, n)
  torch.testing.assert_close(y, ref, atol=1e-3, rtol=8e-3)                       # bf16 rounding of the same exact sum


@gpu
def test_w8a8_linear_keeps_leading_dims_and_accepts_strided_input():
  x = _x(2 * 33, 2 * 512, "cuda").reshape(2, 33, 1024)[..., :512]
  wt = W8A8Weight.from_float(_x(384, 512, "cuda", seed=5) * 0.02)
  y = w8a8_linear(x, wt)
  assert y.shape == (2, 33, 384)
  torch.testing.assert_close(y, w8a8_linear(x.contiguous(), wt))


@gpu
def test_w8a8_gemm_is_deterministic_across_calls():
  x, wt = _x(1000, 4096, "cuda"), W8A8Weight.from_float(_x(2048, 4096, "cuda", seed=6) * 0.02)
  assert torch.equal(w8a8_linear(x, wt), w8a8_linear(x, wt))


def test_auto_backend_picks_triton_below_the_threshold_and_intmm_at_it():
  from w8a8_sm86.linear import AUTO_INTMM_MIN_M, resolve_backend
  assert resolve_backend("auto", AUTO_INTMM_MIN_M - 1) == "triton"
  assert resolve_backend("auto", AUTO_INTMM_MIN_M) == "intmm"
  assert resolve_backend("reference", 10**6) == "reference"


@gpu
@pytest.mark.parametrize("m", [17, 1100])
def test_auto_backend_matches_reference_on_both_sides(m):
  x, wt = _x(m, 4096, "cuda"), W8A8Weight.from_float(_x(1024, 4096, "cuda", seed=7) * 0.02)
  xq, xs = quant.quant_rows(x)
  torch.testing.assert_close(w8a8_linear(x, wt, backend="auto"),
                             quant.w8a8_reference_from_quantised(xq, xs, wt.q, wt.scale), atol=1e-3, rtol=8e-3)


def test_patch_can_keep_the_down_projection_in_bf16():
  model = _fake_lm()
  down = model.lm.layers[0].mlp.down_proj
  report = kev_patch.patch(model, backend="reference", keep=("down_proj",))
  mlp = model.lm.layers[0].mlp
  assert mlp.down_proj is down and isinstance(mlp.gate_up, kev_patch.W8A8Projection)
  assert not hasattr(mlp, "forward")              # kev's own fused MLP forward stays: fla SwiGLU, then the bf16 down
  assert report["linears"] == 2 and report["groups"] == 4


def test_patch_refuses_unknown_names_to_keep():
  with pytest.raises(ValueError):
    kev_patch.patch(_fake_lm(), backend="reference", keep=("lm_head",))
