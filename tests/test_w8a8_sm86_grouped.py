"""Per-token, per-group INT8 activation scales for the W8A8 linear (kernels/w8a8_sm86/grouped.py).

One activation scale per token and per `group` input channels; the int32 partial sum of each K block of `group` channels
is rescaled before it joins the fp32 accumulator. With group = K that is the per-token quantiser exactly.
"""
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
from w8a8_sm86 import grouped, kev_patch, quant  # noqa: E402
from w8a8_sm86.linear import W8A8Linear, W8A8Weight, w8a8_linear  # noqa: E402

nn, F = torch.nn, torch.nn.functional
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _x(m, k, device="cpu", seed=0):
  g = torch.Generator().manual_seed(seed)
  return torch.randn(m, k, generator=g).to(device, torch.bfloat16)


def test_grouped_reference_has_one_scale_per_token_and_group():
  x = torch.zeros(2, 256)
  x[0, :128], x[0, 128:] = 2.0, 0.5
  x[1, 5] = -4.0
  q, s = grouped.quant_groups_reference(x, 128)
  assert s.shape == (2, 2) and q.dtype == torch.int8
  torch.testing.assert_close(s[0], torch.tensor([2.0 / 127, 0.5 / 127]))
  assert q[0].eq(127).all() and q[1, 5] == -127 and s[1, 1] > 0           # an all-zero group stays zero, no NaN


def test_one_group_equals_the_per_token_quantiser():
  x = _x(16, 256) * 3
  q, s = grouped.quant_groups_reference(x, 256)
  qr, sr = quant.quant_rows_reference(x)
  assert torch.equal(q, qr) and torch.equal(s[:, 0], sr)


def test_grouped_reference_applies_the_column_scale_first():
  x, c = _x(8, 256, seed=1), torch.rand(256) + 0.5
  q, s = grouped.quant_groups_reference(x, 128, col_scale=c)
  qr, sr = grouped.quant_groups_reference(x.float() * c, 128)
  assert torch.equal(q, qr) and torch.equal(s, sr)


def test_groups_cut_error_on_token_local_outliers():
  x = _x(128, 512, seed=2).float()
  x[torch.arange(128), torch.randint(0, 512, (128,), generator=torch.Generator().manual_seed(3))] *= 60
  w = _x(64, 512, seed=4).float() * 0.05
  ref = x @ w.T
  wt = W8A8Weight.from_float(w)
  err = lambda y: ((y.float() - ref).norm() / ref.norm()).item()
  assert err(w8a8_linear(x, wt, "reference", group=128)) < 0.6 * err(w8a8_linear(x, wt, "reference"))


def test_grouped_reference_linear_matches_its_dequantised_product():
  x, w = _x(12, 384, seed=5), _x(40, 384, seed=6) * 0.05
  wt = W8A8Weight.from_float(w)
  q, s = grouped.quant_groups_reference(x, 128)
  xd = (q.float().view(12, 3, 128) * s[..., None]).view(12, 384)
  want = (xd.double() @ (wt.q.double() * wt.scale.double()[:, None]).T).to(torch.bfloat16)
  torch.testing.assert_close(w8a8_linear(x, wt, "reference", group=128), want, atol=1e-3, rtol=8e-3)


def test_group_must_divide_k():
  with pytest.raises(ValueError):
    grouped.quant_groups_reference(_x(2, 200), 128)


def test_linear_carries_its_group():
  lin = nn.Linear(256, 64, bias=False).to(torch.bfloat16)
  m = W8A8Linear(W8A8Weight.from_float(lin.weight), "reference", group=128)
  assert m.group == 128 and "group=128" in m.extra_repr()
  torch.testing.assert_close(m(_x(4, 256)), w8a8_linear(_x(4, 256), m.weight8, "reference", group=128))


def _fake_layer():
  lin = lambda i, o: nn.Linear(i, o, bias=False).to(torch.bfloat16)

  class _Concat:
    def __init__(self, ls):
      self.weight, self.splits = torch.cat([l.weight for l in ls], 0), [l.out_features for l in ls]

    def __call__(self, x):
      return F.linear(x, self.weight).split(self.splits, -1)
  layer = types.SimpleNamespace(block_type="full_attention", self_attn=types.SimpleNamespace(
      qkv=_Concat([lin(256, 512), lin(256, 256), lin(256, 256)]), o_proj=lin(256, 256)))
  layer.mlp = types.SimpleNamespace(gate_up=_Concat([lin(256, 512), lin(256, 512)]), down_proj=lin(512, 256))
  return layer


def test_patch_layer_groups_only_the_named_outputs():
  layer = _fake_layer()
  kev_patch.patch_layer(layer, "reference", groups={"down_proj": 128, "o_proj": 128})
  assert layer.mlp.down_proj.group == 128 and layer.self_attn.o_proj.group == 128
  x = _x(8, 256, seed=7) * 0.1
  gate, up = layer.mlp.gate_up(x)
  want = w8a8_linear((F.silu(gate.float()) * up.float()), layer.mlp.down_proj.weight8, "reference", group=128)
  torch.testing.assert_close(kev_patch.mlp_forward(layer.mlp, x), want)


def test_group_env_parses_names(monkeypatch):
  from w8a8_sm86 import kev_smooth
  monkeypatch.setenv("W8A8_GROUP", "down_proj,out_proj,o_proj")
  assert kev_smooth.groups_from_env() == {"down_proj": 128, "out_proj": 128, "o_proj": 128}
  monkeypatch.setenv("W8A8_GROUP", "gate_up")
  with pytest.raises(ValueError):
    kev_smooth.groups_from_env()
  monkeypatch.delenv("W8A8_GROUP")
  assert kev_smooth.groups_from_env() == {}


# ------------------------------------------------------------------ GPU kernels

@gpu
@pytest.mark.parametrize("m,k", [(1, 4096), (37, 4096), (300, 12288)])
def test_triton_grouped_quantiser_matches_reference(m, k):
  x, c = _x(m, k, "cuda", seed=8) * 3, (torch.rand(k) + 0.5).cuda()
  for col in (None, c):
    q, s = grouped.quant_groups(x, 128, col)
    qr, sr = grouped.quant_groups_reference(x, 128, col)
    assert torch.equal(s, sr) and (q.int() - qr.int()).abs().max() <= 1


@gpu
def test_triton_grouped_swiglu_quant_matches_reference():
  gu = _x(40, 2 * 1024, "cuda", seed=9)
  gate, up = gu.split(1024, -1)
  q, s = grouped.swiglu_quant_groups(gate, up, 128)
  qr, sr = grouped.quant_groups_reference(F.silu(gate.float()) * up.float(), 128)
  assert torch.allclose(s, sr, rtol=1e-5) and (q.int() - qr.int()).abs().max() <= 1


@gpu
@pytest.mark.parametrize("m,n,k", [(1, 256, 4096), (200, 4096, 4096), (513, 4096, 12288), (1100, 1024, 4096)])
def test_triton_grouped_gemm_matches_reference(m, n, k):
  x, w = _x(m, k, "cuda", seed=10), _x(n, k, "cuda", seed=11) * 0.02
  wt = W8A8Weight.from_float(w)
  y = w8a8_linear(x, wt, "auto", group=128)
  q, s = grouped.quant_groups(x, 128)
  ref = grouped.gemm_groups_reference(q, s, wt.q, wt.scale, 128)
  assert y.dtype == torch.bfloat16 and y.shape == (m, n)
  torch.testing.assert_close(y, ref, atol=1e-3, rtol=8e-3)
