"""SmoothQuant for the W8A8 linear (kernels/w8a8_sm86): scales, calibration taps, folds and the smoothed Kev patch.

The folds move 1/s into the op before each GEMM (a norm weight, the v or up rows) and s into the GEMM's weight columns,
so in fp32 a folded layer must compute what the unfolded one does, up to rounding. The fake layer below runs kev's fused
decoder math (zero-centred RMSNorm with weight 1 + w, DeltaNet gated norm, GQA attention with a sigmoid output gate,
SwiGLU MLP) in fp32 on small shapes.
"""
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
from w8a8_sm86 import kev_patch, kev_smooth, quant, record, smooth  # noqa: E402
from w8a8_sm86.linear import W8A8Linear, W8A8Weight, w8a8_linear  # noqa: E402

nn, F = torch.nn, torch.nn.functional
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
H, INTER, HD, NQ, NKV, DV = 64, 96, 8, 4, 2, 8          # hidden, MLP, attention head dim, q / kv heads, DeltaNet v dim


def _g(seed):
  return torch.Generator().manual_seed(seed)


# ------------------------------------------------------------------ scales

def test_smooth_scales_follow_the_paper_formula():
  a, w = torch.tensor([4.0, 100.0, 1.0]), torch.tensor([0.5, 0.1, 2.0])
  s = smooth.smooth_scales(a, w, alpha=0.65)
  torch.testing.assert_close(s, a ** 0.65 / w ** 0.35)


def test_alpha_half_equalises_activation_and_weight_ranges():
  a, w = torch.rand(32, generator=_g(0)) * 50 + 0.1, torch.rand(32, generator=_g(1)) + 0.01
  s = smooth.smooth_scales(a, w, alpha=0.5)
  torch.testing.assert_close(a / s, w * s)


def test_smooth_scales_stay_finite_on_dead_channels():
  s = smooth.smooth_scales(torch.tensor([0.0, 3.0]), torch.tensor([1.0, 0.0]), alpha=0.5)
  assert torch.isfinite(s).all() and (s > 0).all()


def test_tied_scales_are_shared_by_the_query_heads_of_one_kv_head():
  act = torch.rand(NQ * HD, generator=_g(2)) * 10
  w = torch.rand(NQ * HD, generator=_g(3))
  s_kv = smooth.tied_scales(act, w, alpha=0.5, groups=NQ // NKV, head_dim=HD)
  assert s_kv.shape == (NKV * HD,)
  full = smooth.expand_tied(s_kv, groups=NQ // NKV, head_dim=HD)
  g = NQ // NKV
  for h in range(NQ):
    torch.testing.assert_close(full[h * HD:(h + 1) * HD], s_kv[(h // g) * HD:(h // g + 1) * HD])
  want = smooth.smooth_scales(act.view(NKV, g, HD).amax(1).flatten(), w.view(NKV, g, HD).amax(1).flatten(), 0.5)
  torch.testing.assert_close(s_kv, want)


def test_stats_merge_keeps_the_running_max_and_round_trips_through_a_file(tmp_path):
  st = smooth.ActStats()
  st.observe("3.in", torch.tensor([[1.0, -5.0], [2.0, 0.5]]))
  st.observe("3.in", torch.tensor([[-4.0, 1.0]]))
  assert st["3.in"].tolist() == [4.0, 5.0]
  st.meta["requests"] = ["a#0"]
  st.save(tmp_path / "s.pt")
  back = smooth.ActStats.load(tmp_path / "s.pt")
  assert back["3.in"].tolist() == [4.0, 5.0] and back.meta["requests"] == ["a#0"]


# ------------------------------------------------------------------ quantiser column scale

def test_reference_quantiser_applies_the_column_scale_before_the_absmax():
  x, c = torch.randn(8, 16, generator=_g(4)), torch.rand(16, generator=_g(5)) + 0.5
  q, s = quant.quant_rows_reference(x, col_scale=c)
  qr, sr = quant.quant_rows_reference(x * c)
  assert torch.equal(q, qr) and torch.equal(s, sr)


@gpu
def test_triton_quantiser_with_column_scale_matches_reference():
  x = torch.randn(33, 4096, generator=_g(6)).cuda().bfloat16() * 3
  c = (torch.rand(4096, generator=_g(7)) + 0.25).cuda()
  q, s = quant.quant_rows(x, col_scale=c)
  qr, sr = quant.quant_rows_reference(x, col_scale=c)
  assert torch.equal(s, sr) and (q.int() - qr.int()).abs().max() <= 1


def test_smoothing_cuts_int8_error_on_outlier_channels():
  x = torch.randn(256, 128, generator=_g(8))
  x[:, [3, 77]] *= 80                                         # two outlier channels, as in Qwen residual streams
  w = torch.randn(64, 128, generator=_g(9)) * 0.05
  ref = x @ w.T
  err = lambda y: ((y.float() - ref).norm() / ref.norm()).item()
  plain = err(w8a8_linear(x, W8A8Weight.from_float(w), "reference"))
  s = smooth.smooth_scales(x.abs().amax(0), w.abs().amax(0), 0.5)
  smoothed = err(w8a8_linear(x / s, W8A8Weight.from_float(w * s), "reference"))
  assert smoothed < 0.5 * plain


# ------------------------------------------------------------------ a fake fused Kev layer (fp32, kev's math)

class _Concat:                                   # stands in for kev.fused_proj.ConcatProjection
  def __init__(self, sizes, k, seed):
    self.weight = torch.randn(sum(sizes), k, generator=_g(seed)) * 0.1
    self.splits = list(sizes)

  def __call__(self, x):
    return F.linear(x, self.weight).split(self.splits, -1)


def _lin(k, n, seed):
  m = nn.Linear(k, n, bias=False)
  with torch.no_grad():
    m.weight.copy_(torch.randn(n, k, generator=_g(seed)) * 0.1)
  return m


def _layer(kind, seed=0):
  layer = types.SimpleNamespace(block_type=kind, input_norm_weight=1 + 0.1 * torch.randn(H, generator=_g(seed)),
                                post_norm_weight=1 + 0.1 * torch.randn(H, generator=_g(seed + 1)))
  if kind == "linear_attention":
    layer.linear_attn = types.SimpleNamespace(in_proj=_Concat([2 * DV * 4, DV * 4, 4, 4], H, seed + 2),
                                              out_proj=_lin(DV * 4, H, seed + 3),
                                              norm=types.SimpleNamespace(weight=1 + 0.1 * torch.randn(DV, generator=_g(seed + 4))))
  else:
    layer.self_attn = types.SimpleNamespace(qkv=_Concat([2 * NQ * HD, NKV * HD, NKV * HD], H, seed + 2),
                                            o_proj=_lin(NQ * HD, H, seed + 3), head_dim=HD)
  layer.mlp = types.SimpleNamespace(gate_up=_Concat([INTER, INTER], H, seed + 5), down_proj=_lin(INTER, H, seed + 6))
  return layer


def _rms(x, w):
  return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def _forward(layer, h, out_scale=None):
  """kev.fused_qwen35.decoder_step in fp32 on [T, H]; DeltaNet's recurrence replaced by a linear stand-in (any linear
  map of the mixed q/k/v keeps the fold exact: only out_proj's input is smoothed there, behind the gated norm)."""
  x = _rms(h, layer.input_norm_weight)
  if layer.block_type == "linear_attention":
    m = layer.linear_attn
    mixed, z, b, a = m.in_proj(x)
    o = mixed[:, :DV * 4].reshape(-1, 4, DV) + mixed[:, DV * 4:].reshape(-1, 4, DV)
    o = (_rms(o, m.norm.weight) * F.silu(z.reshape(-1, 4, DV))).reshape(-1, DV * 4)
    o = o if out_scale is None else o * out_scale
    y = m.out_proj(o) + b.sum(-1, keepdim=True) * a.mean(-1, keepdim=True)
  else:
    m = layer.self_attn
    qg, k, v = m.qkv(x)
    q, gate = qg.reshape(-1, NQ, 2 * HD).chunk(2, -1)
    k, v = k.reshape(-1, NKV, HD).repeat_interleave(NQ // NKV, 1), v.reshape(-1, NKV, HD).repeat_interleave(NQ // NKV, 1)
    p = torch.softmax(torch.einsum("thd,shd->hts", q, k) / HD ** 0.5, -1)
    out = torch.einsum("hts,shd->thd", p, v)
    y = m.o_proj((torch.sigmoid(gate) * out).reshape(-1, NQ * HD))
  r = h + y
  gate, up = layer.mlp.gate_up(_rms(r, layer.post_norm_weight))
  return r + layer.mlp.down_proj(F.silu(gate) * up)


def _stats(layer_index, seed=20):
  st = smooth.ActStats()
  for i, (site, k) in enumerate({"in": H, "out": NQ * HD, "mlp": H, "down": INTER}.items()):   # NQ * HD == 4 * DV
    st.observe(f"{layer_index}.{site}", (torch.rand(1, k, generator=_g(seed + i)) * 30 + 0.05))
  return st


@pytest.mark.parametrize("kind", ["linear_attention", "full_attention"])
@pytest.mark.parametrize("alpha", [0.5, 0.8])
def test_folded_layer_computes_what_the_unfolded_one_does_in_fp32(kind, alpha):
  layer = _layer(kind)
  h = torch.randn(10, H, generator=_g(30)) * 2
  want = _forward(layer, h)
  st = _stats(0)
  unfolded = kev_smooth.fold_layer(layer, 0, st, alpha)
  got = _forward(layer, h, unfolded.get("out"))
  torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
  assert set(unfolded) == ({"out"} if kind == "linear_attention" else set())


def test_fold_moves_the_activation_range_into_the_weights():
  layer = _layer("full_attention")
  st = _stats(0)
  w0 = layer.mlp.gate_up.weight.clone()
  norm0 = layer.post_norm_weight.clone()
  kev_smooth.fold_layer(layer, 0, st, 0.5)
  s = smooth.smooth_scales(st["0.mlp"], w0.abs().amax(0), 0.5)
  torch.testing.assert_close(layer.post_norm_weight, norm0 / s)
  torch.testing.assert_close(layer.mlp.gate_up.weight[:INTER], w0[:INTER] * s)


# ------------------------------------------------------------------ taps and the smoothed patch

def test_taps_record_every_site_and_restore_the_layer():
  layers = [_layer("linear_attention"), _layer("full_attention", seed=40)]
  before = [(l.mlp.gate_up, l.mlp.down_proj._forward_pre_hooks.copy()) for l in layers]
  st = smooth.ActStats()
  with record.taps(layers, st):
    for l in layers:
      _forward(l, torch.randn(6, H, generator=_g(41)))
  assert sorted(st.data) == sorted(f"{i}.{s}" for i in range(2) for s in record.SITES)
  assert st["1.down"].shape == (INTER,) and st["0.out"].shape == (DV * 4,)
  assert all(l.mlp.gate_up is g and not l.mlp.down_proj._forward_pre_hooks for l, (g, _) in zip(layers, before))


def test_taps_keep_a_row_sample_of_long_inputs():
  layer = _layer("full_attention")
  st, samples = smooth.ActStats(), {}
  with record.taps([layer], st, samples=samples, sample_rows=4, min_rows=8):
    _forward(layer, torch.randn(16, H, generator=_g(42)))
  assert samples["0.in"].shape == (4, H) and samples["0.down"].shape == (4, INTER)


def test_taps_sample_the_largest_input_when_none_is_long():
  layer = _layer("full_attention")
  st, samples = smooth.ActStats(), {}
  with record.taps([layer], st, samples=samples, sample_rows=4, min_rows=8):
    _forward(layer, torch.randn(2, H, generator=_g(43)))
    _forward(layer, torch.randn(5, H, generator=_g(44)))
    _forward(layer, torch.randn(3, H, generator=_g(45)))
  assert samples["0.out"].shape == (4, NQ * HD)


def _bf16_lm():
  layers = [_layer("linear_attention"), _layer("full_attention", seed=50)]
  for l in layers:
    for g in (getattr(l, "linear_attn", None) or l.self_attn, l.mlp):
      for name in ("in_proj", "qkv", "gate_up"):
        if hasattr(g, name):
          getattr(g, name).weight = getattr(g, name).weight.bfloat16()
    mixer_out = l.linear_attn.out_proj if hasattr(l, "linear_attn") else l.self_attn.o_proj
    for mod in (l.mlp.down_proj, mixer_out):
      mod.to(torch.bfloat16)
  return types.SimpleNamespace(lm=types.SimpleNamespace(layers=layers))


def test_smoothed_patch_converts_every_layer_and_sets_the_deltanet_quantiser_scale():
  model, st = _bf16_lm(), smooth.ActStats()
  for i in range(2):
    for site, k in (("in", H), ("out", DV * 4 if i == 0 else NQ * HD), ("mlp", H), ("down", INTER)):
      st.observe(f"{i}.{site}", torch.rand(1, k, generator=_g(60 + i)) * 20 + 0.1)
  report = kev_smooth.patch_smooth(model, st, alpha=0.5, backend="reference")
  delta, attn = model.lm.layers
  assert isinstance(delta.linear_attn.out_proj, W8A8Linear) and delta.linear_attn.out_proj.act_scale is not None
  assert attn.self_attn.o_proj.act_scale is None                 # folded into the v rows instead
  assert isinstance(attn.mlp.gate_up, kev_patch.W8A8Projection) and report["smoothed_layers"] == 2


def test_smoothed_patch_leaves_kept_layers_untouched():
  model, st = _bf16_lm(), smooth.ActStats()
  norm0 = model.lm.layers[1].input_norm_weight.clone()
  for site, k in (("in", H), ("out", DV * 4), ("mlp", H), ("down", INTER)):
    st.observe(f"0.{site}", torch.rand(1, k, generator=_g(70)) * 20 + 0.1)
  report = kev_smooth.patch_smooth(model, st, alpha=0.5, backend="reference", keep_layers=(1,))
  attn = model.lm.layers[1]
  assert isinstance(attn.self_attn.o_proj, nn.Linear) and isinstance(attn.mlp.gate_up, _Concat)
  assert torch.equal(attn.input_norm_weight, norm0) and report["layers_kept"] == [1]


def test_keep_layers_env_parses_a_comma_list(monkeypatch):
  monkeypatch.setenv("W8A8_KEEP_LAYERS", "3, 7,11")
  assert kev_smooth.keep_layers_from_env() == (3, 7, 11)
  monkeypatch.delenv("W8A8_KEEP_LAYERS")
  assert kev_smooth.keep_layers_from_env() == ()


def test_keep_env_names_projections_and_refuses_unknown_ones(monkeypatch):
  monkeypatch.setenv("W8A8_KEEP", "down_proj, o_proj")
  assert kev_smooth.keep_from_env() == ("down_proj", "o_proj")
  monkeypatch.setenv("W8A8_KEEP", "lm_head")
  with pytest.raises(ValueError):
    kev_smooth.keep_from_env()


def test_smoothed_patch_with_kept_outputs_smooths_only_the_converted_sites():
  model, st = _bf16_lm(), smooth.ActStats()
  for i in range(2):
    for site, k in (("in", H), ("out", NQ * HD), ("mlp", H), ("down", INTER)):
      st.observe(f"{i}.{site}", torch.rand(1, k, generator=_g(80 + i)) * 20 + 0.1)
  delta, attn = model.lm.layers
  down_w, o_w = delta.mlp.down_proj.weight.clone(), attn.self_attn.o_proj.weight.clone()
  kev_smooth.patch_smooth(model, st, alpha=0.8, backend="reference", keep=("down_proj", "out_proj", "o_proj"))
  assert torch.equal(delta.mlp.down_proj.weight, down_w) and torch.equal(attn.self_attn.o_proj.weight, o_w)
  assert isinstance(delta.mlp.gate_up, kev_patch.W8A8Projection) and isinstance(delta.linear_attn.out_proj, nn.Linear)
