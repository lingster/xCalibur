"""SmoothQuant for Kev's W8A8 patch: fold the smoothing scales into the fused layers, then convert them (kev_patch).

    python -m kev.perf.cli compare ... --candidate fused --cand-set patch=w8a8_sm86.kev_smooth.patch \\
        --cand-set env.W8A8_SMOOTH_ALPHA=0.5 [--cand-set env.W8A8_SMOOTH_STATS=<absmax.pt>] \\
        [--cand-set env.W8A8_KEEP=down_proj,out_proj,o_proj] [--cand-set env.W8A8_KEEP_LAYERS=3,7] \\
        [--cand-set env.W8A8_GROUP=down_proj,out_proj,o_proj] [--cand-set env.W8A8_SMOOTH_SITES=in,mlp]

Per layer and site (record.SITES), with s from smooth.smooth_scales on the calibrated activation absmax, 1/s goes into
the op that produces the activation and s into the GEMM's weight columns:

| site | 1/s goes into | cost |
|---|---|---|
| in (in_proj / qkv) | layer.input_norm_weight (fp32, 1 + w) | none |
| mlp (gate_up) | layer.post_norm_weight (fp32, 1 + w) | none |
| down (down_proj) | the up rows of gate_up: SiLU(g) * (u / s) = (SiLU(g) * u) / s | none (int8 row scales absorb it) |
| out, attention (o_proj) | the v rows of qkv; s tied across the q heads of one kv head | none (as above) |
| out, DeltaNet (out_proj) | the activation quantiser (one multiply per element) | one fp32 multiply |

DeltaNet's gated RMSNorm has one weight per head dim, shared by all 32 heads, so it cannot take a per-channel 1/s; its
1/s is applied in quant.quant_rows instead (W8A8Linear.act_scale). The attention fold is exact because the attention
output is linear in v per channel and the sigmoid gate multiplies element-wise. The folds are computed in fp32 and
stored back in each tensor's dtype (bf16 weights round once more: 2^-9, small next to int8's 2^-7 steps).
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

from . import kev_patch
from .linear import Backend
from .record import SITES, mixer_of
from .smooth import ActStats, column_amax, expand_tied, smooth_scales, tied_scales

STATS_ENV, ALPHA_ENV, SITES_ENV = "W8A8_SMOOTH_STATS", "W8A8_SMOOTH_ALPHA", "W8A8_SMOOTH_SITES"
KEEP_ENV, KEEP_LAYERS_ENV, GROUP_ENV = "W8A8_KEEP", "W8A8_KEEP_LAYERS", "W8A8_GROUP"
GROUPABLE = ("down_proj", "out_proj", "o_proj")      # W8A8Linear; the concatenated groups stay per token
GROUP = 128
DEFAULT_STATS = Path.home() / ".cache" / "xcalibur" / "w8a8_sm86_kev9b_absmax.pt"   # written by diagnose.py
SITE_OF = {"in_proj": "in", "qkv": "in", "out_proj": "out", "o_proj": "out", "gate_up": "mlp", "down_proj": "down"}


def _set(t: torch.Tensor, value: torch.Tensor) -> None:
  t.copy_(value.to(t.dtype))


def _weight(p: Any) -> torch.Tensor:
  return p.weight.data if isinstance(p, nn.Module) else p.weight


@torch.no_grad()
def scale_columns(w: torch.Tensor, s: torch.Tensor) -> None:
  """W[:, j] *= s_j in place (fp32 math)."""
  _set(w, w.float() * s.to(w.device)[None, :])


@torch.no_grad()
def scale_rows(w: torch.Tensor, s: torch.Tensor) -> None:
  """W[n, :] /= s_n in place (fp32 math): the producing GEMM's output channel n comes out divided by s_n."""
  _set(w, w.float() / s.to(w.device)[:, None])


def site_weight(layer: Any, site: str) -> torch.Tensor:
  """The weight [N, K] of the GEMM that reads `site` (the whole group for in / mlp, small parts included)."""
  mixer, group_attr, out_attr = mixer_of(layer)
  return {"in": lambda: getattr(mixer, group_attr).weight, "mlp": lambda: layer.mlp.gate_up.weight,
          "down": lambda: _weight(layer.mlp.down_proj), "out": lambda: _weight(getattr(mixer, out_attr))}[site]()


def gqa_shape(layer: Any) -> tuple[int, int] | None:
  """(q heads per kv head, head dim) of an attention layer's o_proj input; None for a DeltaNet layer."""
  if layer.block_type == "linear_attention":
    return None
  mixer = layer.self_attn
  v = mixer.qkv.splits[2]
  return site_weight(layer, "out").shape[1] // v, mixer.head_dim


def input_scales(layer: Any, index: int, site: str, stats: ActStats, alpha: float) -> torch.Tensor:
  """SmoothQuant's s [K] for the GEMM reading `site` of layer `index`, on the weight's device: smooth_scales of the
  calibrated absmax and the weight's column absmax; for attention's o_proj tied across each kv head's q heads."""
  w, act = site_weight(layer, site), stats.get(index, site)
  if act is None:
    raise KeyError(f"no calibration statistics for layer {index} site {site!r}")
  if act.shape[0] != w.shape[1]:
    raise ValueError(f"layer {index} site {site!r}: statistics for {act.shape[0]} channels, weight has {w.shape[1]}")
  act, gqa = act.to(w.device), gqa_shape(layer) if site == "out" else None
  if gqa is None:
    return smooth_scales(act, column_amax(w), alpha)
  return expand_tied(tied_scales(act, column_amax(w), alpha, *gqa), *gqa)


@torch.no_grad()
def fold_layer(layer: Any, index: int, stats: ActStats, alpha: float, sites: Sequence[str] = SITES) -> dict[str, torch.Tensor]:
  """Smooth one fused decoder layer in place (still in float); returns the scales that could not be folded and must be
  applied in the quantiser: {"out": 1/s} for a DeltaNet layer's out_proj, else {}. All scales come from the weights as
  they were before this call."""
  scales = {site: input_scales(layer, index, site, stats, alpha) for site in sites}
  mixer, group_attr, _ = mixer_of(layer)
  group, gate_up, unfolded = getattr(mixer, group_attr), layer.mlp.gate_up, {}
  if "in" in scales:
    layer.input_norm_weight.div_(scales["in"].to(layer.input_norm_weight.device))
  if "mlp" in scales:
    layer.post_norm_weight.div_(scales["mlp"].to(layer.post_norm_weight.device))
  if "down" in scales:
    scale_rows(gate_up.weight[gate_up.splits[0]:], scales["down"])
  if "out" in scales:
    if (gqa := gqa_shape(layer)) is None:
      unfolded["out"] = 1.0 / scales["out"]
    else:
      groups, head_dim = gqa
      qg, k, v = group.splits
      s_kv = scales["out"].view(-1, groups, head_dim)[:, 0].flatten()
      scale_rows(group.weight[qg + k:qg + k + v], s_kv)
  for site, s in scales.items():
    scale_columns(site_weight(layer, site), s)
  return unfolded


@torch.no_grad()
def patch_smooth(model: Any, stats: ActStats, alpha: float, backend: Backend = "auto", min_out: int = kev_patch.MIN_OUT,
                 keep: Sequence[str] = (), keep_layers: Sequence[int] = (), groups: Mapping[str, int] | None = None,
                 sites: Sequence[str] = SITES) -> dict[str, Any]:
  """Smooth (fold_layer) then convert (kev_patch.patch_layer) every layer of a fused Kev model, one layer at a time.
  keep: projection names left bf16 (their sites are not smoothed); keep_layers: layers left wholly bf16 and unsmoothed;
  groups: per-(token, group) activation scales for these output projections (kev_patch.patch_layer); sites: the
  record.SITES to smooth (default all)."""
  kev_patch.check_keep(keep)
  lm = kev_patch.base_lm(model)
  sites = [site for site in sites if not any(SITE_OF[name] == site for name in keep)]
  smoothed = 0
  for i, layer in enumerate(lm.layers):
    if i in keep_layers:
      continue
    unfolded = fold_layer(layer, i, stats, alpha, sites)
    kev_patch.patch_layer(layer, backend, min_out, keep, out_act_scale=unfolded.get("out"), groups=groups)
    smoothed += 1
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  logger.info("W8A8 + SmoothQuant (alpha {} on {}, {}): {} of {} layers smoothed and in int8{}{}{}", alpha, list(sites),
              backend, smoothed, len(lm.layers), f", kept bf16: {list(keep)}" if keep else "",
              f", per-{GROUP} activation scales: {sorted(groups)}" if groups else "",
              f", layers kept bf16: {sorted(keep_layers)}" if keep_layers else "")
  return {"smoothed_layers": smoothed, "layers": len(lm.layers), "layers_kept": sorted(keep_layers), "alpha": alpha}


def _env_list(name: str) -> list[str]:
  return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def keep_layers_from_env() -> tuple[int, ...]:
  """W8A8_KEEP_LAYERS="3,7,11" -> (3, 7, 11); unset or empty -> ()."""
  return tuple(int(v) for v in _env_list(KEEP_LAYERS_ENV))


def keep_from_env() -> tuple[str, ...]:
  """W8A8_KEEP="down_proj,out_proj,o_proj" -> projection names (kev_patch.NAMES) kept bf16 in every layer."""
  keep = tuple(_env_list(KEEP_ENV))
  kev_patch.check_keep(keep)
  return keep


def groups_from_env() -> dict[str, int]:
  """W8A8_GROUP="down_proj,out_proj,o_proj" -> {name: 128}: those projections get per-(token, 128) activation scales."""
  names = _env_list(GROUP_ENV)
  if bad := set(names) - set(GROUPABLE):
    raise ValueError(f"{GROUP_ENV}: {sorted(bad)} cannot take grouped scales; choose from {GROUPABLE}")
  return {name: GROUP for name in names}


def sites_from_env() -> tuple[str, ...]:
  """W8A8_SMOOTH_SITES="in,mlp" -> the sites to smooth; unset -> all (record.SITES)."""
  sites = tuple(_env_list(SITES_ENV)) or SITES
  if bad := set(sites) - set(SITES):
    raise ValueError(f"{SITES_ENV}: unknown site(s) {sorted(bad)}; choose from {SITES}")
  return sites


def patch(model: Any) -> dict[str, Any]:
  """The kev.perf entry point: SmoothQuant + W8A8 with alpha, statistics and kept layers from the environment, which
  kev.perf records with the variant (W8A8_SMOOTH_ALPHA, default 0.5; W8A8_SMOOTH_STATS, default DEFAULT_STATS;
  W8A8_SMOOTH_SITES: sites to smooth, default all; W8A8_KEEP: projection names kept bf16, default none;
  W8A8_KEEP_LAYERS: layers kept bf16, default none; W8A8_GROUP: output projections with per-128 activation scales)."""
  path = Path(os.environ.get(STATS_ENV) or DEFAULT_STATS)
  stats = ActStats.load(path)
  alpha = float(os.environ.get(ALPHA_ENV) or 0.5)
  logger.info("SmoothQuant statistics {} ({})", path, stats.meta.get("requests", "?"))
  return patch_smooth(model, stats, alpha, keep=keep_from_env(), keep_layers=keep_layers_from_env(),
                      groups=groups_from_env(), sites=sites_from_env())
