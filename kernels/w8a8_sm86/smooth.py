"""SmoothQuant (Xiao et al. 2023) for the W8A8 linear: per-input-channel scales and the calibration statistics.

For $Y = X W^{\\top}$ and a channel scale $s \\in \\mathbb{R}^K_{>0}$, $Y = (X \\operatorname{diag}(s)^{-1})
(W \\operatorname{diag}(s))^{\\top}$ exactly. Choosing

$$ s_j = \\frac{\\max |X_{\\cdot j}|^{\\alpha}}{\\max |W_{\\cdot j}|^{1-\\alpha}} $$

moves the range of an outlier channel of X into W's column j, where the per-output-channel weight scales absorb it
better than the per-token activation scales do. alpha = 0.5 equalises the two ranges per channel; higher alpha pushes
more of it into the weights. max|X_j| comes from calibration requests (ActStats, filled by record.taps); 1/s is then
folded into whatever produces X (kev_smooth.fold_layer), so the smoothing costs nothing at run time.

Grouped-query attention ties the scales of o_proj's input: the q heads that share one kv head read the same v
channels, so a scale folded into v's rows must be equal across them (tied_scales / expand_tied).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

FLOOR = 1e-5          # dead channels (amax 0) get a tiny positive scale, never 0 or inf


def smooth_scales(act_amax: torch.Tensor, w_amax: torch.Tensor, alpha: float, floor: float = FLOOR) -> torch.Tensor:
  """fp32 [K]: s_j = act_amax_j^alpha / w_amax_j^(1 - alpha), both clamped below by `floor`."""
  if not 0.0 <= alpha <= 1.0:
    raise ValueError(f"alpha must be in [0, 1], got {alpha}")
  a = act_amax.float().clamp_min(floor)
  w = w_amax.float().to(a.device).clamp_min(floor)
  return (a.pow(alpha) / w.pow(1.0 - alpha)).clamp_min(floor)


def column_amax(w: torch.Tensor) -> torch.Tensor:
  """max_n |W[n, j]| for a weight [N, K] -> fp32 [K]."""
  return w.abs().amax(0).float()


def tied_scales(act_amax: torch.Tensor, w_amax: torch.Tensor, alpha: float, groups: int, head_dim: int) -> torch.Tensor:
  """Scales for an input laid out as [q heads, head_dim] whose q heads come in `groups` per kv head: the amax of each
  kv head's group, then smooth_scales -> fp32 [kv heads * head_dim]."""
  def reduce(v: torch.Tensor) -> torch.Tensor:
    return v.float().view(-1, groups, head_dim).amax(1).flatten()
  return smooth_scales(reduce(act_amax), reduce(w_amax), alpha)


def expand_tied(s_kv: torch.Tensor, groups: int, head_dim: int) -> torch.Tensor:
  """[kv heads * head_dim] -> [q heads * head_dim], q head h taking kv head h // groups (repeat_interleave order, as
  transformers' repeat_kv and SDPA's enable_gqa)."""
  return s_kv.view(-1, 1, head_dim).expand(-1, groups, head_dim).reshape(-1)


@dataclass
class ActStats:
  """Running per-channel absmax of named activations (key "<layer>.<site>", record.SITES), plus free-form meta."""
  data: dict[str, torch.Tensor] = field(default_factory=dict)
  meta: dict[str, Any] = field(default_factory=dict)

  def observe(self, key: str, x: torch.Tensor) -> None:
    """Fold x [..., K] into the running absmax of `key` (kept in fp32 on x's device)."""
    amax = x.detach().reshape(-1, x.shape[-1]).abs().amax(0).float()
    old = self.data.get(key)
    self.data[key] = amax if old is None else torch.maximum(old, amax)

  def __getitem__(self, key: str) -> torch.Tensor:
    return self.data[key]

  def __contains__(self, key: str) -> bool:
    return key in self.data

  def get(self, layer: int, site: str) -> torch.Tensor | None:
    return self.data.get(f"{layer}.{site}")

  def save(self, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"amax": {k: v.cpu() for k, v in self.data.items()}, "meta": self.meta}, path)

  @classmethod
  def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "ActStats":
    blob = torch.load(path, map_location=device, weights_only=True)
    return cls({k: v.float() for k, v in blob["amax"].items()}, dict(blob.get("meta", {})))
