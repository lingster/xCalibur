"""Where each projection's input lives in a fused Kev decoder layer, and taps that record those inputs.

Four sites per layer (the inputs of the GEMMs kev_patch converts):

| site | DeltaNet layer | attention layer | produced by |
|---|---|---|---|
| in | linear_attn.in_proj | self_attn.qkv | the input RMSNorm (layer.input_norm_weight) |
| out | linear_attn.out_proj | self_attn.o_proj | DeltaNet: the gated RMSNorm; attention: sigmoid(gate) * softmax(qk) v |
| mlp | mlp.gate_up | mlp.gate_up | the post-mixer RMSNorm (layer.post_norm_weight) |
| down | mlp.down_proj | mlp.down_proj | SiLU(gate) * up |

``taps(layers, stats)`` records each site's per-channel absmax into an smooth.ActStats (calibration), and optionally a
row sample of each site's first long input (diagnosis), then puts the layers back as they were. A site that never sees a
long input (Kev's last layer runs its o_proj and MLP on the question rows only) is sampled from its largest one.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
from torch import nn

from .smooth import ActStats

SITES = ("in", "out", "mlp", "down")


def mixer_of(layer: Any) -> tuple[Any, str, str]:
  """-> (mixer, name of its input projection group, name of its output projection)."""
  if layer.block_type == "linear_attention":
    return layer.linear_attn, "in_proj", "out_proj"
  return layer.self_attn, "qkv", "o_proj"


def site_owner(layer: Any, site: str) -> tuple[Any, str]:
  """-> (object, attribute) holding the projection that reads `site`'s activation."""
  mixer, group, out = mixer_of(layer)
  return {"in": (mixer, group), "out": (mixer, out), "mlp": (layer.mlp, "gate_up"), "down": (layer.mlp, "down_proj")}[site]


class _Tap:
  """A projection callable that first shows its input to `see`."""

  def __init__(self, inner: Callable, see: Callable[[torch.Tensor], None]):
    self.inner, self.see = inner, see

  def __call__(self, x: torch.Tensor):
    self.see(x)
    return self.inner(x)


def _observer(key: str, stats: ActStats, samples: dict | None, sample_rows: int, min_rows: int) -> Callable:
  seen = 0                                     # rows of the input the current sample came from

  def see(x: torch.Tensor) -> None:
    nonlocal seen
    stats.observe(key, x)
    x2 = x.detach().reshape(-1, x.shape[-1])
    n = x2.shape[0]
    if samples is not None and seen < min_rows and n > seen:
      rows = torch.linspace(0, n - 1, min(sample_rows, n), device=x2.device).round().long()
      samples[key], seen = x2[rows].to("cpu", copy=True), n
  return see


@contextlib.contextmanager
def taps(layers: Sequence[Any], stats: ActStats, samples: dict | None = None, sample_rows: int = 1024,
         min_rows: int = 1024, sites: Sequence[str] = SITES) -> Iterator[ActStats]:
  """Record every site's input of `layers` (key "<index>.<site>") while the block runs. samples: if a dict, it gets an
  evenly spaced sample of up to `sample_rows` rows (on the CPU) of each site's first input with >= min_rows rows, or
  of its largest input if none is that long."""
  undo: list[Callable[[], None]] = []
  try:
    for i, layer in enumerate(layers):
      for site in sites:
        owner, attr = site_owner(layer, site)
        target = getattr(owner, attr)
        see = _observer(f"{i}.{site}", stats, samples, sample_rows, min_rows)
        if isinstance(target, nn.Module):
          handle = target.register_forward_pre_hook(lambda _m, args, see=see: see(args[0]))
          undo.append(handle.remove)
        else:
          setattr(owner, attr, _Tap(target, see))
          undo.append(lambda owner=owner, attr=attr, target=target: setattr(owner, attr, target))
    yield stats
  finally:
    for u in reversed(undo):
      u()
