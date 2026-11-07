"""Swap a fused Kev backbone's projection GEMMs for W8A8, as a kev.perf variant patch.

    python -m kev.perf.cli compare ... --reference fused --candidate fused --cand-set patch=w8a8_sm86.kev_patch.patch
    (with <xCalibur>/kernels on PYTHONPATH)

After kev.fused_qwen35.fuse each decoder layer holds, per group of projections that read one input, a
kev.fused_proj.ConcatProjection (``.weight`` = the concatenated weights, ``.splits`` = their output sizes), and plain
nn.Linear for the rest:

| layer | group (splits) | becomes |
|---|---|---|
| DeltaNet (24) | in_proj = qkv 8192, z 4096, b 32, a 32 | W8A8 over qkv+z (N = 12,288); b and a stay one bf16 GEMM (N = 64) |
| DeltaNet | out_proj (K = 4096) | W8A8Linear |
| attention (8) | qkv = q+gate 8192, k 1024, v 1024 | W8A8 (N = 10,240) |
| attention | o_proj (K = 4096) | W8A8Linear |
| all (32) | mlp.gate_up = 12288 + 12288 | W8A8 (N = 24,576) |
| all | mlp.down_proj (K = 12,288) | W8A8, its input quantised inside the SwiGLU (quant.swiglu_quant) |

Parts with fewer than ``min_out`` outputs stay bf16 (the DeltaNet's b and a feed exp/softplus gates and cost 0.1% of
the FLOPs). The bf16 weights are dropped as each group is converted, so the swap lowers peak memory rather than raising
it (Kev-9B: 13.8 -> 6.9 GB of projection weights; 20.99 -> 17.67 GB peak per 29k-token request,
run 20261105T122956Z-169e7f). The embedding, norms, conv, DeltaNet recurrence, attention and the pointer head are
untouched.
"""
from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from . import grouped, quant
from .linear import Backend, W8A8Linear, W8A8Weight, gemm_quantised, w8a8_linear

MIN_OUT = 256


def quantised_prefix(splits: Sequence[int], min_out: int = MIN_OUT) -> int:
  """How many leading parts to quantise: those with >= min_out outputs. Small parts must trail the large ones (they
  do in every Kev group), so one W8A8 GEMM and at most one bf16 GEMM cover the group."""
  n = 0
  while n < len(splits) and splits[n] >= min_out:
    n += 1
  if any(s >= min_out for s in splits[n:]):
    raise ValueError(f"a part with < {min_out} outputs precedes a larger one in {list(splits)}; reorder or keep bf16")
  return n


class W8A8Projection:
  """A ConcatProjection with its large leading parts in W8A8 and its small trailing ones (if any) in bf16."""

  def __init__(self, weight: torch.Tensor, splits: Sequence[int], backend: Backend = "auto", min_out: int = MIN_OUT):
    n = quantised_prefix(splits, min_out)
    rows = sum(splits[:n])
    self.w = W8A8Weight.from_float(weight[:rows])
    self.tail: torch.Tensor | None = weight[rows:].contiguous() if rows < weight.shape[0] else None
    self.head_splits, self.tail_splits = list(splits[:n]), list(splits[n:])
    self.backend: Backend = backend

  def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    out = w8a8_linear(x, self.w, self.backend).split(self.head_splits, -1)
    if self.tail is None:
      return tuple(out)
    return tuple(out) + tuple(F.linear(x, self.tail).split(self.tail_splits, -1))


def _swiglu_quant(gate: torch.Tensor, up: torch.Tensor, backend: Backend,
                  group: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
  if backend == "reference":
    h = F.silu(gate.float()) * up.float()
    return quant.quant_rows_reference(h) if group is None else grouped.quant_groups_reference(h, group)
  return quant.swiglu_quant(gate, up) if group is None else grouped.swiglu_quant_groups(gate, up, group)


def mlp_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
  """The fused MLP with the down projection's quantiser inside the SwiGLU: gate|up (W8A8) -> int8 h -> down (W8A8)."""
  gate, up = self.gate_up(x)
  lead, k = gate.shape[:-1], gate.shape[-1]
  down: W8A8Linear = self.down_proj
  hq, hs = _swiglu_quant(gate.reshape(-1, k), up.reshape(-1, k), down.backend, down.group)
  return gemm_quantised(hq, hs, down.weight8, down.backend, down.group).reshape(*lead, down.out_features)


def _project(group: Any, backend: Backend, min_out: int) -> W8A8Projection:
  if not (hasattr(group, "weight") and hasattr(group, "splits")):
    raise TypeError(f"expected a kev.fused_proj.ConcatProjection, got {type(group).__name__}: load with fused=True and "
                    "KEV_FUSED_CONCAT unset/1 (SeparateProjection keeps the bf16 modules registered)")
  return W8A8Projection(group.weight, group.splits, backend, min_out)


def _linear(module: nn.Module, backend: Backend, act_scale: torch.Tensor | None = None,
            group: int | None = None) -> W8A8Linear:
  if not isinstance(module, nn.Linear):
    raise TypeError(f"expected a plain nn.Linear, got {type(module).__name__}")
  if module.bias is not None:
    raise ValueError("W8A8Linear covers bias-free linears only")
  return W8A8Linear(W8A8Weight.from_float(module.weight.detach()), backend, act_scale, group)


NAMES = ("in_proj", "out_proj", "qkv", "o_proj", "gate_up", "down_proj")


def base_lm(model: Any) -> Any:
  """The decoder backbone (``.layers``) of a DecisionModel, unwrapping a peft model if there is one."""
  return model.lm.get_base_model() if hasattr(model.lm, "get_base_model") else model.lm


@torch.no_grad()
def patch_layer(layer: Any, backend: Backend = "auto", min_out: int = MIN_OUT, keep: Sequence[str] = (),
                out_act_scale: torch.Tensor | None = None, groups: Mapping[str, int] | None = None) -> tuple[int, int]:
  """Convert one fused decoder layer in place -> (groups, linears) converted. out_act_scale: an fp32 [K] the mixer's
  output projection multiplies into its input inside the quantiser (kev_smooth: SmoothQuant's 1/s where no fold exists).
  groups: {"down_proj" / "out_proj" / "o_proj": G} for per-(token, G channels) activation scales (grouped.py)."""
  groups = groups or {}
  if layer.block_type == "linear_attention":
    mixer, group_attr, out_attr = layer.linear_attn, "in_proj", "out_proj"
  else:
    mixer, group_attr, out_attr = layer.self_attn, "qkv", "o_proj"
  mlp, n_groups, linears = layer.mlp, 0, 0
  for owner, attr in ((mixer, group_attr), (mlp, "gate_up")):
    if attr not in keep:
      setattr(owner, attr, _project(getattr(owner, attr), backend, min_out))
      n_groups += 1
  for owner, attr, act in ((mixer, out_attr, out_act_scale), (mlp, "down_proj", None)):
    if attr not in keep:
      setattr(owner, attr, _linear(getattr(owner, attr), backend, act, groups.get(attr)))
      linears += 1
  if "down_proj" not in keep:               # else kev's own MLP forward stays: fla SwiGLU, then the bf16 down_proj
    mlp.forward = types.MethodType(mlp_forward, mlp)
  return n_groups, linears


def check_keep(keep: Sequence[str]) -> None:
  if unknown := set(keep) - set(NAMES):
    raise ValueError(f"unknown projection(s) {sorted(unknown)}; choose from {NAMES}")


@torch.no_grad()
def patch(model: Any, backend: Backend = "auto", min_out: int = MIN_OUT, keep: Sequence[str] = (),
          keep_layers: Sequence[int] = (), groups: Mapping[str, int] | None = None) -> dict[str, int]:
  """Convert a fused Kev model (DecisionModel, or anything with ``.lm.layers``) in place; returns counts. ``keep``:
  projection names (NAMES) left in bf16 in every layer; ``keep_layers``: layer indices left wholly in bf16 (ablations)."""
  check_keep(keep)
  lm = base_lm(model)
  n_groups = linears = 0
  for i, layer in enumerate(lm.layers):
    if i not in keep_layers:
      g, n = patch_layer(layer, backend, min_out, keep, groups=groups)
      n_groups, linears = n_groups + g, linears + n
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  logger.info("W8A8 ({}): {} projection groups and {} linears in int8 over {} layers{}{}", backend, n_groups, linears,
              len(lm.layers), f", kept bf16: {list(keep)}" if keep else "",
              f", layers kept bf16: {list(keep_layers)}" if keep_layers else "")
  return {"groups": n_groups, "linears": linears, "layers": len(lm.layers)}


def patch_triton(model: Any) -> dict[str, int]:
  """The same swap with the Triton fused-epilogue GEMM at every M (no int32 buffer: the lowest-memory form)."""
  return patch(model, backend="triton")


def patch_keep_down(model: Any) -> dict[str, int]:
  """Ablation: everything in W8A8 except the MLP down projection (its post-SwiGLU input has the largest outliers)."""
  return patch(model, keep=("down_proj",))
