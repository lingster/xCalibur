"""The reference every design is timed and checked against: stock, as production runs it.

  hub    the kernels-community/finegrained-fp8 Triton kernel, loaded exactly as production loads it
         (transformers.integrations.finegrained_fp8.load_finegrained_fp8_kernel -> matmul_2d), or pinned to a hub
         revision with the ``kernels`` package. Needs sm_89 (fp8e4nv) and the network or a warm hub cache.
  twin   our kernel at the hub's tiling (fused quant, its BLOCK_M rule, 128 x 128 tiles, GROUP_M 8), autotuned over the
         hub's own 12 (num_warps, num_stages) configs per (N, K, BLOCK_M), as the hub does. Runs anywhere (with a
         stand-in dot below sm_89); on Ada, parity.py checks it bitwise against the hub.

Both have the stock contract ``fn(x bf16 [M, K], w fp8 [N, K], ws fp32 [N/128, K/128]) -> y bf16 [M, N]``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable

import torch

import kernel
import space

HUB_REPO = "kernels-community/finegrained-fp8"
HUB_REVISION = "8c178950e8710e26a2210e4e909cd02dc16c8715"        # production's revision (kernel repo type)
# sha256 (base64, as the hub's metadata.json lists them) of build/torch-cuda/{matmul,utils}.py at HUB_REVISION. The v4
# branch head that transformers loads on Ada (03ae3193..., kernel commit db1270d, seen 2026-11-05) has byte-identical
# matmul.py and utils.py: only the packaging (builder version, _ops.py, batched/grouped) differs.
HUB_SOURCE_DIGESTS = {"matmul.py": "D2aLXZ4OfbBnXqRRgFK5Mggw9k8kaPLsvK59CHYzaAc=",
                      "utils.py": "LLh/mKs4XkHJX98eG+x3eiXI4LsAOjV6csB6U8K3tdA="}


def hub_source_matches(module_file: str | None) -> bool | None:
  """Whether the loaded hub kernel's matmul.py and utils.py are the sources hub_ref.py copies (None: no file)."""
  import base64                                                                # noqa: PLC0415
  import hashlib                                                               # noqa: PLC0415
  from pathlib import Path                                                     # noqa: PLC0415
  if not module_file or not Path(module_file).exists():
    return None
  d = Path(module_file).parent

  def digest(f: str) -> str:
    return base64.b64encode(hashlib.sha256((d / f).read_bytes()).digest()).decode()
  return all((d / f).exists() and digest(f) == h for f, h in HUB_SOURCE_DIGESTS.items())


@dataclass
class Reference:
  name: str
  fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
  provenance: dict = field(default_factory=dict)
  choice: Callable[[int, int, int], dict | None] = lambda N, K, M: None      # the autotuner's pick, when known


def load_hub(revision: str | None = None) -> Reference:
  """The production FP8 GEMM. revision=None: transformers' loader (what production runs today, whatever revision
  its hub cache resolves); a revision string: ``kernels.get_kernel`` pinned to it."""
  if revision:
    from kernels import get_kernel                                              # noqa: PLC0415
    matmul = get_kernel(HUB_REPO, revision=revision).matmul_2d
  else:
    from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel   # noqa: PLC0415
    matmul = load_finegrained_fp8_kernel().matmul
  mod = sys.modules[matmul.__module__]
  autotuner = getattr(mod, "w8a8_block_dynamic_fp8_matmul_kernel", None)

  def choice(N: int, K: int, M: int) -> dict | None:
    for k, cfg in (getattr(autotuner, "cache", None) or {}).items():
      if tuple(k[:3]) == (N, K, space.hub_block_m(M)):
        return {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
    return None

  prov = {"loader": "kernels.get_kernel" if revision else "transformers", "revision": revision,
          "module_file": getattr(mod, "__file__", None),
          "source_matches_hub_ref": hub_source_matches(getattr(mod, "__file__", None))}
  if prov["source_matches_hub_ref"] is False:
    print(f"[stock] WARNING: the loaded hub kernel's source differs from {HUB_REVISION[:8]} (hub_ref.py)", file=sys.stderr)
  return Reference("hub", lambda x, w, ws: matmul(x, w, ws, [128, 128], x.dtype), prov, choice)


class Twin:
  """Stock tiling on our kernel, with the hub's autotune grid and key; failures (smem) are skipped like Triton's."""

  def __init__(self, dot: str | None = None):
    self.dot, self.picks = dot, {}

  def _time(self, fn) -> float:
    import triton                                                               # noqa: PLC0415
    return triton.testing.do_bench(fn, warmup=5, rep=25)

  def config(self, x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> space.TileConfig:
    M = x.numel() // x.shape[-1]
    key = (w.shape[0], w.shape[1], space.hub_block_m(M))
    if key not in self.picks:
      best = None
      for warps, stages in space.STOCK_GRID:
        cfg = space.to_config(space.STOCK, M, warps, stages)
        try:
          t = self._time(lambda c=cfg: kernel.block_fp8_matmul(x, w, ws, c, dot=self.dot))
        except Exception:                                                       # noqa: BLE001  (out of resources)
          continue
        if best is None or t < best[0]:
          best = (t, cfg)
      self.picks[key] = best[1]
    return self.picks[key]

  def __call__(self, x, w, ws):
    return kernel.block_fp8_matmul(x, w, ws, self.config(x, w, ws), dot=self.dot)

  def choice(self, N: int, K: int, M: int) -> dict | None:
    c = self.picks.get((N, K, space.hub_block_m(M)))
    return {"num_warps": c.num_warps, "num_stages": c.num_stages} if c else None


def load_twin(dot: str | None = None) -> Reference:
  t = Twin(dot)
  return Reference("twin", t, {"kernel_sha": kernel.KERNEL_SHA, "dot": dot}, t.choice)


def load_reference(kind: str = "auto", dot: str | None = None, revision: str | None = None) -> Reference:
  """kind "auto": the hub on sm_89+ (falling back to the twin if it cannot load), the twin elsewhere."""
  fp8 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
  if kind == "hub" or (kind == "auto" and fp8 and dot in (None, "fp8")):
    try:
      return load_hub(revision)
    except Exception as e:                                                      # noqa: BLE001
      if kind == "hub":
        raise
      print(f"[stock] hub kernel unavailable ({type(e).__name__}: {e}); using the twin", file=sys.stderr)
  return load_twin(dot)
