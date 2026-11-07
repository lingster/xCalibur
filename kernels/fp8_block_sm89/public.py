"""Public tuned configs for sm_89 as seeds: vLLM's block-FP8 tables for the NVIDIA L40S (AD102, the RTX 6000 Ada die).

vLLM ships per-(N, K, device) JSON for its w8a8 block-FP8 Triton GEMM, keyed by M. Its contract is ours with
``quant="once"``: activations quantised per token-group(128) by a separate kernel, then fp8 x fp8 with 128 x 128
block weight scales. Four L40S tables exist (public/vllm_l40s_block_fp8.json, copied at a pinned commit), one of
them for our exact in_proj_qkv shape (N=10240, K=5120). Neither project has RTX 6000 Ada tables. SGLang (commit
874fc07d, python/sglang/kernels/ops/quantization/configs) has sm_89 tables for NVIDIA_L40 (N=2048,K=4096 and
N=5120,K=2048) and NVIDIA_L20Y (DeepSeek shapes); they are not vendored because their K is far from ours, while
vLLM's L40S set covers our exact shape and K up to 25,600. Seeds only nominate: they are measured like any design.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import space

_FILE = Path(__file__).parent / "public" / "vllm_l40s_block_fp8.json"


@functools.lru_cache(maxsize=1)
def _load() -> dict:
  return json.loads(_FILE.read_text())


def provenance() -> dict:
  return {k: v for k, v in _load().items() if k != "tables"}


def _shape(key: str) -> tuple[int, int]:
  n, k = key.split(",")
  return int(n.split("=")[1]), int(k.split("=")[1])


def _pick(table: dict, M: int) -> dict:
  return table[min(table, key=lambda k: abs(int(k) - M))]                # vLLM's own rule: the nearest M key


def _design(c: dict) -> dict:
  return {"quant": "once", "BLOCK_M": c["BLOCK_SIZE_M"], "BLOCK_N": c["BLOCK_SIZE_N"], "BLOCK_K": c["BLOCK_SIZE_K"],
          "GROUP_M": c["GROUP_SIZE_M"], "SPLIT_K": 1, "num_warps": c["num_warps"], "num_stages": c["num_stages"]}


def seeds(N: int, K: int, M: int) -> list[dict]:
  """Legal designs from vLLM's L40S tables at size M, the table nearest in (log N, log K) first."""
  tables = _load()["tables"]
  dist = lambda key: math.hypot(*(math.log(a / b) for a, b in zip(_shape(key), (N, K))))          # noqa: E731
  out: list[dict] = []
  for key in sorted(tables, key=dist):
    d = _design(_pick(tables[key], M))
    if space.why_illegal(d) is None and d not in out:
      out.append(d)
  return out
