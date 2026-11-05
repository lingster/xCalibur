"""Compile the real FP8 kernel for sm_89 on any machine (no Ada needed): the exact half of the search's compile step.

Triton cross-compiles to an sm_89 cubin (``GPUTarget("cuda", 89, 32)``) with its bundled ptxas, and its bundled
cuobjdump reads the resources back. A design that fails here (a Triton error, a ptxas error, or more than 99 KB of
shared memory) never reaches the GPU; one that compiles brings exact features for the ranker: registers, local
memory (spills), shared memory, the e4m3 mma.sync count in the PTX, and the SASS inner-loop mix (sass.py).

The specialisation mirrors what Triton does at launch for contiguous operands: unit strides become constants, and
pointers, N, K and the leading strides are known to be 16-divisible (true for every Kev-27B shape).

  python aot.py --configs '[{"quant": "once", "BLOCK_M": 128, ...}]'      # JSON records, one per config
  python aot.py --sample 200                                               # compile a random sample of the space
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import tempfile

import triton
from triton.backends.compiler import GPUTarget

import hub_ref
import kernel
import sass
from space import QGROUP, SMEM_LIMIT_SM89, TileConfig

SM89 = GPUTarget("cuda", 89, 32)
_MMA = "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"


def _ast(fn, ptr: dict, cons: dict, div16: tuple) -> triton.compiler.ASTSource:
  names = fn.arg_names
  sig = {n: ptr.get(n, "constexpr" if n in cons else "i32") for n in names}
  attrs = {(names.index(n),): [["tt.divisibility", 16]] for n in div16}
  return triton.compiler.ASTSource(fn=fn, signature=sig, constexprs=cons, attrs=attrs)


def _ours(cfg: TileConfig) -> triton.compiler.ASTSource:
  ptr = {"A": "*bf16" if cfg.quant == "fused" else "*fp8e4nv", "As": "*fp32", "B": "*fp8e4nv", "Bs": "*fp32",
         "C": "*fp32" if cfg.SPLIT_K > 1 else "*bf16"}
  cons = {"stride_ak": 1, "stride_bk": 1, "stride_cn": 1, "BLOCK_M": cfg.BLOCK_M, "BLOCK_N": cfg.BLOCK_N,
          "BLOCK_K": cfg.BLOCK_K, "GROUP_M": cfg.GROUP_M, "SPLIT_K": cfg.SPLIT_K, "FUSED": cfg.quant == "fused",
          "DOT": kernel.DOTS["fp8"], "QG": QGROUP}
  return _ast(kernel._gemm_kernel, ptr, cons,
              ("A", "As", "B", "Bs", "C", "N", "K", "stride_am", "stride_bn", "stride_cm", "stride_cs"))


def _hub(block_m: int) -> triton.compiler.ASTSource:
  """The stock kernel as the hub launches it: bf16 A, fp32 block scales, 128 x 128 tiles, GROUP_SIZE_M 8."""
  ptr = {"A": "*bf16", "B": "*fp8e4nv", "C": "*bf16", "Bs": "*fp32"}
  cons = {"stride_ak": 1, "stride_bk": 1, "stride_cn": 1, "stride_bs_k": 1, "BLOCK_SIZE_M": block_m,
          "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}
  return _ast(hub_ref.w8a8_block_dynamic_fp8_matmul_kernel, ptr, cons,
              ("A", "B", "C", "Bs", "N", "K", "stride_am", "stride_bn", "stride_cm"))


def _compile(src, num_warps: int, num_stages: int):
  return triton.compile(src, target=SM89, options={"num_warps": num_warps, "num_stages": num_stages})


def resources(cubin: bytes) -> dict:
  """{"n_regs", "n_spills" (local bytes per thread), "stack"} from ``cuobjdump -res-usage``."""
  with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
    f.write(cubin)
    f.flush()
    out = subprocess.run([sass.cuobjdump(), "-res-usage", f.name], capture_output=True, text=True, check=True).stdout
  get = lambda k: int(re.search(rf"\b{k}:(\d+)", out).group(1))                       # noqa: E731
  return {"n_regs": get("REG"), "n_spills": get("LOCAL"), "stack": get("STACK")}


def record(ck, limit: int = SMEM_LIMIT_SM89) -> dict:
  m = sass.mix(sass.disassemble(ck.asm["cubin"]))
  rec = {"ok": True, "shared": int(ck.metadata.shared), "mma": ck.asm["ptx"].count(_MMA), **resources(ck.asm["cubin"]),
         "loop_instructions": m["loop_instructions"], "loop_mma": sum(m["loop_mma"].values()),
         "loop_convert": m["loop_mix"].get("convert", 0), "native_fp8_mma": m["native_fp8_mma"]}
  if rec["shared"] > limit:
    rec.update(ok=False, error=f"shared {rec['shared']} > {limit}")
  return rec


@functools.lru_cache(maxsize=4096)
def compile_sm89(cfg: TileConfig, limit: int = SMEM_LIMIT_SM89) -> dict:
  """Compile one of our designs for sm_89: {"ok", "error"?, "n_regs", "n_spills", "shared", "mma", loop SASS mix}."""
  try:
    ck = _compile(_ours(cfg), cfg.num_warps, cfg.num_stages)
  except Exception as e:                                                               # noqa: BLE001
    return {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[-1][:200] if str(e) else ''}"}
  return record(ck, limit)


def cubin_sm89(cfg: TileConfig) -> bytes:
  return _compile(_ours(cfg), cfg.num_warps, cfg.num_stages).asm["cubin"]


def hub_cubin(block_m: int, num_warps: int, num_stages: int) -> bytes:
  return _compile(_hub(block_m), num_warps, num_stages).asm["cubin"]


def hub_record(block_m: int, num_warps: int, num_stages: int) -> dict:
  try:
    return record(_compile(_hub(block_m), num_warps, num_stages))
  except Exception as e:                                                               # noqa: BLE001
    return {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[-1][:200] if str(e) else ''}"}


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--configs", help="JSON list of TileConfig dicts")
  ap.add_argument("--sample", type=int, default=0, help="also compile this many random legal designs (seed 0)")
  ap.add_argument("--hub-grid", action="store_true", help="the stock kernel at every BLOCK_M x its 12 autotune configs")
  a = ap.parse_args(argv)
  out = [{"config": c, **compile_sm89(TileConfig(**c))} for c in json.loads(a.configs or "[]")]
  if a.sample:
    import random                                                                       # noqa: PLC0415
    import space                                                                        # noqa: PLC0415
    for d in random.Random(0).sample(space.designs(), a.sample):
      c = space.to_config(d, 128)
      out.append({"config": c.as_dict(), **compile_sm89(c)})
  if a.hub_grid:
    import space                                                                        # noqa: PLC0415
    for bm in (16, 32, 64, 128):
      for w, s in space.STOCK_GRID:
        out.append({"hub": {"BLOCK_SIZE_M": bm, "num_warps": w, "num_stages": s}, **hub_record(bm, w, s)})
  print(json.dumps(out))


if __name__ == "__main__":
  main()
