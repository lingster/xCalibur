"""Read the machine code: does the inner loop issue native e4m3 MMA, or conversions plus FP16/BF16 MMA?

An earlier H200 study (CuTe Sassy) found FP8 ``mma.sync`` on H200 lowered to conversions plus FP16 MMA. PTX can say e4m3 while SASS
does something else, so we read the SASS (``cuobjdump -sass``) of the sm_89 cubin, locate the inner K loop (the
backward branch that encloses the most MMA instructions), and count its instruction mix per K tile.

  python sass.py --hub 128,8,2                          # the stock kernel (hub_ref.py) at BLOCK_M=128, 8 warps, 2 stages
  python sass.py --config '{"quant": "once", ...}'      # one of our builds
  python sass.py --triton-cache ~/.triton/cache         # on Ada: every compiled hub kernel production actually ran
  python sass.py --cubin some.cubin [--dump]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import tempfile

CLASSES = {
  "mma": ("HMMA", "QMMA", "IMMA", "DMMA"),
  "convert": ("F2F", "F2FP", "I2F", "F2I", "I2FP", "F2IP"),
  "permute": ("PRMT",),
  "int": ("SHF", "LOP3", "IADD3", "IMAD", "ISETP", "LEA", "SEL", "IABS", "IMNMX"),
  "ld_shared": ("LDS", "LDSM"),
  "st_shared": ("STS",),
  "ld_global": ("LDG", "LDGSTS"),
  "fp32": ("FFMA", "FMUL", "FADD", "FMNMX", "FSETP", "FSEL"),
  "mufu": ("MUFU",),
  "sync": ("BAR", "DEPBAR", "LDGDEPBAR", "WARPSYNC"),
}
_LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")


def cuobjdump() -> str:
  import triton                                                                 # noqa: PLC0415
  return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")


def disassemble(cubin: bytes) -> str:
  with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
    f.write(cubin)
    f.flush()
    return subprocess.run([cuobjdump(), "-sass", f.name], capture_output=True, text=True, check=True).stdout


def parse(text: str) -> list[tuple[int, str, str]]:
  """[(address, opcode with modifiers, operands)]."""
  return [(int(a, 16), op, rest.strip()) for a, op, rest in _LINE.findall(text)]


def classify(op: str) -> str:
  base = op.split(".")[0]
  return next((c for c, ops in CLASSES.items() if base in ops), "other")


def inner_loop(ins: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
  """The body of the backward branch that encloses the most MMA instructions (the K loop)."""
  best: list = []
  for addr, op, rest in ins:
    m = re.search(r"0x([0-9a-f]+)", rest)
    if op.startswith("BRA") and m and int(m.group(1), 16) < addr:
      body = [i for i in ins if int(m.group(1), 16) <= i[0] <= addr]
      if sum(classify(o) == "mma" for _, o, _ in body) > sum(classify(o) == "mma" for _, o, _ in best):
        best = body
  return best


def mix(text: str) -> dict:
  """Instruction mix of the inner loop (per K tile, as Triton does not unroll the K loop) and of the whole kernel."""
  ins = parse(text)
  loop = inner_loop(ins)
  count = lambda xs: dict(collections.Counter(classify(o) for _, o, _ in xs))                   # noqa: E731
  mma = collections.Counter(o for _, o, _ in loop if classify(o) == "mma")
  cvt = collections.Counter(o for _, o, _ in loop if classify(o) == "convert")
  native = any("E4M3" in o or o.startswith("QMMA") for o in mma)
  return {"loop_instructions": len(loop), "loop_mix": count(loop), "loop_mma": dict(mma), "loop_convert": dict(cvt),
          "native_fp8_mma": native, "kernel_instructions": len(ins), "kernel_mix": count(ins)}


def summary(name: str, m: dict) -> str:
  lm = m["loop_mix"]
  return (f"{name}: K-loop {m['loop_instructions']} instr | mma {lm.get('mma', 0)} {m['loop_mma']} | convert "
          f"{lm.get('convert', 0)} {m['loop_convert']} | ld_shared {lm.get('ld_shared', 0)} ld_global "
          f"{lm.get('ld_global', 0)} fp32 {lm.get('fp32', 0)} mufu {lm.get('mufu', 0)} | native e4m3 MMA: "
          f"{'YES' if m['native_fp8_mma'] else 'NO'}")


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--cubin", nargs="*", default=[])
  ap.add_argument("--hub", action="append", default=[], help="BLOCK_M,num_warps,num_stages of the stock kernel")
  ap.add_argument("--config", action="append", default=[], help="TileConfig JSON of one of our builds")
  ap.add_argument("--triton-cache", help="scan a Triton cache dir for compiled hub kernels (*.cubin)")
  ap.add_argument("--name", default="w8a8_block_dynamic_fp8_matmul_kernel")
  ap.add_argument("--dump", action="store_true", help="print the inner loop's SASS")
  ap.add_argument("--out", default=None)
  a = ap.parse_args(argv)
  jobs = [(p, open(p, "rb").read()) for p in a.cubin]
  if a.triton_cache:
    jobs += [(p, open(p, "rb").read()) for p in glob.glob(os.path.join(os.path.expanduser(a.triton_cache), "*", f"{a.name}*.cubin"))]
  if a.hub or a.config:
    import aot                                                                  # noqa: PLC0415
    from space import TileConfig                                                # noqa: PLC0415
    for h in a.hub:
      bm, w, s = map(int, h.split(","))
      jobs.append((f"hub BM={bm} w={w} s={s}", aot.hub_cubin(bm, w, s)))
    for c in a.config:
      cfg = TileConfig(**json.loads(c))
      jobs.append((f"ours {c}", aot.cubin_sm89(cfg)))
  out = {}
  for name, cubin in jobs:
    text = disassemble(cubin)
    out[name] = mix(text)
    print(summary(name, out[name]), flush=True)
    if a.dump:
      print("\n".join(f"  /*{x:04x}*/ {o} {r}" for x, o, r in inner_loop(parse(text))))
  if a.out:
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
