"""Opt-in production hook: transformers' FP8Linear GEMMs on the kit's confirmed, bitwise-exact tuned table.

Nothing changes unless ``KEV_FP8_TUNED=1``. With it, an FP8Linear call whose (N, K, M-bucket) has an entry in the table
runs the kit's kernel with that entry's tiling; every other call runs transformers' own forward (the hub kernel),
untouched. Every entry is bitwise equal to the hub kernel (BLOCK_K 128, no split-K, parity on Ada), so outputs do
not change by a single bit: under FP8 nothing else is drift-free (kev ledger FP8D).

Three ways in:

  launcher   ``python kev_patch.py SCRIPT [ARGS...]`` (or ``kev_launch.sh`` as KEV_PYTHON): installs the class patch
             when the flag is set, then runs SCRIPT as __main__ (e.g. sec's kev_quant_serve.py).
  kev.perf   variant ``{"patch": "kev_patch.patch", "env": {"KEV_FP8_TUNED": "1"}}``: ``patch(model)`` patches that
             model's FP8Linear instances only, so a same-process A/B leaves the other variant on stock.
  library    ``install()`` / ``uninstall()``.

  KEV_FP8_TUNED          1 to enable (anything else: no-op)
  KEV_FP8_TUNED_TABLE    a table from ship.py (default: tables/ada_rtx6000_bitwise_r1.json next to this file)
  KEV_FP8_TUNED_VERIFY   1: also run stock on every hit, return stock's output, count bitwise mismatches (a slow
                         self-check for one smoke pass before timing; the report at exit must say mismatches 0)

The table is refused (and nothing is patched) unless it is a ship.py table, every entry is in the bitwise class, and
it was measured on a kernel source whose fp8 path is this one (``kernel.fp8_path_compatible``). A call is a hit only
on sm_89, bf16 contiguous input, fp8 e4m3 weight with fp32 128 x 128 block scales, dynamic activations.
"""
from __future__ import annotations

import atexit
import collections
import importlib
import os
import runpy
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_FLAG, ENV_TABLE, ENV_VERIFY = "KEV_FP8_TUNED", "KEV_FP8_TUNED_TABLE", "KEV_FP8_TUNED_VERIFY"
DEFAULT_TABLE = HERE / "tables" / "ada_rtx6000_bitwise_r1.json"
_KIT = ("shapes", "space", "table", "kernel")

try:
  from loguru import logger
except ImportError:                                                         # pragma: no cover
  import logging
  logger = logging.getLogger("kev_patch")


def enabled() -> bool:
  return os.environ.get(ENV_FLAG, "0") == "1"


def import_kit() -> dict[str, types.ModuleType]:
  """The kit's flat modules (kernel, space, table, shapes), imported without leaving generic names such as ``kernel``
  or ``table`` in sys.modules or the kit on sys.path, where they could shadow a production module."""
  saved = {n: sys.modules.pop(n) for n in _KIT if n in sys.modules}
  paths = [str(HERE), str(HERE.parent.parent)]                              # the kit; the repo root (xcalibur)
  sys.path[:0] = paths
  try:
    return {n: importlib.import_module(n) for n in _KIT}
  finally:
    for p in paths:
      sys.path.remove(p)
    for n in _KIT:
      sys.modules.pop(n, None)
    sys.modules.update(saved)


class TunedLinear:
  """The dispatch: table lookup per (N, K, M), kernel for hits, the original forward for misses; with counters."""

  def __init__(self, path: str | Path | None = None, kit: dict | None = None):
    self.kit = kit or import_kit()
    self.path = Path(path or os.environ.get(ENV_TABLE) or DEFAULT_TABLE)
    self.table = self.kit["table"].TuneTable.load(self.path)
    self._check_table()
    self.by_shape: dict[tuple[int, int], list[tuple[int, int, object]]] = collections.defaultdict(list)
    for e in self.table.entries:
      self.by_shape[(e["N"], e["K"])].append((e["lo"], e["hi"], self.kit["space"].TileConfig(**e["config"])))
    self.hits: collections.Counter = collections.Counter()
    self.misses: collections.Counter = collections.Counter()
    self.require_cuda, self.dot = True, "fp8"                              # tests: CPU tensors, interpreter dot
    self.verify, self.mismatches = os.environ.get(ENV_VERIFY, "0") == "1", 0

  def _check_table(self) -> None:
    meta, kernel, space = self.table.meta, self.kit["kernel"], self.kit["space"]
    if meta.get("kind") != "shippable" or not meta.get("bitwise_only"):
      raise ValueError(f"{self.path}: not a ship.py bitwise table (meta kind={meta.get('kind')!r})")
    shas = meta.get("kernel_sha") or []
    if not shas or not all(kernel.fp8_path_compatible(s) for s in shas):
      raise ValueError(f"{self.path}: measured on kernel {shas}, not this fp8 path ({kernel.KERNEL_SHA})")
    for e in self.table.entries:
      if not e.get("config") or not space.bitwise_safe(e["config"]):
        raise ValueError(f"{self.path}: entry N={e['N']} K={e['K']} {e['bucket']} is not in the bitwise class")

  def config(self, N: int, K: int, M: int):
    for lo, hi, cfg in self.by_shape.get((N, K), ()):
      if lo <= M <= hi:
        return cfg
    return None

  def eligible(self, mod, x) -> bool:
    import torch                                                            # noqa: PLC0415
    w, s = mod.weight, mod.weight_scale_inv
    return (x.dtype == torch.bfloat16 and (x.is_cuda or not self.require_cuda) and x.is_contiguous() and w.dtype == torch.float8_e4m3fn
            and s.dtype == torch.float32 and s.dim() == 2 and getattr(mod, "activation_scale", None) is None
            and tuple(mod.block_size or ()) == (128, 128) and w.shape[0] % 128 == 0 and w.shape[1] % 128 == 0)

  def __call__(self, mod, x, original):
    N, K = mod.weight.shape
    cfg = self.config(N, K, x.numel() // K) if self.eligible(mod, x) else None
    if cfg is None:
      self.misses[(N, K)] += 1
      return original(x)
    self.hits[(N, K, cfg.BLOCK_M, cfg.BLOCK_N)] += 1
    calls = sum(self.hits.values())
    if calls == 1 or calls % 100_000 == 0:                                  # grep-able proof the patch is live
      logger.info(f"[kev_patch] {self.report()}")
    out = self.kit["kernel"].block_fp8_matmul(x, mod.weight, mod.weight_scale_inv, cfg, out_dtype=x.dtype, dot=self.dot)
    if mod.bias is not None:
      out.add_(mod.bias)                                                    # as transformers' finegrained_fp8_linear
    if self.verify:
      import torch                                                          # noqa: PLC0415
      ref = original(x)
      if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
        self.mismatches += 1
        logger.error(f"[kev_patch] NOT bitwise: N={N} K={K} M={x.numel() // K} {cfg}")
      return ref
    return out

  def report(self) -> str:
    return (f"hits {sum(self.hits.values())} {dict(self.hits)}; stock {sum(self.misses.values())}"
            + (f"; verify mismatches {self.mismatches}" if self.verify else ""))


def _platform_ok() -> str | None:
  """None when this process can take the table, else why not (the patch then stays off)."""
  import torch                                                              # noqa: PLC0415
  if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
    return "not an sm_89 GPU"
  from transformers.integrations import finegrained_fp8 as tf                # noqa: PLC0415
  if getattr(tf, "is_deepgemm_loadable", lambda: False)():
    return "DeepGEMM is loadable: stock dispatch would not be the hub kernel"
  return None


_STATE: dict = {}


def install(path: str | Path | None = None, *, require_flag: bool = True, check_platform: bool = True) -> bool:
  """Patch FP8Linear.forward process-wide. False (and no change) when the flag is off or the platform/table is refused."""
  if (require_flag and not enabled()) or "original" in _STATE:
    return "original" in _STATE
  if check_platform and (why := _platform_ok()):
    logger.warning(f"[kev_patch] {ENV_FLAG}=1 but not installed: {why}")
    return False
  from transformers.integrations.finegrained_fp8 import FP8Linear           # noqa: PLC0415
  tuned = TunedLinear(path)
  original = FP8Linear.forward

  def forward(self, input):
    return tuned(self, input, lambda x: original(self, x))

  _STATE.update(original=original, tuned=tuned, cls=FP8Linear)
  _STATE["atexit"] = lambda: logger.info(f"[kev_patch] at exit: {tuned.report()}")
  atexit.register(_STATE["atexit"])
  FP8Linear.forward = forward
  logger.info(f"[kev_patch] FP8Linear on the tuned table {tuned.path.name}: {len(tuned.table.entries)} entries, "
              f"kernel {tuned.kit['kernel'].KERNEL_SHA}")
  return True


def uninstall() -> None:
  if "original" in _STATE:
    _STATE["cls"].forward = _STATE["original"]
    atexit.unregister(_STATE["atexit"])
    logger.info(f"[kev_patch] removed; {_STATE['tuned'].report()}")
    _STATE.clear()


def patch(model, path: str | Path | None = None, *, check_platform: bool = True) -> int:
  """kev.perf hook: patch this model's FP8Linear instances (only when the flag is set). Returns how many."""
  if not enabled():
    return 0
  if check_platform and (why := _platform_ok()):
    logger.warning(f"[kev_patch] {ENV_FLAG}=1 but not applied: {why}")
    return 0
  from transformers.integrations.finegrained_fp8 import FP8Linear           # noqa: PLC0415
  tuned, n = TunedLinear(path), 0
  for mod in model.modules():
    if type(mod) is FP8Linear:                                              # not subclasses (emulation, grouped)
      original = mod.forward
      mod.forward = types.MethodType(lambda self, x, _o=original: tuned(self, x, _o), mod)
      n += 1
  model._kev_fp8_tuned = tuned
  logger.info(f"[kev_patch] {n} FP8Linear modules on the tuned table {tuned.path.name}")
  return n


def main(argv: list[str] | None = None) -> None:
  """Launcher: ``kev_patch.py SCRIPT [ARGS...]`` installs (if the flag is set) and runs SCRIPT as __main__."""
  argv = list(sys.argv[1:] if argv is None else argv)
  if not argv:
    raise SystemExit("usage: kev_patch.py SCRIPT [ARGS...]")
  logger.info(f"[kev_patch] {ENV_FLAG}={os.environ.get(ENV_FLAG, '0')}: "
              + ("tuned table installed" if install() else "stock FP8Linear (no patch)"))
  script = str(Path(argv[0]).resolve())
  sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]   # no kit names for the served process
  sys.path.insert(0, str(Path(script).parent))                             # as `python SCRIPT` would have it
  sys.argv = argv
  runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
  main()
