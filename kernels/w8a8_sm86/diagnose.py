#!/usr/bin/env python3
"""Calibrate SmoothQuant on Kev-9B and diagnose where W8A8 loses accuracy, per layer and projection input (record.SITES).

  cd <kev> && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 flock <gpu-lock> \\
      env PYTHONPATH=<xCalibur>/kernels:. python <xCalibur>/kernels/w8a8_sm86/diagnose.py \\
      --bundle <sec_bench>/v1/smoke/requests.jsonl --calib 8 --csv <xCalibur>/data/w8a8_sm86_diagnosis.csv

1. Calibration: the bf16 fused model runs `--calib` requests of the bundle drawn (stratified, seed 1) from those NOT in
   the eval subset (kev.perf's SubsetSource(bundle, --subset, --seed)); every site's per-channel absmax over all their
   tokens goes to --stats (smooth.ActStats; kev_smooth.patch reads it).
2. Diagnosis: the eval subset's longest request runs with the same taps plus a 1,024-row sample of each site's state
   pass. Per site: the channel absmax spread (max / median, channels above 20x and 100x the median, the top channels),
   the median int8 code a typical channel gets under per-token scales, and the relative error of Y = X W^T under W8A8
   (per-token int8 X, per-channel int8 W, against fp32), plain and smoothed with each --alpha (calibrated scales,
   kev_smooth.input_scales: what the patch deploys), and the same with per-token per-128-channel activation scales
   (err_g128*, fake-quantised: no such GEMM here yet).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))              # kernels/ (the w8a8_sm86 package)
from w8a8_sm86 import kev_patch, record  # noqa: E402
from w8a8_sm86.kev_smooth import DEFAULT_STATS, input_scales, site_weight  # noqa: E402
from w8a8_sm86.linear import W8A8Weight, w8a8_linear  # noqa: E402
from w8a8_sm86.quant import quant_rows  # noqa: E402
from w8a8_sm86.smooth import ActStats  # noqa: E402

OUTLIER_X = (20, 100)


def _gemm_rows(layer, site: str) -> int:
  """Rows of the site's weight that run in int8 (the DeltaNet in_proj's b / a stay bf16)."""
  w = site_weight(layer, site)
  if site != "in":
    return w.shape[0]
  splits = record.mixer_of(layer)[0].in_proj.splits if layer.block_type == "linear_attention" else [w.shape[0]]
  return sum(splits[:kev_patch.quantised_prefix(splits)])


@torch.no_grad()
def rel_error(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor | None) -> float:
  """||W8A8(X / s, W s) - X W^T|| / ||X W^T|| in fp32 (s None: no smoothing)."""
  xf, wf = x.float(), w.float()
  ref = xf @ wf.T
  if s is not None:
    xf, wf = xf / s, wf * s
  y = w8a8_linear(xf, W8A8Weight.from_float(wf), backend="triton").float()
  return ((y - ref).norm() / ref.norm()).item()


@torch.no_grad()
def rel_error_grouped(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor | None, group: int = 128) -> float:
  """rel_error with one activation scale per token and `group` input channels (fake-quantised in fp32: what a GEMM
  that rescales its partial sums per K block would compute); the weight stays per output channel."""
  xf, wf = x.float(), w.float()
  ref = xf @ wf.T
  if s is not None:
    xf, wf = xf / s, wf * s
  wq = W8A8Weight.from_float(wf)
  xg = xf.view(xf.shape[0], -1, group)
  sx = xg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
  xd = (torch.round(xg / sx).clamp(-127, 127) * sx).view_as(xf)
  y = (xd @ (wq.q.float() * wq.scale[:, None]).T).bfloat16().float()
  return ((y - ref).norm() / ref.norm()).item()


def median_code(x: torch.Tensor) -> float:
  """The median over tokens of the median |int8 code| across channels: how many levels a typical channel gets."""
  q, _ = quant_rows(x)
  return q.abs().float().median(1).values.median().item()


def spread(amax: torch.Tensor) -> dict:
  med = amax.median().clamp_min(1e-12)
  top = amax.topk(4).indices.tolist()
  return {"amax_max": amax.max().item(), "amax_median": med.item(), "max_over_median": (amax.max() / med).item(),
          **{f"n_over_{k}x": int((amax > k * med).sum()) for k in OUTLIER_X}, "top_channels": " ".join(map(str, top))}


def diagnose_layers(layers, dstats: ActStats, samples: dict, cstats: ActStats, alphas: list[float]) -> list[dict]:
  rows = []
  for i, layer in enumerate(layers):
    for site in record.SITES:
      key = f"{i}.{site}"
      x = samples[key].cuda()
      w = site_weight(layer, site)[: _gemm_rows(layer, site)]
      row = {"layer": i, "kind": "deltanet" if layer.block_type == "linear_attention" else "attention", "site": site,
             "K": x.shape[1], **spread(dstats[key]), "median_code": median_code(x), "err_plain": rel_error(x, w, None),
             "err_g128": rel_error_grouped(x, w, None)}
      for a in alphas:
        s = input_scales(layer, i, site, cstats, a)
        row[f"err_a{a:g}"] = rel_error(x, w, s)
        row[f"err_g128_a{a:g}"] = rel_error_grouped(x, w, s)
        row[f"median_code_a{a:g}"] = median_code(x.float() / s)
      rows.append(row)
      logger.info("L{:02d} {:9s} {:4s} max/med {:8.1f} >20x {:4d} code {:5.1f} err {:.4f} -> {}", i, row["kind"], site,
                  row["max_over_median"], row["n_over_20x"], row["median_code"], row["err_plain"],
                  " ".join(f"{row[f'err_a{a:g}']:.4f}" for a in alphas))
      del x
  return rows


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default="jaredpalmer/kev-9b")
  ap.add_argument("--bundle", required=True)
  ap.add_argument("--subset", type=int, default=8, help="eval subset size (excluded from calibration)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--calib", type=int, default=8, help="calibration requests")
  ap.add_argument("--alpha", type=float, nargs="+", default=[0.5, 0.65, 0.8])
  ap.add_argument("--stats", default=str(DEFAULT_STATS))
  ap.add_argument("--reuse-stats", action="store_true", help="skip calibration, read --stats")
  ap.add_argument("--csv")
  a = ap.parse_args()

  from kev.perf.gpu import require_gpu
  require_gpu()
  from kev.checkpoint import Checkpoint
  from kev.perf.variants import LoadedVariant, resolve
  from kev.perf.workload import BundleSource, SubsetSource, stratified_subset

  items = BundleSource(a.bundle).items()
  evals = SubsetSource(BundleSource(a.bundle), a.subset, a.seed).items()
  eval_ids = {i.id for i in evals}
  calib = stratified_subset([i for i in items if i.id not in eval_ids], a.calib, seed=a.seed + 1)
  target = max(evals, key=lambda i: i.state_tokens or 0)
  v = LoadedVariant(Checkpoint(a.model), resolve("fused"))
  layers = kev_patch.base_lm(v.model).layers

  if a.reuse_stats:
    cstats = ActStats.load(a.stats, "cuda")
  else:
    cstats = ActStats(meta={"model": a.model, "bundle": a.bundle, "requests": [i.id for i in calib],
                            "state_tokens": [i.state_tokens for i in calib], "excluded_eval_subset": sorted(eval_ids)})
    with record.taps(layers, cstats):
      for item in calib:
        logger.info("calibration {} ({} state tokens)", item.id, item.state_tokens)
        v.run(v.encode(item.request))
    cstats.save(a.stats)
    logger.info("saved {} site statistics over {} requests to {}", len(cstats.data), len(calib), a.stats)

  dstats, samples = ActStats(), {}
  logger.info("diagnosis request {} ({} state tokens)", target.id, target.state_tokens)
  with record.taps(layers, dstats, samples=samples):
    v.run(v.encode(target.request))
  torch.cuda.empty_cache()
  rows = diagnose_layers(layers, dstats, samples, cstats, a.alpha)
  for r in rows:
    r["request"] = target.id
  if a.csv:
    with open(a.csv, "w", newline="") as f:
      wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
      wr.writeheader()
      wr.writerows(rows)
  v.close()


if __name__ == "__main__":
  main()
