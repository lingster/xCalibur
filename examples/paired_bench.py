"""Paired CUDA-graph timing: two equivalent skinny BF16 projections, with raw samples and an A/A control."""

import torch
import torch.nn.functional as F

import xcalibur as xc

x = torch.randn(16, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.randn(2304, 8192, device="cuda", dtype=torch.bfloat16) / 90.0

bench = xc.Bench()
aa = bench.aa(lambda: F.linear(x, w), pairs=20)
r = bench.paired(lambda: torch.mm(x, w.t()), lambda: F.linear(x, w), pairs=30)
print(f"protocol: {r['protocol']}; calibration subtraction: {r['subtract_null']}")
print(f"retained {len(r['raw_candidate_samples_us'])} raw candidate/reference pairs")
print(f"A/A ratio {aa['ratio']:.4f}  CI {aa['ci'][0]:.4f}-{aa['ci'][1]:.4f}   (the noise floor)")
print(f"mm vs linear: {r['candidate_us']:.2f} us vs {r['reference_us']:.2f} us, ratio {r['ratio']:.4f}  CI {r['ci'][0]:.4f}-{r['ci'][1]:.4f}")
