"""Judge-faithful paired timing on a CUDA GPU: two equivalent ways to run a skinny BF16 projection, plus the A/A floor."""

import torch
import torch.nn.functional as F

import xcalibur as xc

x = torch.randn(16, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.randn(2304, 8192, device="cuda", dtype=torch.bfloat16) / 90.0

bench = xc.Bench()
aa = bench.aa(lambda: F.linear(x, w), pairs=20)
r = bench.paired(lambda: torch.mm(x, w.t()), lambda: F.linear(x, w), pairs=30)
print(f"null replay cost subtracted: {r['null_us']:.1f} us")
print(f"A/A ratio {aa['ratio']:.4f}  CI {aa['ci'][0]:.4f}-{aa['ci'][1]:.4f}   (the noise floor)")
print(f"mm vs linear: {r['candidate_us']:.2f} us vs {r['reference_us']:.2f} us, ratio {r['ratio']:.4f}  CI {r['ci'][0]:.4f}-{r['ci'][1]:.4f}")
