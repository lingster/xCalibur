# Gemma 4 31B kernels: TG4DAC on SM120

This directory contains **Gemma 4 31B attention and dense W4A16 MLP + LoRA kernels**
for RTX PRO 6000 Blackwell (SM120). TG4DAC is the co-design study behind them:
separate models planned and refined the physical designs, and a frontier coding
model implemented and measured them ([the study](../../docs/studies.md#gemma4)).

The target workload is Gemma 4 31B. The planner and refiner also used Gemma 4 31B,
with separate adapters; their role as designers is distinct from the model whose
operators these kernels execute.

These are hardware-specific benchmark sources. `attention.py` reuses the NVRTC/driver helpers
in the adjacent `mlp.py`; keep both files together. They construct reference and
candidate implementations and time them in the same run. Analytical
speed-of-light bounds are labeled separately from measured implementations.

## `attention.py`: attention, a held-out problem

Gemma 4 31B attention, per layer, against vLLM's Triton attention kernel. Attention was **held out** of the
refiner's training, so these designs are its work on an unseen operator.

```math
o_h = \mathrm{softmax}\!\big(q_h K_{g(h)}^{\top}\big)\, V_{g(h)}, \qquad
\begin{cases}
\text{sliding: } 32\ \text{Q} / 16\ \text{KV heads},\ d = 256,\ \text{last } 1024 \text{ positions}\\
\text{global: } 32\ \text{Q} / 4\ \text{KV heads},\ d = 512,\ \text{all positions}
\end{cases}
```

| Design | Origin | Idea |
|---|---|---|
| Launch constants (prefill) | planner | vLLM's kernel unchanged, run with the planner's tile and launch choices |
| Window trim (sliding decode) | planner | re-base each sequence on the first cache page its 1024-token window touches, so vLLM's kernel stops scanning evicted pages |
| V-only cache (global decode) | planner | cache only $`V`$ and rebuild $`K = \mathrm{RoPE}(w_k \odot V)`$ in the kernel; for the 384 unrotated dims $`K\cdot q = V\cdot(w_k \odot q)`$, so $`w_k`$ folds into $`q`$ and every cached byte is read once |
| WarpSpeed prefill | refiner | a load squad streams Q and K / V through `cp.async` + mbarrier rings into a compute squad running `mma.sync` m16n8k16 |
| WarpSpeed decode | refiner | one load warp fills a KV page ring for three compute warps, window bounds in-kernel (sliding); a 4 + 4 warp split over a V-only ring (global) |

Measured against vLLM in the same run [M]: **4.6×** on sliding-window prefill, **3.1×** on global prefill, up to
**3.6×** on decode, and **1.30×** end-to-end decode throughput at 8 sequences × 32k context inside vLLM. Data:
[`data/tg4dac_attention.csv`](../../data/tg4dac_attention.csv), [`data/tg4dac_e2e.csv`](../../data/tg4dac_e2e.csv).

```bash
python kernels/tg4dac_sm120/attention.py --B 1 8 --L 4096 32768 131072 --prefill-M 7131 15329
```

## `mlp.py`: Gemma 4's dense 4-bit MLP with LoRA

The first TG4DAC round targeted Gemma 4 31B's dense W4A16 MLP (INT4, group-32
scales), with rank-16 LoRA on every projection,

```math
g = XW_g^{\top} + 2\,(XA_g^{\top})B_g^{\top}, \quad u = XW_u^{\top} + 2\,(XA_u^{\top})B_u^{\top}, \quad
Y = \big(\mathrm{GELU}(g) \odot u\big) W_d^{\top} + 2\,(\cdots)A_d^{\top}B_d^{\top},
```

with the 4-bit weights decoded inside every forward. The file holds the hand-written kernels the models were asked to
improve: fused INT4-dequant tensor-core GEMMs with the LoRA and activation fused into the epilogues, a WarpSpeed
variant (two MMA squads and one TMA producer warp with cluster-launch-control work stealing), and an operand-swapped
GEMV for 8 tokens. They already ran near 80% of the matrix units' peak, and no model proposal beat them [M]: the lesson
that sent TG4DAC to attention.

```bash
python kernels/tg4dac_sm120/mlp.py --M 8 15329 --synthetic
```

## Requirements

An SM120 GPU (RTX PRO 6000 Blackwell), PyTorch with CUDA and Triton. The attention
reference imports vLLM's internal Triton attention implementation; the recorded
study used vLLM 0.32 and those internal APIs may differ in other versions.
`mlp.py` compiles with NVRTC through `ctypes` (no `nvcc` needed) and can load real
Gemma 4 weights or use synthetic ones. For local weights use `--weights` or the
standard Hugging Face cache environment variables. The recorded whole-model
vLLM integration and trained planner/refiner adapters are not bundled.

The public sources and analysis are available; reproducing the historical GPU
numbers still requires matching hardware, dependencies and input contracts.

Apache 2.0, like the rest of the repository ([LICENSE](../../LICENSE)).
