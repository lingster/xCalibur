# W8A8 INT8 linears on RTX 3090 (sm_86)

INT8 weights and INT8 activations for the projection GEMMs of Kev-9B (a Qwen3.5-9B hybrid) on a GeForce RTX 3090.
On this card a Kev-9B long prefill is about 81% GEMM time, and its bf16 GEMMs already run at about 95% of the 71
TFLOPS bf16 peak (fp32 accumulate), so tuning bf16 has no headroom left. INT8 tensor cores have **4×** that dense rate
(284 TOPS). W8A8 is therefore the only way left to make those GEMMs faster on Ampere. It is a **method change**: it
moves the model's numbers, so it needs a re-score before production use.

| | |
|---|---|
| What | per-channel INT8 weights, per-token dynamic INT8 activations, int32 accumulate, dequant to bf16 |
| Hardware | RTX 3090 (GA102, sm_86), 300 W cap; PyTorch 2.14.1+cu130, Triton 3.8.0 |
| Kernel speed [M] | **2.4–3.5×** over bf16 `F.linear` per projection (quantiser included), 56–77% of the INT8 peak |
| One state pass of projections [M] | 6,142 → 2,188 ms at 28,672 tokens, **2.81×** |
| End to end, Kev-9B [M] | **1.84×** [90% CI 1.82, 1.86] throughput, 3,825 → 7,051 tok/s on ~29k-token SEC filings; peak memory 20.99 → 17.67 GB |
| Numerics vs bf16 fused [M] | unsmoothed, full smoke bundle (1,198 answers): max \|dp\| 0.469, flips 17.4%, within-1 93.0%, mean \|Δlevel\| 0.142 (fused vs unfused: max 0.016) |
| With SmoothQuant α 0.5 [M] | **1.81×** [1.80, 1.82] on the full smoke bundle (1,198 answers): max \|dp\| 0.158, flips 8.6%, within-1 96.5%, mean \|Δlevel\| 0.073: about half the unsmoothed drift, still 5× the gate |
| Status | **measured, not adopted**: SmoothQuant halves the drift at full speed; nothing measured passes the numerics gate (see Verdict). Needs a re-score as a method change |

## The computation

For a linear $Y = XW^{\top}$ with $X \in \mathbb{R}^{M \times K}$ (tokens × features) and $W \in \mathbb{R}^{N \times K}$:

$$
s^w_n = \frac{\max_k |W_{nk}|}{127},\quad q^w_{nk} = \operatorname{round}\!\Big(\frac{W_{nk}}{s^w_n}\Big), \qquad
s^x_m = \frac{\max_k |X_{mk}|}{127},\quad q^x_{mk} = \operatorname{round}\!\Big(\frac{X_{mk}}{s^x_m}\Big),
$$

$$
Y_{mn} \approx s^x_m\, s^w_n \sum_k q^x_{mk}\, q^w_{nk} .
$$

The weights are quantised once, at load. The activations are quantised on every call, one scale per token. The
integer sum is exact in int32 ($|{\rm acc}| \le K \cdot 127^2 < 2^{31}$ for $K \le 133{,}000$), so every backend and
every autotuned tile gives the same numbers. Only the final bf16 rounding is shared with the reference.

| File | Lines | What |
|---|---:|---|
| [`quant.py`](quant.py) | 152 | Triton per-token quantiser (optional per-column multiplier), SwiGLU fused with the quantiser, dequant kernel, torch reference |
| [`gemm.py`](gemm.py) | 96 | Triton int8 GEMM (`mma.sync` s8, int32 acc) with the dequant in its epilogue; `torch._int_mm` path |
| [`grouped.py`](grouped.py) | 170 | per-(token, 128-channel) activation scales: quantiser, SwiGLU quantiser, Triton GEMM rescaling each K block |
| [`linear.py`](linear.py) | 125 | `W8A8Weight`, `w8a8_linear`, `W8A8Linear` (`act_scale`, `group`); backend choice |
| [`kev_patch.py`](kev_patch.py) | 169 | `patch(model)` / `patch_layer`: swaps a fused Kev backbone's projections, as a `kev.perf` variant patch (`keep=`, `keep_layers=` for ablations) |
| [`smooth.py`](smooth.py) | 83 | SmoothQuant scales (GQA-tied for `o_proj`), `ActStats` (calibrated per-channel absmax) |
| [`record.py`](record.py) | 90 | the four GEMM inputs of a fused layer; taps that record their absmax and a row sample |
| [`kev_smooth.py`](kev_smooth.py) | 194 | `patch(model)`: folds the smoothing into norms / v rows / up rows, then converts each layer (env: α, kept projections, groups) |
| [`diagnose.py`](diagnose.py) | 166 | calibration (writes the absmax file) and the per-site error diagnosis |
| [`microbench.py`](microbench.py) | 139 | paired timing against `F.linear` at Kev-9B shapes (`xcalibur.bench.Bench.paired`) |

## Design choices

1. **Two GEMM paths, chosen by M.** The Triton kernel keeps the int32 tile in registers and writes bf16 directly.
   `torch._int_mm` (cuBLASLt IMMA) writes int32 to memory, then a second kernel dequantises it. In the screen
   (below), cuBLASLt's kernels win at M ≥ 4,096 on five of six shapes, by up to 31% on the down projection. At
   M = 512 the result is mixed: Triton wins three shapes (by 2–6%) and cuBLASLt the other three (by 2–10%). The
   default backend `auto` takes Triton below 1,024 rows, where no int32 buffer is needed, and `_int_mm` from 1,024 up.
   Kev's question rows (a few hundred tokens) therefore run Triton, and its 28k-token states run cuBLASLt. The Triton
   GEMM alone, without the quantiser (`gemm`), beats `_int_mm` on the square 4096 × 4096 shapes. Its weakness is the
   K = 12,288 down projection.
2. **The quantiser sits with what precedes the GEMM where that was cheap.** The MLP's SiLU(gate)·up and the down
   projection's quantiser run as one kernel (`swiglu_quant`): one read of gate and up, one int8 write, and no bf16
   intermediate. The other inputs come out of fla's fused RMSNorm and gated-norm kernels, so those GEMMs get a
   separate quantiser, a memory-bound pass that costs 2–7% of the GEMM (the gap between `triton` and `gemm` below).
   Fusing it into fla's norms would mean forking fla.
3. **Small projections stay bf16.** The DeltaNet's `in_proj_b` / `in_proj_a` (32 outputs each) feed exp / softplus
   gates and carry 0.1% of the FLOPs. They run as one bf16 GEMM next to the INT8 one. The embedding, the norms, the
   conv, the delta rule, attention and the pointer head are untouched.
4. **Memory goes down.** The bf16 weights are dropped group by group as they are converted. On Kev-9B the 6.9 B
   projection weights shrink from 13.8 to 6.9 GB. Some of that space goes to the int32 buffer of `_int_mm` (2.8 GB for
   gate|up at 28,672 tokens). Measured peak per request: 20.99 → 17.67 GB.
5. **Rounding.** Round half away from zero, and IEEE divisions on both sides. A CUDA `tensor / python_scalar`
   multiplies by the reciprocal and is 1 ulp off, which the first test run caught. With that, the Triton quantiser
   reproduces the torch reference's scales exactly.

## Results [M]

### Kernels (microbench, RTX 3090, GPU0)

Each cell is W8A8 against bf16 `F.linear(x, w)` in the same paired session (CUDA graphs, L2 flushed by a 256 MiB
read, ABBA pairs, null replay subtracted), with the median ratio and its 95% bootstrap CI. TOPS = 2MNK / time. Peaks
(GA102 whitepaper, 3090 at its 1,695 MHz boost): 284 dense INT8 TOPS, 71 bf16 TFLOPS with fp32 accumulate.

**Confirm**, run `20261105T121738Z-61fc7d`: the chosen `auto` backend alone, in a fresh process, 10 pairs per cell,
quantiser included. This is the number to quote. It matches the screen within its CIs.

| Projection (N × K) | M | bf16 µs | bf16 TFLOPS (% of 71) | W8A8 µs | W8A8 TOPS (% of 284) | **speedup** [95% CI] | backend |
|---|---:|---:|---:|---:|---:|---|---|
| `deltanet.in_proj[qkv+z]` (12288 × 4096) | 512 | 783 | 65.8 (93%) | 309 | 166.7 (59%) | **2.53× [2.53, 2.54]** | triton |
| `deltanet.out_proj` (4096 × 4096) | 512 | 319 | 53.8 (76%) | 129 | 133.2 (47%) | **2.47× [2.46, 2.49]** | triton |
| `attn.qkv[q+gate,k,v]` (10240 × 4096) | 512 | 652 | 65.8 (93%) | 256 | 167.8 (59%) | **2.54× [2.54, 2.56]** | triton |
| `attn.o_proj` (4096 × 4096) | 512 | 318 | 54.0 (76%) | 129 | 133.2 (47%) | **2.46× [2.46, 2.50]** | triton |
| `mlp.gate_up` (24576 × 4096) | 512 | 1,633 | 63.1 (89%) | 607 | 169.8 (60%) | **2.69× [2.69, 2.69]** | triton |
| `mlp.down_proj` (4096 × 12288) | 512 | 938 | 54.9 (77%) | 364 | 141.8 (50%) | **2.58× [2.57, 2.59]** | triton |
| `deltanet.in_proj[qkv+z]` (12288 × 4096) | 4,096 | 6,113 | 67.4 (95%) | 2,225 | 185.3 (65%) | **2.74× [2.73, 2.76]** | intmm |
| `deltanet.out_proj` (4096 × 4096) | 4,096 | 2,214 | 62.1 (87%) | 808 | 170.0 (60%) | **2.74× [2.73, 2.74]** | intmm |
| `attn.qkv[q+gate,k,v]` (10240 × 4096) | 4,096 | 5,093 | 67.5 (95%) | 1,812 | 189.6 (67%) | **2.81× [2.80, 2.83]** | intmm |
| `attn.o_proj` (4096 × 4096) | 4,096 | 2,214 | 62.1 (87%) | 809 | 169.8 (60%) | **2.74× [2.73, 2.74]** | intmm |
| `mlp.gate_up` (24576 × 4096) | 4,096 | 12,363 | 66.7 (94%) | 4,712 | 175.0 (62%) | **2.62× [2.59, 2.67]** | intmm |
| `mlp.down_proj` (4096 × 12288) | 4,096 | 6,546 | 63.0 (89%) | 2,013 | 204.9 (72%) | **3.25× [3.25, 3.26]** | intmm |
| `deltanet.in_proj[qkv+z]` (12288 × 4096) | 28,672 | 45,155 | 63.9 (90%) | 15,583 | 185.2 (65%) | **2.81× [2.78, 3.06]** | intmm |
| `deltanet.out_proj` (4096 × 4096) | 28,672 | 14,746 | 65.2 (92%) | 5,360 | 179.5 (63%) | **2.75× [2.70, 2.80]** | intmm |
| `attn.qkv[q+gate,k,v]` (10240 × 4096) | 28,672 | 37,789 | 63.6 (90%) | 12,612 | 190.7 (67%) | **2.92× [2.88, 3.14]** | intmm |
| `attn.o_proj` (4096 × 4096) | 28,672 | 14,602 | 65.9 (93%) | 5,361 | 179.5 (63%) | **2.75× [2.70, 2.77]** | intmm |
| `mlp.gate_up` (24576 × 4096) | 28,672 | 87,818 | 65.7 (93%) | 35,170 | 164.1 (58%) | **2.47× [2.42, 2.54]** | intmm |
| `mlp.down_proj` (4096 × 12288) | 28,672 | 46,107 | 62.6 (88%) | 13,004 | 222.0 (78%) | **3.51× [3.28, 3.83]** | intmm |

All projections of one Kev-9B state pass, weighted by calls per pass (24 DeltaNet, 8 attention and 32 MLP layers):
**6,142 → 2,188 ms (2.81×) at 28,672 tokens**, 863 → 309 ms (2.80×) at 4,096, and 116.5 → 44.7 ms (2.61×) at 512.
The session's A/A read 1.0000 [0.9997, 1.0009].

**Screen**, run `20261105T114150Z-c2de74`, 10 pairs per cell. `triton`, `intmm` and `auto` include the per-token
quantiser; `gemm` is the Triton GEMM alone on an already quantised input.

| Projection (N × K) | M | bf16 TFLOPS | triton × [CI] | intmm × [CI] | gemm only × (TOPS) |
|---|---:|---:|---|---|---|
| `deltanet.in_proj[qkv+z]` (12288 × 4096) | 512 | 65.8 | 2.54× [2.53, 2.54] | 2.40× [2.40, 2.41] | 2.59× (171) |
| `deltanet.out_proj` (4096 × 4096) | 512 | 53.9 | 2.47× [2.46, 2.49] | 2.43× [2.41, 2.43] | 2.59× (140) |
| `attn.qkv[q+gate,k,v]` (10240 × 4096) | 512 | 67.4 | 2.55× [2.55, 2.56] | 2.80× [2.79, 2.81] | 2.61× (176) |
| `attn.o_proj` (4096 × 4096) | 512 | 54.0 | 2.48× [2.46, 2.49] | 2.43× [2.40, 2.45] | 2.59× (140) |
| `mlp.gate_up` (24576 × 4096) | 512 | 63.2 | 2.69× [2.69, 2.70] | 2.76× [2.75, 2.77] | 2.72× (172) |
| `mlp.down_proj` (4096 × 12288) | 512 | 54.4 | 2.59× [2.58, 2.59] | 2.64× [2.63, 2.65] | 2.73× (150) |
| `deltanet.in_proj[qkv+z]` (12288 × 4096) | 4,096 | 67.4 | 2.59× [2.58, 2.59] | 2.74× [2.73, 2.75] | 2.65× (180) |
| `deltanet.out_proj` (4096 × 4096) | 4,096 | 62.0 | 2.66× [2.66, 2.66] | 2.74× [2.74, 2.75] | 2.84× (176) |
| `attn.qkv[q+gate,k,v]` (10240 × 4096) | 4,096 | 66.8 | 2.58× [2.58, 2.60] | 2.81× [2.80, 2.82] | 2.65× (180) |
| `attn.o_proj` (4096 × 4096) | 4,096 | 62.0 | 2.66× [2.66, 2.66] | 2.73× [2.73, 2.74] | 2.84× (176) |
| `mlp.gate_up` (24576 × 4096) | 4,096 | 66.3 | 2.64× [2.61, 2.67] | 2.60× [2.58, 2.64] | 2.67× (179) |
| `mlp.down_proj` (4096 × 12288) | 4,096 | 62.7 | 2.74× [2.73, 2.74] | 3.25× [3.25, 3.27] | 2.96× (185) |
| `deltanet.in_proj[qkv+z]` (12288 × 4096) | 28,672 | 64.4 | 2.65× [2.63, 2.84] | 2.83× [2.80, 3.10] | 2.77× (178) |
| `deltanet.out_proj` (4096 × 4096) | 28,672 | 65.9 | 2.46× [2.45, 2.54] | 2.68× [2.66, 2.69] | 2.72× (177) |
| `attn.qkv[q+gate,k,v]` (10240 × 4096) | 28,672 | 63.8 | 2.68× [2.64, 2.90] | 2.94× [2.88, 3.26] | 2.79× (182) |
| `attn.o_proj` (4096 × 4096) | 28,672 | 66.4 | 2.46× [2.44, 2.48] | 2.76× [2.72, 2.80] | 2.64× (175) |
| `mlp.gate_up` (24576 × 4096) | 28,672 | 66.5 | 2.40× [2.32, 2.50] | 2.44× [2.30, 2.52] | 2.44× (160) |
| `mlp.down_proj` (4096 × 12288) | 28,672 | 63.9 | 2.63× [2.58, 2.81] | 3.45× [3.30, 3.63] | 2.85× (189) |

The screen's A/A (bf16 gate|up at M = 4,096 against itself) read 1.0000 [0.9997, 1.0009]. On this card the bf16
GEMMs run at 54–68 TFLOPS (76–96% of peak), and INT8 at 131–219 TOPS (46–77% of peak). The ratio lands at 2.4–3.5×
rather than 4×, because Triton's and cuBLASLt's IMMA kernels on GA102 reach a lower share of their peak than
cuBLAS's bf16 kernels reach of theirs, and the per-token quantiser is extra memory traffic.

### End to end: Kev-9B on SEC filings

Kev's own harness, `kev.perf` (`compare --mode blocks --blocks 4`), ran Kev-9B `fused` (bf16, the 3090 baseline)
against `fused` plus `kev_patch.patch`. Workload: 8 sec_bench smoke requests, stratified (seed 0), about 29k state
tokens each, 78 answers. The two variants ran in ABBA blocks, with one model loaded at a time, 16 pairs. The
reference's throughput ratio is 1.0.

| Run | Candidate | Throughput ratio [90% CI] | tok/s | Peak GB / request | max \|dp\| | flips / 78 | within-1 |
|---|---|---|---|---|---:|---:|---:|
| `20261105T122956Z-169e7f` | **W8A8, every projection** | **1.843 [1.822, 1.864]** | 3,825 → 7,051 | 20.99 → 17.67 | 0.169 | 11 | 93.6% |
| `20261105T124257Z-699504` | W8A8, `mlp.down_proj` kept bf16 (ablation) | 1.477 [1.459, 1.493] | 3,819 → 5,642 | 20.99 → 19.23 | 0.243 | 16 | 91.0% |
| `20261105T123648Z-aeb513` | A/A: `fused` against itself, same session | 0.993 [0.990, 0.997] | 3,816 / 3,790 | 20.99 | 0 | 0 | 100% |

Per form, the full swap gives 10-K 1.86×, 10-Q 1.77×, 20-F 1.84× and DEF 14A 1.83×. The A/A's offset of 0.7% is
small next to the effect. The speedup also matches the profile: GEMMs are 81% of the state pass (B1), and at 2.8×
they predict $1/(0.19 + 0.81/2.8) \approx 2.1\times$ on the state pass and about 1.9× on a request, against 1.84×
measured. The run was power-limited: the 300 W cap throttled the SMs (reason 0x4) during the W8A8 blocks.

## Numerics: what changes

The per-answer drift against bf16 fused (run `20261105T122956Z-169e7f`, 78 answers), next to the drift the house
already accepts, which is fused against unfused (K2 in Kev's kernel-perf ledger):

| | p50 | p90 | p99 | max | mean \|Δlevel\| | flips | within-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fused vs unfused, bf16 (accepted) | 0.0035 | | 0.0109 | 0.0162 | 0.0073 | 2 / 78 | 100% |
| **W8A8 vs fused** | **0.055** | **0.110** | **0.150** | **0.169** | **0.122** | **11 / 78** | **93.6%** |
| W8A8 except down_proj vs fused | 0.062 | 0.132 | 0.212 | 0.243 | 0.128 | 16 / 78 | 91.0% |

W8A8 moves the median answer by 16× the bf16 drift, and five answers move by more than one level. Keeping the down
projection in bf16 does not reduce the drift, which pointed at the GEMMs fed by RMSNorm. The diagnosis below confirms
that for those GEMMs, and finds a second, different source at the down and out projections.

### Diagnosis: where the INT8 error comes from [M]

[`diagnose.py`](diagnose.py) hooks the input of every converted GEMM (four sites per layer, [`record.py`](record.py))
in the bf16 fused model on one 27,995-token 10-K request of the eval subset (request `10k-28k#5`). Per site it
reports the spread of the per-channel absmax over all tokens, the median INT8 code a typical channel gets under
per-token scales, and the relative error $\lVert \hat Y - Y \rVert / \lVert Y \rVert$ of W8A8 against fp32 on a
1,024-token sample. "per 128" is the same with one activation scale per token and 128 channels (fake-quantised).
Means over layers; data in [`data/w8a8_sm86_diagnosis.csv`](../../data/w8a8_sm86_diagnosis.csv) (128 rows).

| site (layers) | absmax max / median, median layer [max] | median code | rel. err | α 0.5 | α 0.8 | per 128 | per 128 + α 0.5 |
|---|---|---:|---:|---:|---:|---:|---:|
| DeltaNet `in_proj` (24) | 21 [27] | 1 | 0.077 | 0.025 | **0.016** | 0.017 | **0.009** |
| attention `qkv` (8) | 25 [44] | 1.5 | 0.062 | 0.019 | **0.015** | 0.015 | **0.009** |
| `gate_up`, DeltaNet layers (24) | 11 [20] | 3 | 0.059 | 0.021 | **0.015** | 0.015 | **0.011** |
| `gate_up`, attention layers (8) | 13 [18] | 2.5 | 0.053 | 0.019 | **0.014** | 0.014 | **0.010** |
| DeltaNet `out_proj` (24) | 21 [441] | 1 | 0.055 | 0.053 | 0.046 | **0.020** | 0.024 |
| attention `o_proj` (8) | 9 [22] | 3.5 | 0.037 | 0.049 | 0.039 | **0.018** | 0.024 |
| `down_proj`, DeltaNet layers (24) | 17 [49] | 1.5 | 0.079 | 0.066 | 0.060 | **0.019** | 0.023 |
| `down_proj`, attention layers (8) | 16 [58] | 1 | 0.071 | 0.063 | 0.057 | **0.019** | 0.023 |

What it says:

- **The norm-fed GEMMs have channel outliers, but they are not huge.** The largest channel's absmax is 10–44× the
  median channel's, not the hundreds to thousands seen in some LLMs, but the same channels lead in every layer
  (3994, 3842, 3986, 3678 of the residual stream). They set each token's scale, so a typical channel gets a median
  code of **1**, i.e. one or two INT8 levels. SmoothQuant fixes these: the error drops 4–5× (0.06–0.08 → 0.015).
- **The down and out projections have token-local outliers, which SmoothQuant cannot remove.** Their median code is
  1–3.5 before smoothing and 2–3.5 after it; at α 0.8 the down projections' error drops only 20% (median layer),
  and the out projections' moves by −52% to +44% per layer (attention `o_proj` gets *worse* on average). Within a
  token, a few post-SwiGLU / post-attention values are large, and *which* channels varies from token to token, so no
  per-channel scale helps. Per-128-channel activation scales do: 0.019.
- **No layer stands out.** After smoothing (α 0.8), the per-layer sum of the four errors lies between 0.12 and 0.17
  (layers 5, 2, 22, 4 highest; the last layer, which sees only the question rows past its mixer, 0.09), so keeping the k worst layers in bf16 would buy little. Layer 2's `down_proj` is the single
  worst GEMM (0.079 at α 0.8). Before smoothing, the early DeltaNet layers' `in_proj` were the worst (0.11–0.15,
  layers 1–8).

### SmoothQuant: folded, calibrated, zero run-time cost

For each converted GEMM, $s_j = \max|X_j|^{\alpha} / \max|W_j|^{1-\alpha}$ ([`smooth.py`](smooth.py)), with
$\max|X_j|$ from 8 calibration requests (about 28k tokens each: the smoke bundle's requests stratified with seed 1
**outside** the 8-request eval subset; the absmax file, 3.2 MB, sha256 `93e98ba9…`, lives outside git and
`diagnose.py` writes it). [`kev_smooth.py`](kev_smooth.py) folds $1/s$ into whatever produces the activation and $s$
into the weight's columns, layer by layer, before converting the layer:

| GEMM | $1/s$ folded into | run-time cost |
|---|---|---|
| `in_proj`, `qkv` | the input RMSNorm weight (`input_norm_weight`, fp32 1 + w) | none |
| `gate_up` | the post-mixer RMSNorm weight (`post_norm_weight`) | none |
| `down_proj` | the **up rows** of `gate_up`: SiLU(g)·(u/s) = (SiLU(g)·u)/s; the per-row INT8 weight scales absorb it | none |
| attention `o_proj` | the **v rows** of `qkv` (attention output is linear in v, the sigmoid gate is element-wise); $s$ tied over the 4 q heads sharing each kv head | none |
| DeltaNet `out_proj` | the activation quantiser (`act_scale`: one fp32 multiply per element) | ~0 |

DeltaNet's gated RMSNorm has one weight per head dimension (128), shared by its 32 heads, so it cannot take a
per-channel $1/s$; that one goes into `quant.quant_rows`. Tests check that a folded layer computes what the unfolded
one does in fp32 (to 1e-5) on kev's fused decoder math, for both layer types and α 0.5 and 0.8.

### End to end [M]

Kev's `kev.perf compare`, `fused` against `fused` + `patch=w8a8_sm86.kev_smooth.patch` (or `kev_patch.patch`
unsmoothed), ABBA blocks. Screens on the 8-request subset (seed 0, 78 answers, 4 blocks); the full smoke bundle is
105 requests, 1,198 answers, 2 blocks. Throughput ratios with 90% bootstrap CIs.

| Run | Workload | Candidate | Throughput [90% CI] | p50 | p99 | p99.9 | max \|dp\| | flips | within-1 | mean \|Δlevel\| |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `20261105T122956Z-169e7f` | 8 req | unsmoothed (before) | 1.843 [1.822, 1.864] | 0.055 | 0.150 | | 0.169 | 11 / 78 | 93.6% | 0.122 |
| `20261105T133739Z-4bbb0f` | 8 req | α 0.5 | 1.828 [1.806, 1.848] | 0.021 | 0.090 | 0.093 | 0.093 | 6 / 78 | 97.4% | 0.060 |
| `20261105T141356Z-09f221` | 8 req | α 0.65 | 1.842 [1.822, 1.861] | 0.024 | 0.093 | 0.120 | 0.123 | 4 / 78 | 97.4% | 0.057 |
| `20261105T145346Z-1d1bb0` | 8 req | α 0.8 | 1.829 [1.805, 1.851] | 0.028 | 0.095 | 0.143 | 0.148 | 7 / 78 | 98.7% | 0.057 |
| `20261105T150833Z-bb8f74` | 8 req | α 0.5, `down_proj` bf16 | 1.483 [1.467, 1.498] | 0.021 | 0.073 | 0.074 | 0.074 | 5 / 78 | 97.4% | 0.059 |
| `20261105T151417Z-a5ba22` | 8 req | α 0.5, `out_proj` + `o_proj` bf16 | 1.728 [1.709, 1.745] | 0.017 | 0.089 | 0.103 | 0.105 | 7 / 78 | 97.4% | 0.054 |
| `20261105T145955Z-97c204` | 8 req | α 0.8, down + out + o bf16 | 1.397 [1.382, 1.411] | 0.009 | 0.042 | 0.045 | 0.045 | 4 / 78 | 100% | 0.025 |
| `20261105T153343Z-6bd3b5` | 8 req | α 0.5 on in / gate_up; down + out + o **per 128** | 1.691 [1.673, 1.707] | 0.017 | 0.100 | 0.119 | 0.121 | 6 / 78 | 98.7% | 0.046 |
| `20261105T152030Z-9bc443` | 8 req | **confirm: α 0.5** (fresh process) | **1.838 [1.816, 1.858]** | 0.021 | 0.090 | 0.093 | 0.093 | 6 / 78 | 97.4% | 0.060 |
| `20261105T152755Z-765997` | 8 req | A/A: `fused` against itself | 0.999 [0.999, 0.999] | 0 | 0 | 0 | 0 | 0 / 78 | 100% | 0 |
| `20261105T164325Z-b89a5e` | full | unsmoothed | 1.829 [1.822, 1.836] | 0.051 | 0.222 | 0.331 | 0.469 | 209 / 1198 (17.4%) | 93.0% | 0.142 |
| `20261105T155559Z-58359a` | full | **α 0.5** | **1.808 [1.799, 1.816]** | 0.025 | 0.117 | 0.143 | 0.158 | 103 / 1198 (8.6%) | 96.5% | 0.073 |
| `20261105T162032Z-731619` | full | α 0.8, down + out + o bf16 | 1.388 [1.384, 1.393] | 0.011 | 0.052 | 0.087 | 0.101 | 55 / 1198 (4.6%) | 98.2% | 0.036 |
| gate for numerics-only changes | full | | | | | ≤ 0.037 | ≤ 0.044 | ≤ 2.9% | ≥ 99.25% | ≤ 0.014 |

The α screens differ by less than 78 answers can resolve: all three halve the mean level drift and take out about
half the flips; α 0.5 has the smallest tail (p99.9, max), so it went to the confirm, which reproduced it exactly (the
INT8 path is deterministic) at 1.84×. Smoothing costs no measurable speed: the screens' ratios overlap the
unsmoothed 1.84×. On the full bundle, against unsmoothed W8A8 measured the same way, α 0.5 **halves the flips
(17.4% → 8.6%) and the mean level drift (0.142 → 0.073), and cuts the tail 2.3–3× (p99.9 0.331 → 0.143, max 0.469 →
0.158)**, at 1.81× against 1.83× (per form: 10-K 1.83×, 20-F 1.81×, DEF 14A 1.82×, 10-Q 1.74×). The A/A's offset of
0.1% is small next to every effect here.

Per-128 activation scales on down / out cut those GEMMs' error 3× in the diagnosis, but end to end they bought
less (mean |Δlevel| 0.060 → 0.046) at a cost of 0.15× in throughput: the grouped GEMM rescales a partial sum per 128-wide
K block and cannot use cuBLASLt, so it runs at 1.87–1.93× over bf16 rather than 2.7–3.3× (microbench run
`20261105T152055Z-e6e321`, [`grouped.py`](grouped.py)). Keeping one of down or out in bf16 changed nothing measurable;
keeping both got closest to the gate but gives back most of the speed.

### Verdict

**SmoothQuant gets W8A8 Kev-9B about halfway to the gate, at no cost in speed, and no variant measured here passes
it.** At α 0.5 on 1,198 answers, the drift is still 5× the gate on mean |Δlevel| (0.073 vs 0.014), 3× on flips
(8.6% vs 2.9%) and 3.6× on max |dp| (0.158 vs 0.044). The closest variant (norm-fed GEMMs smoothed in INT8, down / out
/ o in bf16) is still 2.5× the gate on mean |Δlevel| and 1.6× on flips, at 1.39×. The residual INT8 error is
spread evenly over all 64 layers' worth of GEMMs at 1–2% each, so it cannot be removed by keeping a few layers in
bf16.

Recommendation: **do not adopt W8A8 for Kev-9B serving as a numerics-neutral speedup.** If a 1.8× faster 3090 path
is worth a method change, ship SmoothQuant α 0.5 (`kev_smooth.patch`) and judge it by a full dev/test re-score
(accuracy and calibration), not by this drift gate. The next lever, if the drift has to come down further, is
per-128 activation scales on every GEMM with α 0.5 smoothing (0.009–0.024 per GEMM in the diagnosis, about 1.6–1.7×
end to end), or a weight-and-activation fine-tune (QAT), which this repository does not do.

## Caveats

- **It is a method change, not a kernel swap.** Every projection's inputs and weights are rounded to 8 bits. The
  drift above is measured against the bf16 fused model on 1,198 answers at most. Before any production use, Kev-9B
  needs a re-score on the full dev/test bundle (accuracy, calibration), not just this drift check.
- **Only for Ampere 3090 users.** Production runs FP8 on Ada (RTX 6000 Ada), where FP8 tensor cores serve the same
  purpose with a wider dynamic range per value. This kernel does not carry over there; on Ada, use FP8.
- **Calibration overlaps the full bundle.** The 8 calibration requests are outside the 8-request eval subset, but
  they are 8 of the full smoke bundle's 105. They only set per-channel absmax, so the overlap can flatter the full
  run slightly at most. The statistics come from one model revision: recalibrate (`diagnose.py`) after any change of
  weights.
- 78 answers count flips coarsely (one flip is 1.3%): the screens rank variants only roughly; the full-bundle runs
  are the numbers to quote.
- The grouped GEMM is a first Triton version (autotuned over eight tiles), at 44–45% of the INT8 peak at 28k tokens;
  a tuned one would narrow its 0.15× end-to-end cost, not its numerics.
- The Triton GEMM was tuned only by its autotuner over ten tile shapes. cuBLASLt beats it at large M, so there is
  headroom in a hand-tuned sm_86 IMMA kernel, notably the int32 round trip that `_int_mm` still pays at 28k tokens.
- Synthetic microbench inputs (Gaussian) have no outliers. Their relative error, 1.2–1.3%, says nothing about model
  accuracy.

## Reproduce

```bash
# kernels (GPU0 only, under the shared lock)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 flock <gpu-lock> \
    python kernels/w8a8_sm86/microbench.py --M 512 4096 28672 --pairs 10 --csv data/w8a8_sm86_microbench.csv
# confirm the chosen backend alone, in a fresh process
... microbench.py --candidates auto --pairs 10 --csv data/w8a8_sm86_microbench.csv

# end to end (kev repo; kev.perf takes the GPU0 lock itself)
PYTHONPATH=<kev>:<xCalibur>/kernels:<xCalibur> python -m kev.perf.cli compare --model jaredpalmer/kev-9b \
    --bundle sec_bench/v1/smoke/requests.jsonl --subset 8 --seed 0 --warmup 1 --mode blocks --blocks 4 \
    --reference fused --candidate fused --cand-set patch=w8a8_sm86.kev_patch.patch \
    --one-change "W8A8 int8 GEMMs (per-token act, per-channel weight)" --max-dp 1 --max-flip-rate 1
python -m kev.perf.cli aa ... --variant fused          # the session's noise floor
... --cand-set patch=w8a8_sm86.kev_patch.patch_keep_down   # the ablation (down_proj in bf16)

# SmoothQuant: calibrate and diagnose (bf16 fused model; writes ~/.cache/xcalibur/w8a8_sm86_kev9b_absmax.pt)
cd <kev> && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 flock <gpu-lock> \
    env PYTHONPATH=<xCalibur>/kernels:. python <xCalibur>/kernels/w8a8_sm86/diagnose.py \
    --bundle sec_bench/v1/smoke/requests.jsonl --calib 8 --csv <xCalibur>/data/w8a8_sm86_diagnosis.csv
# SmoothQuant end to end (add --subset 8 --blocks 4 for a screen; the full bundle with --blocks 2)
python -m kev.perf.cli compare ... --cand-set patch=w8a8_sm86.kev_smooth.patch --cand-set env.W8A8_SMOOTH_ALPHA=0.5
... --cand-set env.W8A8_SMOOTH_ALPHA=0.8 --cand-set env.W8A8_KEEP=down_proj,out_proj,o_proj     # mixed precision
... --cand-set env.W8A8_SMOOTH_SITES=in,mlp --cand-set env.W8A8_GROUP=down_proj,out_proj,o_proj  # per-128 scales
# grouped GEMM speed
... microbench.py --M 512 28672 --shapes deltanet.out_proj mlp.down_proj --candidates auto grouped --pairs 10

# tests (the CPU ones run anywhere; the GPU ones need a CUDA card: GPU0 under the lock on the 3090 box)
python -m pytest tests/test_w8a8_sm86.py tests/test_w8a8_sm86_smooth.py tests/test_w8a8_sm86_grouped.py
```

Data: [`data/w8a8_sm86_microbench.csv`](../../data/w8a8_sm86_microbench.csv),
[`data/w8a8_sm86_diagnosis.csv`](../../data/w8a8_sm86_diagnosis.csv).

Apache 2.0, like the rest of the repository ([LICENSE](../../LICENSE)).
