# Block-FP8 GEMM per-shape tuning on RTX 6000 Ada (sm_89)

A kit for one question: production serves Kev-27B in FP8 on RTX 6000 Ada cards ([sm_86/sm_89 writeup](../../docs/writeups/WRITEUP_SM86_SM89.md)), and every FP8 Linear
runs one Triton GEMM. How far is that GEMM from what the card can do, and how much of the gap closes **without
changing a single output bit**? It holds an equivalent kernel with its tiling opened up, xCalibur's measured game
over that tiling (one build per shape and M range), a parity check, and the commands for an Ada card
([RUN_ON_ADA.md](RUN_ON_ADA.md)). Everything except the timing runs and is tested without FP8 hardware.

## The computation

For activations $x \in \mathbb{R}^{M \times K}$ (bf16) and FP8 weights $W$ with 128 × 128 block scales $\sigma^W$:

$$
s_{m,g} = \frac{\max_{k \in g} |x_{m,k}|}{448}, \qquad q_{m,k} = \operatorname{e4m3}\!\Big(\frac{x_{m,k}}{s_{m,g(k)}}\Big), \qquad
C_{m,n} = \operatorname{bf16}\Big(\sum_{g} s_{m,g}\, \sigma^W_{\lfloor n/128 \rfloor, g} \sum_{k \in g} q_{m,k}\, W_{n,k}\Big),
$$

with groups $g$ of 128 along K and fp32 accumulation. This is transformers' `FineGrainedFP8Config` (dynamic
activations, `block_size = [128, 128]`); on sm_89 the GEMM is the `kernels-community/finegrained-fp8` Triton kernel
`w8a8_block_dynamic_fp8_matmul` (revision `8c178950`). Within a group, products of two e4m3 values are exact in fp32,
so two kernels with this contract can differ only by fp32 summation order and the final bf16 rounding.

The shapes (`shapes.py`): 10 FP8 projections per layer kind, 7 distinct (N, K), 48.65 GFLOP per token, called once per
layer of their kind per pass. A request is a ~28k-token state pass plus ~8 question-row passes of ~40 tokens each,
or one packed row pass (B4).

## What the stock kernel does on sm_89 (exact, no Ada needed)

`aot.py` cross-compiles a kernel for sm_89 on any machine (Triton's `GPUTarget("cuda", 89, 32)` and bundled
`ptxas`), and `sass.py` reads the machine code. Run on the hub kernel's own source (`hub_ref.py`, copied verbatim):

| | stock (fused quant), 128×128×128 tile | same tile, quantise-once |
|---|---|---|
| MMA in the K loop | 64 × `QMMA.16832.F32.E4M3.E4M3` (native e4m3) | the same 64 |
| SASS instructions per 128-wide K tile | 892 | 634 |
| fp32 ops / converts / MUFU per K tile | 456 / 64 `F2FP` / 8 | 128 / 0 / 0 |
| shared memory at 2 / 3 / 4 stages | 64 / 112 / 160 KB | 32 / 64 / 96 KB |
| stages that fit the 99 KB per-block limit | **2 only** | 2, 3, 4 |

Three facts follow, all exact:

1. **The MMA is native.** Unlike FP8 `mma.sync` on H200, which an earlier study found lowered to conversions plus FP16 MMA, sm_89
   issues real e4m3 MMA.
2. **The stock kernel quantises inside the K loop**, and every one of the N / 128 column programs re-quantises the same
   A tiles: about 40% more instructions per tile than the MMA work needs.
3. **For every M > 64 the stock tile can only pipeline 2 stages.** It stages bf16 A plus the quantised tile, so stages
   3 and 4 overflow shared memory and the hub autotuner silently drops them. vLLM's tuned L40S table for our exact
   `in_proj_qkv` shape picks 128×128×128, 8 warps, **4 stages** at M ≥ 4096, with A quantised beforehand.

One numerics detail of stock, kept everywhere: Triton lowers fp32 → e4m3 on sm_89 as `F2FP.F16.F32.PACK_AB.RZ` then
`F2FP.SATFINITE.E4M3.F16`. It truncates to fp16, then rounds to nearest, so a value just above an e4m3 tie can round
down (8.5 + 2⁻¹² → 8). The fp8 path uses the same cast; the stand-ins and `reference.py` emulate it. Likewise
Triton's fp32 `/` is `div.full.f32` (within 2 ulp) in both kernels. Ours therefore matches stock bit for bit, while
against the IEEE-dividing CPU reference a rare code on a rounding boundary moves one e4m3 step.

## The Ada probe (2026-11-05) [M]

On one RTX 6000 Ada at its 300 W cap, with transformers' `FP8Linear` against `torch._scaled_mm` (cuBLASLt, per-tensor
scales, no activation quant):

- **M = 28,672:** stock 150–162 TFLOPS on every shape, cuBLAS 190–212, bf16 96–109. Per request, the GEMMs take
  8,967 ms (stock) against 7,222 ms (cuBLAS). The card's practical FP8 ceiling is about 210 TFLOPS: 29% of the 728.5
  datasheet.
- **M = 64:** about 0.17 ms per call on every shape, so 4 TFLOPS on k/v_proj. The probe times with CUDA events around
  Python, so this includes host dispatch; `efficiency.py` separates host and device time.

## The kernel (`kernel.py`)

One Triton kernel with the stock contract and these knobs, as `TileConfig`:

| Knob | Values | Stock |
|---|---|---|
| `quant` | `fused` (bf16 A, quantised per K tile in the loop); `once` (fp8 A + scales from `act_quant`, shared by GEMMs that read the same input) | fused |
| `BLOCK_M`, `BLOCK_N` | 16–256, 32–256 | min(max(next_pow2(M), 16), 128), 128 |
| `BLOCK_K` | 32, 64, 128 (divides the 128 group, ≥ the k32 of `m16n8k32`; `fused` needs 128) | 128 |
| `GROUP_M` | 1, 4, 8, 16, 32, 64 (L2 swizzle) | 8 |
| `SPLIT_K` | 1, 2, 4, 8 (fp32 partials, summed in fixed order: deterministic) | 1 |
| `num_warps`, `num_stages` | 2–8, 2–5 | autotuned over 2–16 × 2–4 |

`dot` picks the matrix instruction: `fp8` (sm_89), or a stand-in that runs the same quantised values without FP8
hardware: `bf16` on sm_86 (e4m3 values are exact in bf16; Triton has no `fp8e4nv` type at all below sm_89, so codes
travel as `uint8` and are decoded in-kernel) and `fp32` under Triton's CPU interpreter.

**Concat groups (`concat.py`)** are a separate lever. q|k|v, gate|up and in_proj_qkv|z read the same input, and every
N is a multiple of 128, so their FP8 weights and block scales concatenate along N exactly. The result is one launch
instead of 3 or 2, with outputs bitwise equal to the separate calls. This works with the stock kernel too, with no
new kernel.

## The numerics contract

A tuned build must match stock to fp32 summation order (`parity.py`). Two classes:

- **Bitwise** (`space.bitwise_safe`: `BLOCK_K = 128`, no split-K). Each output sums the same products in stock's
  order, whatever the M / N tiles, `GROUP_M`, warps, stages or quant path. The tests show it bitwise on the
  stand-ins; on Ada, `parity.py` checks every tuned entry (and `--twin-check` our stock-tiling build against the
  hub kernel itself). This class holds the main levers (quantise-once, 3–4 stages, tile shapes):
  `search.py --bitwise-only`.
- **Order-only** (smaller K tiles, split-K): at most one bf16 ulp from stock. The [writeup's](../../docs/writeups/WRITEUP_SM86_SM89.md#fp8-on-ada) ADA2 run saw FP8 amplify
  one-ulp upstream differences into answer drift, so these builds need the answer gate before they ship.

## The search (`search.py`)

xCalibur's measured game ([measured search](../../docs/search.md#measured-search)) per (N, K), with M buckets as loads and stock as
the incumbent. Each bucket binds its own build (as the [H200 MoE's per-size builds](../moe_w4a16_h200/README.md) do: each problem size its own build).

1. **A/A** per bucket. The margin is `xcalibur.aa_margin` over the A/A ratio and its CI, never below 0.5%.
2. **Seeds:** stock, vLLM's L40S picks nearest the shape (`public.py`), hand anchors, and random legal designs.
   Seed leaders are confirmed next to stock.
3. **Rounds:** the surrogate ranks per bucket (LightGBM, or a numpy ridge ensemble when LightGBM is missing). Exact
   sm_89 compiles (`aot.py`) decide legality and give features (registers, spills, shared memory, the SASS mix).
   Screens nominate, confirms decide, and evidence binds (≥ 2 samples, mean above the margin, none negative).
4. **Winner's curse:** every winner is re-timed in 3 fresh confirm sessions (A/A first) and must bind again on that
   evidence alone. `--confirm-table` repeats this later in a new process.
5. **Parity** against stock. A build that changed the math is dropped. Then the table: `{N, K, bucket, config |
   null}` with gains, sample counts, A/A, times and the kernel's source hash.

Timing is `xcalibur.Bench.paired` (CUDA graphs, cold L2, ABBA, null replay subtracted). By default `once` designs pay
their quant kernel on every call; `--quant-cost shared` drops it, for shapes whose group shares one quantisation.

## Files

| File | Role |
|---|---|
| `shapes.py` | Kev-27B FP8 GEMM inventory, call counts, M buckets, concat shapes |
| `space.py` | `TileConfig`, the design space, legality rules (smem calibrated on sm_89 compiles), features |
| `kernel.py` | the GEMM, `act_quant`, `linear_group` (quantise once per input) |
| `concat.py` | concat groups: one exact GEMM per shared input |
| `reference.py` | the math in torch: quantisers, dequantise → fp32 matmul |
| `stock.py` | the reference: the hub kernel (Ada) or its autotuned twin (anywhere) |
| `hub_ref.py` | the hub kernel's source, verbatim, for AOT and SASS only |
| `aot.py`, `sass.py` | sm_89 compiles on any box; SASS inner-loop instruction mix |
| `plan.py` | the game wiring: per-size reach, A/A margin, seeds, play, fresh confirm |
| `search.py`, `parity.py`, `efficiency.py` | the three Ada runs |
| `timing.py` | eager (host + device) vs graph (device) clocks |
| `public.py`, `public/` | vLLM's tuned L40S block-FP8 tables, as seeds |
| `table.py` | the tuned table: lookup by (N, K, M) |

Tests (`tests/test_fp8_block_*.py`, no GPU needed): the kernel against the CPU reference under Triton's interpreter
(bitwise quantisation with e4m3 edge cases and sm_89 double rounding, every knob, split-K, concat, parity); sm_89
compiles of our kernel and the hub's (native MMA, the smem rule, stock's 2-stage limit); and the game plumbing with a
planted winner and a winner's-curse decoy.

## Results

Pending the Ada runs in [RUN_ON_ADA.md](RUN_ON_ADA.md). Nothing above the exact compile facts is claimed yet.

## Credits and license

- `hub_ref.py` is copied verbatim, and `kernel.py` adapts, the `kernels-community/finegrained-fp8` kernel at revision
  `8c178950e8710e26a2210e4e909cd02dc16c8715` (`build/torch-cuda/matmul.py`, `utils.py`). Copyright 2025–2026 The
  HuggingFace Inc. team, Apache License 2.0; notices are kept at the top of both files. Our changes: tile knobs,
  quantise-once, split-K, stand-in dots for hardware without FP8.
- `public/vllm_l40s_block_fp8.json` holds four of vLLM's tuned block-FP8 tables for NVIDIA L40S, copied unchanged
  from `vllm-project/vllm` at `bdfb02674fc0eaa2b9023420b02a8d4e23a69957`
  (`vllm/model_executor/layers/quantization/utils/configs`), Apache License 2.0.
- The rest is Apache 2.0, like the repository ([LICENSE](../../LICENSE)).
