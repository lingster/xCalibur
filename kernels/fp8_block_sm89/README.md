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

## Round 2: what the Ada run of ce74c95 showed, and the fixes

Data: the Ada run of 2026-11-05 (one card, 15:42–16:32Z, with the other card serving production at 100%).

**Failures and fixes.**

- *`test_e4m3_rounding_edges_match_torch`:* the hardware cast keeps the sign of zero (−0.0 → code 0x80, as torch
  does). `reference.to_e4m3_sm89` and the stand-in `_round_e4m3` returned +0. Both now copy the sign bit. The fp8
  path didn't change: `aot.py --signature` shows the same SASS as ce74c95 (tested), so tables measured on ce74c95
  stay valid (`kernel.FP8_EQUIVALENT_SHAS`). The GEMM is unaffected either way: a ±0 product adds nothing.
- *`test_parity_..._flags_a_real_change`:* sm_89's QMMA doesn't accumulate in IEEE fp32. An order-only build
  (smaller K tile, split-K) is not within one bf16 ulp of stock on hardware; the full down_proj search's split-K builds
  were 4–64 ulp off. The test now asserts the class boundary that holds on every dot. The search's parity gate
  (`ship.parity_gate`) requires **bitwise** equality for `--bitwise-only`, not just "ok".
- *`cudaErrorStreamCaptureInvalidated` in `--confirm-table`:* `confirm_table` timed every bound design at every
  load, so the m64 winner (64×32, 5 stages) was also captured at m28k, where its capture was invalidated. The
  config is legal (smem 48 KB; it ran at m64 in the same session). The failure was caught, but the dead capture's
  state wasn't cleaned up, and the next shape's A/A capture (outside any `try`) killed the process. The fix:
  `safebench.SafeBench` runs each graph eagerly once before capture, captures after `gc` and `empty_cache` (so the
  allocator never calls `cudaFree` inside a capture) in `thread_local` mode, and on failure recovers and runs a
  health check. If that check fails it raises `CudaPoisoned`, and the search saves and exits 2. The session marks
  the design illegal for the session. Confirms time each design only at its bound loads (`Session.restrict`).
  `confirm_table` writes every entry as pending (config null) until it binds, so a crash leaves no unconfirmed build.
- *`confirmed_bitwise.json` is incomplete:* the 8 gate/up and down_proj entries still carried their search configs
  with no `confirm` record, and `eff_tuned` used them. `ship.py` keeps only entries that are confirmed, bitwise on
  hardware against the hub, per-Linear shapes, and faster in steady state. That gives
  `tables/ada_rtx6000_bitwise_r1.json`: 17 entries. The 3 concat-group winners are dropped because production
  doesn't concatenate FP8 weights.
- *Hub revision:* transformers loads the v4 head `03ae3193` (kernel commit db1270d), not `8c178950`. Its
  `matmul.py` and `utils.py` are byte-identical (the digests in `hub_metadata.json` match `stock.HUB_SOURCE_DIGESTS`,
  now checked at load). The twin check is bitwise 27/27, and the 240 cached production cubins all issue native
  `QMMA.16832.F32.E4M3.E4M3` with inline `F2FP` quantisation. The autotuner picked 2 stages at BLOCK_M 128 (the
  only legal choice), as the AOT analysis said.

**"+100%" vs "−7.7%".** The search and confirm gains are a timing artefact plus a different unit. `Bench.paired` timed
one cold replay per sample after a synchronise and an L2 flush. On this 300 W card (in power violation 78% of busy
samples, SM clock 0.6–2.8 GHz) that produced tuned builds at 240–300 TFLOPS (cuBLAS sustains ~200). Stock read 37%
slower next to a candidate (25.4 ms) than in its own A/A (18.5 ms). The pair's total is conserved within 2–6%
against the steady-state sum, so time shifted between neighbours rather than being saved. Steady state
(efficiency.py, back-to-back replays) gives the true per-shape gains: ×1.06–1.09 at m28k, ×1.06–1.10 at m4k,
×1.26–1.52 at m512, ×1.4–2.5 at m64. "+100%" is also a throughput gain (stock/tuned − 1) at one shape, while
−7.7% is per-request time over all shapes, including the stock-kept and unconfirmed entries. `SafeBench`
now times in steady state by default: blocks of back-to-back replays, weights rotated past the 96 MB L2. It drops
any pair whose stock time leaves its own A/A by more than 10%.

**Why only ~8%, not ~19%.**

- *What bound:* nearly every m28k and m4k bucket bound the hand anchor: quantise-once, 64×128×128, 4 warps,
  3 stages. At m512, 128×128 with 8 warps and 3–4 stages also won. At m64, 32/64×32–64 tiles with 5 stages won.
- *Stages:* 3–5 stages were legal with quantise-once and were used. The 128×128 / 8-warp / 4-stage anchor never
  won at m28k.
- *Quantise-once:* it was in the bitwise space. Every winner uses it, and it is bitwise on hardware at every bucket
  (max_ulp 0). It costs its own pass: 0.57 ms per call at K = 5,120 (3% of the GEMM) and 2.0 ms at K = 17,408
  (6.5%), because each GEMM re-quantises its input.
- *What limits the kernel (SASS, `aot.py`):* on a power-capped card every non-MMA instruction costs clock. Stock
  issues 13.9 instructions per QMMA: 7.1 fp32 for the in-loop amax, divide and scale, 1 F2FP, 0.5 PRMT. Quantise-once
  at 64×128 issues 10.0 per QMMA: 2.0 fp32 for the per-128 scale epilogue, which the bitwise contract fixes, plus
  2.7 PRMT byte permutes of the fp8 A tile from shared memory and 2.4 int. 128×256 / 8 warps issues 8.8. cuBLAS
  per-tensor has no per-K scale epilogue at all. Measured: stock 161 TF, once 64×128 ~180 TF GEMM-only (174 with its
  quant pass), cuBLAS 202. With the bitwise per-128 scale epilogue that leaves roughly 10% to cuBLAS, not 19%.
- *Power sharing:* GPU1 ran at its own 300 W cap (mean 268 W, mean SM clock 1.6 GHz while busy). Absolute TFLOPS
  match the earlier probe (stock 161). Ratios within a session are valid; the cold single-replay protocol was the
  one regime it broke.

**Next levers, in order** (bitwise unless noted):

1. Finish the confirm for gate/up/down_proj with steady timing (`search.py --confirm-table`, then parity, efficiency and ship; done in round 3). They are 70% of the GEMM FLOPs.
   gate/up alone are 47%, and their steady ×1.05 is worth ~2.3% of GEMM time. down_proj is not faster while it
   pays its own 2 ms quant pass (164 vs 166 TF).
2. Share one quantisation per input: q|k|v, gate|up and in_proj_qkv|z each read the same input, so one `act_quant`
   per group removes 2 of 3 or 1 of 2 quant passes (`kernel.linear_group`; ~2% of GEMM time). down_proj's quant
   (6.5%) would need fusing into the SwiGLU that produces its input, which is a kev change.
3. A 128×256 / 8-warp / 3-stage tile (8.8 instructions per QMMA vs 10.0; legal at 99 KB) and a weight layout
   pre-permuted at load time, so the B operand needs no PRMT. Expected ×1.05–1.10 more at m28k, i.e. 185–195 TF.
   Both keep each output's summation order: bitwise.
4. Persistent or stream-K scheduling for down_proj (K = 17,408). It is not bitwise if it splits K, so it needs the
   answer gate. Not before 1–3.

Estimate with 1–3 (not measured): tuned ≈ 185–195 TF at m28k, about −12% GEMM time per request vs stock,
≈ −10% end to end.

**Production path** (bitwise only): `kev_patch.py`. With `KEV_FP8_TUNED=1`, `FP8Linear.forward` runs the kit's kernel
for (N, K, M-bucket) entries in the shipped table and transformers' own forward for everything else. It loads
through a launcher (`kev_launch.sh` as `KEV_PYTHON`) or a kev.perf `patch` hook (`kev_patch.patch`, per model). It
refuses tables that aren't ship.py bitwise tables or come from a kernel with a different fp8 path, and it only
installs on sm_89 without DeepGEMM. `KEV_FP8_TUNED_VERIFY=1` checks every hit against stock, bit for bit. Run plan: parity
preflight on the card, a verify pass in the real server, then ABBA pairs gated at |dp| = 0 (results in round 3).

**New files:** `safebench.py` (capture-safe, steady-state bench), `session.py` (the search's hardware half, split out
of search.py), `ship.py` (what may ship), `kev_patch.py` and `kev_launch.sh` (the production hook),
and `tables/ada_rtx6000_bitwise_r1.json`. Tests: `tests/test_fp8_block_{robust,ship,kev_patch}.py`.

**Changes for `xcalibur.Bench`** (made here as a wrapper so the port carries cleanly; the library should adopt them):
(1) `graph()` should run `fn` once eagerly and synchronise before capturing, free cached blocks first, capture in
`thread_local` mode, and on failure synchronise, release memory, health-check and raise a typed error;
(2) a steady-state `paired` mode (blocks of back-to-back replays) for power-capped cards and kernels of milliseconds;
(3) an A/A drift guard: report pairs whose reference time leaves the session's A/A reference time.

## Round 3: the tables in production replay, combining rounds, and review fixes

**E5** (round-2 table alone, 6 MLP and N=1024 entries) against the production FP8 path: ×1.015, ×1.016, ×1.016 in
three ABBA pairs on the full smoke set (105 requests, 599 answers). A/A ×1.000 both ways. max|dp| = 0 in every pair:
the answers are bit-identical.

**E6** (both rounds combined, 21 entries): ×1.049, ×1.052, ×1.049 in three ABBA pairs (2,691–2,695 → 2,826–2,830
tok/s), A/A ×0.999 and ×1.001, max|dp| = 0 on all 599 answers. That is **+5.0% end to end with bit-identical answers**,
above the pre-registered ×1.020–1.035. The prediction weighted round 1 by its share of FLOPs. It should have weighted
by measured time per M bucket: round 1's small-M entries (m64/m512, ×1.4–3.0 steady) carry the question rows, which
run as short passes.

**Combining rounds:** production loads one table, so `combine.py` merges ship.py tables from separate rounds.
- It checks the rules `kev_patch.py` enforces at load time up front: shippable, every config in the bitwise class,
  every kernel hash on this fp8 path.
- Later tables win per (N, K, bucket).
- `tables/ada_rtx6000_bitwise_r2.json` holds round 2; `tables/ada_rtx6000_bitwise_r1r2.json` combines both rounds
  (17 + 6 − 2 overlapping = 21 entries).

**Review fixes** (none can change answers, since only bitwise entries ship, but each could overstate speed):
1. **Margin.** `ship.py` keeps a steady win only above 1 + max(`--steady-margin` 0.01, the entry's confirm-session
   A/A margin), not merely above 1.0 (winner's curse). All 23 entries of r1 and r2 still clear it.
2. **Cold weights.** `efficiency.py` adds `rot_ms` to stock and tuned rows: device time per call while rotating
   enough weight copies to overflow L2, as production reads each layer's own weights. `ship.py` decides on `rot_ms`
   when present and records the clock (`steady_clock`); warm-L2 `graph_ms` overstates small-M gains. Re-run
   efficiency on Ada to re-check the m64/m512 ratios.
3. **Kernel hash.** `search.py --confirm-table` refuses a table searched on another fp8 path and keeps the original
   hash as `searched_kernel_sha`. `ship.py` refuses sources whose hash is not `fp8_path_compatible`.

Tests: `tests/test_fp8_block_{combine,ship,robust}.py`.

## Credits and license

- `hub_ref.py` is copied verbatim, and `kernel.py` adapts, the `kernels-community/finegrained-fp8` kernel at revision
  `8c178950e8710e26a2210e4e909cd02dc16c8715` (`build/torch-cuda/matmul.py`, `utils.py`). Copyright 2025–2026 The
  HuggingFace Inc. team, Apache License 2.0; notices are kept at the top of both files. Our changes: tile knobs,
  quantise-once, split-K, stand-in dots for hardware without FP8.
- `public/vllm_l40s_block_fp8.json` holds four of vLLM's tuned block-FP8 tables for NVIDIA L40S, copied unchanged
  from `vllm-project/vllm` at `bdfb02674fc0eaa2b9023420b02a8d4e23a69957`
  (`vllm/model_executor/layers/quantization/utils/configs`), Apache License 2.0.
- The rest is Apache 2.0, like the repository ([LICENSE](../../LICENSE)).
