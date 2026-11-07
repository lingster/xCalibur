# Construction, search and export for kernel campaigns

What to build, how to search it and how to ship it, from xCalibur's
[formulation](https://github.com/xArmory/xCalibur/blob/main/docs/formulation.md),
[search](https://github.com/xArmory/xCalibur/blob/main/docs/search.md),
[studies](https://github.com/xArmory/xCalibur/blob/main/docs/studies.md) and the writeups
([AUTORESEARCH](https://github.com/xArmory/xCalibur/blob/main/docs/writeups/WRITEUP_AUTORESEARCH.md),
[TG4DAC](https://github.com/xArmory/xCalibur/blob/main/docs/writeups/WRITEUP_TG4DAC.md),
[SM86/SM89](https://github.com/xArmory/xCalibur/blob/main/docs/writeups/WRITEUP_SM86_SM89.md)). The general method
behind each section is in `ooda-coach`'s [agenda](../../ooda-coach/references/agenda.md),
[evidence](../../ooda-coach/references/evidence.md) and [proposers](../../ooda-coach/references/proposers.md).

## 1. The objective is the complete operation

- Contract C = (operation, hardware, workloads, numerics). Objective J = Σ wₚ·log Tₚ over the declared workloads
  (geomean), including unchanged regions and boundary costs. Component timings don't add when stages overlap;
  memory-for-recompute and fusion change lifetimes.
- A complete operator includes everything the workload runs: for training, forward **and backward**, saved tensors,
  casts, gradient accumulation and stream joins. State the timing scope (e.g. fwd+bwd+accum, zero B, no optimizer).
  Test degenerate and non-degenerate parameter cases (LoRA with zero and non-zero B).
- Defining an optimum doesn't show the search found it; passing finite fixtures is a result with that scope.

## 2. Exact layer before timing

The cheapest kernel to reject is the one you never build. In order, before any GPU time:

1. **Plan accounting:** `warpspeed_check(squads, resources, arch=...)` returns pass with every derived number or a typed
   counterexample (`tma_arch`, `wgmma_arch`, `setmaxnreg_arch`, `smem_*`, `sm_threads`, `sm_ctas`, `wgmma_alignment`).
   Formulas: arrivals = 32·Σ warps; smem = Σ kᵣbᵣ + 8Σ kᵣqᵣ + alignment + extra;
   launch register cap = min(255, ⌊65536 / (128·⌈W/4⌉·c)⌋₈) (9 warps → 168, 7 → 255); `setmaxnreg` needs whole
   warpgroups.
   **Accounting, not a performance model**, and not proof of deadlock freedom, memory ordering or lifetimes.
2. **Ceilings for this board:** `ARCHS` tables. GeForce runs bf16 with fp32 accumulate at **half rate** (3090: 71,
   not 142 TFLOPS); unverified datasheet values are marked as such.
3. **Reach:** `ReachMap` (validated against captured builds; dispatch can change the build when kernel fields don't).
4. **Compile:** an error is a contradiction for that lowering in that context, not for the design.
5. **SASS identity:** SASS identical to the parent ⇒ exact no-op, never timed (in one game: 38 no-ops and 18
   contradictions among 325 archived designs before any GPU time). SASS also gives exact features (registers, spills).
   A matching SASS hash still isn't the same packing, launch, dispatch or workload.
6. **Inspect compiled allocation:** a plan can pass accounting and still spill once compiled (a design assumed 224
   registers; the cap was 168). Spills next to a regression are a plausible contributor, not a measured cause.

## 3. Constructions worth trying (by mechanism)

- **Ownership and lifetimes first:** write who owns each value, where it lives, when it is ready and dead
  (ready ≺ consume, lastUse ≺ overwrite). Check **coverage**, not just shapes (same shape, broken ownership ⇒
  duplicate writers). Barriers are part of the design.
- **Coupled moves:** ownership, fragments, packing and staging often have to change together; testing one field
  against its old consumer rejects an unfinished construction. Register the bundle as one change.
- **Structure beats tuning:** retuning gave 1.5–2.4×; warp-specialised structure (load and compute squads,
  producer/consumer rings) gave 2.9–4.6×. The starting point decides the headroom: from a strong hand kernel,
  expect little.
- **Bytes beat loads at the roofline:** a TMA ring can only tie cuBLAS; beating it needs fewer bytes (lossless
  compression, smaller formats). In-register decode can lose at large token counts.
- **Reuse must be a program value:** equal addresses show an opportunity; a constructor must actually share the load.
- **Derive, don't search:** exact GF(2) decoders (`gf2`): equality by basis images, packing T = D⁻¹P, exhaustive
  legal joins (5,989,005 raw combinations → 1,816 legal graphs in 1.3 ms). Finite alphabets (FP4, scale, BF16→FP8
  splits) can be checked exhaustively; a scalar proof is not a matrix-kernel proof.
- **Cross-CTA handoff:** one owner per CTA releasing through a named barrier beat 8 warps polling one flag (+2.2%).
- **Regime awareness (what didn't transfer):** weight-only decoders help memory-bound decode and are expected to lose on
  compute-bound prefill (unmeasured on sm_86/89); Hopper features (TMA, wgmma, setmaxnreg) are absent on sm_86/89; NVFP4
  is Blackwell-only.

## 4. Search the space

- Construct before searching: eliminate determined variables, join boundaries, share proofs across equal
  constructions, cache with full context (source, toolchain, target, flags, enclosing program; for timing also
  packing, dispatch, launch, workload and protocol).
- `Game` loop: explore until a ranker fits (≥12 observations per load) → reach-filter nominations across loads →
  compile a bounded batch (~2× what you'll time) and re-rank on compiled features → screen → confirm next to the
  frozen incumbent → bind by validated build identity. One candidate nominated for several loads is measured once;
  one round gives at most one confirmation per (load, build identity). `Game.confirm` for leaders found elsewhere.
- **Joint sweep, then halving:** sweep the legal grammar with illegal designs rejected by contract, compile on CPU in
  parallel, screen on several GPUs, then successive halving at full confirm cost (e.g. 12 seeds × 2).
- **Neighbourhood sweep** of the incumbent (514 designs: ranked #2 on screen, best at confirm ⇒ near a local optimum
  of what was screened). Then look for new constructors rather than knob values.
- **Recombine on purpose:** +0.4% and +0.9% alone gave +3.3% together; moves that lost alone stay allowed inside
  combinations; moves with an exact witness are never re-proposed.
- **Per-size builds:** give a load (e.g. M = 32 tokens) its own build when its winner hurts others; dispatch is part
  of the design.
- **Admissibility:** ≥0.5% of the load's measured time. Required gain split by headroom is a prioritiser, not proof.
- **Binding variants seen in practice:** 1 + 2×worst A/A (consumer cards); max(3×A/A, 0.5%) with **no load below
  −0.5%** (multi-load game); 0.3% per load at 12×2 confirm. Pre-register which one, and use the margin of the mode
  that actually ran (K-14).

## 5. Timing details

- **Null subtraction:** `Bench` keeps it **off** by default. CUDA-graph replay has a fixed cost (16–19 µs in one
  harness) that compresses µs ratios; subtract `null_us()` only with a justified calibration that matches the judge
  or production timer, and keep the raw samples either way.
- **Measure only what can move:** when a candidate reaches one load, spend samples on that load (12 seeds on one load
  instead of 4 seeds × 6 loads, about half the cost).
- **Protocols don't mix:** direct launches with stream events and a read flush aren't pooled with graph runs.

## 6. End to end

- **Diagnose the baseline:** check whether it runs a generic fallback (vLLM fell back to a generic Triton kernel for
  mixed head sizes 256 and 512); the gap you measure may be the fallback's.
- **Differencing:** time 257 and 1 tokens and divide the difference by 256 to drop prefill and fixed costs.
- **Reconcile:** the end-to-end delta should match the component prediction (5.3 ms measured vs 5.4 ms predicted);
  a fixed per-step cost (weights, LoRA, output layer) caps the end-to-end gain (Amdahl).
- **Verify the run did what you think:** e.g. from the scheduler, that all sequences decoded together.
- **Numerics under nondeterminism:** greedy tokens on random prompts diverge within a few tokens even between the
  reference's own configs; use kernel-level checks or labelled tasks, and judge a candidate's error against exact
  values relative to the reference's own error.
- **Attribute contributions separately** against the same baseline config (e.g. attention alone 1.00–1.02× at short
  context, 1.11–1.17× at long).

## 7. Export: ship exactly what was measured

- `Capture()` records every build's source units, ordered flags and bindings in a dry pass; `Capture.same` compares
  the multiset. It establishes input equality only.
- Edit by AST with `replace_assignment` (byte-preserving, rejects ambiguous targets), then verify the hash.
- Compile per-size builds in `prepare()`, never on the timed path. `lint` the shipped file for syncs, `.item()`,
  side streams, graph capture, timing code and probe/rider/flush scaffolding. `rules_gate` before submitting.
- Predict the exported file's score with a parity bench, then check it on the judge (+1.65% predicted, +1.9%
  measured). A check run is a real candidate, not a probe of the board.

## 8. Proposers for kernel work

A planner (e.g. Gemma 4 with planner/refiner LoRAs via `PlannerLoop`) proposes typed plans; `warpspeed_check` is its
checker tool; the refiner repairs from counterexamples (fine-tuned on ~30 worked tool-run examples from 6 families,
attention held out). Give it the whole design graph listing (~735 nodes, one line each), retrieved neighbours with
evidence grades ("4-seed screen" vs "12×2 confirm"), the code lines each knob controls, and its own past plans and
results. Track legal-but-degenerate plans (7 of 16 were a single squad). Compare against the LightGBM ranker and a
random legal sampler at matched compile and GPU budget.
