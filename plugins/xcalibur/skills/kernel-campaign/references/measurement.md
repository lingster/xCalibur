# Measurement protocol for kernel and inference speedups

Why each rule exists is in the parentheses; the incidents are in [starter-lessons.md](starter-lessons.md). The
xCalibur sources are the [measurement contract](https://github.com/xArmory/xCalibur/blob/main/docs/getting-started.md#measurement),
[measured search](https://github.com/xArmory/xCalibur/blob/main/docs/search.md#measured-search) and the
[evidence chain](https://github.com/xArmory/xCalibur/blob/main/docs/search.md#evidence).

## Contents

1. The contract
2. Paired ABBA timing
3. Noise floor and margin
4. Intervals
5. Screen → confirm → bind
6. Choosing the timer
7. Snapshots and contamination
8. Instruments and parity
9. Numerics gates

## 1. The contract

Fix and record before measuring: model and precision; workload (real requests, stratified; plus a synthetic grid
if useful); metric (e.g. input tokens per GPU-second, per-request latency); reference (commit-pinned, production's
path); environment (library versions; check differences from production numerically); GPU (name, index, power cap).
Changing the reference, workload or protocol starts a new comparison; old ratios don't carry over.

## 2. Paired ABBA timing

- Candidate and reference run in the **same session**, interleaved **A B B A** (cancels linear drift: clocks,
  temperature).
- Record the ratio per pair: `t_ref / t_cand` (> 1 = candidate faster). Keep every raw sample.
- **Same-process** mode (both variants loaded) gives the tightest pairs. **Blocks** mode (reload between blocks,
  when two copies don't fit) gives wider intervals and needs its own A/A and margin.
- Features read from env vars at call time need blocks mode or an in-process switch that is actually exercised.

## 3. Noise floor and margin

- **A/A every session:** the reference against itself, same mode and pairs. A reload-based A/A (new memory
  addresses) is the honest floor; two agreeing in-process controls can understate noise many-fold.
- Margin = 1 + 2 x the worst A/A offset of the session (at least the mode's known noise). An A/A whose CI excludes 1.0
  by 0.1% is "biased but tiny": widen the margin, don't ignore it.
- Repeat noise of an end-to-end replay (e.g. HTTP) is usually larger (≈0.7% in the worked example); require ≥3 ABBA
  pairs there.

## 4. Intervals

- Bootstrap the paired ratios. **Cluster by request/input id** when inputs repeat across pairs; resampling single
  ratios treats repeats as independent and makes the CI too narrow. Stratify by shape only when every stratum has
  ≥5 inputs (stratifying 1–2-input shapes also narrows the CI falsely).
- Report: median ratio [90% CI], token- or time-weighted ratio [CI], per-shape ratios, throughput both sides, n pairs.
- `xcalibur.bootstrap_ci` gives percentile intervals; the clustered form (`clusters=`) is in the sm_86/sm_89 port if
  available. Otherwise group by cluster and resample groups by hand.

## 5. Screen → confirm → bind

- **Screen:** cheap, maybe a proxy or subset; nominates only. Tag numbers "screen".
- **Confirm:** next to the reference, with the same-session A/A, enough pairs, on the real workload; decides.
- **Winner's curse:** a winner picked among N reads high by ≈ sigma x sqrt(2 ln N). Re-time it in a fresh process.
- **Bind:** ≥2 confirm sessions (different subset/seed or a second card), CI low above each session's margin, none
  negative. `Evidence.bind(min_samples=2, margin=...)` implements a repeat-and-margin rule.
- **Transfer:** a gain belongs to the design and hardware it was measured on; re-measure on the target card and
  precision. Interactions can be positive (two small wins → a larger joint one) or negative (fused x FP8 numerics):
  confirm a stack as a stack.
- **Reach:** if a change can't affect the code path a load uses, don't time it (xCalibur `ReachMap`); if it affects
  one phase, report that phase's ratio too.

## 6. Choosing the timer

| Scale | Use | Notes |
|---|---|---|
| µs kernels | `xcalibur.Bench`: CUDA graphs, cold-L2 **read** flush (a write flush charges dirty write-back to the kernel), optional null-replay subtraction | Match the judge or production timer; check the bench against it on a known pair |
| ms–s passes | Plain ABBA with warm-up, CUDA events + wall clock; `LongBench` if available | Graphs, flush and null replay are irrelevant at seconds |
| End to end | Production's replay (e.g. HTTP) | The final arbiter; calibrate the in-process harness against it |

Separate device from host: compare CUDA-event time with wall time, and eager with CUDA-graph replay. GPU-busy % per
phase (from a profiler) tells you whether to attack kernels or launches.

Autotuned kernels: warm up on a short input first (memory), and record the chosen configs per shape. Two processes
can pick different configs, which adds noise and breaks "what you measured is what serves".

## 7. Snapshots and contamination

Before and after each block: power draw and cap, SM clock, GPU memory used, other processes on the GPU, host memory
commit. Record clock dips at the power cap. A second process on the GPU (or a shared board's other GPU under load)
invalidates timing; note it, and rerun if it overlapped a timed block.

## 8. Instruments and parity

- Every instrument (emulator, proxy, in-process harness, cost model) is a hypothesis until it reproduces a real
  measurement of the effect it will be used for.
- Estimators enumerate and rank; they don't select or veto.
- Ship what you measured: same commit, flags, env vars, library versions, autotune picks. xCalibur's `Capture` checks
  build inputs; numerically re-check any environment difference (e.g. a production-only package) on real inputs.

<a id="numerics-gates"></a>
## 9. Numerics gates

- Compare candidate outputs with the reference on the same inputs: distribution of |Δp| (p50/p90/p99/p99.9/max),
  argmax flip rate, within-1 rate (for ordinal outputs), mean level shift; plus task accuracy against external labels
  where available.
- **Calibrate the gate** from an accepted change's drift vs the reference on the same answer set (e.g. 1.5x on each
  metric), using hundreds to thousands of answers across strata. Re-anchor when the answer set grows.
- **Bit-identical** on real inputs (|Δ| = 0, equal caches) needs no gate; say "bit-identical" only when tested on
  real inputs, not just unit shapes.
- **Quantised paths:** dynamic per-token/per-group activation quantisation amplifies any upstream 1-ulp change. Gate
  FP8/INT8 changes against the quantised path's own accepted drift, not the bf16 gate. Expect fused/unfused or
  reordered computations to drift at that level; an exact fix may cost the whole speedup.
- **Outliers under INT8:** norm-fed GEMMs often have persistent outlier channels (fixable by smoothing folded into
  norm weights); token-varying outliers need finer activation scales. Diagnose per GEMM before blaming one layer.
- A gate fail with unchanged external accuracy is a **coach decision** (C1), recorded as COACH/USER, not an
  automatic veto.
