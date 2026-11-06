# Worked example: a one-day prefill campaign on two GPU generations

An anonymised account of a real campaign, written to show the loop and its checkpoints rather than the model.
Run ids are the campaign's own.

## Contract

- **Workload:** long-document requests, about 28k tokens of shared document state followed by 4–20 short question
  rows; scored outputs are per-question level probabilities. A 105-request "smoke" bundle (1,198 answers) for
  tuning, a 561-request "dev" bundle, and a holdout run once at the end by the owner.
- **Models:** a hybrid model (linear-attention layers plus full-attention layers). Development on a 9B bf16 copy on
  one **RTX 3090** (sm_86, 24 GB); production is the 27B **FP8** model on **RTX 6000 Ada** cards (sm_89).
- **Metric:** input tokens per GPU-second; per-request paired ratios; numerics vs production's path.
- **Reference:** production's unfused path. One GPU owned by the campaign; the other card on the box belonged to
  someone else (USER constraint).

## 1. Noise floor first (K1)

A/A in same-process mode: 1.0006 [90% CI 1.0002, 1.0009]; blocks mode (reloads): 0.9990 [0.9987, 0.9993]. Both
"biased" but ≤0.1%, so the margin was set at CI low > 1.002 (same-process) / 1.003 (blocks). Outputs were
deterministic (|Δp| = 0).

## 2. Reference drift and the first gate (K2), and its correction (K5)

The existing fused path vs unfused: **1.140x** [1.138, 1.140], with 2 argmax flips in 78 answers. This drift was the
accepted reference, so the first gate was anchored on it (within-1 100%, max |Δp| ≤ 0.025).

On the full 1,198 answers the same accepted drift showed within-1 99.50% and max 0.029: the 78-answer gate would have
rejected the baseline. **Surprise → C2 → candidate lesson → K-04.** The gate was re-anchored at 1.5x the drift on
1,198 answers (max ≤ 0.044, p99.9 ≤ 0.037, flips ≤ 2.9%, within-1 ≥ 99.25%, mean level shift ≤ 0.014).

## 3. Profile before tuning (B1)

Measured split of the 28k state pass: GEMM 81%, attention 12.5%, linear attention 2.3%, norms 2.5%; the research
estimate had been 70 / 19 / 6. The largest GEMM ran at ~95% of the card's bf16 peak, so **bf16 GEMM tuning was
dropped** (K-09). The question rows spent ~27% of their time copying the cache per row. That pointed at structure:

- **B3:** skip the last layer's dead work in the state pass (only its cache is used). Exact by test.
- **B4:** run all question rows of a request in one packed pass over the shared state, with no per-row cache copy.

## 4. Structural changes: adopted, then bound on two cards

| Change | 3090 session 1 | 3090 session 2 | Ada (27B FP8) | Numerics |
|---|---|---|---|---|
| B3 | 1.0338x [1.032, 1.035] (K3) | 1.0363x [1.0342, 1.0372] (K6; A/A margin 1.008) | 1.015x median of 3 ABBA pairs | bit-identical |
| B4 | 1.0513x [1.0477, 1.0553] (K5b) | — | 1.149x (×1.150, ×1.149, ×1.144) | inside the gate (bf16); inside the FP8-noise gate (FP8) |

- K6's prediction was pre-registered in the run script: "1.025–1.045x, |Δp| = 0; falsified if CI low ≤ 1.003 or
  any |Δp| > 0". Observed 1.0363x, |Δp| 0. Its session A/A was offset −0.42% (vs ≤0.1% before), so the margin rose to
  1.008: K-02 in action.
- B4 predicted 1.04–1.11x and "more under FP8" from the profile (the FP8 rows phase was only 43.5% GPU-busy:
  host-bound, K-06). On Ada it landed just above the top of the range: a surprise in the good direction, still
  reviewed (C2) before the ranges for FP8 row work were widened.
- Both were **bound on two cards, none negative** (K-01) and checkpointed at C1 before adoption. The owner then put
  them into production.

**Cumulative on the 3090:** fused + B3 + B4 vs production's original path, full smoke, run 20261105T121645Z:
**1.2124x** [1.2074, 1.2173] median per request (token-weighted 1.2315x), max |Δp| 0.024, flips 2.25%, within-1
99.50%: **1.21x within the gate.**

## 5. FP8 drift diagnosed, not "fixed"

On Ada, the fused path under FP8 was 1.12x faster but failed the bf16-derived gate: flips 4.5%, max |Δp| 0.073 (599
answers), ~6x the bf16 drift. **C2.** A fake-FP8 screen used earlier (per-row scales) had shown nothing: an
uncalibrated oracle (K-03).

The diagnosis, on the 3090 with an exact emulator of the production FP8 recipe:

1. **Calibrate the emulator first:** it reproduced 5.5x amplification vs the card's ~6x. Only then could it rank fixes.
2. **Localise:** the FP8 inputs were bf16 in both paths; each fused op differed by 1–2 ulp from its reference, and
   per-group dynamic activation quantisation turned that into different e4m3 codes and group scales from layer 1 on.
3. **Ablate:** fixing the 5 hypothesised ops barely helped; leaving out any single op changed little; only putting
   every op back into reference form removed the drift (78/78 identical), and that also removed the speed.

Conclusion: the FP8 model's answers move about this much under **any** 1-ulp change, so production's own answers are
one sample of that noise. External-label accuracy was unchanged (MAE 1.014 vs 1.014 on dev, CI on the difference
[−0.003, +0.003]). New gate for FP8-path changes: 1.5x the accepted FP8-vs-FP8 drift, plus external accuracy (K-08).
The owner kept the fast path (a USER/COACH decision at C1). When the fix branch arrived, it had switched FP8 loads to
the slow exact mode by default; the coordinator caught it, kept the fast default and made exact mode opt-in (K-10,
C3).

## 6. INT8 at 1.8x, rejected by the gate

For 3090 users (no FP8 there), W8A8 INT8 GEMMs ran at 56–78% of the INT8 peak; end to end
**1.843x** [1.822, 1.864] on 78 answers, with the kernel choice re-measured in a fresh confirm (K-05).

Numerics failed: 11/78 flips, within-1 93.6% (~10x the gate). The diagnosis found persistent outlier channels feeding
the norm-fed GEMMs and token-varying outliers elsewhere. SmoothQuant folded into the norm weights at zero run-time
cost brought the full smoke to **1.808x** [1.799, 1.816] with flips 8.6% and within-1 96.5%: still about 3x the
gate's flip limit. Keeping the worst projections in bf16 gave 1.388x at 4.6% flips: still outside.

**Decision at C1: not adopted.** A 1.8x INT8 path is a method change that needs a full re-score, not a speedup.
Recorded as rejected on numerics, with the next options (finer activation scales everywhere, quantisation-aware
training) parked with their estimated cost.

## What the example teaches

- Profile first; the estimate of where time went was wrong in every major category.
- Structure before tuning: the two structural changes carried across GPUs and precisions; bf16 tuning had no
  headroom.
- The gate and the margin are measurements; re-anchor them when the sample grows or the session changes.
- Under quantisation, "numerics-neutral" means "inside the quantised path's own noise", and a coach decides when
  external accuracy and the gate disagree.
- Process slips (a self-matching wait loop, an unlocked test, a silent default change) were each harmless once and
  each became a rule.
