# Agenda: choosing the next experiment and searching a space

The loop can be sound and still ask poor questions: when every review ends with a variant of the current design,
the next experiment is inherited from the last one. Adapted from xCalibur's
[OODA protocol](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#ooda),
[search and games](https://github.com/xArmory/xCalibur/blob/main/docs/search.md) and the reflection
[a sound loop can still choose poor questions](https://github.com/xArmory/xCalibur/blob/main/docs/reflections.md#strategic-loop).

## 1. Keep a frontier, not a to-do list

The **frontier** is every open branch with its estimate and evidence tag: retained alternatives, parked branches
(with a restore path), fresh intel, proposed repairs and neighbours of the incumbent. Keep it in `STATUS.md`
("Backlog") or the knowledge graph. Save the chosen branch **and its siblings**; a new tool, mechanism or repaired
constructor can reopen a region that looked unpromising.

## 2. Rank by expected gain per check

For each branch estimate: expected gain (bias-corrected by the family's median prediction error), its uncertainty,
the cost of the check (runs, GPU time, coach time), and **information value** (what it decides even if it loses).
Rank by gain per check, with uncertainty about unmeasured stages counted as a reason to instrument, not to guess.
Estimators **enumerate and prioritise; they never select or veto**: when a model and a measurement disagree, the
measurement wins.

**Backward reasoning** helps split a target: required log-gain Δ = |W|·log(S*/S₀) over the workloads; allocate it
across components by measured headroom. This prioritises experiments; it proves nothing, and floors measured on one
structure need not survive a change of structure.

**Admissibility:** a mechanism that owns less than ~0.5% of the measured cost isn't worth a run; re-rank after every
adopted change (Amdahl: what was small can become large).

## 3. Two kinds of run

| Kind | Purpose | Brief needs | Scored as a forecast? |
|---|---|---|---|
| **Bet** | beat the incumbent | bias-corrected estimate from a measured analog that beats the incumbent | yes (`e = log(obs/pred)`) |
| **Calibration** | learn a new family's behaviour | declared "no calibration", a **range**, what it will calibrate | no |

A new mechanism family has no calibration: say "unknown, range only" and consider an instrument (profile, isolated
stage, microbenchmark) before betting on it.

## 4. One change per run, and when to break it

One change per run keeps attribution clean. Two exceptions, both registered in the brief:

- **Coupled change.** Some changes only work together (a new output format and its parser; a layout and its
  consumer). Testing one field against its old partner rejects an unfinished construction. Register the bundle as
  one composite change, name its parts, and expect to ablate it later.
- **Deliberate recombination.** Interactions are where gains hide (two moves at +0.4% and +0.9% alone gave +3.3%
  together). After single moves are measured, recombine measured pieces on purpose, including moves that **lost
  alone** (they stay allowed inside combinations). A move with an exact witness (crash, invalid output, identical
  artifact) is never re-proposed.

Winners don't transfer automatically: a gain belongs to the design it was measured on; re-confirm on a new base.

## 5. Spend evaluation budget in stages

- **Construct the space first:** eliminate variables that are determined by others, join compatible boundaries,
  share proofs across equal constructions (but keep their implementations distinct), and cache only with the full
  context (source, tool versions, flags, workload, protocol).
- **Cheap exact filters before measurement** ([evidence.md](evidence.md#exact-checks)): legality, reach, no-op
  detection by diffing the effective artifact against its parent.
- **Two-stage evaluation:** build ~2× what you'll measure, re-rank on cheap post-build features, measure the top half.
- **Successive halving:** screen many cheaply, confirm the survivors at full cost; the screen nominates, the confirm
  decides; re-measure the winner (winner's curse ≈ σ·√(2 ln N)).
- **Measure only what can move:** when a candidate reaches one workload, spend the samples on that workload rather
  than spreading them over all.
- **Free matched pairs:** order edits along the cheapest legal path from the incumbent; every valid midpoint is a
  control.

xCalibur's `Game` runs this loop for you (reach filter → surrogate-ranked nominations → bounded compile →
screen → confirm → bind; [`game.py`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/game.py),
[`notebooks/05_the_measured_game.ipynb`](https://github.com/xArmory/xCalibur/blob/main/notebooks/05_the_measured_game.ipynb)).
Its `Surrogate` (bootstrap LightGBM or ridge ensemble, UCB ranking) **orders** what to measure; its spread is an
exploration signal, not a confidence interval. Start a fresh `Game` when the reference, hardware, toolchain or
protocol changes.

## 6. Is the incumbent a local optimum?

Sweep its neighbourhood (every one-edit neighbour, screened), then confirm the leaders next to it. If none resolves
better, report "no resolved improvement in the screened neighbourhood": not an optimum certificate. Look further
afield (new constructors, coupled changes, new mechanisms) rather than re-tuning the same knobs.

## 7. What may close a branch

| Closes the completions it covers | Does **not** close anything |
|---|---|
| an exact witness: contradiction, proof, invalid output, identical artifact | a sampled loss on one variant |
| a valid upper bound below a qualified alternative's lower bound | a low ranker or surrogate score |
| | a missing measurement, an exhausted budget, a tool or compile failure in one context |

A tool failure may mean a missing lowering or adapter, not an impossible idea. "No eligible completion" and "not
evaluated yet" are different outcomes; an unexecuted cell is unknown, not zero. Parking for budget is not proof.

## 8. Specialise per workload when wins conflict

If a change wins on one workload segment and loses on another, consider routing or a per-segment build rather than
averaging: keep a segment's win without paying for it elsewhere. Guard every adoption with a **no-workload-regresses**
check (e.g. no load below −0.5%) as well as the aggregate (geometric mean across workloads; component timings don't
add when stages overlap).

## 9. Widen the questions

At an AAR or reflection, group the history by **what was physically varied** (structure, precision, schedule, data,
prompt wording), not by proposal name. If the last N briefs all perturb one design, add at least one branch that
changes the construction, the representation or the workload.
