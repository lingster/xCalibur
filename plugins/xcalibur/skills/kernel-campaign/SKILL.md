---
name: kernel-campaign
description: > Playbook for GPU inference and kernel speedup campaigns, from xCalibur, built on the ooda-coach skill
  (load and follow it too). Covers paired ABBA timing with a same-session A/A margin, clustered bootstrap intervals,
  screen-then-confirm with winner's-curse re-measurement, binding only after two sessions, long-pass vs microsecond
  timing, device vs host time, power and clock snapshots, profiling to find the ceiling before tuning, no-op checks
  before timing, bit-identical and structural changes before tuning, numerics gates for bf16/FP8/INT8 quantised paths,
  and GPU ownership and locking for agents. Ships starter lessons from real campaigns. Use it whenever the user wants to
  make a model, kernel, GEMM, attention, prefill/decode, CUDA/Triton/CUTLASS code or an inference server faster, compare
  kernels or configs, autotune, quantise (FP8, INT8, W8A8, FP4), profile GPU time, check whether a speedup is real, or
  run benchmarks on a GPU, even if they don't mention xCalibur.
---

# Kernel campaign

**First load and follow the `ooda-coach` skill.** Everything there applies: the user coaches and you make your reasoning
coachable; read `LESSONS.md` first (COACH > RULE > INFERENCE); pre-register prediction and falsifier; one change (or one
registered bundle) per run; an IAR after every run; append-only ledger and lessons. This skill adds the GPU-specific
measurement protocol, decision rules, discipline and starter rules.

Kernel work hits the coaching checkpoints often. Typical ones: **C1** before adopting a speedup or rejecting one on
numerics; **C2** when drift, an OOM or a speedup lands outside the predicted range; **C3** before changing a default
that production relies on, moving to another GPU or precision, or spending a scarce machine's time; **C4** at
session end. Ask in one line ("Coach: anything to correct before I adopt B4?") and carry on with routine runs
meanwhile.

Method sources: xCalibur's [measurement contract](https://github.com/xArmory/xCalibur/blob/main/docs/getting-started.md#measurement),
[measured search](https://github.com/xArmory/xCalibur/blob/main/docs/search.md#measured-search),
[evidence chain](https://github.com/xArmory/xCalibur/blob/main/docs/search.md#evidence) and
[studies](https://github.com/xArmory/xCalibur/blob/main/docs/studies.md).

## Campaign start

1. Load `ooda-coach`; read `LESSONS.md`. If the project has no kernel rules yet, offer to copy the applicable
   entries from [references/starter-lessons.md](references/starter-lessons.md) (ids K-01…K-17) in as `RULE`,
   keeping each incident so the reason travels with the rule. The coach can overrule or retire any of them; their
   COACH rules always take precedence.
2. Write `CONSTRAINTS.md`: which GPUs you own (by physical index or PCI bus id), the lock path, who else uses the
   machine, production's configuration, holdout rules, budget. Tag RULE / USER / INFERENCE.
3. Fix the **contract**: model and precision, workload (real requests, not only synthetic), metric (e.g. input
   tokens per GPU-second), reference (production's path, commit-pinned), numerics gate (to be calibrated), and the
   environment (library versions match production's, or the difference is checked).
4. **Profile before tuning** (K-09). Get the time split by phase and kernel category, and each big kernel's fraction
   of the hardware ceiling. Anything at ≥90% of its ceiling is done; only less work or lower precision moves it.
5. Measure the **noise floor** (A/A) and the **reference drift** (numerics of an accepted change vs the
   reference, on enough samples), then set the margin and the gate from them (K-02, K-04).

## Order of attack

Spend effort where the measured time is and where headroom exists. Before **any** candidate gets GPU time, run the
**exact layer**: `warpspeed_check` accounting (with the board's real ceilings; GeForce bf16 with fp32 accumulate
runs at half rate), reach, compile, and a SASS / captured-build diff against the parent; identical ⇒ exact no-op,
never timed (K-15). Details:
[references/construction-and-search.md](references/construction-and-search.md#2-exact-layer-before-timing).

1. **Bit-identical and structural first:** remove work that doesn't contribute to the output (dead last-layer
   compute, per-row copies, repeated passes, unneeded casts). These are often bit-identical, so they need no numerics
   gate and they carry to other hardware and precisions.
2. **Fewer launches / less host time** where the device is idle (batched or packed passes, CUDA graphs, fused
   elementwise). Check GPU-busy % first (K-06).
3. **Lower precision** (FP8, INT8, FP4) only with a calibrated numerics gate (K-08); prefer variants whose numerics
   are bit-identical to the current production path.
4. **Tuning** (tile configs, stages, autotune) last, and only on kernels below their ceiling. Expect small gains;
   apply the winner's curse (K-05).

Structure beats tuning (warp-specialised redesigns gave 2.9–4.6× where retuning gave 1.5–2.4×); at the roofline,
fewer bytes beat better loads. Register coupled changes (ownership + packing + staging) as one change, recombine
measured moves on purpose (interactions hide gains), and give a load its own build when its winner hurts others.
Search with the measured game (screen → confirm → bind) and a neighbourhood sweep of the incumbent:
[references/construction-and-search.md](references/construction-and-search.md).

A mechanism class that owns less than ~0.5% of the time isn't admissible for kernel work. Re-rank after each
precision change: if FP8 halves GEMM time, attention's share grows (Amdahl), and a lever that was small becomes worth
doing.

## Measurement protocol

Details and the reasons behind each rule: [references/measurement.md](references/measurement.md).

- **Paired, interleaved:** candidate and reference in the same session, ABBA order, ratios per pair (ratio > 1 =
  candidate faster). Never compare against a number from another day.
- **A/A in every session** (K-02); margin = 1 + 2 x the worst A/A offset (at least the noise of the mode used).
- **Interval:** bootstrap the paired ratios; **cluster by input/request id** when an input is repeated, else the
  interval is too narrow. Report the median ratio, the token-weighted ratio and the per-shape ratios.
- **Screens nominate, confirms decide.** A screen (short, few samples, maybe a proxy) only picks candidates. A confirm
  runs next to the reference with the A/A, enough pairs, on the real workload.
- **Winner's curse** (K-05): the best of N screened candidates reads high by about sigma x sqrt(2 ln N); re-measure it in
  a fresh confirm before reporting it.
- **Bind on ≥2 sessions, none negative** (K-01): a single-session pass is "adopted, pending confirm". Pre-register
  the margin rule (e.g. 1 + 2×worst A/A; or max(3×A/A, 0.5%) with no load below −0.5% for multi-load games) and
  apply the margin of the mode that actually ran (K-14).
- **Right timer for the scale:** µs kernels → CUDA graphs, cold-L2 *read* flush, null-replay awareness (xCalibur's
  `Bench`; null subtraction is off by default and needs a calibration that matches the judge's timer). Long passes
  (ms to s) → plain ABBA with warm-up, no graphs or flush (`LongBench`).
- **Device vs host:** time CUDA events and wall clock separately; a "fixed cost" can be Python dispatch (K-06).
- **Snapshots:** power, SM clock, memory used and other processes on the GPU before and after each block; note power
  caps and clock dips in the entry.
- **Measure what can move:** if a change touches only one phase, report the per-phase ratio too; whole-request ratios
  dilute it.
- **Calibrate every instrument** (emulator, proxy, in-process harness) against the real measurement on a known pair
  before it chooses anything (K-03).
- **Parity:** time the exact artifact that will ship (same commit, flags, env vars, library versions); record
  autotune picks. Export with `Capture` + `replace_assignment` + hash check, compile per-size builds in `prepare()`,
  `lint` the shipped file, predict its score with a parity bench, then check it on the judge.
- **End to end:** check the baseline isn't on a fallback path (K-17); use differencing to drop fixed costs; reconcile
  the end-to-end delta with the component prediction (Amdahl).

## Numerics gates

Details: [references/measurement.md#numerics-gates](references/measurement.md#numerics-gates).

- Calibrate the gate from the **accepted reference drift** measured on **enough answers** (hundreds to thousands,
  stratified), e.g. gate = 1.5 x the accepted drift on each metric (max/p99.9 |Δp|, argmax flip rate, within-1 rate,
  mean level shift). A gate set on a few dozen samples is too tight on more (K-04).
- **Bit-identical** (|Δ| = 0 on real inputs) is the strongest claim and needs no gate; prefer such speedups.
- Under **dynamic quantisation** (FP8/INT8 activations), any 1-ulp change upstream can flip codes and group scales,
  amplifying drift several-fold. Gate FP8 changes against **FP8's own noise** (the accepted FP8-vs-FP8 drift), not
  the bf16 gate, and add an accuracy check against external labels where available (K-08).
- A gate fail is a decision for the owner, not an automatic veto, when external accuracy is unchanged; record it
  as a USER decision.

## GPU discipline (agents and humans)

Full rules: [references/gpu-discipline.md](references/gpu-discipline.md).

- **One GPU process per agent**, under an **exclusive lock** on a **named GPU** (physical index or bus id, with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` pinned). Never touch a GPU you don't own, even for "a quick
  test" (K-12).
- **Wait on a PID or the lock**, never on `pgrep -f <name>` when the name appears in your own command line (K-11).
- **Autotune on a short input first**, then run long ones (K-07).
- **Never let a default change silently remove a production choice** (K-10). Defaults in shared code match what
  production runs; new behaviour is opt-in until confirmed and the coach approves it at C3.
- No background GPU processes left running; check memory headroom (host commit limit, GPU memory) before launching.

## xCalibur modules to use (by upstream name)

From the `xcalibur` package (`pip install -e ".[gpu,search]"` in a checkout; all of these ship in current
checkouts: `python -c "import xcalibur as x; print(x.LongBench, x.aa_margin, x.DesignGraph)"`):

| Need | Module / API |
|---|---|
| µs kernel paired timing, A/A | `Bench` (`paired`, `aa`, `null_us`), `paired()` in [`bench`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/bench.py) |
| ms–s passes | `LongBench` (`paired`, `aa`), warm-up `small_first` / `warm_repeat`, `cuda_clock` / `wall_clock`, `smi_snapshot` |
| Ratios, intervals, margins, binding | `geomean`, `bootstrap_ci(..., clusters=)`, `aa_margin`, `Evidence.bind` in [`stats`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/stats.py) |
| Screen → confirm search loop | `Game` (`round`, `confirm`) in [`game`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/game.py); `Surrogate` (lightgbm or ridge), `bits`, `one_hot` |
| Does a change reach the build? | `ReachMap` in [`reach`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/reach.py) |
| Same build inputs as measured; exact export edits | `Capture`, `replace_assignment` in [`capture`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/capture.py) |
| Run card, rules version, timed-path lint | `Card`, `rules_gate`, `lint`, `LINT` in [`gates`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/gates.py) |
| Smem/register/barrier plan check; boards | `warpspeed_check`, `plan_of`, `launch_register_cap`, boards `H200`, `RTX3090`, `A6000`, `RTX6000ADA`, `L40S`, `RTX4090` (`ARCHS` keys and `arch=` strings are lowercase, e.g. `"rtx3090"`), `execute` + `SCHEMA` as a model tool, in [`warpspeed`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/warpspeed.py) |
| Exact decoders and packing | `gf2` (`packing_transform`, `join_all`, `compose`, `as_cute_layout`) in [`gf2`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/gf2.py) |
| Design graph, retrieval, model planner | `DesignGraph`, `Node`, `Retriever`, `embedding_gemma`, `PlannerLoop` in [`graph`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/graph.py) / [`planner`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/planner.py) |

Kits: `kernels/fp8_block_sm89/` (block-FP8 GEMM per-shape tuning restricted to bitwise-equal builds, fresh confirm),
`kernels/w8a8_sm86/` (INT8 W8A8 for Ampere, with outlier diagnosis), `kernels/moe_w4a16_h200/` (NVFP4 MoE with a
GF(2)-searched decoder), `kernels/tg4dac_sm120/` (Gemma 4 attention and MLP). Notebooks `03` (noise, winner's
curse), `04` (interactions), `05` (the game), `07` (knowledge graph) and `12` (consumer-card prefill), plus the
`data/` analogs, are listed in `ooda-coach`'s [toolkit](../ooda-coach/references/toolkit.md). If a module is
missing in an older copy, implement the same protocol by hand (it's short; see measurement.md) rather than skipping it.

## Worked example

A real campaign (prefill of long-document requests on a hybrid linear-attention model; 3090 and RTX 6000 Ada) shows
the loop end to end: profile → structure first → 1.21x within the gate on the 3090 → two structural changes bound on
two cards → FP8 drift diagnosed as quantisation sensitivity → a 1.8x INT8 path rejected by the gate. Read
[references/worked-example.md](references/worked-example.md) when planning a campaign or when a checkpoint needs a
concrete precedent.

## Ledger entry additions for kernel work

On top of the ooda-coach template, every kernel entry records: run id, commit, workload id, GPU (name, index, power
cap), library versions, mode (same-process / reload blocks), pairs, same-session A/A and margin, median and
token-weighted ratios with clustered CI, per-shape ratios, numerics vs gate, GPU-busy % if relevant, and the
snapshot notes (clocks, power, other processes), the protocol version, the source hash, and the exact-layer result
(plan check, reach, SASS/no-op diff).
