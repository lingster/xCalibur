# Starter rules for kernel campaigns (K-01…K-17)

Each rule came from a miss or near-miss in a real two-GPU inference campaign (a 9B bf16 model on an RTX 3090 and its
27B FP8 sibling on RTX 6000 Ada cards; long-document prefill). Run ids are that campaign's. Copy the applicable ones
into the project's `LESSONS.md` as `RULE` (format: [lessons-format.md](../../ooda-coach/references/lessons-format.md)),
keeping the incident. The coach's COACH rules take precedence, and the coach can overrule or retire any of these.

Each rule names the coaching checkpoint where it usually matters (C1 adopt/reject, C2 surprise, C3 plan or default
change, C4 session end).

---

## K-01 · Bind an adoption only after ≥2 sessions, none negative
- **Incident:** a structural change (skip the last layer's dead work) was adopted from one session, "screen = confirm
  here". It held (1.0338x, then 1.0363x in a second session on different requests), but the rule had allowed a
  one-session adoption that a noisier change would have failed.
- **Check:** Before marking a change adopted, verify ≥2 sessions (different seed/subset, ideally a second card), each
  with CI low above that session's margin, none negative. Otherwise "adopted, pending confirm". Checkpoint C1.
- **Scope:** speed adoptions. xCalibur's `Evidence.bind` encodes the same idea (≥2 samples, mean above margin, none
  negative).

## K-02 · Run an A/A in every session; set the margin from it
- **Incident:** run 20261105T142735Z. The first session's A/A offset was ≤0.1% (margin 1.003); the second session's
  was −0.24 to −0.42%, so the margin had to be 1.008. Carrying the old margin would have accepted noise.
- **Check:** Before a compare is used for a decision, verify a same-session A/A for the same code, model, mode and
  GPU; margin = 1 + 2 x the worst A/A offset. If missing, run the A/A first.
- **Scope:** all paired timing.

## K-03 · Calibrate any oracle, emulator or screen against a real measurement before it chooses
- **Incident:** a fake-FP8 screen (per-row scales) predicted that a fusion would be numerics-neutral under FP8. On the
  real FP8 card the drift was ~6x the bf16 drift (flips 4.5% vs the 2.6% gate). Later, an exact emulator of the
  production FP8 recipe (128x128 weight blocks, per-token per-128-group activation quant, the card's double-rounded
  cast) reproduced 5.5x vs the card's ~6x before it was allowed to rank fixes.
- **Check:** Before an emulator, proxy or in-process harness chooses between changes, verify it reproduces a real
  measurement of the same effect on the same inputs (within its CI). If not, it may nominate only. Checkpoint C2.
- **Scope:** timing harnesses vs the production replay, numerics emulators, cost models. xCalibur's write-flush vs
  read-flush incident is the timing version of this rule.

## K-04 · Set numerics gates from the measured reference drift, on enough samples
- **Incident:** a gate anchored on 78 answers ("within-1 100%, max |Δp| ≤ 0.025") would have rejected the accepted
  baseline itself on 1,198 answers (within-1 99.50%, max 0.029). Re-anchored at 1.5x the accepted drift on 1,198:
  max ≤ 0.044, p99.9 ≤ 0.037, flips ≤ 2.9%, within-1 ≥ 99.25%, mean level shift ≤ 0.014.
- **Check:** Before applying a numerics gate, verify it was calibrated as 1.5x the accepted reference drift on the
  same answer set, with at least several hundred answers across the workload strata. Re-anchor when the set grows.
- **Scope:** numerics-only changes vs a fixed reference.

## K-05 · Winner's curse: re-measure the winner in a fresh confirm
- **Incident:** an INT8 GEMM backend was picked among several by screen; the pick was re-measured in a fresh process
  before the end-to-end run. In a per-shape config search with ~50 configs at ~1% noise, the best reads ~2.8% high by
  luck alone (sigma x sqrt(2 ln N)).
- **Check:** Before reporting a winner chosen among N candidates, re-time it against the reference in a new session
  (new process). Report screen gain and confirm gain side by side.
- **Scope:** any selection among several candidates: configs, kernels, seeds.

## K-06 · Time device and host separately
- **Incident:** an FP8 GEMM showed a "fixed ~0.17 ms per call" at 64 tokens and looked like a bad tile config. The
  question-row phase was only 43.5% GPU-busy (60,680 launches in 7.1 s): the cost was host dispatch through Python
  layers. Packing the rows into one pass (fewer launches) gave 1.149x on that card, not a tile change.
- **Check:** Before attributing a cost to a kernel, verify GPU-busy % for the phase and compare CUDA-event time with
  wall time (or eager vs CUDA-graph replay). If the device is idle, target launches and host work.
- **Scope:** short inputs, small M, many launches.

## K-07 · Autotune on a short input before long ones
- **Incident:** the first long (28k-token) request ran Triton autotuning of a fused SwiGLU on a 24 GB card already
  holding 16.5 GB of weights and went out of memory. Warm-up on a ≤2,048-token input first fixed it (tuning keys
  didn't depend on length); peak afterwards 20.95 GB.
- **Check:** Before the first long run in a process, verify warm-up ran on a short input that triggers the same
  autotune keys. Note the risk for production servers whose first request is long.
- **Scope:** Triton/autotuned kernels whose keys don't include sequence length.

## K-08 · Under quantisation, any 1-ulp change can be amplified: gate against the quantised noise; prefer bit-identical speedups
- **Incident:** fused norms/gates changed some bf16 values by 1 ulp before dynamic FP8 activation quantisation. Under exact
  emulation, from the second layer on 60–85% of e4m3 codes and 85–97% of group scales differed; on the FP8 card drift
  was ~6x bf16's. Putting every op back in
  reference form removed the drift (78/78 answers identical) and all of the speed (≈1.0x instead of 1.12x). External
  label accuracy was unchanged, and the owner kept the fast path.
- **Check:** Before gating an FP8/INT8-path change, verify the gate is calibrated on the quantised path's own
  accepted drift (e.g. 1.5x the accepted FP8-vs-FP8 drift), plus an external accuracy check if available. Prefer
  changes that are bit-identical on real inputs. Checkpoint C1/C2.
- **Scope:** dynamic per-token or per-group activation quantisation (FP8, INT8, FP4).

## K-09 · Profile and find the ceiling before tuning
- **Incident:** a research estimate split time as GEMM 70 / attention 19 / linear-attention 6%. The profile measured
  81 / 12.5 / 2.3%, and the big gate+up GEMM ran at ~67 of the 3090's ~71 dense bf16 TFLOPS (~95%). bf16 GEMM tuning
  was dropped; only lower precision or less work could move it.
- **Check:** Before tuning a kernel, verify a profile of the real workload gives its time share (≥0.5% to be
  admissible) and its fraction of the hardware ceiling (≥90% → stop tuning it). Re-rank after a precision change
  (Amdahl). Keep the scope label on the ceiling claim ("bf16 on the 3090").
- **Scope:** all kernel work.

## K-10 · Never let a default change silently remove a production choice
- **Incident:** a fix branch for FP8 drift switched FP8 loads to the exact (slow) mode by default. Merged as written,
  it would have removed production's 1.12x without anyone deciding to. The coordinator kept the fast path as the
  default and made exact mode opt-in.
- **Check:** Before merging any change to a default (env var, config, flag), verify what production runs and keep
  it; new behaviour is opt-in until the coach approves. Review subagent diffs for default changes. Checkpoint C3.
- **Scope:** shared and production-facing code.

## K-11 · Wait on a PID or a lock, never `pgrep -f` a name in your own command line
- **Incident:** run K6 sat idle ~2 h: its `pgrep -f run_k5.sh` waiter matched its own command line (and another
  waiter matched itself), so neither ever exited.
- **Check:** Before writing a wait loop, verify it waits on a PID file, `wait <pid>`, or the lock (`flock`), not a
  pattern that can match the waiter itself.
- **Scope:** any scripted sequencing of runs.

## K-12 · One GPU process per agent, under an exclusive lock on a named GPU; never touch a GPU you don't own
- **Incident:** several agents' torch processes (10–25 GB virtual each) exhausted a strict host commit limit
  (167 GB); process spawns and mmaps failed. Separately, one agent ran tests without pinning the GPU (may have touched
  a card belonging to someone else), and another ran ~17 s of GPU tests without the lock (no timed run was affected,
  by luck).
- **Check:** Before any GPU command, verify `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=<owned index>`, the
  exclusive lock held (`flock <lock> <cmd>`), no other process of yours alive, and host/GPU memory headroom.
- **Scope:** all agents and humans sharing a machine. Details: [gpu-discipline.md](gpu-discipline.md).

## K-13 · Pre-register prediction and falsifier in every entry
- **Incident:** early entries had estimates but no falsifiers, so a result could always be explained after the fact.
  From K6 on, each entry carried "expect X; if Y, the hypothesis is wrong" in the run script; e.g. "1.025–1.045x,
  |Δp| = 0; falsified if CI low ≤ 1.003 or any |Δp| > 0" → observed 1.0363x, |Δp| 0.
- **Check:** Before a run starts, verify the ledger entry (or run script header) holds the prediction with range,
  its analog, the falsifier and the decision rule. A run without them is `unregistered`.
- **Scope:** every run.

---

K-14…K-17 come from xCalibur's other studies (the sm_86/89 writeup, the H200 MoE AUTORESEARCH game and TG4DAC), not
from the two-GPU campaign above. General versions of many K-rules for non-GPU work: `ooda-coach`'s
[starter-rules.md](../../ooda-coach/references/starter-rules.md).

## K-14 · An automated decision uses the margin of the mode that actually ran
- **Incident:** run K4 (sm_86/89 writeup). The run table said ADOPT against the same-process margin (1.002); the
  blocks-mode run needed the blocks margin (1.003), under which the result was a null.
- **Check:** Before a script or agent records adopt/reject, verify the margin it compared against came from an A/A
  in the same mode (same-process vs reload blocks) and session.
- **Scope:** every automated decision on paired timing. Checkpoint C1.

## K-15 · Identical artifact ⇒ no-op: diff before timing
- **Incident:** H200 MoE game. 8 of 9 screened plans produced byte-identical target builds yet "moved" ±1.4%; later
  the game diffed SASS and caught 38 no-ops and 18 compile contradictions among 325 archived designs before any GPU time.
- **Check:** Before timing a candidate, verify its captured build inputs (`Capture.same`) or SASS differ from the
  parent's for the load it targets; identical ⇒ record "exact no-op", don't time it, and tell the proposer.
- **Scope:** all candidate timing. Checkpoint C2 if a no-op was already timed and "won".

## K-16 · A plan that passes accounting can still spill: inspect the compiled allocation
- **Incident:** TG4DAC Altair MLP. The design assumed 224 registers for a 9-warp kernel; the launch cap is 168, and
  the compiled kernel spilled. The checker counts registers but does not compile.
- **Check:** Before attributing a result to a passed plan, verify the compiled registers, spills and occupancy
  (SASS / `-Xptxas -v`), and recompute the register cap for the warp count.
- **Scope:** warp-specialised and register-heavy kernels.

## K-17 · Check the baseline isn't on a fallback path
- **Incident:** TG4DAC end to end. vLLM ran a generic Triton attention kernel because the model mixed head sizes 256
  and 512; part of the measured headroom was the fallback, and one decode shape still favoured the simpler kernel.
- **Check:** Before claiming headroom or a speedup over a framework, verify which kernel the baseline actually
  dispatches for each shape (profiler trace or logs), and say so in the ledger.
- **Scope:** comparisons against serving frameworks and libraries. Checkpoint C3 before a campaign is planned on it.
