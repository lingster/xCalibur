# Starter rules for any experiment loop (G-01…G-14)

General rules distilled from xCalibur's campaigns and from kernel-campaign's
[K-rules](../../kernel-campaign/references/starter-lessons.md), worded for any domain (LLM prompts and configs, ML
training, systems benchmarks). G-02, G-08, G-09, G-10 and G-12 come from a separate LLM document-extraction campaign run
with this skill, not from the xCalibur repo; their incidents can't be replayed from `data/`. At session start, offer to
copy the applicable ones into `LESSONS.md` as `RULE`, keeping the incident. The coach can overrule or retire any of
them; COACH rules always win.

Each rule: the incident it came from, then the decision-time check.

## G-01 · Same-session A/A before any comparison (K-02)
Noise floors move between sessions (reloads, clocks, caches, other tenants). **Check:** before a compare decides
anything, verify an A/A from this session, same code, mode and device; margin from it.

## G-02 · Count distinct conditions, not repeats
An LLM extraction campaign at temperature 0: warm-cache repeats were byte-identical; only the cold first pass
differed. Three "runs" were two samples. **Check:** before planning N repeats, list what varies between them (cache
state, input order, seed, process); vary that (shuffle order, reload) and don't count identical repeats.

## G-03 · Calibrate any proxy before it chooses (K-03)
An emulator, an in-process harness and a write-flush timer each ranked candidates wrongly until checked against the
real measurement. **Check:** before a proxy picks or vetoes, verify it against the real instrument on a known pair.

## G-04 · Re-measure the winner (K-05)
The best of N screened candidates reads high by about σ·√(2 ln N) (a 1% screen over 30 candidates fakes 2.6%).
**Check:** before reporting a screened winner, confirm it fresh next to the reference.

## G-05 · Bind on two sessions, none negative (K-01, K-14)
**Check:** before marking adopted, verify ≥2 sessions above that session's margin and none negative; else "adopted,
pending confirm". Use the margin of the mode that actually ran.

## G-06 · Never let a default change silently remove a production choice (K-10)
**Check:** before changing a shared default, list every caller (a server flag can serve several products); keep
production's choice and make new behaviour opt-in until the coach approves at C3.

## G-07 · Wait on a PID or a lock, never `pgrep -f` a name in your own command line (K-11)
The waiter matched its own shell and never woke; a gap between two chained runs can also let a queued job start early.
**Check:** before a wait loop, wait on a PID, a lock file or a completion marker.

## G-08 · Read what a knob does before betting on it
For that model's chat template, "reasoning effort" turned out to be one system-prompt sentence, not a thinking budget;
"high" doubled latency and cut accuracy. **Check:** before a flag/config lever, read its implementation (template,
props, source) and predict from that, with a range that includes harm.

## G-09 · Rescore before auditing
Stored per-row scores were from an older scorer; an error audit built on them proposed a fix that wasn't needed.
**Check:** before auditing errors, recompute scores from stored raw outputs with the current scorer.

## G-10 · Attribute a change only after a second perturbation
A prompt rule was blamed for breaking a field; a different prompt edit broke the same field: that field flips under
any prompt change. **Check:** before attributing a changed result to one edit, see whether another edit of similar
size flips it too.

## G-11 · Diff the effective artifact before spending on it (K-15)
8 of 9 screened "changes" were byte-identical builds that still "moved" ±1.4%. **Check:** before measuring, diff the
resolved config / rendered prompt / binary against the parent; identical ⇒ no-op, don't run it.

## G-12 · Tuned-on-test is a screen
Rules written from the errors of the evaluation set score on the data they were written from. **Check:** a lever
derived from evaluation-set errors is a screen until data it wasn't derived from agrees.

## G-13 · Estimators enumerate, measurements decide Analytic estimates were optimistic in 30 of 30 cases (and another
predicted −4.6 µs, measured +15.5 µs). **Check:** before an estimate selects or vetoes a branch, measure; give the
estimator a calibration entry.

## G-14 · Pre-register prediction and falsifier (K-13)
**Check:** before running, the ledger has the prediction with a range, its analog, the falsifier and the decision
rule; an unregistered run can nominate, not confirm.
