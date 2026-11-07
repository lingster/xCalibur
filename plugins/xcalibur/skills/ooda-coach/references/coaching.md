# Being coached: checkpoints and how to show reasoning

The user is the coach. Claude is the practitioner: it shows its reasoning at a few defined checkpoints so the coach
can correct it, records the correction, and applies it from then on. Claude doesn't quiz, test or challenge the
coach. Between checkpoints it works without waiting.

## Principles

- **Make reasoning easy to correct.** Separate facts (with error bars) from inferences, and say what you're unsure
  of. A coach can only correct what they can see.
- **Short.** A checkpoint is at most about eight lines plus a one-line question. Point to the ledger entry for the
  detail.
- **One question, specific.** "Coach: anything to correct before I adopt X?", not an open-ended survey.
- **Don't stall routine work.** Briefs, planned runs and IARs proceed. Only the four checkpoints wait for the coach.
- **Record the answer verbatim.** Paraphrase loses what the coach meant. Quote, date, then add your one-line
  reading as a check.
- **Silence is not approval of a rule.** If the coach doesn't answer, proceed on existing rules where the step is
  reversible, mark it `uncoached`, and raise it again next session.

## The four checkpoints

### C1: before adopting or rejecting a change

```text
Checkpoint C1 · adopt B3 (skip the last layer's dead work) · ledger K6
Observe [M]: 1.036x [90% CI 1.034, 1.037], 16 pairs; same-session A/A 0.9976 -> margin 1.008; numerics |dp| = 0.
Decide: rule "CI low > margin, numerics pass, 2 sessions none negative". Session 1 (K3) 1.034x; this is session 2.
Rules applied: K-01 (bind on 2 sessions), K-02 (A/A this session).
Unsure: both sessions used the same GPU; the second card hasn't run it yet.
Coach: anything to correct before I adopt B3?
```

### C2: after a surprise

```text
Checkpoint C2 · B2 under FP8 fails the numerics gate · ledger ADA2
Predicted [H]: drift about the bf16 level (flips <= 2.6%). Observed [M]: flips 4.5%, max |dp| 0.073 (599 answers).
Why (best reading, INFERENCE): fused ops round differently by 1 ulp; dynamic FP8 activation quant amplifies it.
Unsure: whether any single op dominates; my fake-FP8 screen couldn't show this at all.
Candidate lesson L-05: calibrate any emulator against a real measurement before it chooses.
Coach: is this the right reading of the miss?
```

### C3: before changing the plan or a default

```text
Checkpoint C3 · make exact-rounding mode the default for FP8 loads
State: production runs FP8 + fused (1.12x), chosen by the user (USER, 11-05). Exact mode removes the drift and the speed.
Premise: "drift matters more than speed" [H], unmeasured against accuracy; external-label MAE was unchanged [M].
Plan change: none to production; exact mode stays opt-in. Default unchanged.
Coach: OK to keep exact mode opt-in, rather than the default?
```

### C4: at session end

```text
Checkpoint C4 · session end 2026-11-05
Decided: B3 confirmed (2 sessions), B4 adopted pending confirm, INT8 rejected by the gate.
Open: B5 (FP8 GEMM efficiency) waits for the second card; the gate for FP8 changes needs the coach's choice.
Candidate lessons: L-11 (pgrep waiter matched itself), L-12 (unpinned GPU test).
Prediction error: median e = -0.02 over 7 speed predictions; numerics 2 misses of 4.
Uncoached decisions today: none.
Coach: any lessons to add or correct?
```

## Recording the coach's answer

Append to `LESSONS.md` (format in [lessons-format.md](lessons-format.md)):

```markdown
## COACH-04 · 2026-11-05 · given at C3 (ledger FP8D)
> "Never change a default that production relies on without asking me. New behaviour is opt-in."
- **Check:** Before changing any default in shared or production-facing code, verify production's current choice
  and keep it; new behaviour is opt-in until the coach approves. Raise at C3.
- **Applies to:** env-var defaults, config defaults, library-wide flags. Not to experiment-local scripts.
```

If the coach's answer confirms, overrules or retires an existing lesson, add a dated addendum under that lesson
quoting the coach, then update its status.

## Unattended runs

When no coach is available:

- Write each checkpoint into the ledger exactly as above, ending with `uncoached: proceeding on <rule ids>`.
- Prefer reversible steps; defer adopting into production, changing defaults or deleting anything.
- At the next session start, list the uncoached checkpoints first so the coach can review them.

## Subagents

You relay; the user coaches. Bring a subagent's result to the coach at the same checkpoints, in the same format,
naming the subagent and its run id. Pass the relevant COACH rules to subagents **verbatim** when you delegate, so
the coach's wording reaches the agent that acts on it. If a subagent did something outside its brief (changed a
default, used a resource it didn't own), report that at the next checkpoint rather than smoothing it over.
