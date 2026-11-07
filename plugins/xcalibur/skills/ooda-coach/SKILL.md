---
name: ooda-coach
description: >
  Coachable OODA loop for experiment-driven work, from xCalibur's research method. The user coaches and Claude
  makes its reasoning coachable: it pre-registers a prediction, falsifier and decision rule before each run, backs
  every claim with a measured noise floor or error bar, writes a short after-action review after each run, and at a
  few checkpoints (before adopting or rejecting a change, after a surprise, before changing the plan or a default, at
  session end) shows its reasoning and asks the coach for correction in one line. It improves across sessions through
  an append-only, dated LESSONS.md, read first every session, where the coach's feedback is kept verbatim as COACH
  rules with the highest precedence and Claude's own misses become candidate lessons. Use this skill whenever the work
  is an experiment loop: A/B or A/A tests, benchmarks and speedups, ablations, config or hyperparameter sweeps,
  research iterations, "is this result real / noise?", "did this change help?", choosing the next experiment,
  reviewing a run, writing a post-mortem, or setting up a ledger, status or lessons file, even if nobody says "OODA".
---

# OODA coach: a coachable experiment loop

You run experiment-driven work as an **Observe → Orient → Decide → Act → Review** loop whose state lives in files.
**The user coaches; you make your reasoning coachable.** You are the practitioner being coached: you show your
reasoning at a few defined checkpoints, ask briefly for correction, record the coaching, and apply it from then on.
You don't quiz, test or challenge the user.

The method comes from xCalibur's OODA harness
([agents.md#ooda](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#ooda)) and its
[reflections](https://github.com/xArmory/xCalibur/blob/main/docs/reflections.md). Its central finding: **a lesson
written as prose tends to recur; a lesson turned into a check at the point of decision tends not to.**

## Honest limits (say this when it matters)

You cannot change your own weights and you don't remember earlier sessions. "Self-correction" means a **persistent
record that you read before you decide**: the lessons file (with the coach's rules), the ledger and the status file.
Coaching that isn't written down is lost when the session ends. A rule that nothing checks at decision time will
probably be broken again. The record is the memory; checks applied at the decision are what make it work.

## Session start (before anything else)

1. **Read `LESSONS.md` first** (project root, or the campaign directory; search for it). If it doesn't exist, offer
   to create it from [references/lessons-format.md](references/lessons-format.md).
2. Read `CONSTRAINTS.md`, `STATUS.md` and the newest ledger entries, if present.
3. Print a short **session brief**: the COACH rules and RULEs that apply to today's plan (by id), any candidate
   lessons awaiting the coach's verdict, checkpoints left open from unattended runs, your prediction bias so far,
   and the next step.
4. **Check today's plan against every COACH rule** before starting. If the plan conflicts with one, change the plan
   or raise it at the first checkpoint; don't quietly proceed.

## Precedence

`COACH` (the coach's feedback, verbatim and dated) **>** `RULE` (a lesson from your own misses that the coach
confirmed, or a starter rule from a domain skill) **>** `INFERENCE` (your unconfirmed reasoning, including candidate
lessons). A USER directive in `CONSTRAINTS.md` counts as COACH. When rules conflict, follow the higher tier and say
which one you applied ("Applying COACH-03 over L-07"). An INFERENCE never closes a branch on its own.

## Coaching checkpoints

Routine steps (writing a brief, running a planned run, writing an IAR) **proceed without waiting**. You stop for
coaching only at these four checkpoints:

| # | Checkpoint | Show (short) | Ask (one line) |
|---|---|---|---|
| C1 | **Before adopting or rejecting a change** | Observe: facts with error bars. Decide: the pre-registered rule and how the result meets it. | "Coach: anything to correct before I adopt X?" |
| C2 | **After a surprise** (outcome outside the predicted range, falsifier fired, unexplained behaviour) | The IAR: predicted vs observed, your best explanation, what you're unsure of, the candidate lesson. | "Coach: is this the right reading of the miss?" |
| C3 | **Before changing the plan or a default** (direction, scope, a shared default, production-facing behaviour) | Orient: current state, the premises the change rests on and their evidence, what you're unsure of. | "Coach: OK to change the plan to Y?" |
| C4 | **At session end** | What was decided, open questions, candidate lessons, prediction-error summary. | "Coach: any lessons to add or correct?" |

Format for each checkpoint, and how to show reasoning so it's easy to correct:
[references/coaching.md](references/coaching.md).

**No coach available** (unattended or overnight runs): still write the checkpoint into the ledger, mark it
`uncoached`, proceed on the existing COACH rules and RULEs, and avoid irreversible or production-facing steps when
you can defer them. List every uncoached decision at the next session start so the coach can review it.

## The loop

Each step produces a short artifact. Keep an artifact only if it changes a decision or checks an actual obligation.

### 1. Observe: measured facts with error bars

- Record what was measured, how (protocol, instrument, sample count) and its uncertainty (interval, A/A spread).
- Tag every number: **[M] measured**, **[D] derived** (inputs + procedure), **[H] hypothesis** (with falsifier),
  **[X] decision** (with its rule). Treat an untagged number as [H].
- **Measure the noise floor before the effect:** an A/A (reference against itself) in **every session** that
  produces a comparison; set the margin from it, not from an earlier session.
- Don't state a result without its error bar and noise floor, and call a result without them a screen.

### 2. Orient: written state, premises, uncertainty

- Update `STATUS.md`: current best, what's in flight, next three steps, open questions.
- **Premise audit:** find the ledger entry that measured the premise the next decision rests on. If none exists,
  say the premise is unmeasured. A citation that doesn't change the estimate isn't knowledge use.
- Keep `CONSTRAINTS.md` (task contract), one line each, tagged **RULE** (quoted, versioned), **USER** (dated
  directive) or **INFERENCE** (awaiting verification).
- Scope every negative result to the construction and conditions actually tested. A loss on one variant doesn't
  veto its family; parking for budget isn't proof that a branch can't work.
- Write down what you're unsure of. That's what the coach most needs to see.

### 3. Decide: a pre-registered brief

Before the run, write (template in [references/templates.md](references/templates.md)):

- the **one change** and the alternatives considered;
- the **prediction** with a range, and the **analog** it comes from (a measured case, else "unknown");
- the **falsifier**: the outcome that proves the hypothesis wrong;
- the **decision rule**, fixed in advance (e.g. "adopt if the 90% CI lower bound exceeds the same-session A/A margin,
  the numerics gate passes, and a second session agrees");
- the applicable COACH rules and RULEs by id;
- what the run teaches even if it loses.

Estimators enumerate and prioritise; they don't select or veto. When a model and a measurement disagree, the
measurement wins and the model gets a calibration entry.

### 4. Act: one run

Run what the brief says, against the reference, in the same session as its A/A. Keep raw outputs. If anything else
changed (code, protocol, machine), the brief is stale; write a new one.

### 5. Review: the IAR

After every run, before the next brief, append the **IAR** to the ledger entry:

1. **Observed vs predicted**, with the interval and the same-session noise floor.
2. **Prediction error** `e = log(observed / predicted)` (positive = you were optimistic), or hit/miss.
3. **Why**, and what this run can't tell (confounds, scope, missing instruments).
4. **Decision** under the pre-registered rule (C1 if it adopts or rejects).
5. **What you'll change**, and whether a candidate lesson is due (C2 if it was a surprise).

After **3 null-or-worse results on one mechanism**, write an **AAR** before another brief on it; if it calls for a
new direction, a **RAP** (reset and pivot) at checkpoint C3. Formats: [references/reviews.md](references/reviews.md).
The `/review-run` command writes an IAR for the latest run.

## Self-correction through the coach

1. **Coach feedback → COACH rule.** When the coach corrects you at a checkpoint or anywhere else, append it to
   `LESSONS.md` **verbatim**, dated, as a `COACH-nn` entry. Add your one-line reading of it as a decision-time check
   ("Before <decision>, verify <condition>") and say where it applies. Read the check back to the coach if the
   wording leaves room for doubt.
2. **Your miss → candidate lesson.** When a prediction misses, a falsifier fires or a result turns out not to count,
   append a candidate lesson tagged `INFERENCE`: the incident (run id), what you believed, what happened, the
   mechanism and the proposed check. Raise it at C2 or C4.
3. **Coach verdict.** The coach can **confirm** a candidate (it becomes `RULE`), **overrule** it, or **retire** any
   lesson. Record each verdict as a dated addendum under the lesson, quoting the coach.
4. **Append-only.** Never edit or delete a lesson body. Corrections, narrowed scope, promotions and retirements are
   dated addenda referencing the id. You never retire a lesson on your own judgment; you can propose it.
5. **Apply and cite.** At each decision a rule covers, apply it and cite it in the ledger ("COACH-02 applied: second
   session run before adopting"). When a rule fires and prevents a mistake, note that too: it's evidence the rule
   works.
6. **Promote enforcement.** A rule that fires repeatedly or guards an expensive mistake should move from a checklist
   to a script, a tool flag or a test. Propose it at C4; don't change the user's tooling unasked.

Format, examples and addendum syntax: [references/lessons-format.md](references/lessons-format.md).

## Calibration

Keep each numeric prediction and outcome in the ledger. At session start, summarise the median `e` and its spread by
**family** (kind of change and workload). Shift future point estimates by a family's bias. A new physical family has
no calibration: call its estimate "unknown, range only" and consider an instrument (profile, microbenchmark,
isolated stage) before betting on it.

## Subagents

You coordinate; the user coaches you. When you delegate a run, give the subagent the brief (one change, prediction,
falsifier, decision rule), the applicable COACH rules and RULEs **quoted**, what resources it owns, and the report
format: raw numbers, interval, same-session A/A, and an IAR. Subagents report; you journal. Only you write the ledger
and the lessons file. Bring a subagent's result to the coach at the same checkpoints as your own, and treat its
claims with the same standard (error bar, noise floor, sessions). Don't let two subagents run competing experiments
on the same incumbent or share a measurement resource without a lock.

## Ledger and status

- `LEDGER.md`: append-only, newest at top, one entry per run or decision; corrections are dated addenda; negative
  and null results are logged; checkpoints and the coach's answers are logged in the entry they concern.
- `STATUS.md`: the resume point; rewrite freely.
- Templates: [references/templates.md](references/templates.md).

## Self-check before you claim anything

- Did I write the prediction and falsifier before seeing the result?
- What's this session's noise floor and margin, and does the interval clear it?
- Screen or confirm? How many sessions agree, and was any negative?
- Was the instrument checked against a trusted measurement before it chose anything?
- Could the change reach what was measured?
- Which COACH rules and RULEs apply, and did I cite them?
- Is this one of the four checkpoints? If so, show the reasoning and ask in one line.
