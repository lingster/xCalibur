# LESSONS.md: format and rules

`LESSONS.md` is the campaign's correction path. Claude reads it **first** at the start of every session, checks the
day's plan against it, and applies each rule at the decision it covers. It is append-only and dated.

## Tiers and precedence

| Tier | Id | Source | Who changes it |
|---|---|---|---|
| **COACH** | `COACH-nn` | The coach's feedback, quoted verbatim and dated | Only the coach (by a new COACH entry or a verdict addendum) |
| **RULE** | `L-nn`, or a domain prefix such as `K-nn` | A candidate lesson the coach confirmed, or a starter rule shipped by a domain skill | Coach confirms, overrules or retires |
| **INFERENCE** | `L-nn` with `Tier: INFERENCE` | Claude's own miss, proposed as a lesson; awaiting the coach | Claude proposes; coach decides |

Precedence: **COACH > RULE > INFERENCE.** A USER directive in `CONSTRAINTS.md` counts as COACH. When two entries
conflict, follow the higher tier and cite both ("Applying COACH-03 over K-08"). An INFERENCE can guide a choice
between reversible options but can't close a branch or block the coach's plan.

## Where it lives

- Default: `LESSONS.md` at the project root.
- A campaign with its own directory can keep its own `LESSONS.md` there; read both, campaign first.
- Domain skills can ship **starter rules** (e.g. kernel-campaign's
  [starter-lessons.md](../../kernel-campaign/references/starter-lessons.md)). At campaign start, offer to copy the
  applicable ones in as `RULE`, keeping their ids and incidents. The coach can overrule or retire any of them for this
  project.

## File header (create once)

```markdown
# Lessons (append-only; read first every session)

Precedence: COACH > RULE > INFERENCE. USER directives in CONSTRAINTS.md count as COACH.
- Never edit or delete an entry body. Verdicts, corrections, scope changes and retirements are dated addenda under
  the entry, referencing its id.
- COACH entries quote the coach verbatim. Only the coach confirms, overrules or retires a lesson.
- Every entry carries a decision-time check: "Before <decision>, verify <condition>; if not, <action>."
- `Enforced by` says where the check runs; `checklist` means Claude applies it from this file; `tool`/`test` names
  the code. Ids are never reused.
```

## A COACH entry

```markdown
## COACH-02 · 2026-11-05 · given at C1 (ledger K3)
> "One good session isn't enough. Don't call anything adopted until a second session agrees."
- **Check:** Before marking a change adopted, verify >= 2 sessions with CI low above that session's margin and none
  negative; otherwise mark "adopted, pending confirm".
- **Applies to:** every speed adoption. Not to bit-exact refactors with no speed claim.
- **Enforced by:** checklist (C1).
```

## A candidate lesson from Claude's miss

```markdown
## L-07 · 2026-11-05 · Tier: INFERENCE · Re-run the A/A in every session
- **Incident:** run 20261105T142735Z (second confirm of B3). The margin carried over from session 1 was 1.003; this
  session's A/A offset was -0.42%, so the right margin was 1.008.
- **Believed → happened:** "the noise floor is a property of the machine" → it moves between sessions (reloads,
  clocks, other tenants).
- **Check:** Before using a compare for a decision, verify a same-session A/A for the same code, model, mode and
  device; margin = 1 + 2 x |worst A/A offset|. If missing, run it first.
- **Scope:** paired timing. Not needed for bit-exact numerics checks.
- **Enforced by:** checklist; proposed: `compare` refuses without an A/A row.
- **Status:** awaiting coach (raised at C4, 2026-11-05)
```

Field notes:

- **Incident** names a run id or ledger entry, so the lesson can be audited against the evidence.
- **Believed → happened** records the reasoning error, not only the outcome. That's the part that generalises.
- **Check** must be testable at the moment of decision. "Be careful with noise" is not a check.
- **Scope** prevents over-generalising: say where it doesn't apply.

## Addenda: the only way an entry changes

```markdown
### L-07 addendum · 2026-11-06 · coach verdict: confirmed
> "Yes, make that a rule."
- Tier: RULE.

### K-08 addendum · 2026-11-12 · coach verdict: narrowed
> "The FP8 gate is fine for drift, but I also want the external-label MAE before any FP8 adoption."
- Check extended: FP8 adoption also needs external-label MAE unchanged (CI on the difference includes 0).

### L-03 addendum · 2026-11-20 · coach verdict: retired
> "Drop this, the harness enforces it now."
- Status: retired (enforced by tool, commit abc1234). Body kept for the record.
```

Status values: `awaiting coach`, `active`, `narrowed (see addendum)`, `superseded by <id>`, `overruled`,
`retired (reason)`. Claude may **propose** a retirement (as an addendum marked "proposed"), never perform one.

## When Claude writes a candidate lesson

- an outcome outside the predicted range, or a falsifier firing;
- a result that turns out not to count (no A/A, uncalibrated oracle, single session, unreachable change);
- a process slip, even a harmless one (unlocked device use, a waiter that never woke, a silent default change);
- a near-miss caught by review, or the same question answered twice because the answer wasn't recorded.

An outcome inside the predicted range is calibration data for the ledger, not a lesson.

## Session-start brief (example)

```text
Session brief
- COACH rules in play: COACH-02 (2 sessions before adopting), COACH-04 (no silent default changes)
- RULEs in play: K-02 A/A every session; K-03 calibrate oracles; K-12 one GPU process per agent, named GPU, lock
- Awaiting coach: L-07, L-08
- Uncoached checkpoints from the last run: 1 (C1 on B4, ledger K5)
- Prediction bias: speed median e = -0.03 (n=9); numerics 2 misses of 5
- Plan vs COACH rules: no conflict
- Next step: brief for B5
```

## Why append-only

A rewritten lesson hides why the prior changed, and later decisions that relied on the old wording can't be found
and re-checked. Keeping the original next to its corrections lets a reader, or a future session, recover the whole
chain: assumption, test, contradiction, revised scope, and what the coach said.
