# Templates: brief, ledger, status, constraints

Copy these into the project. Keep each artifact short; a field earns its place by changing a decision.

## Brief (written before the run; goes into the ledger entry)

```markdown
### <ID>: <one change, in words> ⏳ briefed <date>
- **Kind:** bet (must beat the incumbent) | calibration (range only, not scored as a forecast). *(Fields marked
  "when relevant" can be omitted from a routine brief.)*
- **Change:** <the one variable, or one registered coupled bundle: parts>; reference = <incumbent id / commit>.
  Alternatives considered: <2-3, one line each>. Rank on the agenda: <gain per check, information value>.
- **Prediction:** <number with range, or direction> [H]. **Analog:** <ledger id it comes from, or "unknown">.
- **Falsifier:** <the outcome that proves the hypothesis wrong>.
- **Pre-mortem** (when relevant): <the likeliest way this run misleads: no-op, wrong protocol, proxy, cache state,
  leakage>.
- **Knowledge used** (when relevant): <query → records → the fact that changed this choice>, or "none".
- **Exact checks before running** (when relevant): <legality, reach, artifact differs from parent: how verified>.
- **Decision rule:** <e.g. adopt if 90% CI low > same-session A/A margin AND numerics gate passes; bind after a 2nd
  session, none negative>.
- **Protocol:** <instrument, workload, pairs/samples, mode>; A/A planned in this session: yes.
- **Budget / risk** (when relevant: scarce runs; coach approves, e.g. an xCalibur `Card`): <runs, GPU-hours, what could
  break>.
- **Rules applied:** <COACH-ids, RULE ids> (conflicts resolved by COACH > RULE > INFERENCE).
- **Teaches even if it loses:** <one line>.
```

## Ledger entry after the run (append to the brief; never rewrite it)

```markdown
- **Run:** <run id> · <commit / source hash> · <workload id> · <device> · <protocol version> · <date-time>
- **A/A (same session):** <ratio [CI]> → margin <value>.
- **Observed:** <ratio [CI]> [M]; secondary metrics <...>; numerics <...> vs gate <...>.
- **Predicted vs observed:** <pred> vs <obs>; e = log(obs/pred) = <value> (or hit/miss).
- **Falsifier fired?** <yes/no>.
- **IAR:** means <...>; can't tell <...>; decision <adopt-pending / confirm / reject / park / instrument> under the
  rule; next <...>; candidate lesson: <none / L-nnn>.
- **Checkpoint (if C1/C2/C3):** shown <date>; coach: "<verbatim answer>" -> <COACH-nn / verdict on L-nnn>, or `uncoached: proceeding on <ids>`.
```

Status marks for the entry title: ⏳ briefed / running, ✅ adopted (pending confirm), ✅ CONFIRMED (n sessions),
❌ rejected (why: speed / numerics / gate), ⏸ parked (reason; restore path), 🔬 instrument needed.

## LEDGER.md header

```markdown
# <Campaign> ledger (append-only, newest at top)

Rules
- One entry per run or decision. Never delete or overwrite; corrections are dated addenda referencing the id.
- Log negative and null results.
- Decision rule (fixed): <rule>. A/A noise floor of the same session is reported next to every delta.
- Bind on >= 2 sessions, none negative. Single-session adoptions are "adopted, pending confirm".
- Every entry states a prediction and its falsifier before the run.
- Only <coordinator> writes this file; subagents report. Result rows come from tool output, never typed by hand.
- Every row carries its protocol version; a changed protocol is requalified, not pooled.
- Structured rows: <path to the tool's CSV/SQLite/JSONL>; artifacts: <path>/<run id>/.
```

## STATUS.md (the resume point; rewrite freely)

```markdown
# <Campaign> status (read this first to resume; LESSONS.md before it)
Updated <date>.

## Current best
- <incumbent>, <metric> [CI], bound on <n> sessions. Gate: <numerics gate>. Margin rule: <rule>.

## In flight
- <who / which agent>: <run>, <resource owned>, expected done <time>.

## Next 3
1. <step> — why: <what it decides>.

## Handoff
- Exact source: <commit / hash>. Unresolved obligations: <list>. Retained alternatives: <ids>. Next valid step: <...>.
- Blocked mechanisms (open AARs): <mechanism → AAR id>.

## Open questions / unmeasured premises
- <premise> [H]: what would measure it.

## Frontier (estimates tagged; ranked by gain per check)
| # | Branch | Kind | Expected | Cost | Info value | Evidence tag (analog id or "unknown") | Status (open/parked: restore path) |
```

## CONSTRAINTS.md

```markdown
# Constraints (one line each; tag RULE / USER / INFERENCE)
- RULE (<source, version>): "<quoted rule>"
- USER (<date>): <directive>, e.g. "device 1 belongs to someone else; never touch it".
- USER (<date>): holdout runs once, on the final configuration only.
- INFERENCE (<date>, unverified): <claim> — verify by <check>. Cannot close a branch until verified.
```

## Calibration table (optional; can be generated from the ledger)

| Family | n | median e | spread | note |
|---|---|---|---|---|
| structural skips | 4 | -0.01 | 0.02 | well calibrated |
| low-precision GEMM | 2 | +0.15 | — | optimistic; widen ranges |

## Knowledge claim (KNOWLEDGE.md or JSONL; see [knowledge.md](knowledge.md))

| id | claim | source (run / URL) | scope | evidence type | status | mechanism tags |
|---|---|---|---|---|---|---|
| KN-01 | <one sentence> | <E5–E7> | <model, version, workload> | confirm (4 orders) | established | <tags> |

## Obligation register (per candidate; see [evidence.md](evidence.md))

| obligation | evaluator | falsifier | status | receipt (source hash) |
|---|---|---|---|---|
| <output valid / no workload regresses / numerics within gate> | <tool or test> | <outcome that fails it> | pass / pending / fail | <hash> |
