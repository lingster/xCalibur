# Reviews: IAR, AAR, RAP and reflections

Different scales of review answer different questions. Adapted from xCalibur's
[IAR / AAR / RAP](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#reviews) and
[reflections](https://github.com/xArmory/xCalibur/blob/main/docs/reflections.md).

## IAR: immediate after-action review (every run)

Written right after a run, before the next brief. Five short items:

1. **Observed vs predicted.** Both numbers, the interval, the same-session noise floor.
2. **Model error.** `e = log(observed / predicted)`; positive = the prediction was optimistic. Or hit/miss.
3. **Inference and limits.** What it means; confounds; what this run cannot tell (scope, missing instrument,
   unreached code path).
4. **Decision** under the pre-registered rule.
5. **What changes.** Next action, ledger/status updates, and whether a candidate lesson is due. A surprise goes to the coach at checkpoint C2.

Rule: no new brief while a run is waiting for its IAR.

## AAR: after-action review (a mechanism or a campaign)

Trigger: **3 null-or-worse results on one mechanism**, a change of direction, a predeclared stop condition, or the
user asking "why isn't this working?".

- Reconstruct the decisions across runs: what each brief predicted, what it tested, what it got.
- Separate **what was tested** (exact constructions and conditions) from **what remains open**.
- Find the **habit** behind the repeated failure (e.g. "kept tuning the component we'd already measured at the
  ceiling", "kept trusting the proxy").
- Output: a candidate lesson expressed as a check (for the coach to confirm), and a scoped decision per branch: **open**, **parked** (reason + restore
  path) or **closed** (only with a witness covering the whole branch, not a sampled loss).

## RAP: reset and pivot

When the AAR calls for a new direction:

- Restate the objective and constraints (re-read CONSTRAINTS.md; check USER directives are current).
- Recompute headroom from measurements (profile, ceilings), not from the old plan's estimates.
- Revise whatever misled the previous decisions: the estimator, the catalog, the brief template.
- Name the **new question**, its **falsifier**, milestones and a **stop-loss**.
- Append what happened at close-out, including a failed first prediction.

## Reflection (occasional, cross-campaign)

Pause execution and ask whether the method is learning:

- Are we asking better questions, or re-running variants of the last design?
- Did retrieved knowledge change any estimate, or was it only cited?
- Did a narrow negative result become a veto on a whole family?
- Did a lesson change the next decision? Did a later correction reach every place the lesson was used?

Group the history by **what was physically varied** (structure, precision, schedule), not by the latest proposal's
name; repeated assumptions become visible that way. Keep the reflection as written; add later corrections as
addenda.
