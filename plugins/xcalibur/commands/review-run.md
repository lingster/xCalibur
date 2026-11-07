---
description: Write the after-action review (IAR) for the latest experiment run, compare it with its registered prediction, and bring it to the coach if it needs a checkpoint
argument-hint: "[run id or ledger entry id]"
---

Use the `ooda-coach` skill (and `kernel-campaign` too if this is GPU/kernel work). The user is the coach; make your
reasoning coachable.

Target run: $ARGUMENTS (if empty, the newest ledger entry or run directory that has results but no IAR).

1. Read `LESSONS.md` first, then the run's brief (prediction, falsifier, decision rule) in the ledger, then its raw
   results and the same-session A/A. If no prediction was registered before the run, say so and mark the entry
   `unregistered`: it can nominate, not confirm.
2. Append the IAR to the ledger entry (never rewrite earlier text): observed vs predicted with interval and noise
   floor; `e = log(observed/predicted)` or hit/miss; whether the falsifier fired; why, and what the run can't tell;
   the decision under the pre-registered rule; what you'll change. Cite the COACH rules and RULEs you applied.
3. If the outcome was a surprise (outside the range, falsifier fired, or the result doesn't count), append a
   candidate lesson (Tier: INFERENCE) to `LESSONS.md` with a decision-time check.
4. If the decision adopts or rejects a change, or the run was a surprise, show the checkpoint (C1 or C2 format) and
   end with a one-line request, e.g. "Coach: anything to correct before I adopt X?". Record the coach's answer
   verbatim as a COACH entry or verdict addendum. If no coach is available, mark the checkpoint `uncoached`.
5. Update `STATUS.md` if the decision changed the current best or the next steps, and refresh the knowledge records
   the run touched (claims tested, design-graph node with its evidence grade; see the skill's `knowledge.md`).
6. If this is the third null-or-worse result on one mechanism, say an AAR is due before another brief on it.

Keep it to one screen.
