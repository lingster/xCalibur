# xcalibur: Claude Code skills for xCalibur's research method

Two skills and one command that bring xCalibur's [OODA harness](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#ooda) and
[reflections](https://github.com/xArmory/xCalibur/blob/main/docs/reflections.md) into Claude Code.

| Component | What it does |
|---|---|
| [`ooda-coach`](skills/ooda-coach/SKILL.md) | A coachable experiment loop for any experiment-driven work. **You coach; Claude makes its reasoning coachable**: pre-registered prediction, falsifier and decision rule before each run; measured noise floors and error bars; a short after-action review after each run; one-line coaching requests at four checkpoints (before adopting or rejecting, after a surprise, before changing the plan or a default, at session end). |
| [`kernel-campaign`](skills/kernel-campaign/SKILL.md) | GPU inference and kernel speedup campaigns on top of `ooda-coach`: paired ABBA timing, per-session A/A margins, clustered bootstrap intervals, screen → confirm, winner's curse, profile-and-ceiling first, quantised-numerics gates, GPU ownership rules, 13 starter rules with their incidents, and a worked example. |
| [`/review-run`](commands/review-run.md) | Writes the after-action review for the latest run and brings it to the coach when it needs a checkpoint. |

## How it corrects itself

Claude can't change its weights or remember past sessions. The skills keep a persistent record instead:

- `LESSONS.md` in your project, append-only and dated, read **first** every session.
- Your feedback is recorded **verbatim** as `COACH` rules, the highest precedence. Claude's own prediction misses
  become candidate lessons (`INFERENCE`); you confirm them into `RULE`s, overrule them or retire them. Nothing is
  edited or deleted: changes are dated addenda.
- Each lesson carries a decision-time check, which Claude applies and cites at the decision it covers.

Unattended runs still write their checkpoints into the ledger, marked `uncoached`, for you to review next session.

## Install

In Claude Code:

```text
/plugin marketplace add xArmory/xCalibur
/plugin install xcalibur@xcalibur
```

Or in one step (Claude Code 2.1.275 or later):

```text
/plugin install xcalibur --marketplace xArmory/xCalibur
```

A local checkout works for testing: `/plugin marketplace add ./path/to/xCalibur`. See also
[Use with Claude Code](https://github.com/xArmory/xCalibur#use-with-claude-code) in the repository README.
