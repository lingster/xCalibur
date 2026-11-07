# Evidence: exact checks, claim types, obligations and receipts

"Models propose and measurements decide", and between them sits an **exact layer**: checks that are cheap,
deterministic and decisive, run before any expensive measurement. Adapted from xCalibur's
[what the connected checks establish](https://github.com/xArmory/xCalibur/blob/main/docs/formulation.md#constraints),
[evidence chain](https://github.com/xArmory/xCalibur/blob/main/docs/search.md#evidence),
[physical board](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#physical-board),
[studies](https://github.com/xArmory/xCalibur/blob/main/docs/studies.md) and the reflection
[a gate must engage with the construction](https://github.com/xArmory/xCalibur/blob/main/docs/reflections.md#substance-of-gates).

<a id="exact-checks"></a>
## 1. Exact checks before spending

The cheapest candidate to reject is the one you never build or run. In order:

1. **Legality / contract:** does the candidate satisfy the declared constraints (schema, resource budget, interface)?
   A checker should return *pass with every derived number* or *the first violation as a counterexample*, so a
   proposer can repair it (xCalibur: `warpspeed_check`, `execute`; for LLM work: schema validators, config linters).
2. **Reach:** can the change affect what is measured at all? Derive it from the code path (xCalibur `ReachMap`),
   and validate the map against captured builds before trusting it; a dispatch or routing change can alter the
   build even when the obvious fields didn't change.
3. **No-op detection:** diff the **effective artifact** against its parent: the resolved config, the rendered prompt,
   the compiled binary/SASS, the captured build inputs (`Capture.same`). Identical artifact ⇒ exact no-op: never time
   it. (In one campaign 8 of 9 screened "changes" were byte-identical builds that still "moved" ±1.4%.)
4. **Build/compile:** a failure is a contradiction *for that lowering in that context*, not proof the design is
   impossible.

Only then measure. Feed counterexamples and no-op verdicts back to whoever proposed the candidate.

## 2. Claim types stay distinct

Label every claim by what backs it, and never let one stand in for another:

| Claim | Establishes | Does not establish |
|---|---|---|
| proof / exhaustive check | the stated property over the stated domain | anything outside the domain (a scalar proof isn't a matrix-kernel proof; enumerating components isn't enumerating the system) |
| resource calculation | the accounting balances | correctness, absence of deadlock, the compiler's actual allocation |
| tool / compiler observation | what the tool did in that context | the design's value; an identity hash ≠ same packing, dispatch or workload |
| fixture / unit test | behaviour on those fixtures | behaviour on the real distribution |
| screen | a nomination | adoption |
| confirm (n sessions) | the delta under that protocol | transfer to another base, model or workload |

Positive results carry scope too: "passed on finite fixtures" is a test result with that scope. Validity is not task
success (a valid layout, grid or JSON can still be semantically wrong; keep the denominator visible).

## 3. Obligations and receipts

- Keep an **obligation register** per candidate: each obligation (correct output, within budget, no regression on
  workload X, numerics within gate…) with status `pass / pending / fail`, its evaluator and its falsifier. A filled
  field points to evidence; it isn't evidence.
- A passed check produces a **receipt bound to the source identity** (commit or content hash). A changed source
  can't inherit an old receipt: **a patch invalidates every receipt whose premises it touches**; re-run those.
- **A proposal can't declare itself correct, compiled or fast;** the evaluator supplies those.
- **Credit assignment:** a timing is credited to the action that produced that exact source. A candidate introduced
  by hand outside the search grammar is registered as a new action, not credited to an abstract one.
- **Protocol identity is data:** record the protocol version (timer, flush, warm-up, mode, margin) on every row.
  Repetition under the wrong protocol reduces sampling noise around the wrong number; when the protocol changes,
  requalify instead of pooling. An automated decision must use the margin of the mode that actually ran.
- **Ledger rows come from tool output** (JSON, CSV), never typed by hand.

## 4. Things that look like evidence and aren't

- **Replay vs live.** A replay of unchanged arithmetic is not a new live trajectory; post-hoc replays are not pooled
  with live runs.
- **Partial evaluations stay partial.** An early-stopped holdout or a subset run is labelled as such.
- **Same work?** Check the candidate does the same useful work as the reference (one fast candidate repeated work
  across tiles; another skipped it). Account for useful and issued work.
- **Plausible ≠ causal.** A visible symptom (spills, long outputs) next to a regression is a plausible contributor
  until an instrument isolates it.
- **Isolation changes context.** A stage measured alone loses its concurrency and resource context; reconcile with
  the complete operation before claiming the stage explains the total.
- **Leaderboard / holdout probing.** A check run is a real, prepared candidate, not a probe for gaming the board.
  Count holdout looks (general practice, not an xCalibur incident).
- **Evaluation order for anything with labels:** fit → solve → **freeze predictions** → score. Never look at
  labels before freezing.
- **Baseline on a fallback path.** Before claiming a speedup or a quality gap, check the baseline isn't running a
  generic fallback (wrong kernel, default config, truncated context).
- **The run did what you think.** Verify from logs that the intended config was in effect (batching, flags, prompt
  version, model alias, cache state).
- **Nondeterministic outputs.** When outputs can't be compared token-for-token, choose a quality check that tolerates
  it (labelled tasks, scored fields) and judge a candidate's error against ground truth relative to the reference's
  own error.

## 5. Determinism and differencing

- Before attributing drift to a change, check the reference is deterministic: repeat it and compare with stored
  outputs. If it isn't, the reference's own drift is the gate's floor.
- Cancel fixed costs by differencing (e.g. time N+1 and 1 steps and divide the difference by N).
- Reconcile end-to-end deltas with component predictions (Amdahl); a large mismatch means the model of the system is
  wrong, not that the measurement is.

## 6. Designing a new gate or required field

Before adding one, answer: **which recurring failure does it detect, using which artifact, at which decision point?**
A lightly edited copy of the last brief can pass a gate while omitting the new design's dominant cost; a citation
field can be filled without auditing the premise. More required prose doesn't fix either. Tie gates to concrete
objects (source identity, obligations, evaluators, receipts) so a stale transition is rejected mechanically.
Hard gates on anything public or scarce: submit only through the tool, check the live rules version matches the copy
you read (`rules_gate`), lint the shipped artifact for scaffolding (`lint`).
