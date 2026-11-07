# Proposers: planner, refiner, implementer and challenger

When a model (an LLM, a surrogate, a subagent) proposes candidates, structure its role so that its guesses become
checkable and its record improves it. Adapted from xCalibur's
[planner/refiner](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#planner-refiner),
[Anvil](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#anvil),
[Rebel](https://github.com/xArmory/xCalibur/blob/main/docs/search.md#rebel) and the
[Gemma 4 / TG4DAC study](https://github.com/xArmory/xCalibur/blob/main/docs/studies.md#gemma4).

## 1. Separate the roles

| Role | Does | Gets back |
|---|---|---|
| **Planner** | chooses the design: decomposition, structure, the one change | exact diagnostics on its plan |
| **Refiner** | repairs a plan against a checker's counterexample | pass, or the next counterexample |
| **Implementer** | builds the artifact (code, config, prompt) from the accepted plan | build, correctness and timing evidence |
| **Challenger** | attacks a committed candidate with a registered query and falsifier | the referee's verdict |
| **Referee / evaluator** | answers exactly (checker, test, benchmark); never a player | — |

Every result keeps its attribution to a role (planner idea, refiner fix, implementer code). The models that propose
designs are distinct from the workload being optimised. Delegated roles return artifacts and evidence; only the
coordinator writes the ledger and lessons.

## 2. Give a proposer the right context

A proposer given only the latest snapshot chases noise (in one campaign every such proposal lost). Assemble:

- the **whole history** compactly (`DesignGraph.listing()`: one line per design plus one-edit effects);
- **retrieved** nearest designs and code, each **with its evidence grade** ("4-seed screen" vs "12×2 confirm"), and
  say in words which rows are screens and which are confirms;
- the **incumbent** and the reach note (what each knob actually touches, with the code lines it controls);
- the proposer's **own past plans, verdicts, fixes and results** (planner memory);
- measured transfer facts (what did and didn't carry over between bases).

## 3. A checker it can call

Expose the exact layer as a tool: "pass, with every derived number" or "the first violation, as a counterexample"
([evidence.md](evidence.md#exact-checks)). A checker turns guesses into fixes: the refiner repairs once, and a plan
that still fails is dropped. A checker pass is scoped to what it checks (a plan can pass resource accounting and
still spill when compiled).

xCalibur's `PlannerLoop` ([`planner.py`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/planner.py)) wires
this up against any OpenAI-compatible endpoint: forced tool calls (`chat`, `tool`), planner and refiner models, a
`check` function, `retrieve`, a reach note, memory with `record_result`, and a **control file re-read every round**
for sampling temperature (e.g. 0.2 careful, 0.7 explore) so a human can steer mid-run.

## 4. Judge the proposer, not just its proposals

- **Compare against cheap baselines at matched budget:** a surrogate ranker and a random legal sampler with the same
  build and measurement budget. Without that, a proposer's wins can't be attributed to it.
- **Track telemetry:** episodes finished, tool calls that ran, plans passing the checker, plans that fixed an error,
  and **legal-but-degenerate** outputs (valid but trivial, e.g. no specialisation at all).
- **Use models for new constructors, not new knob values.** Knob values are what search and surrogates do well.
- **Check the model's own estimates.** Model-estimated resources and speedups are hypotheses for the checker.
- **Training a small proposer** on tool-run traces: hold out a whole task family to test generalisation; split by
  source ancestry before generating examples; keep construction replays apart from real search traces; a
  construction label ("passes its checks") is not a performance label ("is faster"); compare against deterministic
  and untrained controls under the same budget.

## 5. A challenger (red team)

After a constructor commits a candidate, a challenger picks a **registered query** that names the binding it doubts
and the falsifier: wrong value association, missing output, conflicting writer, premature overwrite, no progress,
unsupported tool claim. A trusted referee answers. Use negative controls (planted defects) to show the challenger
can find them. Surviving the challenge is verification, not a measured gain: it supplies no unmeasured payoff.
"Keep the incumbent" is always an explicit option.

## 6. Subagent briefs

When delegating any of these roles, pass: the brief (one change, prediction, falsifier, decision rule), the
applicable COACH rules **verbatim**, the resources it owns (GPU index, port, directories), the checker or evaluator
to use, and the report format (raw numbers, interval, same-session A/A, IAR). Don't let two proposers run competing
experiments on the same incumbent or share a measurement resource without a lock.
