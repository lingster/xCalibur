# Toolkit: sub-skills, modules, notebooks, data and scripts

Everything xCalibur ships that an experiment loop can use, and when to reach for it. Paths are relative to an
[xCalibur checkout](https://github.com/xArmory/xCalibur). If the package isn't installed, implement the protocol by
hand (it's short) rather than skipping it.

## Sub-skills and commands (this plugin)

| Name | Use when |
|---|---|
| `ooda-coach` (this skill) | any experiment loop: benchmarks, A/B or A/A tests, sweeps, prompt/config tuning, ablations |
| `kernel-campaign` | GPU inference or kernel speed work; loads on top of `ooda-coach` |
| `/review-run [run id]` | write the IAR for the newest run and bring it to the coach if it needs a checkpoint |

References of this skill: [coaching](coaching.md) · [lessons format](lessons-format.md) · [reviews](reviews.md) ·
[templates](templates.md) · [knowledge](knowledge.md) · [agenda](agenda.md) · [evidence](evidence.md) ·
[proposers](proposers.md) · [starter rules](starter-rules.md).

## Install

```bash
python -m pip install -e ".[test]"          # core: numpy only
python -m pip install -e ".[search]"        # + lightgbm for Surrogate / Game
python -m pip install -e ".[retrieval]"     # + sentence-transformers for embedding_gemma
python -m pip install -e ".[gpu]"           # + torch for Bench / LongBench
python -m pip install -e ".[cute]"          # + nvidia-cutlass for as_cute_layout / PyCuTe
python -m pip install -e ".[notebooks,test]" # + pandas, matplotlib, nbclient to rebuild notebooks
```

Then `import xcalibur as xc`; every name below is `xc.<name>`. API contracts:
[getting-started.md](https://github.com/xArmory/xCalibur/blob/main/docs/getting-started.md).

## Modules by need

| Need | API | Notes |
|---|---|---|
| µs kernel paired timing, A/A | `Bench(flush_mib, warmup)`: `.paired`, `.aa`, `.null_us`; `paired()` one-shot | CUDA-graph replay, cold-L2 read flush; null subtraction off by default |
| ms–s passes (prefill, requests) | `LongBench`: `.paired(c, r, pairs, block)`, `.aa`; `warm_repeat`, `small_first`; `cuda_clock`, `wall_clock`; `smi_snapshot` | ABBA, device vs host time, power/clock snapshots |
| intervals and margins | `geomean`, `bootstrap_ci(ratios, level, clusters=)`, `aa_margin(aa_ratios, k, step)` | cluster by input id; inputs must be finite and positive |
| binding a gain | `Evidence().add(load, identity, gain, design)`, `.bind(min_samples, margin)` | ≥n samples, mean > margin, none negative |
| does a change reach the build? | `ReachMap(loads, cutoff_field, fields)`: `.reaches`, `.identity`, `.build_of` | validate against captured builds |
| ship exactly what was measured | `Capture()` (`.keyed`, `Capture.same`), `replace_assignment(source, name, value)` | byte-preserving AST edit; re-qualify after export |
| run card, rules, timed-path lint | `Card(...).render()`, `rules_gate(live, read)`, `lint(source)`, `LINT` | Card fields: parent, target, estimate, budget, risk, noise, teaches, falsifier |
| ranking candidates | `Surrogate(members, backend="lightgbm"/"ridge")`: `.fit`, `.predict`, `.ucb`, `.top`; `Ridge`; features `bits`, `one_hot` | orders measurement; never decides |
| screen → confirm → bind loop | `Game(...)`: `.round()`, `.confirm(designs)` | frozen incumbent; new reference ⇒ new Game |
| design graph, retrieval | `Node`, `DesignGraph`: `.one_edit`, `.effects`, `.listing`; `Retriever(nodes, embed)`, `embedding_gemma()` | see [knowledge.md](knowledge.md) |
| model proposer loop | `PlannerLoop`, `chat`, `tool` | OpenAI-compatible endpoint; see [proposers.md](proposers.md) |
| GPU plan accounting | `warpspeed_check(squads, resources, arch=)`, `plan_of`, `launch_register_cap`, `Arch`, boards `H200`, `RTX3090`, `A6000`, `RTX6000ADA`, `L40S`, `RTX4090` (`ARCHS` keys / `arch=` strings lowercase), `SCHEMA`, `execute` | accounting, not a performance model |
| exact bit-level maps | `gf2`: `rank`, `apply`, `compose`, `inverse`, `from_function`, `rotation`, `equal`, `packing_transform`, `compatible`, `join_all`, `as_cute_layout` | proofs for decoders and packing |

## Notebooks (CPU; `notebooks/`, rebuilt by `python notebooks/build.py [prefix]`)

| Notebook | Read it for |
|---|---|
| `01_decoders_over_gf2` | exact equality and composition; deriving a packing instead of searching it |
| `02_warpspeed_plans` | checking a plan before compiling; counterexamples |
| `03_measurement_and_noise` | ABBA, bootstrap, screen vs confirm, **winner's curse** with real data |
| `04_moe_on_h200` | multi-load geomean, **interactions** (+0.4%, +0.9% → +3.3%), landscape of 514 designs |
| `05_the_measured_game` | the `Game` loop on a planted 2% gain |
| `06_tg4dac` | planner + refiner + checker; kernel vs end-to-end gain (Amdahl) |
| `07_knowledge_graphs` | design graph, neighbourhood, field effects, retrieval with evidence grades |
| `08_layout_ownership` | correct ≠ fast; check coverage, not just shapes |
| `09_structural_search` | failures in the ledger; missing latency ≠ zero; don't pool separate campaigns |
| `10_operator_and_model_feedback` | complete operators (incl. backward); model repairs from a counterexample |
| `11_arc_transfer` | fit → solve → freeze → score; validity ≠ task success |
| `12_prefill_on_sm86_sm89` | adopt rule, answer-drift gate, profile first, clustered bootstrap |
| `docs/writeups/TG4DAC_WarpSpeed.ipynb` | full SM120 record; imports a private tool, so read-only |

<a id="data-analogs"></a>
## Data as analogs for predictions (`data/`; ratio > 1 = candidate faster; empty = failed, not zero)

Use these when a brief needs an **analog** and the project has none yet; quote the file and its scope line from
`data/README.md`.

| File | Analog for |
|---|---|
| `noise.csv` | repeated identical builds; screen vs confirm shrinkage |
| `moe_sweep.csv` (545 rows) | screen→confirm regression to the mean; building a design graph |
| `moe_interaction.csv` | interaction effects between measured moves |
| `structural_search.csv` | 36 ownership + 40 compute-family records, failures included; ratios vs each campaign parent |
| `deterministic_search.json` | base rate for a no-model campaign: 44 compiled, 28 measured, 0 demonstrated gains |
| `decoder_search.json` | exact search denominators, solution counts and setup vs search costs |
| `moe_judge.csv`, `scores.json` | judge rows (SGLang, candidate, analytical bound) and archived aggregate scores |
| `load_patterns.csv`, `load_composition.csv` | 168 copy-only TMA patterns; 54 full-operator load rows (separate protocol) |
| `l4_f1.csv` | 9 revision comparisons: correct but slower in 8 (correct ≠ fast) |
| `kev_prefill_runs.csv`, `kev_prefill_profile.csv` | prefill margins, drift, time shares on consumer cards |
| `w8a8_sm86_microbench.csv`, `w8a8_sm86_diagnosis.csv` | INT8 speedups per shape; activation outliers |
| `tg4dac_attention.csv`, `tg4dac_e2e.csv` | kernel gain vs end-to-end gain |
| `arc_transfer.csv` | model tool use: grid validity vs task accuracy (15 conditions, 4 puzzles; not independent) |
| `model_tool_feedback.*`, `lora_operator*.csv` | a model-repaired plan (correct, slower); complete fwd/bwd operator costs |
| `evidence_sources.json` | SHA-256 provenance for the above |

## Scripts, examples and tests

- `examples/`: `paired_bench.py` (Bench A/A and paired, GPU), `measured_game_toy.py` (Game, needs `[search]`),
  `warpspeed_plan.py`, `decoder_packing.py`, `finite_alphabet.py` (exhaustive checks of finite numeric contracts).
- `tools/check_public.py`: audit files before publishing (home paths, tokens, keys, emails, notebook outputs, broken
  links and anchors). Run it before sharing a ledger or a repo.
- `tests/test_core.py`: the shortest usage reference for every module (CPU only).
- `kernels/`: kits with their own READMEs (`fp8_block_sm89`, `w8a8_sm86`, `moe_w4a16_h200`, `tg4dac_sm120`).

## Docs map

[formulation](https://github.com/xArmory/xCalibur/blob/main/docs/formulation.md) (contracts, physical design, exact
checks) · [search](https://github.com/xArmory/xCalibur/blob/main/docs/search.md) (games, measured search, what may
close a branch) · [agents](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md) (knowledge base, OODA,
planner/refiner, Anvil) · [studies](https://github.com/xArmory/xCalibur/blob/main/docs/studies.md) ·
[reflections](https://github.com/xArmory/xCalibur/blob/main/docs/reflections.md) · writeups:
[AUTORESEARCH](https://github.com/xArmory/xCalibur/blob/main/docs/writeups/WRITEUP_AUTORESEARCH.md),
[TG4DAC](https://github.com/xArmory/xCalibur/blob/main/docs/writeups/WRITEUP_TG4DAC.md),
[SM86/SM89](https://github.com/xArmory/xCalibur/blob/main/docs/writeups/WRITEUP_SM86_SM89.md).
