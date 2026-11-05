# Executable research notebooks

The notebooks combine a mechanism, a runnable example or recorded dataset, and
an explicit statement of what the result establishes. All eleven run on CPU.
Saved outputs let GitHub readers inspect the figures without installing anything.

## Run from a fresh checkout

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks,test]"
python notebooks/build.py             # generate and execute all notebooks
python notebooks/build.py 09          # generate and execute one prefix
python -m pytest tests
python tools/check_public.py
```

`build.py` is the source of truth for notebooks 01–07;
[`research.py`](research.py) supplies 08–11. Change those sources and rebuild.
The builder selects the active Python interpreter, uses deterministic cell IDs,
removes timing metadata, and writes figures into `docs/figures/`.

For interactive use, open a generated notebook with this virtual environment's
Python kernel. The setup cells support running from the repository root or from
the notebook directory. No model, account, GPU or network call is needed to
execute the notebooks after installing dependencies.

## Choose a question

| Notebook | What you execute | Evidence |
|---|---|---|
| [01 · Decoders over GF(2)](01_decoders_over_gf2.ipynb) | Construct a decoder map, derive its packing, verify exact composition | CPU proof + recorded production-search counts |
| [02 · WarpSpeed plans](02_warpspeed_plans.ipynb) | Compute squads/ring storage/register budgets; reject an over-budget plan | CPU accounting example |
| [03 · Measurement and noise](03_measurement_and_noise.ipynb) | Plot repeated builds and screen/confirm regression | Recorded observations + labeled statistical illustration |
| [04 · MoE on H200](04_moe_on_h200.ipynb) | Recompute archived six-load scores and inspect interaction effects | Recorded judge and paired-study data |
| [05 · The measured game](05_the_measured_game.ipynb) | Run the search loop over a seeded toy problem | Synthetic rewards; no GPU performance claim |
| [06 · TG4DAC](06_tg4dac.ipynb) | Compare attention plans and whole-model throughput | Recorded SM120 results; distinct scopes |
| [07 · Knowledge graphs](07_knowledge_graphs.ipynb) | Query neighbours, field effects and retrieval | Recorded design data; offline hashed retrieval illustration |
| [08 · Layout and ownership](08_layout_ownership.ipynb) | Compose coordinate maps, find collisions, replay nine L4 cases | CPU teaching map + historical F1 data |
| [09 · Structural search](09_structural_search.ipynb) | Compare search/setup costs, execute finite-alphabet contracts and inspect load-pattern/complete-program outcomes | CPU proofs + recorded H200 summaries, failures and intervals |
| [10 · Operators and model feedback](10_operator_and_model_feedback.ipynb) | Recompute LoRA medians; replay a resource repair and its six-load outcome | Recorded SM120/H200 studies + CPU checker replay |
| [11 · ARC transfer](11_arc_transfer.ipynb) | Check a transpose map; distinguish grid validity from task accuracy | Synthetic map + four-task repeated-condition pilot |

These notebooks reproduce **analysis**, not the original hardware campaigns.
The [documentation](../docs/README.md#reading-order) lists the shipped implementations and
the studies whose original harnesses are not distributed.
