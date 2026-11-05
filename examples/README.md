# Small executable entry points

Run these from the repository root after `python -m pip install -e .`.

| Example | What it does | Extra requirement |
|---|---|---|
| `python examples/decoder_packing.py` | Derives a lossless packing and checks GF(2) composition | None |
| `python examples/warpspeed_plan.py` | Accounts for squads, shared rings, phases and registers | None |
| `python examples/finite_alphabet.py` | Exhausts 65,024 finite paired FP4/scale cases, 512 NaN-class cases and bounded BF16/FP8 decompositions | Python standard library only; writes fixtures under ignored `build/finite_alphabet` |
| `python examples/measured_game_toy.py` | Exercises the ranking/confirmation loop on seeded synthetic rewards | `python -m pip install -e ".[search]"` |
| `python examples/paired_bench.py` | Times two functions with paired GPU replays and an A/A control | CUDA-enabled PyTorch and a CUDA GPU |

The scalar examples prove the properties they explicitly assert. They do not
establish native fragment ownership, synchronization or performance. The
[notebooks](../notebooks/README.md) connect these mechanisms to recorded studies.
