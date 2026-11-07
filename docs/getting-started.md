# Start with a complete example

The public package runs its exact maps, resource checks and statistical tools
on CPU. GPU timing, learned ranking and embedding retrieval are optional.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest tests
python examples/decoder_packing.py
python examples/warpspeed_plan.py
```

## Prove a packing transformation

```python
import xcalibur as xc

reference = tuple(1 << bit for bit in range(32))
decoder = xc.rotation(5)
packing = xc.packing_transform(decoder, reference)
assert xc.compose(decoder, packing) == reference
```

The assertion checks equality of the complete linear maps. The public
[`gf2`](../xcalibur/gf2.py) module also includes rank, inverse, composition and
a small exact atom join. The production GPU enumeration discussed in the
research studies is not part of this module.

## Check a physical plan

```python
plan = xc.plan_of(
    squads=[("mma", 16), ("w_load", 1), ("x_load", 1)],
    resources=[
        ("w_ring", 3, 36864, [("w_load", 36864), ("mma", None)]),
        ("x_ring", 3, 32768, [("x_load", 32768), ("mma", None)]),
    ],
)
result = xc.warpspeed_check(**plan)
assert result["pass"]
print(result["registers"]["launch_cap"])  # 96
print(result["smem"]["total_bytes"])     # 208992
```

This uses the shipped H200 resource profile, which is the default for
`warpspeed_check`; it does not check SM120 limits. Pass `arch="rtx3090"`,
`"a6000"`, `"rtx6000ada"`, `"l40s"` or `"rtx4090"` for the sm_86/sm_89
[profiles](formulation.md#architecture-profiles); the plan above then fails with
`tma_arch` and shared-memory counterexamples. It accounts for the declared
rings and barriers. Include auxiliary shared storage when checking a complete
plan; compiler allocation and native lifetime correctness need additional
validation.

## Choose the next layer

| Goal | Install / entry point | Requirements |
|---|---|---|
| Rebuild all analyses | `python -m pip install -e ".[notebooks]"`; [notebook guide](../notebooks/README.md) | CPU; no runtime network calls |
| Run the toy learned search | `python -m pip install -e ".[search]"`; `python examples/measured_game_toy.py` | CPU, LightGBM; synthetic objective |
| Retrieve related designs | `python -m pip install -e ".[retrieval]"`; [`Retriever` and `embedding_gemma`](../xcalibur/graph.py) | sentence-transformers; access to the selected embedding model and a download unless cached |
| Time two GPU functions | Install a suitable CUDA-enabled PyTorch; `python examples/paired_bench.py` | CUDA GPU; [measurement protocol](getting-started.md#measurement) |
| Inspect the H200 operator | [MoE kernel guide](../kernels/moe_w4a16_h200/README.md) | H200, compiler, PyTorch, benchmark-compatible inputs |
| Run Gemma 4 31B attention / MLP | [TG4DAC kernel guide](../kernels/tg4dac_sm120/README.md) | SM120, CUDA/NVRTC, PyTorch, Triton; attention baseline needs vLLM |
| Add a model proposer | [`PlannerLoop`](../xcalibur/planner.py), [integration scope](agents.md#public-code) | Model endpoint, schema, construction/check/evaluation callbacks |

`Game` also requires workload-specific construction and measurement callbacks.
Installing an optional dependency does not install a production co-design agent.

## Reusable code

| Module | Responsibility |
|---|---|
| [`gf2`](../xcalibur/gf2.py) | Linear bit maps and packing proofs |
| [`warpspeed`](../xcalibur/warpspeed.py) | Resource and phase accounting |
| [`reach`](../xcalibur/reach.py) | Which fields affect which compiled build |
| [`bench`](../xcalibur/bench.py), [`stats`](../xcalibur/stats.py) | Paired timing (graph replay and long calls), intervals (optionally clustered), A/A margins and evidence pooling |
| [`surrogate`](../xcalibur/surrogate.py), [`game`](../xcalibur/game.py) | Learned ranking (LightGBM, or a numpy ridge fallback) and the callback-based search loop |
| [`graph`](../xcalibur/graph.py) | Design relationships and retrieval |
| [`planner`](../xcalibur/planner.py) | Model proposal/refinement interface |
| [`capture`](../xcalibur/capture.py), [`gates`](../xcalibur/gates.py) | Captured-build identity and release checks |

<a id="measurement"></a>
## Measurement contract

`Bench` captures a candidate and reference as CUDA graphs, performs a read flush
before each timed replay, and alternates their order. Use it when that execution
path matches the benchmark contract. It returns every raw timing pair alongside
the median ratio and bootstrap interval:

```python
bench = xc.Bench()
control = bench.aa(reference, pairs=20)
result = bench.paired(candidate, reference, pairs=30)
raw = result["raw_candidate_samples_us"], result["raw_reference_samples_us"]
ratio, interval = result["ratio"], result["ci"]
```

A ratio above one favors the candidate. Calibration subtraction is disabled by
default. The optional `null_us()` times a tiny-add graph, including real work;
`subtract_null=True` needs a justified calibration and rejects nonpositive
corrected times. Raw observations remain available.

<a id="long-calls"></a>
`LongBench` covers calls of milliseconds to seconds, such as a whole model pass,
where graphs, the flush and the null replay do not apply. It keeps ABBA pairs,
times each call with CUDA events plus wall time, runs a warm-up policy, and
records an nvidia-smi snapshot per block of pairs:

```python
warm = xc.small_first({"candidate": lambda: new(short), "reference": lambda: old(short)})
long = xc.LongBench(warmup=warm)
result = long.paired(lambda: new(full), lambda: old(full), pairs=8, block=4)
ratio, interval, clocks = result["ratio"], result["ci"], result["snapshots"]
```

`small_first` lets autotuning happen on a short input; tuning on a full-size one
can exhaust a 24 GB card. Snapshots show whether a power-capped board's clock
drifted within a block. When several inputs are timed repeatedly, pass
`bootstrap_ci(ratios, clusters=input_ids)` so whole inputs are resampled rather
than single ratios; the default call is unchanged. `aa_margin(aa_ratios, k=2,
step=0.001)` turns A/A medians or interval ends into an adoption threshold that
a candidate's lower bound must clear. The [sm_86/sm_89 study](studies.md#sm86-sm89)
applies this protocol.

`geomean` and `bootstrap_ci` require finite positive ratios. `Evidence.bind`
uses a repeat-and-margin heuristic; it is separate from an interval test.
`Game` records at most one confirmation per load and declared identity per round.
Its reference remains fixed, and its measurement callback must return one load
record per requested design: the reference covers every load, and another design
every load it reaches (a per-size build that does not apply to a size has nothing
to time there). `Game.confirm(designs)` applies the round's confirmation step to
designs chosen outside a round, such as seed-screen leaders. Changing the reference or measurement protocol
requires fresh evidence. The [search chapter](search.md#measured-search) explains
the policy around those interfaces.

<a id="build-and-export"></a>
## Build and export contract

`ReachMap` describes which fields affect a build. Its identity is a declared
lowering signature; validate it against captured inputs or compiled code under
a fixed compiler context before pooling observations.

`Capture` replaces a compiler callback during a dry construction pass. It records
source units, ordered flags, bindings and other options. `Capture.same(a, b)`
compares the complete multiset, including repeated builds. It establishes input
equality; compiler environment and native behavior require their own checks.

`replace_assignment` changes one unambiguous module-level value while preserving
surrounding source bytes. It supports annotated and multiline assignments and
UTF-8 offsets, and rejects repeated or chained targets. The exported program
must still be qualified and timed as the actual artifact to be used.

`Card`, `rules_gate` and `lint` expose small pieces of the
[OODA protocol](agents.md#ooda). The complete agent controller, model serving,
native construction and publication integrations are supplied by the caller.

[Documentation](README.md) · [Implementation availability](studies.md#availability)
