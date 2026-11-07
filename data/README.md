# Data and provenance

These files support CPU replay of the published analyses. Ratios greater than
one mean the candidate is faster. Units are carried in column names. Empty
timings are missing observations or failures, never zero-latency results.

## Archived public datasets

| File | Contents | Scope |
|---|---|---|
| [moe_judge.csv](moe_judge.csv) | 3 submissions × 6 loads; SGLang, candidate and analytical bound in µs | Recompute each submission's score using its own SGLang column; `sol_us` is a bound, not another measured kernel |
| [scores.json](scores.json) | Archived aggregate MoE submission scores | Historical snapshots, not current leaderboard positions; rounded timing rows may differ slightly from these scores |
| [moe_sweep.csv](moe_sweep.csv) | Screen and confirm observations of decoder/compute/activation combinations | Evidence grade belongs to each row; do not pool screen winners as confirms |
| [moe_interaction.csv](moe_interaction.csv) | Decoder × arithmetic interaction | Paired gains on the stated load; not an independent whole-model speedup |
| [noise.csv](noise.csv) | Repeated observations of identical builds and screen/confirm outcomes | Timing-noise analysis |
| [tg4dac_attention.csv](tg4dac_attention.csv) | 13 attention cases, vLLM/planner/refiner timings in ms | One SM120 study and the recorded vLLM backend |
| [tg4dac_e2e.csv](tg4dac_e2e.csv) | 5 whole-model decode throughput cases | Tokens/second; historical engine integration |

These datasets predate the expanded research snapshot. Their original raw run
bundles and full environment locks are not distributed. The shipped notebook
analyses are reproducible; repeating the GPU experiments needs more than a CSV.

## Expanded research snapshot

| File | Contents | Scope |
|---|---|---|
| [decoder_search.json](decoder_search.json) | Exact denominator, solution count, CPU controls, GPU samples and setup costs | Search latency; warm and cold costs kept separate |
| [structural_search.csv](structural_search.csv) | All 36 ownership and 40 compute-family records from the exported ledgers | Includes failures, repeats and different implementation contexts; ratios are relative to each campaign parent |
| [deterministic_search.json](deterministic_search.json) | Separate 44-plan / 28-measurement campaign summary | Aggregate evidence; no model inference in this campaign |
| [load_patterns.csv](load_patterns.csv) | All 168 copy-only TMA patterns | Canonical byte checks; no MMA or full-operator speedup claim |
| [load_composition.csv](load_composition.csv) | 54 load rows for six full-operator finalists and three factorial cells | Direct launches, stream events and read flush; separate from graph-replay protocols |
| [model_tool_feedback.csv](model_tool_feedback.csv), [aggregate](model_tool_feedback.json) | Six load results from one repaired model plan | Complete MoE, correct on tested fixtures but slower than its parent |
| [lora_operator.csv](lora_operator.csv), [samples](lora_operator_samples.csv) | Four plans and 40 candidate timing samples | One complete forward/backward LoRA linear; no optimizer or whole-model training |
| [l4_f1.csv](l4_f1.csv) | Nine xR38 revision comparisons | F1 only: routing, gate/up and SwiGLU; excludes down/combine |
| [arc_transfer.csv](arc_transfer.csv) | 15 conditions on the same four public puzzles | Exact task top-1/top-2, validity and tool-use fields; conditions are not independent tasks |

[evidence_sources.json](evidence_sources.json) identifies each extraction input
by SHA-256 and records its scope and public exports. The export selected named
technical and numeric fields; it did not copy raw operator logs, prompts,
credentials or infrastructure metadata. Source hashes bind the extraction to
retained records; they do not make those original records publicly retrievable.

The later notebooks read these public files directly. They need no access to
private experiment directories. See the [notebook guide](../notebooks/README.md)
and [evidence guide](../docs/studies.md#evidence).

## Consumer-card prefill study

| File | Contents | Scope |
|---|---|---|
| [kev_prefill_runs.csv](kev_prefill_runs.csv) | Run ids, paired median ratios with 90% intervals, margins, answer drift and decisions | RTX 3090, Kev-9B bf16, 16 pairs from 8 requests (intervals clustered by request); one interim RTX 6000 Ada FP8 row; pending rows stay empty |
| [kev_prefill_profile.csv](kev_prefill_profile.csv) | Request split (state pass / rows) and state-pass CUDA time by kernel category | Profiled shares per workload, not timings of a candidate |
| [w8a8_sm86_microbench.csv](w8a8_sm86_microbench.csv) | Paired W8A8 and bf16 `F.linear` timings per projection shape and M | Kernel microbenchmark on synthetic inputs; says nothing about model accuracy |
| [w8a8_sm86_diagnosis.csv](w8a8_sm86_diagnosis.csv) | Per-layer, per-site activation range and INT8 error under plain, per-128 and SmoothQuant scales | One 28k-token request; fake-quantised error on a 1,024-token sample |

See the [writeup](../docs/writeups/WRITEUP_SM86_SM89.md) and [W8A8 guide](../kernels/w8a8_sm86/README.md).
