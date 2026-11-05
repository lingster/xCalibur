# 4 · Studies: from a formulation to an implementation

The earlier chapters define constructions, search policies and agent workflows.
This chapter follows what happens when those ideas are applied to an operator:
which boundary changes, what must remain true, what was built, and what the
result established. Detailed observations remain in the linked datasets; this
chapter explains their meaning.

<a id="gemma4"></a>
## Gemma 4 and TG4DAC

TG4DAC developed separate planner, refiner and implementer roles around executable
code graphs. The kernel workload was **Gemma 4 31B on RTX PRO 6000 Blackwell
(SM120)**: first its dense W4A16 MLP with LoRA, then sliding-window and global
attention. Gemma planner/refiner adapters also supplied design proposals; the
workload model and its designers are distinct roles.

The planner decomposed computation and proposed fusion, tiling and ownership.
The refiner revised the hardware plan using PyCuTe expressions and WarpSpeed
squads, resources and phases. A coding model implemented those plans and ran the
compiler, numerical checks and measurements. A checker diagnostic therefore
became a concrete design revision, rather than another unverified estimate.

For grouped-query attention,

```math
o_h=\operatorname{softmax}(q_hK_{g(h)}^\top)V_{g(h)}.
```

The design choices affected the physical work behind that expression:

- Trim sliding-window decode to the cache pages its window can read.
- Exploit the global-attention key/value relation through a V-only cache and
  explicit key reconstruction, preserving RoPE and weight association.
- Split prefill into load and compute squads, with staged Q/K/V transport.
- Choose producer/consumer ownership and ring depths for decode's smaller work.

Attention was held out of the refiner's training examples. The study recorded
checker repairs and native implementations. Selected attention workloads improved
against the measured vLLM backend; the whole-model decode experiment had a
separate timing scope. The earlier MLP round began from strong hand-built kernels
and its model proposals did not improve them. These outcomes belong to their
operators and baselines.

**Available:** [attention and MLP source](../kernels/tg4dac_sm120/README.md),
[attention observations](../data/tg4dac_attention.csv), and
[whole-model observations](../data/tg4dac_e2e.csv). The trained adapters and
historical vLLM integration harness are not bundled.

<a id="h200"></a>
## H200: the complete NVFP4 MoE

The challenge operator combines routing, gate/up projection, activation, down
projection, normalization and weighted accumulation. For token $`t`$ and routed
expert $`e(t,s)`$, its structure is

```math
y_{t,s}=W_2^{(e)}\bigl(\operatorname{silu}(W_g^{(e)}x_t)
\odot W_u^{(e)}x_t\bigr),
```

```math
\operatorname{out}[t]=\sum_s g_{t,s}
\operatorname{RMSNorm}\bigl((1+w)\odot y_{t,s}\bigr).
```

Weights use NVFP4 codes with group scales and per-expert global scales. A legal
repacking must preserve both the values and the original scale association.

The public persistent implementation combines a searched bit decoder, matching
weight packing, packed-BF16 arithmetic for the one-token path, and per-size
builds. Dispatch is part of the design: a field change matters only to the builds
that read it. Export must preserve the exact source units and build options that
were measured.

The [co-design game](search.md#codesign-game) grew around this compound operator.
Archive designs became search roots; one-field effects informed proposals;
constructors exposed new physical choices. Combining a decoder and arithmetic
change exposed an interaction that their separate measurements did not capture.

The campaign also exposed an error in its measurement oracle. Write-flushing L2
charged dirty-line writeback to the timed operation, changing candidate order
relative to the intended read flush. Old and corrected observations therefore
carry different protocol identities. Replication under a wrong protocol did not
make the original interpretation valid.

**Available:** [kernel, ancestry and recorded results](../kernels/moe_w4a16_h200/README.md),
[judge timings](../data/moe_judge.csv), [design sweep](../data/moe_sweep.csv), and
[dataset descriptions](../data/README.md). The full search worker and construction
harness are documented through the formulations, rather than distributed with
the small public `Game` interface.

<a id="l4-moe"></a>
## L4: ownership before search

The Supersonic/xR38 lineage explored routed BF16 gate/up and SwiGLU through token
compaction, packed weights, warp-local activation panels and explicit output
ownership. The design had to connect the logical route to the lane's source
value, the MMA consumer and the final writer.

The retained F1 study passed its recorded correctness and sanitizer checks but
regressed in most comparisons. Compiled allocation exposed local-memory spills;
that was a plausible contributor, not an isolated causal measurement. The
experiment motivated explicit attention to compiler allocation and the complete
physical path even when the coordinate maps were correct.

**Available:** [nine F1 cases](../data/l4_f1.csv). F1 includes routing/alignment,
gate/up and SwiGLU; it excludes down projection and final combination. Historical
kernel sources and the original execution harness are outside the public package.

<a id="decoder-search"></a>
## Exact decoder construction

The finite decoder grammar supplies atoms for output bit fields. Pairwise
compatibility bitsets prune combinations; full-rank checks establish complete
maps. For an invertible decoder $`D`$ and reference $`P`$, the construction derives
$`T=D^{-1}P`$ instead of searching packing independently.

The production CUDA solver uses count, prefix and emission passes. Its full
solution set matched the CPU enumerator for the declared grammar. Canonical
semantic maps retain their source aliases and output permutations, because those
continuations may compile differently.

Complete enumeration of decoder maps did not enumerate every enclosing kernel.
The initial native study sampled only a few complete replacements and did not
establish a latency improvement. Search cost, setup cost and generated-kernel
latency remain separate observations.

**Available:** [search coverage, setup and timing samples](../data/decoder_search.json),
plus the CPU analogue in [`gf2.py`](../xcalibur/gf2.py). The production GPU solver
is not part of that module.

<a id="structural-search"></a>
## CuTe Sassy: coordinated physical changes

The structural constructor widened the action space from decoder choices to
ownership, compute families, activation residence, resource lifetimes and
dispatch. A move carries its dependent interfaces to a complete translation unit.

Three constructions make that coupling concrete:

- **Packed GEMV:** avoid unused tensor-core columns at small token counts, then
  reconstruct the K reduction and output ownership around packed BF16 operations.
- **Complementary sparse MMA:** derive a lossless K permutation, matching original
  scale owners and complementary metadata that covers the dense computation.
- **Activation residence:** trade global/register loading against shared prefetch
  or shared residence with shorter register lifetimes, rebuilding buffer rings
  and checking actual compiled occupancy for the new launch.

MCTS allocated complete-program evaluations across compatible partial choices.
The ledger retains compiler failures, runtime failures, repeated contexts and
losing candidates. A generator repair creates a new implementation context;
it does not overwrite evidence from the earlier context.

A later deterministic campaign compiled and qualified another set of complete
plans without model inference or MCTS. It demonstrated no operator gain under its
confirmation interval. This is a separate experiment from both the tree search
and the archived challenge submission.

**Available:** [constructor campaign ledger](../data/structural_search.csv) and
[deterministic campaign summary](../data/deterministic_search.json).
The [conditional-subgame formulation](search.md#conditional-subgames) explains
what the grammar can search and which continuations must be retained.

<a id="finite-alphabet"></a>
### Finite alphabets as exact inner problems

E2M1's small value alphabet supports exhaustive paired-code/scale tables. Another
construction decomposes supported BF16 values into E4M3 components,

```math
x=h+2^{-r}l,\qquad h,l\in\operatorname{E4M3}.
```

The finite sumset determines exact scalar coverage in a stated exponent window.
A native consumer additionally needs compatible fragments, scale boundaries,
rounding, accumulation and actual instruction lowering. Scalar reconstruction
alone does not establish a matrix kernel's behavior.

The public [`finite_alphabet.py`](../examples/finite_alphabet.py) checks the paired
decoder, signed-zero/NaN-class handling and supported decomposition windows.
Native table, GEMV and instruction probes were research studies, with complete
MoE integration still a separate obligation.

<a id="load-patterns"></a>
### Load patterns and composition

The load-pattern study varied TMA streams, lookahead, copies per stream, issue
staggering and packet order. Copy-only probes checked every consumer byte before
selected layouts were composed into the complete operator.

For $`S`$ streams in the studied construction, each stream expected $`36864/S`$
bytes and $`16/S`$ consumer-warp retirement arrivals; its ring organization required
$`6S+6`$ barriers including activation rings. Codes and scales moved together when
packet coordinates changed. A decoder transform acted within the packed word.

Dedicated producer warps restored intended overlap in the emitted code, but the
qualified complete compositions did not demonstrate a latency gain. The protocol
used direct native launches and stream events, a read flush and no graph-replay
correction. Those observations are separate from the earlier graph-based runs.

**Available:** [copy-pattern observations](../data/load_patterns.csv) and
[complete-operator load rows](../data/load_composition.csv).

<a id="complete-operators"></a>
## LoRA: the graph includes backward

For frozen quantized $`W`$ and trainable adapters $`A,B`$,

```math
Y=XW^T+s(XA^T)B^T.
```

A complete plan includes forward, backward, saved tensors, casts, adapter-gradient
accumulation and stream joins. Retaining decoded $`W`$ removes a possible second
decode; retaining $`XA^T`$ removes a possible repeated projection. Both decisions
extend lifetimes and consume memory. Fusion and scheduling must preserve the
values and parameter versions required by $`dX,dA,dB`$.

The checkpoint-linear study compared related recompute/framework and retain/CuTe
plans. All passed the recorded zero/nonzero-B numerical checks and cumulative
FP32 gradient checks. Timing included forward, backward and gradient accumulation,
used zero B, and contained no optimizer update. One retained-weight plan reduced
latency within that comparison; whole-model saved-state admission remained open.

**Available:** [plan contracts](../data/lora_operator.csv) and
[repeat samples](../data/lora_operator_samples.csv). The full checkpoint-linear
execution harness is not bundled. This is an operator study, not a completed
training run or a general claim about learned planning superiority.

<a id="model-feedback"></a>
### Model-to-native feedback

A later tool loop let a model select a complete-operator construction or supply a
CUDA function behind a fixed interface. In the constructor route, a request for
two resident CTAs violated the declared shared-memory budget:

```math
2(174848+1024)>233472.
```

The model reduced residency, and the worker generated, compiled, checked and timed
the complete operator. The candidate was correct on its fixtures and slower than
the frozen parent. In the CUDA-function route, a wrapper failure was repaired
and the unchanged arithmetic was replayed successfully on component fixtures.
That replay is distinct from a successful new live trajectory.

**Available:** [per-load results](../data/model_tool_feedback.csv) and
[aggregate contract](../data/model_tool_feedback.json).

<a id="forge-and-vega"></a>
### Forge and Vega

Other pilots explored blind proposals, code-graph construction, counterexample
refinement and bounded policy training. A bit-exact decoder-LUT proposal was
slower. Forge's recorded holdout evaluation stopped before completing all tasks.
A Vega tiled construction repeated gate/up work across downstream tiles, changing
the actual amount of computation being timed.

These studies sharpened the admission rule: account for useful and issued work,
verify the selected implementation, and keep partial evaluations partial. Their
training artifacts and original harnesses are not distributed here.

<a id="arc"></a>
## ARC: a representation still needs a semantic rule

ARC transfers the construction approach to grid reasoning. Given demonstrations
$`\mathcal D=\{(x_i,y_i)\}`$, a candidate program must first satisfy

```math
\forall(x_i,y_i)\in\mathcal D,\qquad p(x_i)=y_i.
```

It must then produce the required query outputs without using their labels during
construction. A transpose illustrates the distinction: its read and write maps
$`R(r,c)=rW+c`$ and $`D(r,c)=cH+r`$ can have exact coverage, while transpose may still
be the wrong transformation for the task.

The work explored bounded layout/bit programs, demonstration replay, AST proposals,
co-design graphs, vision input and repair loops. The exported pilot repeats the
same four puzzles across fifteen conditions; every condition had zero exact task
solves. Valid grids and exact coordinate maps did not establish semantic success.
Post-hoc program replays retain their own labels and are not pooled with live runs.

**Available:** [condition/task observations](../data/arc_transfer.csv). The evaluation
order is fit on demonstrations, solve query inputs, freeze predictions, then
score against labels. The wider teacher-data and adapter branches are separate
from this pilot and are not packaged as a general ARC solver.

<a id="evidence"></a>
## Reading and reproducing the evidence

A record needs a source revision, workload, reference, numerical contract,
measurement protocol and denominator. The relevant claims are distinct:

- Algebra establishes a property within its declared model.
- A resource checker establishes its modeled constraints.
- Compilation establishes a lowering and exposes its allocation.
- Numerical tests qualify the recorded fixtures.
- Timing establishes an observation under its protocol.
- Independent confirmation assesses whether an apparent gain persists.

An archived judge score is a snapshot. A source digest is an identity, not an
execution harness. Ratios with different parents or protocols are not interchangeable.
The [data guide](../data/README.md) describes all exported tables and provenance;
[visual analyses](../notebooks/README.md) accompany those observations.

<a id="availability"></a>
## What can be used from this checkout

The public library provides GF(2) maps, resource accounting, reach signatures,
captured-build comparison, paired timing, a callback-based search loop, design
graph queries and planner/refiner interfaces. [Getting started](getting-started.md)
shows their contracts and dependencies.

The [kernel directories](../kernels/README.md) distribute H200 MoE and Gemma 4
SM120 implementations. Their guides identify runtime requirements and baselines.
The full KB ingestion system, OODA controller, Rebel solver, native structural
constructors, trained adapters and serving stack are documented research systems;
they are not installed by the Python package.

The public surface contains technical explanations, attributed source and curated
observations. Local operator state, credentials and model conversations remain
outside it. The [publication checker](../tools/check_public.py) scans the selected
files and local links; git history is outside that check.

[← Knowledge and agents](agents.md) · [Contents](README.md) · [Run the code →](getting-started.md)
