# CuTe, SASS, and fast graph synthesis

## Goal and current status

**The objective remains open: discover new complete-kernel designs and achieve
2.7x or better on `muse_moe_w4a16` under the target evaluation protocol.** Fast
enumeration is supporting infrastructure. It does not satisfy that objective.

The latest deterministic batch compiled two spellings of each of the 1,816
canonical decoder graphs, compiled 44 complete physical plans, and measured
28 complete-operator candidates on H200. All passed the tested correctness
fixtures; none demonstrated a latency gain. The official best remains 2.3654x.
The [current report](../garage/cute_sassy/results/TARGET27_SEARCH.md) retains all
168 timing rows, compiler receipts, and runtime release evidence. This batch
used exact search and compiler feedback without Gemma inference.

The implemented decoder solver is one inner subgame. It froze dataflow,
ownership, MMA geometry, squads, buffers, schedules, and dispatch; sampled eight
decoder classes; and measured two complete lowerings. Their paired speedups
were 0.95921x and 1.00511x relative to the frozen parent, with no demonstrated
gain. All sixteen isolated SASS samples had the same total instruction count
(40), so the final selection used the ID tie-break rather than distinguishing
hardware latency. These results neither meet 2.7x nor establish that the broader
design problem lacks a solution. See [execution evidence](../garage/cute_sassy/results/REPORT.md).

The problem is to realize a semantic computation graph in GPU space and time.
Source text is one encoding of a realization. A layout, instruction sequence,
pipeline, packing transform, and launch are coupled parts of that realization.

The outer implementation now exists in [`outer/`](../garage/cute_sassy/outer/STATUS.md):
a persistent MCTS selects compute family, activation delivery, and conditional
ownership, residency, pipeline, dispatch, work-map, and decoder decisions.
Its first native family jointly rewrites
warpgroups, weight packing, gather/combine ownership, publication, and retirement.
The expanded 2,460 ownership/pipeline contracts are checked before compilation, and actual
occupancy is checked before persistent launch. Additional constructors replace
the small-load tile with packed BF16 GEMV or complementary sparse MMA, and change
activation transport and register lifetime. Their exact CuTe maps, native SASS,
and complete-operator results are in the [compute board](../garage/cute_sassy/outer/COMPUTE_BOARD.md)
and [program report](../garage/cute_sassy/results/PROGRAM_REPORT.md).
The grammar's existence is not evidence of a score improvement.

## 1. The object being searched

Let $G=(V,E)$ be the required semantic dataflow graph, $H$ the hardware, and
$\Gamma$ a finite grammar of permitted implementations. Each value edge carries
dtype, logical domain, numerical obligations, and a physical map

$$\Phi_e(i)=(\operatorname{owner}(i),\operatorname{storage}(i),
\operatorname{address}(i),\operatorname{ready}(i),\operatorname{lastUse}(i)).$$

Ownership includes CTA, warpgroup, warp, lane, and register slot. Storage includes
global/shared/register memory. Time is expressed as a dependency/event partial
order before a concrete instruction schedule exists. CuTe layouts describe
coordinate-to-location maps; they do not by themselves prove lifetime or barrier
correctness. WarpSpeed-style squad/resource/phase contracts add those obligations
and resource accounting. A successful accounting check is not a synchronization
proof.

A realization $\Pi$ contains the selected operation implementations, maps,
pack/unpack transforms, communication, synchronization, buffer lifetimes,
schedule, and dispatch. For fixed compiler context $\kappa$ (toolchain, target,
flags, complete translation unit), its correctness obligation is

$$\operatorname{unpack}_{out}\circ
\operatorname{Exec}(C_\kappa(\Pi))\circ\operatorname{pack}_{in}
\simeq_{\mathrm{required\ numerics}}G.$$

For the six equally weighted MoE loads, minimizing geometric mean latency is
equivalent to minimizing

$$\Pi^*=\arg\min_{\Pi\in\operatorname{Realize}_\Gamma(G,H)}
\frac{1}{6}\sum_{\ell=1}^6\log T_\ell(\Pi).$$

The finite realizations and compiler results are deterministic under a frozen
context. Measured latency has noise, and compiler determinism does not make the
compiler invertible or latency additive.

## 2. Search the constraints before paying for executions

Use a typed AND/OR graph: an OR node chooses an implementation; an AND node
requires all its inputs, resource contracts, and obligations. A partial state
retains every boundary fact that can affect a continuation. In particular,
register pressure and shared-memory lifetime cannot be discarded from a state
and later recovered by adding isolated kernel timing deltas.

Four reductions are sound when their stated conditions hold:

1. **Eliminate variables algebraically.** If a consumer map determines a unique
   compatible producer packing, derive it instead of enumerating both.
2. **Join compatible boundaries.** Construct producers forward and consumer
   requirements backward. Match canonical boundary maps, ownership, and events.
   Bitsets perform many compatibility tests at once; exact predicates discharge
   the remaining obligations.
3. **Share proofs, retain implementations.** Equal semantic maps share algebra
   work. Their distinct source recipes and scheduling continuations remain
   available because the compiler may produce different machine programs.
4. **Cache complete compiler outcomes.** Key compilation by exact source,
   toolchain, architecture, flags, and enclosing context. Reusing a measurement
   additionally requires equivalent executable code, packing/dispatch/launch,
   and workload. A shared SASS hash alone is insufficient.

Bidirectional Dijkstra is appropriate inside a finite transition graph with
nonnegative additive costs and sufficient state. It can minimize movement or
instruction derivation cost under an explicit model. It does not establish the
fastest full kernel from context-free per-edit timings. A boundary join is often
cheaper than shortest-path search when the inverse relation can be solved
directly. Use admissible lower bounds for exact branch-and-bound; an empirical
ranker may order unresolved work, but its guesses cannot prove rejection.

## 3. The first executable instance: decoder DAG plus weight representation

The native boundary is the existing complete MoE kernel's packed 32-bit word to
four BF16-pair bit fragments. Its scale bytes, ownership, MMA consumers,
pipelines, dispatch, and output arithmetic are frozen. This is a real subgraph
replacement inside a full operator, with an untimed lossless weight repacker.

For the bit-linear part, a decoder is a $32\times32$ matrix over $\mathbb F_2$.
Each output contributes eight rows, corresponding to bit positions
$\{6,7,8,15,22,23,24,31\}$. These are bit-routing maps, not floating-point
linearizations; ordinary CuTe integer-stride maps need not be GF(2)-linear.

Let $P$ be the frozen parent's decoder and $D$ a proposed decoder. Exact
compatibility requires $DT=P$. If $D$ has full rank, eliminate packing:

$$T=D^{-1}P,\qquad D(Tw)=Pw\quad\text{for every }w\in\mathbb F_2^{32}.$$

This proves every possible code word, including sign bits. Zero and basis tests
validate the implementation of the linear expressions; independent random words
and native packing checks catch mistakes outside that model. Scale-group and
physical word ordering must also agree with the consumer.

The v1 grammar has identity, shifts/rotates by 1..31, optional XOR with the input,
the output field mask, and the parent's split-field output. Individual rank-eight
maps are canonicalized by their **ordered rows**, retaining all source aliases.
Four independent output maps form a legal decoder. Register permutations remain
explicit lowering continuations, as do arithmetic/bit-shift source spellings.

Search works as follows:

1. Compute pair compatibility (rank 16) as upper-triangular bitsets.
2. For each compatible pair, intersect continuation bitsets and join the next
   two outputs. Reject rank-deficient triples and quartets exactly.
3. Emit every legal canonical quartet, with counts for each rejection. No
   sampled failures eliminate a family, and no top-k buffer silently drops work.
4. Derive packing only for graphs being materialized; the full-rank predicate
   already proves that the unique transform exists for every emitted graph.
5. Compile a declared sample of implementation continuations, record actual
   SASS/resource outcomes, and run complete survivors on H200.

The CUDA search assigns a compatible pair and an eight-atom continuation shard
to each thread. It uses a count pass,
prefix offsets, then an emission pass, so there is no global atomic append
bottleneck or output-capacity truncation. The CPU reference directly enumerates
all four-element subsets; solution-set equality is checked, not just counts.

## 4. Evidence and limits

The implementation lives in [`garage/cute_sassy`](../garage/cute_sassy/README.md).
It separates graph construction, compiler outcomes, correctness, and measured
latency. The search report includes the complete finite denominator, rejected
mass, surviving graphs, cold setup, warm count-plus-emission, and a flat
enumeration comparison. GPU event time and host wall time are reported separately.

This v1 completely enumerates its decoder contract graph space. It does **not**
exhaust every source spelling, output permutation, pipeline, MMA geometry, or
full-kernel schedule. A bounded compile/measurement sample cannot certify a
latency optimum. Existing SASS is useful evidence about compiler behavior, not
an exact compositional latency oracle. A small isolated decoder is not sufficient
evidence of its register allocation, spills, or throughput in the full kernel.

The route to the broader system is to add typed graph rewrites at new boundaries:
CuTe ownership/movement, tensor-core fragments, producer/consumer squads,
buffer stages and lifetimes. Each extension must supply its exact compatibility
predicate, native lowering, continuation state, and evidence obligations. The
same join/cache machinery then searches those factors without restarting from
the current fastest kernel.

The search is fast in this implemented finite space: the H200 warm count/scan/
emission pipeline takes **1.346 ms host wall time** for **5,989,005 canonical
quartets**, emitting **1,816 legal graphs**. Flat C++ enumeration on the same
machine takes **350.011 ms** and the factored CPU join takes **7.009 ms**; complete
solution sets agree. Compilation and first setup are additional costs. See the
[timing and scope report](../garage/cute_sassy/README.md) for the exact denominator,
proofs, setup costs, and native evaluation evidence. These measurements establish
fast search, not a full-kernel latency improvement or global optimality.

## 5. The outer search required for the actual goal

The partial-state controller and the constructors identified above are executable.
The broader action space below remains the formulation: fusion-cut synthesis,
new MMA orientations, and shared/shared MMA fed by a separate dequantization
squad still need native constructors. The controller cannot discover a design
that its grammar cannot express.

### Partial realizations and coordinated moves

A search state is a partial physical realization of the whole operator, with
unresolved choices and explicit obligations. Its desired value is

$$V(s)=\sup_{\Pi\in\operatorname{Complete}(s)}\log S(\Pi).$$

The fastest known completion is evidence about this value; a slow sampled
completion does not bound the whole branch from above. A partial realization
need not compile until enough choices have been made to close its obligations.

The outer action grammar must expose structural decisions:

- Work ownership: expert/token/output tiles, persistent versus partitioned work,
  and split/reduction placement.
- Dataflow: fusion cuts and the storage/lifetime of the FF1-to-FF2 intermediate.
- Compute: MMA family and operand placement, tile geometry, accumulator ownership.
- Pipeline: producer/consumer squads, register redistribution, staged resources,
  publication and retirement.
- Layout/instructions: producer and consumer maps, lossless packing, fragment
  construction, and movement/decode instruction DAGs.

A structural move carries its dependent choices. For example, changing MMA
ownership must jointly reconstruct fragment layouts, weight and scale mapping,
accumulator storage, shared-memory staging, and synchronization. Testing an
isolated packing change against an unchanged consumer would reject an incomplete
design. A typed partial state or coordinated rewrite preserves that continuation.

### Conditional subgames

A subgame is a region of the implementation graph plus its boundary contract:

$$B=(\text{semantics},\text{input/output maps},\text{owners},
\text{storage},\text{ready/last-use events},\text{resource budgets}).$$

The outer search chooses among boundary contracts. An inner solver constructs
compatible implementations conditional on the selected contract. It retains
distinct resource/layout/implementation alternatives that may matter to the
continuation. The current GF(2) solver supplies one such inner solver. CuTe
movement and fragment synthesis and WarpSpeed phase/resource construction would
supply others, each requiring a native lowering.

Register allocation, scheduling, and overlap depend on the enclosing program.
Inner microkernel timings therefore cannot be added to obtain the parent's
value. Complete-program compilation and matched measurements resolve those
interactions. Cache identities must include the relevant boundary and compiler
context; semantic equivalence alone does not justify merging search continuations.

The implemented semantic cuts make this concrete:

- **d1 packed GEMV:** replace wasted tensor-core columns with HFMA2 pair
  operations, then reconstruct the four-lane K reduction and output ownership.
  Groupwise scale placement is another implementation with a separate rounding
  obligation, not a free algebraic rewrite of floating-point arithmetic.
- **Complementary sparse MMA:** derive a lossless K permutation and matching
  scale owners from the sparse fragment map. Complementary metadata covers every
  dense term. The doubled nominal K does not alone imply doubled useful work.
- **Activation residence:** compare global-to-register loading, shared prefetch,
  and shared residence with one K64 chunk in registers. The last construction
  couples shorter lifetimes, two-slot weight and partial-sum rings, and a two-CTA
  launch whose actual compiled occupancy must be at least two.

MCTS chooses these families before selecting an enclosing MAIN realization.
The lowered small-load program and the other five load paths are measured as one
complete dispatch policy. A family can win on d1 and still lose the geometric
mean; all six rows remain visible. A compiler-generator defect is repaired in a
new recorded implementation context rather than treated as physical impossibility.

### MCTS, regret, and abstraction refinement

Use MCTS to allocate expensive complete-program evaluations across partial
realizations. Maintain explicit edges, visits, return statistics, unresolved
alternatives, and backup paths. Expand architectural families and compatible
completions, including branches with currently poor samples. Preserve a
nonzero exploration allocation and widen the action abstraction as evidence
exposes missing choices. For a graph with transpositions, update the selected
decision path and retain edge statistics; merging states requires equivalent
continuations. Estimated values and exploration bonuses are ordering heuristics,
not certified bounds on the best descendant.

Regret matching can be another allocation policy. CFR+ supplies equilibrium
guarantees for the appropriate extensive-form game; inventing opposing layout
and scheduler players does not turn this fixed benchmark objective into such
a zero-sum game. Counterfactual action values must include compatible downstream
completions. Unmeasured siblings have unknown values, not zero regret. A genuine
adversarial workload game would optimize a different robustness objective unless
its payoff is explicitly matched to the required score.

The relevant Libratus precedent is its combination of an abstract blueprint,
nested subgame solving, and repairs to the action abstraction. Its 2017 result
was superhuman heads-up no-limit play. The design-search counterpart is a coarse
whole-program grammar, conditional refinement, and adding new implementation
actions when the abstraction excludes useful designs. Poker's subgame safety
theorems do not automatically transfer to compiler optimization.

References: [Libratus architecture](https://ijcai.org/proceedings/2017/4),
[safe and nested subgame solving](https://arxiv.org/abs/1705.02955),
[UCT](https://aima.cs.berkeley.edu/~russell/classes/cs294/s11/readings/Kocsis%2BSzepesvari%3A2006.pdf),
[CFR+](https://ijcai.org/proceedings/2015/97).

### Bounds and evidence that may close a branch

Only proven infeasibility or a valid bound covering every remaining completion
can eliminate an architectural branch exactly. A measured baseline time slice,
instruction count, or empirical bandwidth ceiling is not such a bound. Changing
fusion, ownership, or reuse can change the bytes moved and the overlap itself.
Those measurements remain useful priors, with their scope recorded.

The progress measure is distinct valid complete-program designs evaluated per
compile/GPU budget and their confirmed score, including losing designs. A new
design must change the physical graph or its maps/protocol, not just a source
label. Final success requires complete correctness and a reproducible 2.7x+
score under the target protocol; private gains relative to a frozen parent are
intermediate evidence. The six-load objective stays fixed throughout the search.

## 6. Finite alphabets: paired decode and FP8 component products

The E2M1 alphabet is $\pm\{0,\tfrac12,1,\tfrac32,2,3,4,6\}$.
For two packed codes, the unscaled byte-to-BF16x2 map has only 256 inputs.
Separating the signs leaves 64 magnitude pairs: a 256-byte table plus sign
insertion, rather than a general 8-bit-to-32-bit synthesis problem.

There is also an exact linear embedding already exploited by the decoder family:

```text
BF16bits(E2M1(q) * 2^-126) = ((q & 7) << 6) | ((q & 8) << 12)
E4M3bits(E2M1(q) * 2^-6)  = ((q & 7) << 2) | ((q & 8) << 4)
```

The compensating scale and the treatment of subnormals belong to the enclosing
contract. A cheap embedding is not permission to lose those factors.

For a normal E4M3 scale, $s=(-1)^z2^{e-7}(1+m/8)$. Its sign and exponent can
be restored after looking up only $(q_0\bmod8,q_1\bmod8,m)$:

- 512 packed BF16x2 entries, or **2 KB**, cover normal scales.
- A second 512-entry table covers the subnormal-scale branch: **4 KB total**.
- Zero codes suppress exponent addition; signs, including signed zeros, are
  restored separately. The two NaN scale encodings retain their NaN class.

`format_search.py` exhaustively checks all 65,024 finite byte/scale pairs and
512 NaN cases. An independent Torch FP8/BF16 comparison checks the finite cases.
The H200 shared-table probe passes all 65,536 cases. This establishes the value
map, not its latency inside a consumer. Bank layout, query ownership, table
publication, register pressure, and stage residence remain search decisions.

### BF16 as FP8 components

For normalized inputs in a supported exponent window, search constructions

$$x=h+2^{-r}l,\qquad h,l\in\operatorname{E4M3}.$$

Enumerating the finite sumsets for $r=0,\ldots,8$ gives exact coverage before
compilation. Two implemented constructors are:

| Construction | Complete BF16 exponent bins after common scaling | Scalar cases |
|---|---|---:|
| Clear the low four BF16 fraction bits; $l=16(x-h)$ | -6 through 7 | 3,584 |
| Round $x$ to nearest E4M3; $l=32(x-h)$ | -7 through 7 | 3,840 |

Each count includes both signs and all 128 BF16 significands per exponent.
These are bounded exact-reconstruction statements, not a representation of
arbitrary BF16 exponent spread with one shared scale. Block normalization,
outlier handling, conversion placement and numerical fallback are explicit
constructor obligations.

For one NVFP4 K16 group $g$, let $Q_g$ contain the raw E2M1 values, $s_g$ its
per-output-row weight scales, and $a_g$ activation scales per token. Then the
real-arithmetic factorization is

$$C_g=\operatorname{diag}(s_g)\left[Q_gH_g+2^{-r}Q_gL_g\right]
\operatorname{diag}(a_g).$$

Raw E2M1 is exactly representable in E4M3, so this boundary needs two component
products. **K16 scales cannot be pulled outside a reduction that mixes different
scale groups.** Alternatively, absorbing the scale into each weight produces
up to six significant bits; those scaled weights generally need a second FP8
component too. Decomposing both operands then gives four products. Dropping a
cross term is an approximate action requiring the complete numerical gate.

The compiler is a required part of this search. On the recorded H200/CUDA 12.8
stack, FP8 `mma.sync` K16 and K32 probes compile and pass the component tests,
but their actual SASS contains `F2FP` conversions and ordinary FP16 `HMMA`.
The PTX datatype therefore does not establish native FP8 throughput. The
`wgmma` probe passes both component constructions and emits two actual
`QGMMA.64x8x32.F32.E4M3.E4M3` instructions. It explicitly publishes shared
operands through the async-proxy fence and retires the WGMMA group before use.
Preserving two independent K16 scale groups within this native K32 instruction
remains a complete-constructor obligation. Candidate factorizations may carry
the group into an output-column axis before scaling/combining, or retain
separate reductions; their extra storage and unused products must be counted.

Evidence and reproduction: [format report](../garage/cute_sassy/results/FORMAT_REPORT.md),
[finite search](../garage/cute_sassy/format_search.py),
[native probes](../garage/cute_sassy/format_probe.cu).
The native probes use constant tiles to isolate arithmetic and lowering; they
are not a full fragment-map or complete-operator qualification.

Primary references: [PTX 8.7 MMA and WGMMA](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma),
[component-product decomposition](https://arxiv.org/abs/1904.06376).

## 7. Construct the action space before allocating search

An MCTS policy over a fixed grammar cannot propose a transformation absent from
that grammar. Responsibility for enlarging it belongs to the co-designer.
Before scheduling more trials, derive candidate rewrites from semantic facts:
finite alphabets, shared operands, homogeneous paths, reduction boundaries,
numerical tolerance, and ownership/admission constraints. Each proposal needs
an identity or approximation bound, an actual source constructor, expected
removed work, introduced work, and a falsifiable experiment. Instruction counts
and local timings remain evidence at their own scope.

For NVFP4 GEMV, activation reuse across rows suggests memoizing **products**:

$$T_{k,c}=x_k q(c),\qquad
y_r=\sum_g s_{r,g}\sum_{k\in g}T_{k,c_{r,k}}.$$

A warp's 32 table registers hold two signed 16-entry alphabets, or four unsigned
8-entry alphabets. The latter restores signs after lookup. This derives K2/K4
chunks from the format and warp width. Weights become K-quad-major; lanes own
output rows; shuffles query the register tables; scale stays outside its whole
K16 sum. Split-K supplies parallel CTAs and adds a measured final reduction.
The [implemented frontier](../garage/cute_sassy/opportunities.py) checks every
short codeword, constructs nonconstant native GEMV probes, and records
[H200 evidence](../garage/cute_sassy/results/OPPORTUNITY_REPORT.md). Product-table
reuse has prior art in [LUT-GEMM](https://arxiv.org/abs/2206.09557); no novelty or
complete-operator advantage follows from implementing this variant.

A second rule exposes scale placement as a format decision. Let
$N_\epsilon(z)=z/\sqrt{\operatorname{mean}(z^2)+\epsilon}$. Then

$$N_\epsilon(cz)=N_{\epsilon/c^2}(z),\qquad c>0.$$

A positive global scale can move through the up input of SwiGLU, the down
projection, and post-norm row multiplication. It cannot move through the SiLU
gate or combine independent K16 scales before their separate reductions. Keep
the epsilon correction. The resulting exponent gauge can be searched jointly
with FP8 component encodings, subject to finite-range and rounding gates. This
identity has a CPU check; its complete native constructor is still missing.

These opportunities are separate from the existing MCTS action set until their
complete physical lowerings exist. Codex expands the explicit rewrite library;
the allocator chooses trials within it. Neither role alone establishes 2.7x.

## 8. Gemma-assisted improvement of the executable language

Let the search state contain a rewrite library $\mathcal L_t$, a proposal/ranking
policy $q_t$, and source-bound evidence $E_t$. Searching more leaves of
$\mathcal L_t$ changes the selected kernel; adding a verified constructor changes
which kernels can be expressed. The latter is the improvement sought here:

$$\Delta_t\sim q_t(\Delta\mid C,\mathcal L_t,E_t),\qquad
\mathcal L_{t+1}=\mathcal L_t\cup
\{\Delta_t:\operatorname{verified}(\Delta_t,C)\}.$$

Codex owns the construction and acceptance. Gemma assists with bounded proposals
and criticism. CuTe supplies coordinate/value identities and equivalences;
WarpSpeed supplies explicit squad/resource/phase accounting. These constrain
both one-shot proposals and learned search. The compiler and complete hardware
measurement decide properties those abstractions do not establish.

The first pilot uses Deneb's pinned Gemma 4 31B QAT base and saved WarpSpeed
refiner adapter, with two proposals and one round of actual checker feedback.
Weights stay frozen. Responses, checker counterexamples, adapter-effect probes,
authorship, and subsequently implemented constructors are retained separately.
The model cannot alter the reference, tolerance, timed work, or acceptance gate.

There are three distinct outcomes to measure: a new valid constructor; a faster
complete kernel; and improved search efficiency on fresh tasks under matched
proposal, compile, and GPU budgets. None implies the others. Reflection in a
prompt is not by itself evidence of recursive self-improvement or reinforcement
learning. Training/ranking on verified continuations is a later measurable step;
it should improve the held-out search curve against a frozen policy and the
deterministic selector before receiving credit.

## 9. Make refiner plans executable through GPU tools

The refiner now has a direct execution interface. `evaluate_program` accepts a
typed physical plan and invokes the existing complete-operator constructor,
compiler, occupancy/reference checks and paired H200 measurement.
`evaluate_codelet` accepts a new CUDA K16 arithmetic implementation; a generated
wrapper supplies the exact input coordinates, independent group scales, stores
and final reduction. Both tools return actual evidence to the next model turn.
This replaces the manual plan-to-source handoff within the supported interfaces.

The model stays resident on an inference GPU and invokes separate native CUDA
workers. Codex owns the formulation, constructors and acceptance gate; GPU
access is supplied by the tool layer. Repeated rounds retain losing evidence,
reject duplicate implementations, and can revise compiler or numerical failures
without returning to the host agent for a kernel edit.

The [implementation interface](../garage/cute_sassy/AUTO_IMPLEMENTATION.md)
documents its boundaries. A complete-program ratio and a GEMV-component ratio
have different scopes. New fusion/layout protocols still need a corresponding
typed constructor. The first native run demonstrates implementation and repair;
it establishes neither a speedup nor a learned-search advantage.
