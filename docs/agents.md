# 3 · Knowledge and agents

A co-design agent needs more than a proposal model. It needs a representation of
what can be built, access to the sources behind that representation, and a record
of what its experiments established. This chapter connects those pieces:
**research memory → a justified decision → a physical construction → feedback**.

The [formulation](formulation.md) defines the physical objects and obligations.
[Search](search.md) determines how their alternatives are explored. Here, the
knowledge base supplies context, OODA governs the research process, and separate
planner and refiner roles operate inside that process.

<a id="knowledge-base"></a>
## 1. A knowledge base for co-design

The research KB connects kernel source, layout algebra, hardware semantics,
physical plans, compiler observations, experiments and reviews. Its useful unit
is a claim with a source and a scope. A record should let an agent recover what
a construction means, where it came from, and what happened when it was tried.

A compact representation is

$$G=(V,E,\tau_V,\tau_E,p),$$

where $\tau_V$ and $\tau_E$ give node and relation types, and $p$ retains
provenance. Several connected graphs have distinct meanings:

| Graph | Objects and relations | Use |
|---|---|---|
| **Source graph** | Modules, symbols, definitions and examples; contains, imports, calls and references | Find an operation's implementation and dependencies |
| **API dependency graph** | Callable contracts and candidate output-to-input bindings | Test whether operations compose under supplied shapes and preconditions |
| **Co-design graph** | Computation, movement, storage and synchronization; data, ordering and reuse edges | Describe a physical realization |
| **Design/evidence graph** | Designs, code moves, compiler records and observations; one-edit and provenance links | Recover what changed and what a comparison established |
| **Concept and hardware graphs** | CuTe definitions, PyCuTe code, WarpSpeed resources, PTX instructions and scoped hardware facts | Resolve algebraic and hardware obligations |
| **Source-discovery graph** | Public repositories, papers, issues and implementations linked by references and mechanisms | Find related work outside the current design family |

A source call edge does not establish API compatibility. Executing a coordinate
expression does not establish a legal buffer lifetime. A neighbouring measured
design does not establish the value of an unmeasured completion. These boundaries
are preserved when records move between graphs.

### Ingestion preserves the unit of reasoning

The research implementation splits Markdown by section, CUDA by kernel or
embedded namespace, and Triton/CuTe Python by decorated function. Repeated kernel
bodies are deduplicated while retaining a source identity. Briefs, reviews,
phase catalogs and ledger rows remain different record kinds, connected through
run, mechanism and source links.

CuTe, PTX, WarpSpeed, TLX and SASS records supply definitions, target conditions,
source fragments and compiler observations. Public GPU MODE kernels retain their
problem, GPU, correctness status and board context. Mechanism tags connect names
such as low-bit decode, activation residence, TMA rings and warp specialization
across libraries and weight formats.

Symbolic fields stay outside the embeddings: shapes, instruction targets,
source identities and observed values remain inspectable. The embedding index
helps locate those records; the underlying records carry their meaning.

<a id="retrieval"></a>
## 2. Retrieve a source, then audit its evidence

BM25 supplies a lexical ranking for exact names and mechanisms. Normalized query
and document encoders supply a semantic ranking:

$$s_{\mathrm{dense}}(q,v)=\langle E_q(q),E_d(v)\rangle.$$

The local research KB combines them with reciprocal-rank fusion:

$$
s_{\mathrm{RRF}}(v)=
\sum_{r\in\{\mathrm{lexical},\mathrm{dense}\}}
\frac{1}{c+\operatorname{rank}_r(v)}.
$$

Its implementation uses $c=60$ and zero-based ranks, limits repeated hits from
one source, and supports record-kind filters. An agent can inspect CuTe/PTX
sources without mixing them with an earlier empirical cost model.

EmbeddingGemma supplies the query and document encodings. The research KB also
has a lexical-centroid fallback when a query embedding is unavailable. That
route expands lexical hits and must be distinguished from an independently
encoded query. A separate GPU MODE index explored a dense score plus normalized
lexical score; the two index policies are kept distinct.

### Premise audits

A premise audit follows retrieval into the experiment record. It finds a claim,
walks the hits' run and mechanism links, and displays the associated ledger
outcomes and missed phases. If no judged record sits behind the hits, the audit
reports that the claim is unmeasured.

For an extra prefetch stage, this should recover the candidate source, why it
was proposed, its resource/compile observations, and the review of the complete
operation. The next design can then test whether the same ownership, working
set and target make the earlier result applicable.

### Design neighbourhoods

For one-edit relations between complete designs,

$$N(v)=\{u:(v,u)\in E_{\mathrm{one\text{-}edit}}\}.$$

When observations share a reference and measurement contract, the observed
effect of a field change $f:a\to b$ can be summarized by

$$
\delta_\ell(f:a\to b)=
\operatorname{median}_{(u,v)}\frac{\rho_\ell(v)}{\rho_\ell(u)},
$$

where $\rho_\ell$ is speedup against that reference for load $\ell$.
This can reveal which loads a field affects or recover a useful part of a design
that lost overall. It describes observed contexts; it is not an additive latency
law or a proof that the neighbourhood contains a global optimum.

### Attribution to graph use

A graph-informed move records its provenance before selection:

$$
\text{query or seed}\to\text{source edges}\to\text{applicable fact}
\to\text{construction choice}\to\text{obligations}\to\text{outcome}.
$$

Actual traversal establishes navigation. A fact that changes the selected
construction establishes a decision role. A performance benefit from graph
structure requires a matched comparison with alternatives such as flattened
source access. Some research comparisons reached the same mechanism through
both routes. Those outcomes remain part of the [studies](studies.md).

<a id="ooda"></a>
## 3. OODA as an execution protocol

xCalibur adapts **Observe–Orient–Decide–Act** into an artifact-backed protocol for
research agents. Orientation becomes inspectable state, a decision becomes a
falsifiable brief, and a lesson becomes a check where the mistake could recur.

A compact description of the agent's state is

$$X_t=(C_t,F_t,E_t,K_t,B_t,q_t),$$

with task/constraints $C_t$, design frontier $F_t$, experiment record $E_t$,
retrievable knowledge $K_t$, remaining budget $B_t$, and execution phase $q_t$.
This notation describes the protocol; the research implementation stores its
state and artifacts in files.

An action advances that state when its applicable gate passes:

$$
X_{t+1}=\operatorname{Transition}(X_t,a_t,r_t),
\qquad \operatorname{Gate}(q_t,a_t,C_t,E_t)=\mathrm{pass}.
$$

The gate checks prerequisites and evidence bindings. The action's result
supplies the observation. The cycle has five concrete artifacts:

1. **Entry:** current task, rules, target, source revision, baseline, budget and
   recent outcomes.
2. **Orientation:** relevant sources, prior experiments, physical obligations
   and unresolved alternatives.
3. **Agenda and brief:** ranked alternatives, the selected change, its scope,
   prediction or explicit unknown, falsifier and expected learning.
4. **Experiment:** exact candidate identity, declared evaluator and raw result.
5. **Review:** what happened, what it means, and what changes next.

The implemented loop refuses a new brief while the previous run lacks its
review, and refuses an experiment without an accepted brief. Repeated misses on
a mechanism trigger a deeper after-action review. A reset-and-pivot plan records
the revised frontier, retired assumptions, next milestone and stop condition.
The later physical-board wrapper also rejects stale or incomplete entry context.

### Lessons become invariants

The update is operational:

$$
\text{failure}\to\text{reviewed mechanism}\to
\text{decision-time predicate}\to\text{tool check}.
$$

A no-op measured on an unaffected build becomes a reach check. A timing mismatch
becomes a protocol identity and requalification requirement. A forgotten prior
experiment becomes a premise audit. An edited candidate becomes a requirement
for receipts bound to the new source.

After an accepted review, the research harness rebuilds the KB and identifies
new records awaiting embeddings. The next session can recover both the lesson
and the evidence behind it. Brief-time predictions stay beside outcomes;
hindsight cannot silently replace foresight.

### Authority and retained alternatives

Constraints record whether they come from a versioned rule, an operator
directive, or an inference awaiting verification. The operator sets the
objective, scope, hardware and spending authority. Within that contract, agents
can gather evidence, propose constructions and execute authorized checks.

The agenda allocates attention. A witnessed contradiction can eliminate its
covered completions; a measured losing leaf applies to that leaf's context;
a heuristic cutoff spends budget elsewhere without a completeness guarantee.
Reasons, source states and restore paths remain available for backtracking.

A handoff preserves the current state, exact source, unresolved obligations,
retained alternatives and next valid transition. Delegated work returns artifacts
and evidence to that state. It does not authorize competing updates to the same
incumbent or conflicting experiments. Required artifacts earn their place by
changing a decision or checking an actual obligation.

<a id="physical-board"></a>
## 4. Bind the reasoning to one physical board

The Rebel investigations extended OODA with a versioned technical board. CuTe,
WarpSpeed and PTX reasoning must all describe the same complete operator.

- **Operator:** computation coverage, precision, tails and dispatch.
- **Maps:** logical values, owners and storage, including packed coordinates and
  original scale-group identity through every transformation.
- **Lifetimes:** actual thread squads, buffer generations, producers, consumers,
  publication, last use and overwrite.
- **Admission:** shared bytes, register expectations, residency and progress.
- **Lowering:** operations bound to applicable instructions, operand maps,
  ordering scopes and sources.
- **Evidence:** prior-art applicability and each obligation's pass/pending/fail
  status, evaluator and falsifier.

For a buffer generation $g$, the lifetime obligation includes

$$
\operatorname{publish}(g)\prec\operatorname{read}(g),\qquad
\operatorname{lastRead}(g)\prec\operatorname{overwrite}(g+1),
$$

under the actual participant sets and memory visibility rules. A filled field
points to the reasoning or test that establishes these relations; it does not
prove them by itself.

At evaluation time, a passed receipt binds the candidate's source identity:

$$h_{\mathrm{source}}(\mathrm{receipt})=h(\mathrm{candidate}).$$

The wrapper checks file hashes and required evidence states. Offline coordinate
and lifetime obligations remain distinct from target compilation and runtime
checks. A source change cannot silently inherit a passed receipt for an older
candidate.

<a id="planner-refiner"></a>
## 5. Separate planning, refinement and implementation

TG4DAC used separate planner and refiner adapters while targeting **Gemma 4
31B's operators** on SM120. The Gemma 4 models proposing designs and the Gemma 4
workload whose operators are executed have distinct roles.

The planner chooses decomposition, fusion, tiling, ownership and layout
expressions. The refiner revises those choices against resource and dependency
constraints. The implementer constructs the native program and obtains compiler,
correctness and timing evidence. Each result retains that attribution.

A plan transition can be written

$$
P_0\sim q_\theta(\cdot\mid C,h,R),\qquad
\Delta P\sim r_\phi(\cdot\mid P_0,C,h,R,\mathrm{checks}),\qquad
P_1=P_0\oplus\Delta P,
$$

where $C$ is the computation/numerical contract, $h$ the hardware context, and
$R$ the retrieved source context. A patch invalidates affected facts; its new
obligations must be checked again.

The planner can execute PyCuTe expressions while constructing the plan. For
ownership map $O$ and storage layout $L$, the address map is $A=L\circ O$.
An execution record connects the source symbol and expression to the affected
plan field. Layout checks, resource calculations and synchronization claims
retain their separate evaluators.

Training investigations covered graph navigation, executable API composition,
completion of missing plan decisions, and repair from concrete counterexamples.

### Teach a policy over connected completions

For the partial state $z$ in the [formulation](formulation.md#graph-completion),
retrieval $\mathcal R(z)$ supplies source-linked construction records. The
proposed policy ranks live actions:

$$a\sim\pi_\phi(\cdot\mid z,\mathcal R(z)),\qquad
z'=\operatorname{Close}_C(z\cup\{a\}).$$

The teacher erases selected decisions from a verified construction, preserving
shared holes and surrounding constraints. Closure propagates a proposed binding
through the connected program and returns consequences or a contradiction.

The supervision has two meanings. A **construction label** identifies a
completion that satisfies its stated checks; it need not be unique or fastest.
A **performance label** needs measured complete continuations. For the observed
continuations $\mathcal M(z,a)$ of an action,

$$\widehat Q(z,a)=
\min_{\Pi\in\mathcal M(z,a)}\widehat T_C(\Pi).$$

Here $\widehat T_C$ is complete-operation latency under the fixed contract.
Unmeasured actions have unknown labels. Timing noise and unequal exploration
remain attached to the evidence; the minimum observed is not an optimality proof.
For a workload aggregate, use the declared objective $J_C$ in place of latency.

A weighted supervised objective is

$$\mathcal L_{\mathrm{SFT}}(\phi)=
-\sum_{d\in\mathcal D_{\mathrm{train}}}
w_d\log\pi_\phi(a_d\mid z_d,\mathcal R_d),\qquad w_d\ge0.$$

Split source and implementation ancestry **before** generating descendants or
retrieval indexes. Keep construction replay separate from actual search traces.
Compare policies with the same executable language, checks and research budget,
including deterministic and untrained controls. This teaching formulation does
not imply that a general completion compiler or trained scheduler is distributed.

<a id="anvil"></a>
## 6. Anvil: place the models inside the search

Anvil combines this memory and role split with exact checks, constructors and
measured search. For a workload $p$, assemble context from the design listing,
retrieved sources, the incumbent and the planner's own earlier outcomes:

$$C_p=(\operatorname{listing}(G),\operatorname{retrieve}(p),c_p,M_p).$$

The planner proposes a typed design $x$. The checker returns diagnostics
$\Delta(x)$; the refiner proposes a repair; search completes the fields left
open. The constructor then lowers the completed design, followed by compiler,
numerical and complete-operation checks.

A reach predicate rejects changes that cannot affect the serving build. A
declared field map supports this predicate only when it covers the actual
lowering and routing choices. Candidate and reference comparisons retain the
same measurement contract. Screening nominates; repeated confirmation supplies
evidence for a replacement. Every result, including rejection and regression,
returns to the design graph and planner memory.

The proposed deployment uses one host and GPU: the host handles ordinary checks
and compilation, while models and candidate kernels use the device. Model
residency, available memory and benchmark interference belong to that experiment
contract. A successful tool call is one step in the loop; the full native
construction and evaluation path must be established separately.

<a id="public-code"></a>
## 7. Public interfaces and implemented research

The public package exposes composable pieces:

| Code | Responsibility |
|---|---|
| [`Node`, `DesignGraph`](../xcalibur/graph.py) | Design fields, observations, one-edit queries, median effects and compact listings |
| [`Retriever`, `embedding_gemma`](../xcalibur/graph.py) | Embedding retrieval with a pluggable encoder |
| [`PlannerLoop`](../xcalibur/planner.py) | Typed model proposals, checker feedback, one repair attempt and per-problem result memory through callbacks |
| [`ReachMap`](../xcalibur/reach.py) | Declared build reach and lowering signatures |
| [`Game`, `Evidence`](../xcalibur/game.py) | Screening, repeated confirmation and fixed-reference evidence accounting |
| [`Card`, `rules_gate`, `lint`](../xcalibur/gates.py) | Compact run summaries, rules-version checks and selected timing-path pattern checks |

The full KB ingester, hybrid indexes, research corpus, OODA state machine and
physical-board validator were implemented in research harnesses. They are
documented here rather than bundled into these small interfaces. Anvil's native
constructors, model serving setup and trained adapters are also separate from
`PlannerLoop`.

Use [getting started](getting-started.md) for the callable interfaces, and
[the studies](studies.md) for TG4DAC, Rebel and the native implementation work.
The [references](references.md) identify the primary layout, GPU and retrieval
sources behind the system.

[← 2 · Search](search.md) · [1 · Formulation](formulation.md) · [4 · Studies →](studies.md)
