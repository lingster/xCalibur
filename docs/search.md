# 2 · Search and games

The [physical formulation](formulation.md) describes what a kernel must realize.
Search decides which compatible realizations to construct, which unknowns to
resolve, and which experiment to run next. In xCalibur, this developed from a
measured graph of complete designs into partial-program construction, conditional
subgames, and Rebel's constructor/challenger experiments.

The common object is a design with explicit alternatives and obligations. A
search policy can rank those alternatives; a checker can establish a scoped
fact; a complete implementation can earn a measured value.

<a id="codesign-game"></a>
## The co-design game

Fix a context $`C=(P,H,W,\mathcal G,\mathcal E,B)`$: the operator and numerical
contract, hardware/compiler context, workloads and metric, construction grammar,
evaluators, and research budget. In the complete-design game, group the physical
choices as

```math
\Pi=(\Gamma,\sigma,L,\tau,I),
```

where $`\Gamma`$ is dataflow and materialization, $`\sigma`$ is the CTA/squad plan,
$`L`$ contains storage and fragment layouts, $`\tau`$ is scheduling and dispatch,
and $`I`$ selects instruction/decode constructions. These fields are coupled:
changing a consumer's fragment can require a different packing and producer.

A graph node is a complete design. An edge is a declared move: rebind a field,
replace a subgraph, change a schedule, compose compatible mechanisms, or start
from another archived construction. An archive root carries its history and
unexplored neighbourhood; a losing overall result can still contain a useful
component. The constructor grammar determines which moves are executable.

For a load $`\ell`$, the recorded paired value is

```math
\widehat J_\ell(\Pi)=\log\mathrm{median}_i
\frac{t^{(i)}_{\mathrm{ref},\ell}}{t^{(i)}_{\Pi,\ell}}.
```

Under an equal geometric-mean objective over loads $`W`$,

```math
\widehat{\mathcal S}(\Pi)=
\exp\!\left(\frac{1}{|W|}\sum_{\ell\in W}\widehat J_\ell(\Pi)\right).
```

The reference, workload and measurement protocol stay fixed for a comparison.
The log objective separates algebraically by load. The design decisions separate
only while their shared packing, dispatch, resource and launch interfaces do.
A coordinated cross-build change must be evaluated as a coordinated design.

## A position can contain an unfinished program

The structural formulation retains choices before they form a complete kernel:

```math
s=(X,D,K^+,K^-,O,R,E,b,h).
```

$`X`$ holds bound choices and remaining domains; $`D`$ records dependencies;
$`K^+,K^-`$ contain established and refuted contextual claims; $`O`$ lists open
obligations; $`R`$ describes resources, allocations and lifetimes; $`E`$ contains
applicable evidence; $`b`$ is remaining budget; and $`h`$ is the action history.
A claim in neither knowledge plane is unknown, and $`K^+\cap K^-=\varnothing`$.

An action names a typed constructor or checker and the binding versions it read.
For Boolean prerequisites $`P_a`$, forbidden bits $`F_a`$, and clear/set masks
$`\Delta_a^-,\Delta_a^+`$, a compact guard and update are

```math
(X\land P_a)=P_a,\qquad X\land F_a=0,
```

```math
X'=(X\land\neg\Delta_a^-)\lor\Delta_a^+.
```

Bitsets make these checks cheap. They index richer records: integer extents,
original scale coordinates, allocation aliases, event order and source identity.
GF(2) matrices encode binary-linear maps; arbitrary strides and ragged routing
retain their integer relations and predicates.

A rewrite changes coupled bindings together, adds obligations and invalidates
receipts whose premises changed. The old observations remain in history. A
proposal cannot declare itself correct, compiled or fast; the evaluator supplies
those observations. Identical source with a different pending check or remaining
budget can therefore be a different research position.

<a id="conditional-subgames"></a>
## Conditional subgames

Let $`\mathrm{Complete}_C(s)`$ be the eligible complete constructions
reachable from a partial state. Its ideal engineering continuation value is

```math
V_C(s)=\sup_{\Pi\in\mathrm{Complete}_C(s)}\mathcal S(\Pi).
```

The best measured completion gives evidence about this value. A slow sampled
completion does not bound all its siblings from above. No eligible completion
and an unevaluated completion are different outcomes. A qualified incumbent can
remain an explicit outside option.

A subgame freezes a boundary contract

```math
B_s=(\text{values and precision},\text{owners and maps},
\text{ready/retirement events},\text{resource envelope},
\text{external bindings},\text{evidence versions}).
```

The outer search chooses the boundary; an inner solver returns compatible
implementations under it. It keeps alternatives whose resource use, layout or
lowering may matter to the continuation. A decoder solver, a movement/fragment
solver and a squad/phase constructor can be different inner solvers.

For example, an invertible bit-linear decoder $`D`$ with reference map $`P`$ determines its
packing $`T=D^{-1}P`$. Deriving $`T`$ eliminates a search variable. It does not decide
which ownership, decode instruction spelling or enclosing pipeline will be fast.
Those remain continuations of the exact map.

A structural move must carry its dependencies. Changing MMA ownership can
require new fragments, code/scale packing, accumulator storage, gather loops,
staging and synchronization. A partial state retains these obligations until a
compatible completion exists. Rejecting an isolated packing edit against an
unchanged consumer would reject an unfinished design rather than its possible
completion.

Component latencies do not add when regions overlap or contend. Cache keys for
algebra facts, compiler outcomes and measurements consequently differ: reuse of a
coordinate proof does not authorize reuse of a whole-program timing.

<a id="structural-search"></a>
## Construct the space before allocating search

CuTe Sassy expresses construction as a typed AND/OR graph. An OR node chooses an
implementation; an AND node requires all of its input contracts and obligations.
The action space exposes work ownership, fusion/materialization, compute family,
operand placement, activation residence, pipeline phases and dispatch.

Four reductions give the search structure:

1. **Eliminate determined variables.** Derive packing, inverse maps or forced
   bindings instead of enumerating them independently.
2. **Join compatible boundaries.** Construct producers forward and consumer
   requirements backward, then match maps, ownership and event constraints.
3. **Share proofs while retaining implementations.** Equal semantic maps can
   share algebra work; different source recipes and schedules can compile
   differently and remain distinct continuations.
4. **Cache with context.** Compilation depends on exact source, toolchain, target,
   flags and enclosing program. Measurement additionally depends on packing,
   dispatch, launch, workload and timing protocol.

The implemented families included packed-BF16 GEMV for small loads,
complementary sparse-MMA constructions, and alternatives for activation residence.
These change the physical computation and its lifetimes. New fusion cuts or MMA
orientations need corresponding constructors before search can explore them.

MCTS allocates complete-program evaluations across partial realizations. Each
visited decision retains its alternatives, edge visits, return statistics and
backup path. For a graph with transpositions, equivalent continuation states may
share work, while the selected decision edges retain their statistics. A state
merge must preserve the choices and obligations available afterward.

A learned prior orders expansion; exploration and widening leave room for poorly
sampled families. A failed compiler-generator attempt may identify a missing
lowering rather than an impossible physical construction. The engine returns
its retained frontier and unresolved alternatives when its budget ends.

Shortest-path methods fit a different subproblem. Dijkstra can minimize a stated
nonnegative additive movement or rebuild cost when its state is sufficient. It
cannot recover whole-kernel latency by adding context-free timings of isolated
edits. Direct boundary joins are often enough when compatibility can be solved
algebraically.

<a id="measured-search"></a>
## Rank candidates, then measure their completions

The measured game uses a bootstrap ensemble of LightGBM regressors. For features
$`x(\Pi)`$ and ensemble predictions $`f_k`$, define

```math
\mu_\ell(\Pi)=\frac1K\sum_{k=1}^K f_{k,\ell}(x(\Pi)),\qquad
s_\ell(\Pi)=\mathrm{sd}_k f_{k,\ell}(x(\Pi)).
```

Its exploration score is

```math
A_\ell(\Pi)=\mu_\ell(\Pi)+\kappa s_\ell(\Pi)-J_\ell(\Pi_{\mathrm{ref}}).
```

Features include design fields, decoder-map bits, resource counts and compiler
observations. Before compilation, a ranker sees what the design declares; after
compilation it can also see registers, spills and instruction features. Ensemble
spread is an exploration signal, not a certified confidence interval for latency.

The public [`Game`](../xcalibur/game.py) implements the following progression:

1. Explore the supplied frontier until a load has enough history to fit a ranker.
2. Keep candidates that affect the build serving that load, using a declared
   [`ReachMap`](../xcalibur/reach.py); collect nominations across loads.
3. Compile a bounded batch and re-rank successful records using compiled features.
4. Screen candidates, then confirm promising designs alongside the fixed reference.
5. Pool confirmation evidence by validated build identity and recommend bindings.

For repeated relative gains $`g_i=t_{\mathrm{ref},i}/t_{\Pi,i}-1`$, the public
[`Evidence`](../xcalibur/stats.py) binding rule is

```math
n\ge n_{\min},\qquad \overline g>m,\qquad \min_i g_i>0.
```

This repeat-and-margin rule is a selection heuristic. It is separate from the
bootstrap intervals provided by the statistics module. One candidate nominated
for several loads is measured once; one round contributes at most one
confirmation per load/build identity. Changing the reference or protocol requires
fresh comparable evidence.

The public loop takes construction, feature and measurement callbacks. Research
MCTS controllers and complete-operator constructors belong to separate harnesses.

## Backward reasoning and what may close a branch

Forward construction asks what can be built. Backward reasoning asks what would
have to change to meet a target. Let $`\Delta=|W|\log(\mathcal S^*/\mathcal S_0)`$
be the required total log-gain. Given stated per-load floors $`F_\ell`$, a planning
allocation can use modeled headroom $`h_\ell=\log(T_\ell/F_\ell)`$, assuming positive
floors, nonnegative headroom and $`\sum_j h_j>0`$:

```math
r_\ell=T_\ell\left(1-\exp\left[-\Delta
\frac{h_\ell}{\sum_j h_j}\right]\right).
```

$`r_\ell`$ is the assigned latency reduction. It is useful when the floors and
allocation assumptions are meaningful. A floor derived from empirical bandwidth
or a baseline time slice need not survive a change in reuse, ownership or fusion.
The allocation therefore prioritizes experiments rather than proving that other
architectures cannot work.

For a move affecting only loads $`U\subseteq W`$, the corresponding optimistic
scope screen is

```math
\mathcal S_{\mathrm{screen}}(U)=\mathcal S(\Pi)
\exp\left[\frac1{|W|}\sum_{\ell\in U}\log
\frac{T_\ell(\Pi)}{F_\ell}\right].
```

This becomes a pruning certificate only if each floor applies to every remaining
completion under the same metric and context. A sampled loss, low ranker score,
instruction count, missing measurement or exhausted budget supplies no such
certificate. Exact infeasibility needs a witness covering the represented
completion set; exact performance pruning needs a valid upper bound below a
qualified alternative's supported lower bound.

Keep completed, refuted and unexpanded regions distinct. Save the chosen child
and its siblings. A new mechanism or a repaired constructor can reopen an
empirically unpromising region without changing the task's semantics.

<a id="rebel"></a>
## Rebel: constructing and challenging designs

Rebel made this research state into an explicit constructor/challenger game.
A constructor commits a physical design or a bounded partial construction. The
challenger observes it and chooses a registered query; a trusted referee supplies
the result. Queries can expose a wrong value/scale association, missing output,
conflicting writer, premature overwrite, progress failure or unsupported compiler
claim. A challenge names a source binding and a falsifier.

Codex and Claude served as researcher roles in the outer protocol. The tabular
solver used mathematical max/min actors for those roles. Hardware facts, the
knowledge base and compiler receipts are shared evidence, rather than additional
players with invented preferences.

For the finite PyCuTe verification game,

```math
g(c,q)=\begin{cases}
+1 & \text{construction }c\text{ passes query }q,\\
-1 & \text{query }q\text{ produces a scoped contradiction}.
\end{cases}
```

Retaining the qualified parent without making a new construction claim was a
separate zero-valued action. An unexecuted cell was unknown, not zero. The
constructor maximizes $`g`$ and the challenger minimizes it. Since the challenger
sees $`c`$, its policy depends on that construction; a simultaneous matrix with one
fixed query per column would describe a different game.

A proposed performance-synthesis utility used a bounded transform of a qualified
score, $`j(c)=\widehat{\mathcal S}(c)/(1+\widehat{\mathcal S}(c))`$, and assigned
hard-contract violations $`-1`$. The executed PyCuTe learning episode instead used
the verification utility. This keeps the question the agent learned to answer
explicit: surviving those queries did not supply an unmeasured hardware payoff.

The constructor used actual PyCuTe composition, `right_inverse` and
`logical_divide` calls to couple output ownership, activation slices, token
capacity and independent weight/activation ring depths. One branch combined a
K256 weight stage with a permuted-K64 activation stage. It had to change global
activation packing, copy scheduling and gather/final ownership together.
Negative controls caught wrong original-K16 scale association, incomplete output
coverage and ring reductions unsupported by the input packing. Native register
allocation and publication/progress remained separate obligations.

<a id="cfr"></a>
## Recursive CFR and learned continuation targets

The ReBeL adaptation implemented recursive depth-limited CFR searches, tabular
value/policy targets and self-play. It supplied a finite perfect-information
construction tree in which both players observe the admitted history. The
original ReBeL method and its upstream implementation are credited in the
[references](references.md).

At each public state, the solver builds a depth-limited subgame. Known terminal
queries keep their exact utilities; nonterminal cutoffs use the current value
table. Fresh local regrets produce an average strategy. For player $`i`$ at history
$`h`$, the update uses the opponent's reach probability:

```math
R_i^{t+1}(h,a)=R_i^t(h,a)+\rho_{-i}^t(h)
\bigl(v_i^t(h,a)-v_i^t(h)\bigr),
```

```math
\pi_i^{t+1}(a\mid h)=
\frac{[R_i^{t+1}(h,a)]_+}{\sum_{a'}[R_i^{t+1}(h,a')]_+}.
```

A zero denominator selects a uniform strategy. Average-policy targets weight
strategies by the player's own reach. The subgame supplies a value target and
policy targets; random-iteration sampling with explicit exploration selects the
next continuation. Recursive self-play updates the lookup tables and records
which histories were visited. An untrained cutoff's initial zero is labeled an
estimate, separate from a known terminal payoff.

The final policy is extracted in coherent depth-sized chunks, with new solves at
chunk boundaries. Exact full-tree best responses then audit this learned policy.
They were not substituted for the policy during training.

For constructor policy $`\pi_C`$, challenger policy $`\pi_Q`$ and constructor utility
$`U`$, the improvement gaps are

```math
\epsilon_C=\max_{\sigma_C}U(\sigma_C,\pi_Q)-U(\pi_C,\pi_Q),
```

```math
\epsilon_Q=U(\pi_C,\pi_Q)-\min_{\sigma_Q}U(\pi_C,\sigma_Q).
```

The audit reports both and their sum over the complete admitted tree, including
histories unvisited during self-play. The recorded runs had nonzero gaps. These
quantify exploitable choices within the frozen validation game; they do not
certify globally optimal GPU code. The lookup targets estimate validation
utility, not native latency, and are not neural planner training results.

<a id="evidence"></a>
## Evidence connects the game to the machine

Each observation belongs to a chain: context and action, complete construction,
source, compiler configuration, qualification, workload and measurement. A
coordinate proof, a compiler receipt and a target timing answer different parts
of that chain. A native candidate introduced outside the current grammar must be
registered as a new action or identified as an outer-protocol nomination; its
timing cannot be credited to an abstract action that never produced that source.

The experiments exposed this boundary directly. An earlier bargaining formulation
explored construction owners and conditional agreements. The general engineering
contract retained one complete-operator objective. The later zero-sum
constructor/challenger pilot supplied a concrete finite game and an executable
learning loop. Its formulation review then identified the missing connection
from validation-game survivors to qualified, measured native completions.

The same discipline applies to measured search. A historical cache-flush mismatch
made repeated apparent gains disappear when the target protocol was restored.
Repetition had reduced sampling uncertainty around the wrong measurement.
Protocol identity, A/A controls and reference comparisons therefore belong in the
state, not just in an experiment's prose description.

Public code exposes [`Game`](../xcalibur/game.py),
[`Surrogate`](../xcalibur/surrogate.py), [`ReachMap`](../xcalibur/reach.py),
[`Evidence`](../xcalibur/stats.py) and [`Bench`](../xcalibur/bench.py).
The research ReBeL engine, structural constructors and original episode bundles
are documented studies rather than bundled package modules. The
[implementation studies](studies.md) connect these methods to the operators used
to test them.

The next chapter explains how the [knowledge base and agent operating loop](agents.md)
retain this state, retrieve useful mechanisms and turn observations into the next
co-design decision. See [getting started](getting-started.md) for the public APIs.
