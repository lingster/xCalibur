# 1 · Physical formulation

**Kernel co-design is a constraint problem with a measured objective over its
executable solutions.** The choices include dataflow, ownership, layouts,
storage, instruction families, schedules and synchronization. They are coupled:
a new packing changes its decoder; a new owner changes its fragments; a deeper
pipeline changes lifetime and resource admission.

This chapter defines the physical objects and the relations that make those
choices compatible. [Search](search.md) develops partial programs and conditional
subgames; [agents](agents.md) connects them to models, knowledge and OODA;
[studies](studies.md) follows the resulting implementations and experiments.

<a id="contract"></a>
## The fixed contract and complete objective

Fix

```math
C=(\mathcal O,H,\mathcal W,N),
```

where $`\mathcal O`$ specifies computation and state effects, $`H`$ specifies hardware
and compiler/runtime context, $`\mathcal W`$ specifies workloads, and $`N`$ specifies
numerical behavior. Checkpoint decoding, scale associations, casts, reductions,
adapter behavior and state updates belong to the contract.

For a versioned construction language $`\nu`$, let $`\Omega_\nu(C)`$ contain its
supported complete candidates. One objective is

```math
\Pi^*\in\arg\min_{\substack{\Pi\in\Omega_\nu(C)\\
\mathrm{Pass}_C(\Pi)}}J_C(\Pi),\qquad
J_C(\Pi)=\sum_{p\in\mathcal W}w_p\log T_p(\Pi),
```

with $`w_p\ge0`$ and $`\sum_p w_p=1`$. Here $`T_p`$ is complete-operation latency under
the declared protocol, and $`\mathrm{Pass}_C`$ is implementation admission.
Equal weights give geometric-mean latency; a speedup objective against a fixed
reference gives the same ordering.

The objective includes unchanged regions and boundary costs. Saving decoded
weights for backward can remove work while increasing memory use. Fusion can
remove a materialization while extending register lifetimes. Independent kernel
timings cannot be added to recover those effects.

A bounded search returns its best qualified measured candidate. Defining an
optimum does not establish that the search found it; passing finite numerical
fixtures is recorded as a test result with that scope.

<a id="physical-design"></a>
## A physical realization

Represent a complete candidate as

```math
\boxed{\Pi=(\Gamma,\mathbf S,\boldsymbol\alpha,\omega).}
```

| Object | What it fixes |
|---|---|
| $`\Gamma`$ | Typed operations, produced values, consumers, fusion boundaries and explicit sharing |
| $`\mathbf S`$ | Nested shapes and their selected refinements |
| $`\boldsymbol\alpha`$ | Coordinate-factor maps and bindings shared across the program |
| $`\omega`$ | Hardware roles, storage, lifetimes, schedules, instructions, communication and launches |

The co-design game's alternative notation $`\pi=(\Gamma,\sigma,L,S,I)`$ separates
CTA organization $`\sigma`$, layouts $`L`$, schedule $`S`$ and instructions $`I`$.
It describes the same kinds of coupled decisions; the tuple above makes shared
coordinate construction explicit.

For each value edge, record

```math
\Phi_e(i)=\bigl(\mathrm{owner}(i),\mathrm{storage}(i),
\mathrm{address}(i),\mathrm{ready}(i),
\mathrm{lastUse}(i)\bigr).
```

Ownership includes CTA, warpgroup, warp, lane and local slot, with intentional
replication stated explicitly. Storage names registers, shared or global memory.
Addresses carry units and packed-bit/scale-group meaning. Readiness and last use
are events in a partial order, before a concrete instruction schedule exists.
For storage generation $`g`$, require

```math
\mathrm{ready}_{e,g}\prec\mathrm{consume}_{e,g},\qquad
\mathrm{lastUse}_{e,g}\prec\mathrm{overwrite}_{e,g+1}.
```

The implementation must realize that order at the participating hardware scope.
A named barrier, transaction completion or publication protocol is part of the
design; coordinate equality cannot supply it.

For compiler context $`\kappa`$, the enclosing correctness obligation is

```math
\mathrm{unpack}_{out}\circ
\mathrm{Exec}(C_\kappa(\Pi))\circ\mathrm{pack}_{in}
\simeq_N\mathcal O.
```

A structural move may therefore need to change ownership, fragments, packing,
accumulators, staging and publication together. Testing one isolated field
against its old consumer can reject an incomplete construction rather than the
intended complete design.

<a id="layouts"></a>
## Ownership, storage and executable layout algebra

A layout maps logical coordinates to storage. An ownership map identifies the
logical value handled by a physical worker. Their composition gives its address:

```math
\text{worker coordinate}\xrightarrow{O}\text{logical coordinate}
\xrightarrow{L}\text{storage offset},\qquad A=L\circ O.
```

For a row-major matrix of width $`H`$, let $`L(r,c)=rH+c`$. If lane $`\ell`$ holds four
consecutive columns, with local slot $`s\in\{0,1,2,3\}`$, then

```math
O(r,\ell,s)=(r,4\ell+s),\qquad A(r,\ell,s)=rH+4\ell+s.
```

This is a simple coverage map. An MMA fragment requires the instruction's own
ownership contract. Domains, predicates, intentional replication and offset
units remain explicit in either case.

For a producer's logical-value map $`p`$, consumer map $`c`$ and transfer $`R`$, value
agreement is

```math
p\circ R=c.
```

The relation has declared domains; a general inverse need not exist. A fusion
also requires compatible ownership, schedule and lifetime. Otherwise it needs
conversion, materialization or another supported recipe.

[PyCuTe](https://github.com/NVIDIA/cutlass/tree/main/python/pycute) makes supported
coordinate algebra executable while constructing a plan. Source graphs connect
the expression to library definitions and examples; execution records the actual
result. Ordinary integer-stride layouts and XOR-linear bit layouts use different
algebras: integer carries prevent treating them interchangeably.

<a id="graph-completion"></a>
## Shared coordinate choices as graph completion

The proposed graph-completion environment exposes a bounded language of
construction recipes. A model ranks unresolved shape, map and execution choices;
deterministic tools propagate their consequences through the connected program.
The categorical layout definitions are inherited from Colfax; the constrained
search, execution bindings and learning procedure are the xCalibur formulation.

### Factor maps and derived layouts

Use ordered flattenings $`S^\flat=(s_1,\ldots,s_m)`$ and
$`T^\flat=(t_1,\ldots,t_n)`$. Factor positions are one-based, coordinate values are
zero-based, and the first leaf varies fastest. The basepoint $`*`$ discards a
factor; $`?`$ denotes an unresolved choice.

The supported coordinate choice is a pointed map

```math
\alpha:\{*,1,\ldots,m\}\longrightarrow\{*,1,\ldots,n\},
```

subject to

```math
\alpha(*)=*;\qquad
\alpha(i)=j\ne *\Rightarrow s_i=t_j;\qquad
\#\alpha^{-1}(j)\le1\quad(j\ne *).
```

It is a partial matching between equal-extent factor positions. Unused target
coordinates are zero. These tractable **Tuple** morphisms extend to **Nest**
through ordered flattening. [Colfax, §3.1.1](https://arxiv.org/html/2601.05972v1#S3.SS1.SSS1)

For the morphism $`f`$ represented by $`\alpha`$, derive $`L_f=S:D_f`$ with leaf strides

```math
(d_f)_i=\begin{cases}
0,&\alpha(i)=*,\\
\displaystyle\prod_{j<\alpha(i)}t_j,&\alpha(i)\ne *.
\end{cases}
```

For flat coordinates, $`\varphi_{L_f}(x)=\sum_i(d_f)_ix_i`$ and
$`\Phi_{L_f}=\varphi_{L_f}\circ\mathrm{colex}_S^{-1}`$. Thus selecting the
wiring determines the encoded layout. [Colfax, Construction 3.1.2.1](https://arxiv.org/html/2601.05972v1#S3.SS1.SSS2)

### Composition propagates a decision

For a required path $`S\xrightarrow{f}T\xrightarrow{g}U`$, a morphism-level
constraint becomes

```math
\boxed{\beta\circ\alpha=\gamma.}
```

Binding one edge restricts its neighbours. A target shape shared with the next
edge is one object. If the contract only requires equality of realized coordinate
functions, check that weaker condition; extent-one factors can make it distinct
from morphism identity.

For an allowed refinement $`T'\twoheadrightarrow T`$, the categorical construction
produces $`S'=f^*T'`$ and $`f':S'\to T'`$ while preserving the indicated integral
layout function, $`\Phi_{L_{f'}}=\Phi_{L_f}`$. Here the pullback acts on refinements.
The search bounds shape depth, leaf count, extents and allowed refinements so its
domains remain finite. [Colfax, §3.2.5](https://arxiv.org/html/2601.05972v1#S3.SS2.SSS5)

### Worked finite completion

Take $`S=(3,128,128)`$ and $`T=(3,2,128,2,128)`$. Fourteen factor maps satisfy the
rules above. Requiring a return path with $`\beta\circ\alpha=\mathrm{id}`$ leaves:

| $`\alpha`$ | Derived $`\beta`$ | Derived stride of $`L_f`$ |
|---|---|---|
| $`(1,3,5)`$ | $`(1,*,2,*,3)`$ | $`(1,6,1536)`$ |
| $`(1,5,3)`$ | $`(1,*,3,*,2)`$ | $`(1,1536,6)`$ |

The shapes and first layout follow Colfax's example. The enumeration and
return-path constraint are our worked derivation. Both completions preserve the
required composite while exposing different intermediate layouts. Algebra gives
no performance preference between them; the surrounding construction decides
what can use each one.

### Reuse must be a program value

Equal addresses do not establish equal issued work:

```text
Load(address) -> Extract0        word = Load(address)
Load(address) -> Extract1        word -> Extract0
                                word -> Extract1
```

The right-hand recipe declares sharing through one produced value. Its scope,
consumers and lifetime belong to $`\Gamma`$ and $`\omega`$, and the lowering must
preserve them.

For an INT4 projection, write $`h=b+8w+32g`$ with coordinate shape
$`S_q=(8,4,H/32,I)`$. Packed-word and scale projections give

```math
\begin{aligned}
\alpha_P&=(*,1,2,3),&\varphi_P(b,w,g,i)&=w+4g+(H/8)i,\\
\alpha_s&=(*,*,1,2),&\varphi_s(b,w,g,i)&=g+(H/32)i.
\end{aligned}
```

Discarded factors expose repeated addresses. A constructor still has to turn
that opportunity into shared loaded instances. Extraction, scale association and
BF16 rounding remain fixed numerical operations around those projections.

### The partial state and closure

A completion state retains

```math
z=(C,\Pi_{\mathrm{partial}},\mathcal D_z,\mathcal F_z,\mathcal E_z),
```

where $`\mathcal D_z`$ holds live finite domains, $`\mathcal F_z`$ holds derived facts
and dependencies, and $`\mathcal E_z`$ holds permitted evidence. An action selects
a supported constructor or binds a shared decision. The transition is

```math
z'=\mathrm{Close}_C(z\cup\{a\}).
```

Closure propagates composition/refinement constraints, derives layouts and
applies available dependence/resource rules. It returns an updated state or a
concrete contradiction. Forced choices need no model decision. Unresolved
siblings remain available to the [search policy](search.md).

<a id="gf2"></a>
## Exact decoder construction over GF(2)

A bit-linear decoder is a matrix $`D`$ over $`\mathbb F_2`$. Row $`i`$ is the mask of
input bits XORed into output bit $`i`$:

```math
y_i=\bigoplus_j D_{ij}x_j.
```

Equality of two known linear maps can be checked on their basis images. This
covers every word in the declared domain. Zero and basis evaluations alone do
not prove an arbitrary bit program linear; nonlinear operations need a separate
contract or exhaustive finite check.

Let $`P`$ be the reference map and $`D`$ an invertible candidate of matching width.
The packing relation determines

```math
\boxed{DT=P\quad\Longrightarrow\quad T=D^{-1}P,}\qquad D(Tx)=Px.
```

When $`P`$ is also full rank, $`T`$ is invertible and the repacking is lossless.
Packing is therefore derived from the decoder instead of independently searched.
Scale groups, physical word ordering, consumers and numerical operations still
belong to the enclosing kernel contract.

A finite grammar can supply partial decoder atoms. If atom row sets $`A`$ and $`B`$
have disjoint output fields, their independent join requires

```math
\mathrm{rank}(A\cup B)=|A|+|B|.
```

Pairwise compatibility bitsets prune the product, but pairwise independence alone
does not establish independence of a larger tuple. The final combined rows must
reach the required rank. Semantically equal maps can share proof work while
retaining distinct source implementations for compiler evaluation.

[`gf2.py`](../xcalibur/gf2.py) provides row maps, rank, composition, inversion,
packing transforms and a small exact join. The production decoder campaign is
covered with the other [studies](studies.md).

<a id="warpspeed"></a>
## WarpSpeed as a physical plan language

A CTA is partitioned into **squads**, contiguous warp ranges with assigned roles.
Squads exchange values through **resources**, staged shared-memory rings. Each
resource has **phases** with named owner squads. A stage's phase transition
connects production, consumption and release of that storage generation.

This vocabulary gives the model, constructor and checker one physical plan to
work on. It also exposes search variables: squad sizes, stage counts, strides,
ownership and register targets. Bounded domains make a particular plan grammar
finite. WarpSpeed supplies the vocabulary; the project's checker and construction
interfaces build on it. [CCCL reference](references.md#cccl)

For squad warp counts $`w_s`$, resource stages $`k_r`$, byte strides $`b_r`$ and phase
counts $`q_r`$, the modeled arrival and shared-memory accounting is

```math
\mathrm{arrivals}(p)=32\sum_{s\in\mathrm{owners}(p)}w_s,
```

```math
B_{\mathrm{smem}}=\sum_r k_rb_r+8\sum_r k_rq_r+
B_{\mathrm{alignment}}+B_{\mathrm{extra}}.
```

The first term is ring storage; the second is one eight-byte mbarrier per stage
and phase. Alignment padding and auxiliary storage must be included. An optional
TMA phase records expected transaction bytes as well as thread arrivals. Ring
phase ownership must agree with the native arrival/wait/release protocol.

For the public H200 accounting profile, with $`W=\sum_s w_s`$ warps and a target
of $`c`$ CTAs per SM, the modeled launch register cap is

```math
r_{\mathrm{launch}}=\min\!\left(255,
\left\lfloor\frac{65536}{128\lceil W/4\rceil c}\right\rfloor_8\right),
```

where $`\lfloor x\rfloor_8`$ rounds down to a multiple of eight. This accounts for
the declared launch under that architecture profile. `setmaxnreg` targets require
compatible whole-warpgroups and a redistribution that fits the available pool.
Actual compiler allocation and occupancy remain observations to check.

[`warpspeed.py`](../xcalibur/warpspeed.py) places squads, derives phase arrivals,
allocates rings/barriers, and checks CTA/SM budgets. It returns derived quantities
and typed violations that a refiner can act on. The shipped profile is H200;
the Gemma 4 SM120 studies use their own hardware context.

Resource accounting does not prove deadlock freedom, memory ordering or lifetime
correctness. Cross-CTA publication additionally requires grid-level ownership,
progress and retirement arguments. Those obligations stay attached to the full
physical realization rather than disappearing after a checker pass.

<a id="constraints"></a>
## What the connected checks establish

The candidate must satisfy

```math
\mathcal K_C=
\mathcal K_{\mathrm{maps}}\land\mathcal K_{\mathrm{values}}\land
\mathcal K_{\mathrm{dependences}}\land\mathcal K_{\mathrm{resources}}\land
\mathcal K_{\mathrm{lowering}}.
```

Map checks establish declared coordinate relations. Value checks preserve scale
association, casts, reductions and state effects. Dependence constraints require
$`(i,j)\in\Delta_\Gamma\Rightarrow\theta(i)\prec_{\mathrm{lex}}\theta(j)`$,
and the native synchronization must realize that order. Resource checks account
for the selected physical plan. Lowering binds it to supported generated code.

**Reach is a dependency question.** For workload $`p`$, compare the selected build
and the fields read by its lowering. A dispatch change can alter the build even
when its kernel fields are unchanged. [`ReachMap`](../xcalibur/reach.py) records
this declared dependency; validate its completeness against captured inputs or
compiled code before using it to pool observations.

**Compilation fixes a context.** A compile failure rejects that lowering in that
context; it need not prove the mathematical design impossible. A matching SASS
hash alone does not establish the same packing, launch, dispatch or workload.
[`Capture`](../xcalibur/capture.py) compares complete captured build inputs;
compiler and runtime identity require the additional context.

**Evidence retains its scope.** Exact coordinate proofs, resource calculations,
compiler observations, numerical fixtures and complete-operation timings are
different records. A failure or bound can close only the choices it actually
covers. The [co-design search](search.md) preserves that distinction when
selecting, pruning and refining partial programs.

[Next: search and co-design games →](search.md) · [Setup and code](getting-started.md)
