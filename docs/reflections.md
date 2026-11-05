# Big-picture reflections

xCalibur's work includes a research method as well as kernels, algebra and search
procedures. The campaign reflections asked whether that method was learning:
were better questions being asked, were earlier lessons changing decisions, and
was the same failed assumption returning under a new name?

This file synthesizes those reflections and their later corrections. The
[agent chapter](agents.md#ooda) describes the OODA harness and its artifacts;
this account considers the patterns across campaigns and what they imply for
co-design agents.

<a id="distinct-record"></a>
## Why reflection has its own file

An **immediate after-action review (IAR)** reviews one run. An **after-action
review (AAR)** reconstructs a mechanism's or campaign's history. **Reset and pivot
(RAP)** changes the working model and next plan.
A big-picture reflection compares those histories, including the assumptions
shared by apparently different projects.

The original cross-campaign reflection deliberately paused execution. It could
leave questions open without immediately turning each observation into another
candidate. That separation matters: when every review ends with a variant of
the current design, the next experiment is inherited from the last one before
the wider frontier has been reconsidered.

A reflection record keeps the campaigns and source revisions it covers, the
patterns supported by their evidence, competing interpretations, and unresolved
strategic choices. Follow-up decisions belong in the next agenda or RAP, linked
back to the reflection. The reflection remains readable as the view held at that
point; subsequent corrections are appended or linked.

<a id="strategic-loop"></a>
## A sound experiment loop can still choose poor questions

The early reflections found a split between tactical and strategic progress.
Briefs could state falsifiers, candidates could pass correctness checks, and
reviews could report losses honestly, while the agenda kept returning to the
same physical structure. Better accounting inside that loop did not by itself
broaden the questions it asked.

The useful comparison was often structural: item geometry, ownership, producer
and consumer roles, synchronization, and work per SM. Different kernel names
could retain all of these choices. Grouping the history by physical construction
made repeated assumptions visible; grouping it only by the latest proposed
optimization concealed them.

This motivates the shared representation in the
[physical formulation](formulation.md#physical-design). The same design objects
used to construct a kernel can also describe what an experiment actually varied
and what its result leaves unresolved.

<a id="knowledge-transfer"></a>
## Knowledge has to alter a decision

The intelligence reviews exposed two separate failures. Discovery missed useful
work when searches used a weight-format label instead of visiting relevant
source channels and searching by mechanism. Later, retrieval found an earlier
design, but its measured behavior never entered the new design's cost estimate.
A citation was present; the decision was unchanged.

The resulting research discipline connects three activities. **Gather intel**
finds sources across implementation families. **Curate knowledge** preserves
their contracts, provenance, evidence type and unresolved claims.
**Extrapolate** identifies a possible transfer and states the conditions under
which it should hold. The [knowledge pipeline](agents.md#intel) carries
those records into orientation and premise audits.

For example, a successful bulk-copy stage in isolation supplies evidence about
that stage. Transferring it into a complete producer/consumer pipeline introduces
register residence, synchronization, overlap and tail behavior. Those become
new obligations. A useful extrapolation names the changed conditions and an
experiment that can distinguish the competing explanations.

The broader lesson is that knowledge curation and search belong in the same
system. Search produces observations that revise the KB; the KB changes which
constructions deserve examination and which obligations must be checked.

<a id="revisable-memory"></a>
## Memory needs a correction path

The later de-biasing review corrected an overreaction to repeated losses.
Interpretations in earlier IARs had become reasons to veto whole design
families. Some runs had implementation defects; others tested a particular wait
schedule, ring depth or staging path. Their results did not settle every
completion sharing a mechanism name.

Let $`s`$ be a partial design and $`\Pi_0`$ one of its eligible completions. Using the
[physical objective](formulation.md#contract) $`J_C`$ and the
[search completion set](search.md#conditional-subgames), even if one completion
has a worse objective,

```math
J_C(\Pi_0) > J_C(\Pi_{\mathrm{ref}})
\quad\not\Rightarrow\quad
\forall\Pi\in\operatorname{Complete}_C(s),\;
J_C(\Pi) > J_C(\Pi_{\mathrm{ref}}).
```

The campaign record therefore needs both the observation and its **binding
conditions**: what construction was actually tested, whether the proposed
mechanism was exercised, and what an instrument isolated. A budget decision to
park a branch remains distinct from a proof that its completions cannot work.

This correction runs in both directions. Narrowing an old negative result does
not justify a more optimistic prediction for a new design. New constructions
still owe legality, correctness and matched measurement. Keeping original
reviews alongside their corrections lets an agent recover why its prior changed.

<a id="models-and-instruments"></a>
## Predictions need a domain, and surprises need instruments

The reflections repeatedly found models calibrated on an incumbent family being
used to price a different physical structure. Instruction counts and byte rates
remained useful observations, but they did not capture every dependency chain,
shared-resource interaction or decomposition cost introduced by the new design.

A compact calibration diagnostic is the log prediction error

```math
e_i=\log\frac{T_i^{\mathrm{observed}}}{\widehat T_i},
```

where both times refer to the same workload and evaluation protocol. Positive
values mean the prediction was optimistic. Grouping these errors by physical
family and workload is more informative than one aggregate score. This is a
way to describe the calibration question, not an additional estimator shipped
by the package.

When several edits leave a result unchanged, the next useful action may be an
instrument that separates causes. Replacing one unmeasured explanation with
another leaves the uncertainty intact. The research record also shows the
limits of instruments: isolating a stage changes its concurrency and resource
context, so its behavior must be reconciled with the complete operator.

One RAP moved attention from a hot-expert path to the larger cold-expert
population. Its first experiment contradicted the expected benefit, and the
close-out revised the headroom model and closed that branch. The useful artifact
is the whole chain—assumption, test, contradiction and revised scope—including
the failed prediction.

<a id="substance-of-gates"></a>
## A gate must engage with the construction

An earlier reflection observed that a lightly edited copy of the previous brief
could satisfy a gate while omitting the new design's dominant cost. A source
field could contain a citation without auditing its premise. More required prose
would not necessarily resolve either failure.

The later harness connects review to concrete objects: current source identity,
physical-board obligations, declared evaluators and receipts. Those checks make
stale or incomplete transitions rejectable. The
[harness account](agents.md#ooda) distinguishes enforced checks from the
interpretation that reviewers must still supply.

The design question for each additional gate is practical: which recurring
failure can it detect, using which artifact, at the point where the decision is
made? A gate that answers that question can turn a lesson into a reusable
invariant. A required paragraph without that connection can merely accumulate.

<a id="open-questions"></a>
## What these reflections leave open

The work brings exact constructions, empirical search and agent memory into one
research process. Its next questions concern how well that process learns:

- Can a shared physical representation retrieve the right prior experiment
  when a new design uses different names or comes from another library?
- Can search retain useful unmeasured alternatives while respecting the scope
  of exact contradictions, measured losses and limited budgets?
- Can calibration recognize a new physical family early enough to ask for an
  instrument instead of assigning it an inherited cost?
- Can a lesson change the next decision reliably, and can a later correction
  reach every place where that lesson was used?

The [studies](studies.md) document parts of this process in use. Answering these
questions requires comparisons that expose the decisions, their supporting
knowledge and the resulting implementations.

[Knowledge and agents](agents.md) · [Documentation](README.md)
