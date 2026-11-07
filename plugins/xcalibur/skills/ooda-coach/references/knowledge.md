# Knowledge: gather, curate, retrieve, extrapolate

The ledger records what happened. The knowledge layer makes what happened **change the next decision**. Adapted from
xCalibur's [knowledge base](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#knowledge-base),
[intel](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#intel),
[retrieval](https://github.com/xArmory/xCalibur/blob/main/docs/agents.md#retrieval) and the reflection
[knowledge has to alter a decision](https://github.com/xArmory/xCalibur/blob/main/docs/reflections.md#knowledge-transfer).

The discipline has three steps, in order: **gather intel → curate it into records → extrapolate to a scoped,
falsifiable hypothesis**. A citation that leaves the estimate unchanged isn't knowledge use.

## 1. Gather intel

- **Map sources by actor and channel** before searching: the workload owner (what they run, how they score), the
  engines and libraries you build on (release notes, issues, the versions actually deployed), research (papers,
  preprints), public implementations and other teams' **measured** results, and hardware or platform documentation.
- **Sweep channels first, then search by mechanism.** Release indexes, changelogs and issue trackers before keyword
  search. Then query by mechanism, **across labels**: the same construction hides under different names (INT4 /
  MXFP4 / NVFP4 / NF4; "few-shot" / "in-context examples"; "reasoning effort" / "thinking budget").
- **Never dismiss a source by its label**, and read primary sources in full; summaries drop the conditions.
- **Record each lead:** source, when checked, what it contributes, and its scope (hardware, model, version, workload).
- A source-coverage checklist proves a process was followed, not that the literature was covered or read.

## 2. Curate: claims with source, scope and evidence type

The unit of knowledge is a **claim with a source and a scope**, not a paragraph. Keep:

| Field | Example |
|---|---|
| claim | (illustrative) "low reasoning effort shortens list values without fewer thinking tokens" |
| source | ledger E5–E7, commit, run ids; or paper / issue URL |
| scope | model, quantisation, prompt version, workload, hardware |
| evidence type | proof · resource calculation · tool/compiler observation · fixture test · screen · confirm (n sessions) |
| status | established · refuted · unknown (never mix the three) |
| mechanism tags | e.g. `prompt-sentence`, `token-budget`, `cache-state` |

- **Symbolic fields stay outside embeddings.** Shapes, versions, ids and observed values must stay inspectable; an
  embedding only helps find a record, the record carries the meaning.
- **Link records** by run id, mechanism tag and source id, so a premise audit can walk from a claim to its evidence.
- **Keep each record's meaning when it crosses contexts:** a call edge doesn't prove compatibility; a measured
  neighbour doesn't give the value of an unmeasured one; a result on one model or prompt version isn't a result on
  another.
- **Refresh after every accepted IAR:** add the run's outcome to the records it touches (including losses and
  nulls), and update the status of the claims it tested.

Minimum viable form: a `KNOWLEDGE.md` (or JSONL) of claim rows next to `LEDGER.md`. Grow into a graph (below) when
the design space has more than a few dozen measured variants.

## 3. The design graph

Record every measured design as a node and connect designs that differ in **exactly one field** (one edit). This is
xCalibur's `DesignGraph` ([`xcalibur/graph.py`](https://github.com/xArmory/xCalibur/blob/main/xcalibur/graph.py);
walkthrough in [`notebooks/07_knowledge_graphs.ipynb`](https://github.com/xArmory/xCalibur/blob/main/notebooks/07_knowledge_graphs.ipynb)).

- **Node:** id, kind (design, code move, submission, function, concept, instruction), fields (the design's knobs),
  per-workload results, **evidence grade** ("screen 4×1", "confirm 12×2", "A/A") and free text for retrieval.
- **Edges:** one-edit, touches, scored-as, uses, is-a.

Three queries do most of the work:

1. **Neighbourhood** (`one_edit()`): what is one edit away from the incumbent, and what did each edit do? If many
   neighbours were measured and none resolves better, the incumbent sits near a local optimum *of what was screened*
   (not a proof of an optimum).
2. **Field effects** (`effects()`): the median effect of each field change, per workload, under one reference and
   protocol. It shows which workloads a field reaches (the graph rediscovered the reach map from data) and recovers a
   useful component from a design that lost overall. It is not an additive law.
3. **Retrieval** (`Retriever`, `embedding_gemma`): nearest past designs to a proposal. **Similarity is not quality:**
   always return the evidence grade with each hit, and let exact checks decide legality.

`listing()` renders the graph compactly (one line per node plus one-edit effects) so the whole history fits in a
model's context; give proposers that listing, not a snapshot (see [proposers.md](proposers.md)).

Non-kernel example: nodes = (prompt version, reasoning effort, temperature, OCR source) with per-document scores;
an edge per single-field change. "Did we already try this?" becomes a lookup, and a field that flips one document
under every edit (prompt-perturbation sensitivity) shows up as a pattern, not a one-off.

## 4. Retrieve, then audit the premise

1. Retrieve records for the mechanism you're about to rely on (by tag and by text; filter by record kind).
2. Walk their links to the ledger outcomes, including losses and any phase/stage the estimate missed.
3. If no judged record stands behind the hits, say the premise is **unmeasured**.
4. Retrieval finding a claim doesn't verify it: its [M]/[D]/[H] tag is an authoring convention.

## 5. Extrapolate to a new design

When prior evidence comes from a different design, model, workload or machine:

- Recover the prior source, its measured observations and its full-operation (end-to-end) review.
- **List what carries over and what changed** (resources, working set, data, interfaces, lifetimes, budget).
- List the **new obligations** the change creates (in isolation a component may succeed; inside the full pipeline it
  inherits synchronization, overlap, budget and tail costs).
- Write a scoped, falsifiable hypothesis for the agenda, then design the check that separates it from the rivals.
- **De-bias both ways:** narrowing an old negative to its scope does not justify an optimistic prediction for the new
  design; new constructions still owe legality, correctness and matched measurement.

## 6. Record what knowledge did

When knowledge informs a choice, record the chain in the brief: query or seed → records found → the applicable fact
→ the choice it changed → new obligations → outcome. Traversal alone shows navigation; a fact that changed the
choice shows a decision role. Claiming the knowledge base helped needs a matched comparison against flat access.
