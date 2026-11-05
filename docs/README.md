# Read xCalibur

Four chapters connect the mathematics, search procedures, agent system and
experiments. Read them in order, or enter at the question you care about.

## Reading order

1. **[Physical formulation](formulation.md)** — What is a kernel design, and
   which choices must agree? Start with the [contract](formulation.md#contract),
   then [layouts](formulation.md#layouts), [graph completion](formulation.md#graph-completion),
   [GF(2) packing](formulation.md#gf2) and [WarpSpeed](formulation.md#warpspeed).
2. **[Search and games](search.md)** — How do we explore that space? Follow the
   [co-design game](search.md#codesign-game), [conditional subgames](search.md#conditional-subgames),
   [structural search](search.md#structural-search) and [Rebel](search.md#rebel).
3. **[Knowledge and agents](agents.md)** — How does work accumulate into better
   decisions? Read the [knowledge base](agents.md#knowledge-base),
   [retrieval](agents.md#retrieval), [OODA protocol](agents.md#ooda) and
   [Anvil integration](agents.md#anvil).
4. **[Studies](studies.md)** — Where did these ideas meet an implementation?
   [Gemma 4 / TG4DAC](studies.md#gemma4), [H200 MoE](studies.md#h200),
   [structural construction](studies.md#structural-search),
   [complete operators](studies.md#complete-operators) and [ARC](studies.md#arc).

Each chapter defines its notation and distinguishes a formulation, its
implementation, and the evidence collected from it. Equations and worked
constructions stay with the explanation. Visual analyses live in the
[notebooks](../notebooks/README.md); measurements and their provenance live in
[data](../data/README.md).

## Run or extend it

**[Getting started](getting-started.md)** covers installation, CPU examples,
optional dependencies, measurement and build/export contracts. The
[kernel guides](../kernels/README.md) cover GPU-specific requirements.
The [implementation guide](studies.md#availability) identifies which systems
are distributed as code and which are documented studies.

**[References](references.md)** credits the mathematical foundations, libraries,
prior kernels and learning/search methods.

[Repository](../README.md) · [Start with the formulation →](formulation.md)
