# xCalibur

**Co-designing computation through executable mathematics, search and agents.**

xCalibur explores automatable algorithms for tractable problems, with the aim of
making more computation practical on accessible hardware. GPU kernels are the
main proving ground: their layouts, ownership, movement, lifetimes and instruction
choices make the consequences of a design concrete.

## The organizing problem

Given a computation and hardware contract $`C`$, construct a compatible physical
realization and optimize its complete execution:

```math
\Pi^*(C)\in\arg\min_{\Pi\in\Omega_\nu(C),\ \mathrm{Pass}_C(\Pi)}J_C(\Pi),
\qquad \Pi=(\Gamma,\mathbf S,\boldsymbol\alpha,\omega).
```

Here $`J_C`$ aggregates complete-operation latency over the declared workloads,
and $`\Omega_\nu(C)`$ is the construction language's candidate space.
$`\Gamma`$ is the construction program, $`\mathbf S`$ its shapes,
$`\boldsymbol\alpha`$ its coordinate mappings, and $`\omega`$ its execution bindings.
These choices share constraints. CuTe/PyCuTe makes the coordinate algebra
executable; WarpSpeed expresses squads, resources and phases. Search explores
compatible completions, while the knowledge base and agent workflow preserve
what each experiment teaches.

## Read the work

1. **[Physical formulation](docs/formulation.md)** — the constraint problem,
   ownership and lifetime maps, graph completion, GF(2) packing proofs and
   WarpSpeed resource equations.
2. **[Search and games](docs/search.md)** — the co-design game, CuTe Sassy's
   conditional subgames, and Rebel's constructor/challenger and tabular ReBeL/CFR
   experiments.
3. **[Knowledge and agents](docs/agents.md)** — the research knowledge base,
   intel gathering and curation, the OODA harness with IARs, AARs and reset and
   pivot (RAP), and the Anvil planner/refiner loop.
4. **[Studies](docs/studies.md)** — how these ideas were exercised in Gemma 4,
   MoE, complete LoRA operators, structural construction, ARC, and a long-document
   prefill on RTX 3090 and RTX 6000 Ada cards.

The [documentation index](docs/README.md) provides section-level navigation.
[Big-picture reflections](docs/reflections.md) examine what the campaigns taught
us about choosing experiments, transferring knowledge and revising assumptions.

## Use the code

[`xcalibur/`](xcalibur/) contains reusable algebra, checking, search, graph and
measurement tools, with resource profiles for H200, sm_86 and sm_89 boards.
[`kernels/`](kernels/README.md) contains the H200 NVFP4 MoE,
**TG4DAC's Gemma 4 31B attention and W4A16 MLP + LoRA kernels** for SM120, and
block-FP8 (sm_89) and INT8 W8A8 (sm_86) GEMM kits.

[Getting started](docs/getting-started.md) covers installation, runnable examples
and API contracts. The [data](data/README.md) and
[notebooks](notebooks/README.md) hold the supporting observations and visual analyses.

## License

[Apache 2.0](LICENSE). The H200 MoE builds on work by vjeux, alygg77 and Matsvei;
their notices remain in the source. See the [kernel credits](kernels/moe_w4a16_h200/README.md)
and [references](docs/references.md).
