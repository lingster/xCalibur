# xCalibur

**Co-designing computation through executable mathematics, search and agents.**

xCalibur explores automatable algorithms for tractable problems, with the aim of
making more computation practical on accessible hardware. GPU kernels are the
main proving ground: their layouts, ownership, movement, lifetimes and instruction
choices make the consequences of a design concrete.

## The organizing problem

Given a computation and hardware contract $C$, construct a compatible physical
realization and optimize its complete execution:

$$
\Pi^*(C)\in\arg\min_{\Pi\in\Omega(C),\ \operatorname{Pass}_C(\Pi)}T_C(\Pi),
\qquad \Pi=(\Gamma,\mathbf S,\boldsymbol\alpha,\omega).
$$

Here $\Gamma$ is the construction program, $\mathbf S$ its shapes,
$\boldsymbol\alpha$ its coordinate mappings, and $\omega$ its execution bindings.
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
   retrieval and premise audits, OODA for agents, and the Anvil planner/refiner loop.
4. **[Studies](docs/studies.md)** — how these ideas were exercised in Gemma 4,
   MoE, complete LoRA operators, structural construction and ARC.

The [documentation index](docs/README.md) provides section-level navigation.

## Use the code

[`xcalibur/`](xcalibur/) contains reusable algebra, checking, search, graph and
measurement tools. [`kernels/`](kernels/README.md) contains the H200 NVFP4 MoE and
**TG4DAC's Gemma 4 31B attention and W4A16 MLP + LoRA kernels** for SM120.

[Getting started](docs/getting-started.md) covers installation, runnable examples
and API contracts. The [data](data/README.md) and
[notebooks](notebooks/README.md) hold the supporting observations and visual analyses.

## License

[Apache 2.0](LICENSE). The H200 MoE builds on work by vjeux, alygg77 and Matsvei;
their notices remain in the source. See the [kernel credits](kernels/moe_w4a16_h200/README.md)
and [references](docs/references.md).
