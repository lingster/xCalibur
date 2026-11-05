"""Notebook sources for the broader research studies; imported by build.py."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def notebooks(setup):
    return {
        "08_layout_ownership": [
            md(r"""# Layout, ownership, and the complete operation

Ownership identifies the logical element handled by a lane and slot. A storage
layout turns that element into an address:

$$O(r,\ell,s)=(r,4\ell+s),\quad L(r,c)=rH+c,\quad A=L\circ O.$$

**Evidence:** a synthetic CPU teaching map, not an MMA fragment, followed by nine
recorded L4 F1 comparisons. [Physical design](../docs/formulation.md#layouts)."""),
            code(setup + r"""
rows, lanes, slots = 8, 32, 4
width = lanes * slots
owners = np.empty((rows, width), dtype=int)
addresses = []
for row in range(rows):
    for lane in range(lanes):
        for slot in range(slots):
            col = slots * lane + slot
            addresses.append(row * width + col)
            owners[row, col] = lane
assert sorted(addresses) == list(range(rows * width))
print(f"{len(addresses)} elements, exactly one owner each; all addresses in bounds.")
fig, ax = plt.subplots(figsize=(11, 3))
im = ax.imshow(owners, aspect="auto", interpolation="nearest", cmap="viridis")
ax.set_xlabel("logical column"); ax.set_ylabel("logical row")
ax.set_title("Four consecutive elements per lane")
fig.colorbar(im, ax=ax, label="owner lane")
save(fig, "layout_ownership"); plt.show()
"""),
            md("## Same shape, broken ownership\n\nReplacing `4 * lane + slot` with `lane + slot` preserves the loop bounds but creates duplicate writers. Check addresses and coverage, not just shapes."),
            code(r"""bad = [row * width + lane + slot
       for row in range(rows) for lane in range(lanes) for slot in range(slots)]
assert len(set(bad)) < rows * width
print("colliding writes:", len(bad) - len(set(bad)))
print("uncovered elements:", rows * width - len(set(bad)))
"""),
            md("""## Correctness and performance remain separate

The historical xR38 revision passed its recorded correctness and sanitizer
checks, but regressed against its preceding revision in eight of nine cases.
Compiler spills are a plausible contributor, not an isolated causal measurement.

**Timed scope:** routing/alignment, gate/up and SwiGLU (F1). Down projection and
final combination are excluded. This replays the data; the historical kernel
and its harness are not distributed here."""),
            code(r"""f1 = pd.read_csv(DATA / "l4_f1.csv")
assert len(f1) == 9
prev, engine = f1.previous_us / f1.xr38_us, f1.vllm_us / f1.xr38_us
print("faster than preceding revision:", int((prev > 1).sum()), "of", len(f1))
print("faster than vLLM:", int((engine > 1).sum()), "of", len(f1))
y = np.arange(len(f1))
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(y - .18, prev, .34, label="previous / revision", color=C["ours"])
ax.barh(y + .18, engine, .34, label="vLLM / revision", color=C["cyan"])
ax.axvline(1, ls="--", color=INK)
ax.set_yticks(y, [f"{m}, N={n}" for m, n in zip(f1.model, f1.N)])
ax.set_xlabel("speedup (>1 is faster)")
ax.set_title("Nine L4 F1 cases: retain the regressions"); ax.legend()
save(fig, "l4_f1_results"); plt.show()
display(f1[["model", "N", "xr38_us", "previous_us", "vllm_us", "rel_l2"]])
"""),
            md("A uniquely owned address can still be overwritten before its consumer finishes. [WarpSpeed plans](02_warpspeed_plans.ipynb) add resource/phase accounting; [structural search](09_structural_search.ipynb) tests complete implementations."),
        ],
        "09_structural_search": [
            md("""# Fast search, complete construction, measured outcomes

The inner decoder problem has a finite grammar. The outer problem chooses
ownership, compute, transport and lifetime together.

**Evidence:** recorded H200 summaries and complete-program campaign ledgers,
replayed on CPU. The production CUDA enumerator and MCTS engine are not run here.
[Method and scope](../docs/studies.md#structural-search)."""),
            code(setup + r"""
import json
search = json.loads((DATA / "decoder_search.json").read_text())
assert search["cpu_cuda_solution_sets_equal"] and search["legal_graphs"] == 1816
print(f"Exact coverage: {search['raw_combinations']:,} combinations -> {search['legal_graphs']:,} legal graphs")
methods = ["flat CPU", "factored CPU", "warm CUDA (host wall)"]
times = [search[k] for k in ["flat_cpu_ms", "factored_cpu_ms", "warm_gpu_wall_ms"]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].barh(methods[::-1], times[::-1], color=[C["ours"], C["cyan"], C["ref"]])
axes[0].set_xscale("log"); axes[0].set_xlabel("ms, log scale")
axes[0].set_title("Same-machine search methods")
for i, value in enumerate(times[::-1]): axes[0].text(value * 1.08, i, f"{value:.3f}", va="center")
axes[0].set_xlim(.8, 1000)
costs = [search["compile_ms"], search["first_transfer_and_pair_build_ms"], search["warm_gpu_wall_ms"]]
axes[1].bar(["compile", "first transfer /\npair build", "warm search"], costs,
            color=[C["ref"], C["accent"], C["ours"]])
axes[1].set_yscale("log"); axes[1].set_ylabel("ms, log scale")
axes[1].set_title("Setup is outside the warm number")
note(axes[1], "CUDA initialization and allocation are additional.")
save(fig, "structural_search_costs"); plt.show()
"""),
            md("""## Coverage includes failures and distinct implementation contexts

The ledger contains 36 ownership and 40 compute-family entries, including
repeated designs and failures. A missing latency is not zero. Early ownership
measurements retain their original context; a generator repair does not
retroactively change their values."""),
            code(r"""ledger = pd.read_csv(DATA / "structural_search.csv")
assert len(ledger) == 76
display(ledger.groupby(["family", "status"]).size().rename("entries").to_frame())
display(ledger[["family", "batch", "context", "compute", "activation", "status", "ratio"]])
"""),
            md("""## Corrected ownership campaign: values carry uncertainty

These are the recorded intervals for the corrected two-round ownership campaign.
The vertical line is parity with its frozen parent. They are complete six-load
ratios, not official scores."""),
            code(r"""selected = ledger[(ledger.family == "ownership") &
                  ledger.batch.str.startswith("outer_campaign_v2/") &
                  ledger.status.eq("measured")].copy().sort_values("ratio")
assert len(selected) == 12
y = np.arange(len(selected))
fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(selected.ratio, y,
    xerr=np.array([selected.ratio - selected.interval_low, selected.interval_high - selected.ratio]),
    fmt="o", color=C["ours"], ecolor=C["ref"], capsize=3)
ax.axvline(1, color=INK, ls="--")
ax.set_yticks(y, [f"{w} WG / {t} row tiles | {p}" for w,t,p in zip(selected.warpgroups, selected.row_tiles, selected.pipeline)])
ax.set_xlabel("paired six-load speedup (>1 is faster)")
ax.set_title("A complete implementation can be correct and slower")
save(fig, "structural_search_outcomes"); plt.show()
"""),
            md(r"""## Execute the finite-alphabet contracts

Unlike the production GPU search, this scalar constructor is shipped. It checks
every paired FP4/scale encoding and searches exact decompositions
$x=h+2^{-r}l$ with E4M3 components. Coverage is bounded by the exponent window;
it does not establish arbitrary BF16 matrix accumulation equivalence."""),
            code(r"""from tempfile import TemporaryDirectory
sys.path.insert(0, str(ROOT / "examples"))
from finite_alphabet import lut_contract, fp8_contract
with TemporaryDirectory(prefix="finite-alphabet-") as temp:
    lut = lut_contract(Path(temp))
    components = fp8_contract(Path(temp))
assert lut["finite_pair_scale_cases"] == 65024 and lut["nan_pair_scale_cases"] == 512
print("Paired finite/NaN-class cases:", lut["finite_pair_scale_cases"], lut["nan_pair_scale_cases"])
coverage = np.array([[b["exact_of_128"] for b in row["binades"]] for row in components["coverage"]])
exponents = [b["exponent"] for b in components["coverage"][0]["binades"]]
fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(coverage, aspect="auto", vmin=0, vmax=128, cmap="viridis")
ax.set_xticks(range(0, len(exponents), 2), exponents[::2])
ax.set_ylabel("residual shift r"); ax.set_xlabel("BF16 exponent bin")
ax.set_title("Exact FP8-pair sumset coverage per 128 BF16 significands")
fig.colorbar(im, ax=ax, label="exact values out of 128")
save(fig, "finite_alphabet_coverage"); plt.show()
display(pd.DataFrame(components["constructions"])[["residual_shift", "high_method", "scalar_cases", "exponent_window"]])
"""),
            md("""## Copy traces and whole-operator composition

All 168 copy-only patterns passed their byte checks. They omit MMA. The later
operator study has nine candidates/cells × six loads = 54 rows, using direct
launches and stream events. This is a different protocol from graph replay;
the figures below keep the two measurement scopes separate."""),
            code(r"""patterns = pd.read_csv(DATA / "load_patterns.csv")
transport = pd.read_csv(DATA / "load_composition.csv")
assert len(patterns) == 168 and (patterns.bad_threads == 0).all()
assert len(transport) == 54
groups = transport.groupby(["stage", "design"])
assert groups.size().eq(6).all()
whole = groups.agg(ratio=("ratio", lambda x: np.exp(np.log(x).mean())), recorded=("all_load_ratio", "first"))
assert np.allclose(whole.ratio, whole.recorded)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(patterns.geomean_speedup, bins=24, color=C["cyan"])
axes[0].axvline(1, color=INK, ls="--"); axes[0].set_xlabel("copy-only speedup")
axes[0].set_ylabel("patterns"); axes[0].set_title("168 byte-checked load traces")
axes[1].scatter(np.arange(len(whole)), whole.ratio, color=C["ours"])
axes[1].axhline(1, color=INK, ls="--"); axes[1].set_xlabel("candidate / factorial cell")
axes[1].set_ylabel("six-load speedup"); axes[1].set_ylim(.97, 1.02)
axes[1].set_title("Nine complete-operator comparisons")
note(axes[1], "Recorded intervals overlap parity; no demonstrated gain.")
save(fig, "load_pattern_results"); plt.show()
display(whole)
"""),
            md("## A separate deterministic campaign\n\nThis later batch used neither model inference nor MCTS. It compiled 44 complete plans, measured 28, and found no demonstrated gain. Do not combine it with the earlier ledger as one policy evaluation."),
            code(r"""det = json.loads((DATA / "deterministic_search.json").read_text())
assert det["full_operator_candidates"] == det["qualified"] == 28
assert det["demonstrated_gains"] == 0 and not det["target_achieved"]
lo, hi = det["best_private_interval"]
assert lo < 1 < hi
print(f"Best paired ratio: {det['best_private_ratio']:.6f}, interval [{lo:.6f}, {hi:.6f}]")
print("The 2.7x target remains open; fast enumeration is a separate result.")
"""),
        ],
        "10_operator_and_model_feedback": [
            md("""# Complete operators and model-to-native feedback

Forward, backward, saved tensors, decode reuse, gradient accumulation and stream
joins belong in the same executable graph.

**Evidence:** a recorded SM120 checkpoint-linear study and one H200 native model
round. The resource repair is replayed arithmetically on CPU. No model or GPU is
launched. [Operator co-design](../docs/studies.md#complete-operators)."""),
            code(setup + r"""
import json
plans = pd.read_csv(DATA / "lora_operator.csv")
samples = pd.read_csv(DATA / "lora_operator_samples.csv")
medians = samples.groupby("plan")[["gpu_ms", "wall_ms"]].median()
assert len(plans) == 4 and len(samples) == 40
assert np.allclose(medians.loc[plans.plan, "gpu_ms"], plans.gpu_ms)
assert np.allclose(medians.loc[plans.plan, "wall_ms"], plans.wall_ms)
assert (plans.optimizer_updates == 0).all()
display(plans[["plan", "gpu_ms", "wall_ms", "reference_before_ms", "reference_after_ms", "qualified"]])
fig, ax = plt.subplots(figsize=(10, 4))
for i, label in enumerate(plans.plan):
    values = samples[samples.plan == label].gpu_ms.to_numpy()
    ax.scatter(np.full(len(values), i) + np.linspace(-.12, .12, len(values)), values,
               color=C["ours"] if label.startswith("model") else C["ref"], s=24)
    ax.hlines(np.median(values), i-.3, i+.3, color=INK, linewidth=2)
ax.set_xticks(range(len(plans)), plans.plan); ax.set_ylabel("GPU ms")
ax.set_title("Complete forward/backward linear: ten repeats per plan")
note(ax, "M=191, N=21,504, K=5,376, rank=16; fresh inputs, zero B timed. No optimizer updates.")
save(fig, "lora_operator_results"); plt.show()
reference = plans.set_index("plan").loc["deterministic_1", "gpu_ms"]
candidate = plans.set_index("plan").loc["model_0", "gpu_ms"]
print(f"Latency reduction vs deterministic 1 in this run: {100 * (1-candidate/reference):.2f}%")
"""),
            md("""The model plan retained decoded W and used supported CuTe epilogues; large
GEMMs remained library calls. Decoded W alone occupied 220.5 MiB. This four-plan
study does not establish whole-model fit or policy superiority.

## A resource counterexample the model could repair

The native plan requested two CTAs per SM with 174,848 shared bytes per CTA plus
1,024 auxiliary bytes. The recorded diagnostic rejected two CTAs, and the model
reduced residency to one. Recompute that diagnostic; native correctness comes
from the separate measured record, not this arithmetic."""),
            code(r"""per_cta, sm_capacity = 174848 + 1024, 233472
for ctas in [2, 1]:
    required = ctas * per_cta
    print(f"{ctas} CTA(s): {required:,} / {sm_capacity:,} bytes ->", "fits" if required <= sm_capacity else "reject")
assert 2 * per_cta > sm_capacity >= per_cta
"""),
            md("## Execution succeeded; performance regressed\n\nAll six recorded loads passed numerical checks. The complete operator was slower than its parent. The intervals and A/A control remain attached to that outcome."),
            code(r"""loads = pd.read_csv(DATA / "model_tool_feedback.csv")
summary = json.loads((DATA / "model_tool_feedback.json").read_text())
assert len(loads) == 6 and (loads.numerical_failures == 0).all()
assert np.isclose(np.exp(np.log(loads.ratio).mean()), summary["geomean_speedup"])
fig, ax = plt.subplots(figsize=(9, 4))
ax.errorbar(loads.load, loads.ratio,
    yerr=np.array([loads.ratio-loads.interval_low, loads.interval_high-loads.ratio]),
    fmt="o", color=C["ours"], capsize=4)
ax.axhline(1, color=INK, ls="--"); ax.set_ylabel("parent / candidate")
ax.set_title("Repaired model plan: all six complete-operator loads")
save(fig, "model_tool_feedback"); plt.show()
print(f"Geometric speedup: {summary['geomean_speedup']:.5f}; interval: {summary['interval']}")
print(f"A/A: {summary['baseline_AA']:.5f}")
display(loads)
"""),
            md("The CUDA-function route was a separate component experiment: unchanged model arithmetic passed after a wrapper repair, but was also slower than its control. The [write-up](../docs/studies.md#model-feedback) distinguishes that replay from the original live trajectory."),
        ],
        "11_arc_transfer": [
            md(r"""# Valid layouts, valid grids, and exact task success

A layout proves where a proposed transform reads and writes. It does not prove
that the transform is the rule hidden in a puzzle. For transpose:

$$R(r,c)=rW+c,\qquad D(r,c)=cH+r.$$

**Evidence:** a synthetic transpose illustration, then 15 recorded conditions on
the same four public puzzles. [Study and contract](../docs/studies.md#arc)."""),
            code(setup + r"""
x = np.array([[1, 2, 3], [4, 5, 6]])
height, width = x.shape
out = np.empty((width, height), dtype=int)
written = []
for row in range(height):
    for col in range(width):
        read_address, write_address = row * width + col, col * height + row
        out.flat[write_address] = x.flat[read_address]
        written.append(write_address)
assert sorted(written) == list(range(x.size)) and np.array_equal(out, x.T)
reflection = x[:, ::-1]
assert out.shape != reflection.shape
print("Transpose coverage passes; this does not select transpose over reflection.")
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, grid, title in zip(axes, [x, out, reflection], ["input", "transpose", "reflection"]):
    ax.imshow(grid, vmin=0, vmax=6, cmap="viridis")
    for (r,c), value in np.ndenumerate(grid): ax.text(c, r, str(value), ha="center", va="center", color="white")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
save(fig, "arc_layout_example"); plt.show()
"""),
            md("""## Keep the denominator visible

Count exact tasks separately from rows tagged `valid_grid_predictions`. Other
conditions return programs or post-hoc replays, so this validity count is not a
measure of all successful tool executions. Budgets differ across conditions;
the comparison is descriptive, not causal. Sixty rows are still four puzzles."""),
            code(r"""arc = pd.read_csv(DATA / "arc_transfer.csv")
assert arc.task.nunique() == 4 and arc.condition.nunique() == 15 and len(arc) == 60
assert not arc.duplicated(["condition", "task"]).any()
arc["valid_grid_row"] = arc.status.eq("valid_grid_predictions").astype(int)
summary = arc.groupby("condition").agg(tasks=("task", "nunique"),
    valid_grid_rows=("valid_grid_row", "sum"), exact_top1=("pass1", "sum"), exact_top2=("pass2", "sum"))
assert (summary.tasks == 4).all() and (summary.exact_top2 == 0).all()
display(summary)
fig, ax = plt.subplots(figsize=(11, 7))
y = np.arange(len(summary))
ax.barh(y, summary.valid_grid_rows, color=C["cyan"], alpha=.65, label="rows tagged valid grid predictions")
ax.scatter(summary.exact_top2, y, color=C["bad"], zorder=3, label="exact top-2 tasks")
ax.set_yticks(y, [s.replace("_", " ") for s in summary.index])
ax.set_xticks(range(5)); ax.set_xlim(-.15, 4.4)
ax.set_xlabel("tasks out of four, per condition")
ax.set_title("Formatting success did not become exact query success")
ax.legend(loc="upper left", bbox_to_anchor=(0, -.13))
save(fig, "arc_transfer_results"); plt.show()
"""),
            md("""The reusable boundary is `fit(demonstrations) -> state -> solve(state,
query_input) -> freeze -> score`. Demonstration replay is an admission check;
query labels belong to the scorer. Interrupted attempts and post-hoc replays
retain their own status. A valid map, a successful execution and useful task
performance are separate claims. [Evidence guide](../docs/studies.md#evidence)."""),
        ],
    }
