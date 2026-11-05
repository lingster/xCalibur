"""Build and execute the notebooks (figures are saved to docs/figures as they run).

  python notebooks/build.py            # all
  python notebooks/build.py 02         # one, by prefix
"""
import os
import sys
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from research import notebooks as research_notebooks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = """import sys
from pathlib import Path
ROOT = Path.cwd()
if not (ROOT / "xcalibur").is_dir():
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks")]
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
import xcalibur as xc
from style import C, SEQ, MUTED, INK, DATA, save, note"""

NB = {}

# ---------------------------------------------------------------------------------------------------------------------
NB["01_decoders_over_gf2"] = [
md(r"""# Weight decoders are linear maps over GF(2)

A 4-bit weight format like NVFP4 packs eight E2M1 codes into each 32-bit word. Before the math, a kernel turns those
codes into 16-bit values with a handful of bit operations: masks, shifts, rotates. Every recipe that only *moves and
selects* bits is a **linear map over GF(2)**, the field of single bits where addition is XOR:

$$
D : \mathbb{F}_2^{32} \to \mathbb{F}_2^{32}, \qquad y = D\,w \pmod 2, \qquad y_i = \bigoplus_{j} D_{ij}\, w_j .
$$

That turns a messy design space into linear algebra:

* **equality** of two decoders is equality of matrices, a check over the 32 basis words that covers all $2^{32}$ inputs;
* **composition** is matrix product, so a decoder after a repacking is $D \circ T$;
* **a new decoder comes with its packing.** If $P$ is the reference decode and $D$ is invertible, the repacking that makes
  $D$ produce exactly the same values is

$$
D\,T = P \quad\Longrightarrow\quad T = D^{-1} P .
$$

In CuTe terms the map is `Layout((2,)*32, F2-strides)`: the stride of input bit $j$ is the image of the basis vector
$e_j$, the $j$-th column of $D$."""),
code(SETUP + r"""

MASK, FIELDS = 0x81C081C0, (6, 7, 8, 15, 22, 23, 24, 31)   # bit positions the decode writes in each output word
rotl = lambda q, k: ((q << (k % 32)) | (q >> ((32 - k % 32) % 32))) & 0xFFFFFFFF

def nvfp4_decode(q):
    # One packed word of eight 4-bit codes -> four output words, each holding two codes' sign/exponent/mantissa bits
    return (q & MASK, (q << 4) & MASK, rotl(q, 10) & MASK, ((q << 6) & 0x80C080C0) | (rotl(q, 14) & 0x01000100))

def as_matrix(fn):
    images = [fn(1 << j) for j in range(32)]
    return tuple(sum(((images[j][r] >> b) & 1) << j for j in range(32)) for r in range(4) for b in FIELDS)

P = as_matrix(nvfp4_decode)
bits = lambda rows: np.array([[(r >> j) & 1 for j in range(32)] for r in rows])
print("rank of the reference decode:", xc.rank(P), "of 32")"""),
code(r"""fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.imshow(bits(P), cmap=plt.matplotlib.colors.ListedColormap(["#f4f5f9", C["ours"]]))
ax.set_title("The reference NVFP4 decode as a 32 x 32 bit matrix")
ax.set_xlabel("input bit of the packed word"); ax.set_ylabel("output bit (4 words x 8 field bits)")
ax.set_xticks(range(0, 32, 4)); ax.set_yticks(range(0, 32, 8), ["word 0", "word 1", "word 2", "word 3"])
ax.grid(False)
for y in (7.5, 15.5, 23.5): ax.axhline(y, color="white", lw=2)
save(fig, "gf2_reference_decode"); plt.show()"""),
md(r"""## A new decoder and its proven packing

Take a decoder that reads the packed word rotated by 8 bits first: $D = P \circ R_8$. It produces different values from
the same bits, but $T = D^{-1}P$ repacks the weights in `prepare()` so that $D(T(w)) = P(w)$ for **every** word.
Here $T$ turns out to be a permutation: a pure bit shuffle, exactly the kind of lossless repacking the rules allow."""),
code(r"""D = xc.compose(P, xc.rotation(8))
T = xc.packing_transform(D, P)
assert xc.compose(D, T) == P  # exact equality of the complete linear maps
rng = np.random.default_rng(0)
words = [int(w) for w in rng.integers(0, 2**32, 2000)]
print("D(T(w)) == P(w) on 2,000 random words:", all(xc.apply(D, xc.apply(T, w)) == xc.apply(P, w) for w in words))
print("T is a permutation:", all(bin(r).count("1") == 1 for r in T))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
cmap = plt.matplotlib.colors.ListedColormap(["#f4f5f9", C["ours"]])
for ax, (name, M, col) in zip(axes, [("D: the new decoder", D, C["ours"]), ("T = D⁻¹P: the packing", T, C["accent"]),
                                        ("D·T = P: the proof", xc.compose(D, T), C["good"])]):
    ax.imshow(bits(M), cmap=plt.matplotlib.colors.ListedColormap(["#f4f5f9", col])); ax.set_title(name); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
save(fig, "gf2_packing_proof"); plt.show()"""),
md(r"""## Searching the space exactly

The production grammar builds decoders from **atoms**: partial recipes that each produce some output fields. Two
atoms can be joined only if their rows stay linearly independent, so a pairwise-compatibility bitset prunes the
product before any combination is built:

$$
\text{legal} = \{\, a_1 \cup \dots \cup a_k \;:\; \operatorname{rank}\big(\textstyle\bigcup_i a_i\big) = 32,\ \text{pairwise compatible} \,\}.
$$

On an H200 this exhausted **5,989,005** raw combinations to **1,816** legal decoder graphs in about **1.3 ms** [M].
Each one is a candidate kernel with a proven packing, and the measured search later found one that, combined with a
different dequant arithmetic, gave **+3.3%** on the one-token load (each piece alone: +0.4% and +0.9%)."""),
code(r"""fig, ax = plt.subplots(figsize=(7, 3.2))
vals, labels = [5_989_005, 1_816, 1], ["raw combinations", "legal decoder graphs", "the one we shipped"]
ax.barh(labels[::-1], vals[::-1], color=[C["ours"], C["cyan"], C["sglang"]][::-1])
ax.set_xscale("log"); ax.set_title("Exact pruning of the decoder space")
for i, v in enumerate(vals[::-1]): ax.text(v * 1.15, i, f"{v:,}", va="center", color=INK)
ax.set_xlim(0.5, 6e7); note(ax, "Exhausted in about 1.3 ms on one H200 [M].")
save(fig, "gf2_search_space"); plt.show()"""),
md(r"""## The CuTe view

The same map as a CuTe layout: shape $(2,)^{32}$, strides in $\mathbb{F}_2^{32}$. The stride of input bit $j$ is the
column $D e_j$. Evaluating the layout at a coordinate (the bits of $w$) XORs the strides of the set bits, which is
$D w$."""),
code(r"""strides = xc.columns(P)
print("first eight F2 strides (hex):", [hex(s) for s in strides[:8]])
w = 0x12345678
acc = 0
for j in range(32):
    if (w >> j) & 1: acc ^= strides[j]
print("layout evaluation == matrix application:", acc == xc.apply(P, w))
try:
    print(xc.as_cute_layout(P))
except ImportError:
    print("(install NVIDIA's pycute to print the same object as a CuTe Layout)")"""),
]

# ---------------------------------------------------------------------------------------------------------------------
NB["02_warpspeed_plans"] = [
md(r"""# WarpSpeed plans, checked exactly

WarpSpeed (NVIDIA CCCL) describes a warp-specialized kernel as **squads** (contiguous groups of warps with one job)
that hand **staged shared-memory resources** to each other through **phases**. Every quantity a plan implies can be
derived, so a plan can be checked before it is compiled:

$$
\text{arrivals}(\text{phase}) = 32 \sum_{s \in \text{owners}} \text{warps}_s, \qquad
\text{smem} = \sum_{r} \text{stages}_r\, \text{stride}_r \;+\; 8 \sum_{r} \text{phases}_r\, \text{stages}_r ,
$$

$$
\text{register cap} = \min\!\Big(255,\ \Big\lfloor \frac{65536}{128\,\lceil \text{warps}/4 \rceil\, \text{CTAs per SM}} \Big\rfloor_8 \Big).
$$

Here is the large-batch build of our H200 MoE kernel: four MMA warpgroups fed by a weight-producer warp and an
activation-producer warp through two TMA rings."""),
code(SETUP + r"""
plan = xc.plan_of([("mma", 16), ("w_load", 1), ("x_load", 1)],
                  [("w_ring", 3, 36864, [("w_load", 36864), ("mma", None)]),
                   ("x_ring", 3, 32768, [("x_load", 32768), ("mma", None)])])
r = xc.warpspeed_check(**plan)
print("pass:", r["pass"], "| register cap:", r["registers"]["launch_cap"], "| mbarriers:", r["mbarriers"],
      "| shared memory:", r["smem"]["total_bytes"], "of", r["smem"]["cta_limit"])"""),
code(r"""fig, ax = plt.subplots(figsize=(12, 2.4))
col = {"mma": C["ours"], "w_load": C["accent"], "x_load": C["cyan"]}
for s in r["squads"]:
    for wrp in range(s["warp_begin"], s["warp_end"]):
        ax.add_patch(plt.Rectangle((wrp, 0), 0.92, 0.92, color=col[s["name"]]))
        ax.text(wrp + 0.46, 0.46, f"{wrp % 4}", ha="center", va="center", color="white", fontsize=8)
for g in range(5):
    ax.text(g * 4 + 2, 1.2, f"warpgroup {g}", ha="center", color=MUTED, fontsize=9)
ax.set_xlim(-0.2, 18.2); ax.set_ylim(-0.4, 1.5); ax.axis("off")
for i, (n, cc) in enumerate(col.items()):
    ax.add_patch(plt.Rectangle((i * 3.4, -0.38), 0.3, 0.22, color=cc)); ax.text(i * 3.4 + 0.45, -0.27, n, va="center", fontsize=9)
ax.set_title("Squads in warp order (number = sub-partition)")
save(fig, "ws_squads"); plt.show()"""),
code(r"""fig, ax = plt.subplots(figsize=(11, 3.6))
for row, res in enumerate(r["resources"]):
    y = 1.6 - row * 1.6
    for st in range(res["stages"]):
        x = st * 3.2
        ax.add_patch(plt.Rectangle((x, y), 2.8, 0.9, fc="#eef0ff" if row == 0 else "#e6f6fb", ec=MUTED))
        ax.text(x + 1.4, y + 0.62, f"{res['name']}[{st}]", ha="center", fontsize=9)
        ax.text(x + 1.4, y + 0.25, f"{res['stride_bytes'] // 1024} KiB", ha="center", fontsize=8, color=MUTED)
    p0, p1 = res["phases"]
    ax.annotate("", xy=(9.9, y + 0.65), xytext=(9.2, y + 0.65), arrowprops=dict(arrowstyle="->", color=INK))
    ax.text(10.1, y + 0.62, f"phase 0: {p0['owners'][0]} fills ({p0['arrival_count']} arrivals, TMA {p0.get('tx_bytes', 0) // 1024} KiB)", fontsize=9)
    ax.text(10.1, y + 0.22, f"phase 1: {p1['owners'][0]} consumes ({p1['arrival_count']} arrivals)", fontsize=9)
ax.set_xlim(-0.3, 21); ax.set_ylim(-0.2, 2.8); ax.axis("off")
ax.set_title("Two rings, two phases each: produce, then consume, around the ring")
save(fig, "ws_rings"); plt.show()"""),
code(r"""fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
ws = np.arange(1, 33)
for cps, cc in ((1, C["ours"]), (2, C["cyan"])):
    axes[0].step(ws, [xc.launch_register_cap(xc.H200, w, cps) for w in ws], where="mid", color=cc, label=f"{cps} CTA per SM")
axes[0].scatter([18], [96], color=C["accent"], zorder=5); axes[0].annotate("this plan: 18 warps → 96", (18, 96), (19.5, 140), arrowprops=dict(arrowstyle="->", color=MUTED))
axes[0].set_xlabel("warps per CTA"); axes[0].set_ylabel("registers per thread at launch"); axes[0].legend()
axes[0].set_title("The launch register cap")
deep = dict(plan); deep["resources"] = [dict(plan["resources"][0], stages=6), plan["resources"][1]]
bad = xc.warpspeed_check(**deep)
for i, (name, res) in enumerate((("3-stage weight ring", r), ("6-stage weight ring", bad))):
    w = res["resources"][0]["bytes"]; x = res["resources"][1]["bytes"]
    axes[1].bar(i, w / 1024, 0.6, color=C["ours"], label="weight ring" if i == 0 else None)
    axes[1].bar(i, x / 1024, 0.6, bottom=w / 1024, color=C["cyan"], label="activation ring" if i == 0 else None)
    axes[1].text(i, (res["smem"]["total_bytes"]) / 1024 + 6, "pass" if res["pass"] else "fail", ha="center",
                 color=C["good"] if res["pass"] else C["bad"], fontweight="bold")
axes[1].axhline(r["smem"]["cta_limit"] / 1024, color=C["bad"], ls="--")
axes[1].text(-0.45, r["smem"]["cta_limit"] / 1024 + 5, "limit: 232,448 B per CTA", color=C["bad"], fontsize=9)
axes[1].set_ylim(0, 360); axes[1].legend(loc="upper left", fontsize=9, bbox_to_anchor=(0, 0.97))
axes[1].set_xticks([0, 1], ["3-stage weight ring", "6-stage weight ring"]); axes[1].set_ylabel("KiB of shared memory")
axes[1].set_title("Shared memory, and a counterexample")
save(fig, "ws_limits"); plt.show()
print("counterexample:", bad["counterexample"]["message"])"""),
]

# ---------------------------------------------------------------------------------------------------------------------
NB["03_measurement_and_noise"] = [
md(r"""# Measurement when the noise is as large as the gains

A paired comparison times candidate $c$ and reference $b$ alternately (ABBA), each replay from a cold L2, and keeps
the per-pair ratios

$$
\rho_i = \frac{t_b^{(i)}}{t_c^{(i)}}, \qquad \hat\rho = \operatorname{median}_i \rho_i, \qquad
\text{CI}_{95} = \text{bootstrap percentiles of } \hat\rho .
$$

Late in an optimization, real gains are 0.5–2% per load, about the same as run-to-run noise. Three consequences follow,
and this notebook shows each one in our own data."""),
code(SETUP + r"""
noise = pd.read_csv(DATA / "noise.csv")
fig, ax = plt.subplots(figsize=(9, 3.4))
groups = [("identical_d16_build_confirm_12x2", "one new 16-token build,\nfour 12×2 confirms", C["cyan"], "truth: one number"),
          ("identical_d1_build_screen_4x1", "reference's own one-token\nbuild, 4-seed screens", C["accent"], "truth: 0"),
          ("aa_six_load", "reference vs itself (A/A),\nsix-load score", C["good"], "truth: 0")]
for i, (k, lab, cc, truth) in enumerate(groups):
    v = (noise[noise.kind == k].ratio.values - 1) * 100
    ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(v)), s=70, color=cc, zorder=3)
    ax.plot([v.min(), v.max()], [i - 0.22, i - 0.22], color=cc, lw=1.5)
    ax.text(v.max() + 0.08, i - 0.22, f"spread {v.max() - v.min():.1f} pts · {truth}", va="center", fontsize=8.5, color=MUTED)
ax.axvline(0, color=MUTED, lw=1); ax.set_yticks(range(3), [g[1] for g in groups]); ax.set_xlabel("measured change vs the reference (%)")
ax.set_xlim(-1.8, 2.9); ax.set_title("Measuring the same code again")
note(ax, "Each row is one piece of code measured several times. Any spread inside a row is noise [M].")
save(fig, "noise_identical_builds"); plt.show()"""),
md(r"""## Screens nominate, confirms decide

Every design in the 514-design sweep was screened with 4 seeds; 30 finalists were re-timed at 12 seeds × 2 repeats
next to the control. For each load, take the 8 finalists with the best screen reading and measure them again: their
average gain shrinks, and on three loads it changes sign. That is **regression to the mean**: part of every screen
leader's lead was luck."""),
code(r"""sw = pd.read_csv(DATA / "moe_sweep.csv")
key, LOADS = ["compute", "activation", "decoder"], ["d1", "d16", "d32", "v1", "v16", "v32"]
scr, con = sw[sw.grade == "screen_4x1"], sw[sw.grade == "confirm_12x2"]
cs = scr[scr.group == "control"].iloc[0]; cc_ = con[con.group == "control"].iloc[0]
m = scr.merge(con, on=key, suffixes=("_s", "_c")); m = m[m.group_s != "control"]
fig, ax = plt.subplots(figsize=(10, 4))
for i, L in enumerate(LOADS):
    x = (m[f"{L}_s"] / cs[L] - 1) * 100; y = (m[f"{L}_c"] / cc_[L] - 1) * 100
    top = x.nlargest(8).index
    for a, b in zip(x[top], y[top]): ax.plot([i - 0.15, i + 0.15], [a, b], color="#d9dce5", lw=1, zorder=1)
    ax.scatter(np.full(8, i - 0.15), x[top], s=14, color=C["accent"], alpha=0.5, zorder=2)
    ax.scatter(np.full(8, i + 0.15), y[top], s=14, color=C["cyan"], alpha=0.5, zorder=2)
    ax.annotate("", xy=(i + 0.15, y[top].mean()), xytext=(i - 0.15, x[top].mean()),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=4)
    ax.scatter([i - 0.15], [x[top].mean()], s=80, color=C["accent"], zorder=5, label="screen (4 seeds)" if i == 0 else None)
    ax.scatter([i + 0.15], [y[top].mean()], s=80, color=C["cyan"], zorder=5, label="confirm (12 × 2)" if i == 0 else None)
ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(range(6), LOADS); ax.set_ylim(-3.2, 2.6)
ax.set_ylabel("gain vs the control (%)"); ax.legend(loc="lower right", ncol=2)
ax.set_title("The 8 best screen readings per load, measured again")
note(ax, "Big dots: the mean of the 8. Small dots: each design. On the one-token load even the best 8 read below the control.")
save(fig, "noise_screen_vs_confirm"); plt.show()"""),
md(r"""## The winner's curse

Pick the best of $n$ designs that are all truly equal and measured with noise $\sigma$, and the winner reads high by
about

$$
\mathbb{E}\Big[\max_{i \le n} \varepsilon_i\Big] \approx \sigma \sqrt{2 \ln n}.
$$

With a 1% screen and 30 candidates, that is a 2.6% "gain" that does not exist. Replication, and a binding rule that
needs two samples above a margin with none negative, is what removes it."""),
code(r"""rng = np.random.default_rng(1)
ns = np.array([2, 5, 10, 30, 100, 300, 1000])
fig, ax = plt.subplots(figsize=(8, 3.8))
for sigma, lab, cc in ((1.0, "4-seed screen (σ ≈ 1%)", C["accent"]), (0.4, "12×2 confirm (σ ≈ 0.4%)", C["cyan"])):
    sim = [np.mean(np.max(rng.normal(0, sigma, (4000, n)), axis=1)) for n in ns]
    ax.plot(ns, sim, "o-", color=cc, label=lab)
    ax.plot(ns, sigma * np.sqrt(2 * np.log(ns)), ":", color=cc)
ax.set_xscale("log"); ax.set_xlabel("designs compared (all truly equal)"); ax.set_ylabel("best measured 'gain' (%)")
ax.set_title("The best of many noisy measurements reads high"); ax.legend()
note(ax, "Solid: simulation. Dotted: σ√(2 ln n).")
save(fig, "noise_winners_curse"); plt.show()"""),
]

# ---------------------------------------------------------------------------------------------------------------------
NB["04_moe_on_h200"] = [
md(r"""# The MoE kernel on H200

The challenge times one routed-experts layer (8 of 256 experts per token, 32 owned per GPU, NVFP4 weights) at six
token counts, and scores the geometric mean of SGLang's time over ours:

$$
S = \exp\!\Big(\frac{1}{6}\sum_{l=1}^{6} \ln \frac{T^{\text{SGLang}}_l}{T_l}\Big).
$$

Each load counts equally, so a 1% gain on any one load moves the score by about 0.17%."""),
code(SETUP + r"""
NAMES = ["decode 1", "decode 16", "decode 32", "verify 1\n(8 tokens)", "verify 16\n(128 tokens)", "verify 32\n(256 tokens)"]
j = pd.read_csv(DATA / "moe_judge.csv"); j["speedup"] = j.sglang_us / j.ours_us
runs, loads = list(dict.fromkeys(j.run)), list(dict.fromkeys(j.load))
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.42})
x, wd = np.arange(len(loads)), 0.26
for k, (run, cc) in enumerate(zip(runs, (C["ref"], C["ours"], C["cyan"]))):
    s = j[j.run == run].set_index("load").loc[loads, "speedup"]
    axes[0].bar(x + (k - 1) * wd, s, wd, color=cc, label=f"{run}: {np.exp(np.log(s).mean()):.3f}×")
axes[0].set_xticks(x, NAMES, fontsize=9); axes[0].set_ylabel("speedup vs SGLang"); axes[0].legend(loc="upper center", ncol=3, fontsize=9)
axes[0].set_ylim(0, 3.9); axes[0].set_title("Speedup per load on the judge (legend: six-load geomean)")
last = j[j.run == runs[-1]].set_index("load").loc[loads]
axes[1].barh(range(6)[::-1], last.sol_us / last.ours_us * 100, color=C["ours"])
axes[1].set_yticks(range(6)[::-1], NAMES, fontsize=8.5); axes[1].set_xlim(0, 100)
axes[1].set_xlabel("% of speed of light"); axes[1].set_title("Distance to the bound")
for i, v in enumerate((last.sol_us / last.ours_us * 100).values[::-1]): axes[1].text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=9)
save(fig, "moe_judge_speedups"); plt.show()
print(j.groupby("run", sort=False).speedup.apply(lambda s: round(float(np.exp(np.log(s).mean())), 4)).to_dict())"""),
md(r"""## Pieces that only work together

Two changes were tested alone and together at 12 seeds × 2 repeats: a grouped HFMA2 dequant for the one-token build,
and a different decoder graph (§01). Alone they gave little. Together they gave **+3.3%** on the one-token load: an
interaction no single-change search would find, which is why the measured search recombines pieces on purpose."""),
code(r"""ia = pd.read_csv(DATA / "moe_interaction.csv")
DEC = {"parent": "reference decoder", "1683507725:1": "new decoder A", "1246039045:0": "new decoder B\n(shipped)"}
ARITH = {"dense": "reference dequant", "hfma2_group_d1": "grouped HFMA2\ndequant (shipped)"}
ia["dec"], ia["arith"] = ia.decoder.map(DEC), ia.compute.map(ARITH)
ref = {c: 1.0 for c in ia.columns if c not in ("compute", "decoder", "dec", "arith")}
ia = pd.concat([pd.DataFrame([{**ref, "dec": DEC["parent"], "arith": ARITH["dense"]}]), ia], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, col, title in ((axes[0], "d1", "One-token decode"), (axes[1], "six_load", "Six-load score")):
    pv = ia.pivot_table(index="arith", columns="dec", values=col, aggfunc="first").loc[list(ARITH.values()), list(DEC.values())]
    g = (pv.values - 1) * 100
    ax.imshow(g, cmap="Purples", vmin=-0.5, vmax=g.max() * 1.05)
    ax.set_xticks(range(3), pv.columns, fontsize=9); ax.set_yticks(range(2), pv.index); ax.grid(False)
    for (a, b), v in np.ndenumerate(g):
        txt = "starting\nkernel" if (a, b) == (0, 0) else f"{v:+.1f}%"
        ax.text(b, a, txt, ha="center", va="center", fontsize=10.5 if (a, b) != (0, 0) else 8.5,
                color="white" if v > 0.55 * g.max() else INK, fontweight="bold" if (a, b) == (1, 2) else None)
    ax.set_title(f"{title}, vs the starting kernel")
save(fig, "moe_interaction"); plt.show()"""),
md(r"""## The landscape around the design

A sweep of 514 designs around the shipped design (new decoders, and compute × activation for the one-token build)
says it sits near a local optimum: it ranks second of 514 on the six-load screen, and first of the 31 designs re-timed
at 12 × 2, tied within noise with the next two (1.0147 vs 1.0146 and 1.0144 over the starting kernel) [M]."""),
code(r"""sw = pd.read_csv(DATA / "moe_sweep.csv"); scr = sw[sw.grade == "screen_4x1"]
ctl = scr[scr.group == "control"].iloc[0]
L = ["d1", "d16", "d32", "v1", "v16", "v32"]
fig, ax = plt.subplots(figsize=(11, 4))
for i, l in enumerate(L):
    v = (scr[scr.group != "control"][l] - 1) * 100
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.28, 0.28, len(v)), v, s=6, alpha=0.35, color=C["ours"])
    ax.scatter([i], [(ctl[l] - 1) * 100], marker="D", s=70, color=C["accent"], zorder=5)
ax.set_xticks(range(6), NAMES, fontsize=9)
ax.set_ylabel("vs the starting kernel (%)"); ax.set_ylim(-25, 8)
ax.set_title("514 neighbours (dots) and the shipped design (diamond), 4-seed screen")
save(fig, "moe_landscape"); plt.show()"""),
]

# ---------------------------------------------------------------------------------------------------------------------
NB["05_the_measured_game"] = [
md(r"""# The measured game

A learned model chooses **what to measure next**; the GPU decides **what is faster**. Per load $l$, a bootstrap
ensemble of $K$ LightGBM models gives a mean and a spread for every unmeasured design $d$, and the frontier is ranked
by an upper confidence bound

$$
\text{UCB}_l(d) = \mu_l(d) + \kappa\, \sigma_l(d) - J_l(\text{incumbent}), \qquad
\mu_l = \tfrac1K \textstyle\sum_k f_k(d),\ \ \sigma_l = \operatorname{sd}_k f_k(d).
$$

A design **binds** for load $l$ when its build identity has at least two confirm-grade samples $g_1, \dots, g_n$
next to the incumbent with

$$
\bar g > m, \qquad \min_i g_i > 0 .
$$

This notebook runs the loop on a synthetic problem with one planted 2% gain
(decoder 11 on the 16-token load) and many look-alikes. The configuration below
requires three confirmation rounds. A design nominated for multiple loads is
measured once, and each load/build identity contributes at most one observation
per round. The trace and figure show when the rule is met. This repeat-and-margin
rule is a heuristic, not a confidence-interval test; selection can still bias the
reported gain above the planted truth."""),
code(SETUP + r"""
import itertools, math, random
LOADS = ("m1", "m16", "m32")
SPACE = {"cutoff": [8, 17], "tile": [64, 128, 256], "stages": [2, 3, 4], "unroll": [1, 2, 4], "decoder": list(range(16))}
INC = {"cutoff": 17, "tile": 128, "stages": 3, "unroll": 2, "decoder": 0}
REACH = xc.ReachMap(loads={"m1": 1, "m16": 16, "m32": 32}, cutoff_field="cutoff",
                    fields={"large": ("tile", "stages"), "small": ("unroll",), "all": ("decoder",)})
rng = random.Random(0)
def truth(d, l):
    g = 1.0
    if l == "m16" and d["decoder"] == 11: g *= 1.02
    if l == "m32" and REACH.build_of(l, d) == "large": g *= 1 + 0.003 * (d["stages"] - 3) - 0.004 * abs(math.log2(d["tile"] / 128))
    if l == "m1": g *= 1 - 0.01 * abs(d["unroll"] - 2)
    return g
measure = lambda ds, grade: [{l: truth(d, l) * math.exp(rng.gauss(0, 0.010 if grade == "screen" else 0.003)) for l in LOADS} for d in ds]
featurize = lambda d, c: [d["cutoff"], d["tile"], d["stages"], d["unroll"]] + xc.bits(d["decoder"], 4)
allD = [dict(zip(SPACE, v)) for v in itertools.product(*SPACE.values())]
frontier = lambda h: [d for d in allD if repr(sorted(d.items())) not in {repr(sorted(x["design"].items())) for x in h}]
game = xc.Game(loads=LOADS, reach=REACH, incumbent=INC, frontier=frontier, featurize=featurize,
               compile=lambda ds: [{"ok": True} for _ in ds], measure=measure, per_load=6, compile_n=18, screen_n=12,
               margin=0.008, min_samples=3)
seed = [INC] + rng.sample(allD, 40)
for d, l in zip(seed, measure(seed, "screen")): game.history.append({"design": d, "loads": l, "weight": 1.0})
trace = []
for rnd in range(1, 9):
    rep = game.round(); trace.append(rep); print(rnd, rep)"""),
code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.5})
axes[0].plot(range(1, 9), [t["bound"].get("m16", 0) for t in trace], "o-", color=C["ours"], label="bound gain on m16")
axes[0].axhline(2.0, color=C["good"], ls="--", label="planted truth: +2%"); axes[0].set_ylim(-0.2, 3)
axes[0].set_xlabel("round"); axes[0].set_ylabel("% over the incumbent"); axes[0].legend(); axes[0].set_title("Binding finds the real gain")
ev = game.evidence.rows.get("m16", {})
real = sorted([k for k in ev if ev[k]["design"]["decoder"] == 11], key=lambda k: -len(ev[k]["gains"]))[:4]
fake = sorted([k for k in ev if ev[k]["design"]["decoder"] != 11], key=lambda k: -max(ev[k]["gains"]))[:4]
rows = [(k, "look-alike") for k in fake[::-1]] + [(k, "carries the +2%") for k in real[::-1]]
for i, (k, kind) in enumerate(rows):
    g = np.array(ev[k]["gains"]) * 100
    ok = len(g) >= 3 and g.mean() > 0.8 and g.min() > 0
    axes[1].scatter(g, np.full(len(g), i), color=C["good"] if ok else C["ref"], s=42, zorder=3)
    axes[1].scatter([g.mean()], [i], marker="|", s=380, color=INK, zorder=4)
axes[1].axvline(0.8, color=C["accent"], ls="--"); axes[1].axvline(0, color=MUTED, lw=1)
axes[1].set_yticks(range(len(rows)), [f"{kind}, build {i % 4 + 1}" for i, (_, kind) in enumerate(rows)], fontsize=9)
axes[1].set_xlabel("confirm-grade gain on m16 (%)"); axes[1].set_title("Pooled evidence: binds (green) or not (grey)")
note(axes[1], "Dash: margin 0.8%. Bar: mean. A build binds with ≥ 3 samples, mean above the margin, none negative.")
save(fig, "game_binding"); plt.show()"""),
]

# ---------------------------------------------------------------------------------------------------------------------
NB["06_tg4dac"] = [
md(r"""# TG4DAC: language models planning WarpSpeed kernels

On an RTX PRO 6000 Blackwell, a Gemma 4 31B **planner** proposed what to change, a Gemma 4 31B **refiner** turned the
proposal into a WarpSpeed plan checked by `warpspeed_check`, and a frontier model implemented the plan as written.
Attention was held out of the refiner's training, so these are results on unseen work. Each design was timed against
vLLM's kernel in the same run [M]."""),
code(SETUP + r"""
tg = pd.read_csv(DATA / "tg4dac_attention.csv")
pre = tg[tg.phase == "prefill"]
fig, ax = plt.subplots(figsize=(10, 3.8))
lab = [f"{l} prefill, {m:,} tok" for l, m in zip(pre.layer, pre.M)]; x = np.arange(len(pre)); wd = 0.36
ax.bar(x - wd / 2, pre.vllm_ms / pre.planner_ms, wd, color=C["ref"], label="planner's idea (vLLM kernel retuned)")
ax.bar(x + wd / 2, pre.vllm_ms / pre.refiner_warpspeed_ms, wd, color=C["ours"], label="refiner's WarpSpeed plan")
for i, v in enumerate(pre.vllm_ms / pre.refiner_warpspeed_ms): ax.text(i + wd / 2, v + 0.08, f"{v:.1f}×", ha="center", fontweight="bold")
ax.set_xticks(x, lab); ax.set_ylabel("speedup vs vLLM"); ax.legend(loc="upper right"); ax.set_title("Structure beats tuning")
save(fig, "tg4dac_prefill"); plt.show()"""),
code(r"""dec = tg[tg.phase == "decode"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.1), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.22})
k = lambda c: f"{c // 1024}k"
sp = dec.vllm_ms / dec.refiner_warpspeed_ms
axes[0].bar(range(len(dec)), sp, color=[C["cyan"] if l == "sliding" else C["ours"] for l in dec.layer])
for i, v in enumerate(sp): axes[0].text(i, v + 0.05, f"{v:.1f}×", ha="center", fontsize=8.5)
axes[0].set_xticks(range(len(dec)), [f"{m}×{k(c)}" for m, c in zip(dec.M, dec.context.astype(int))], fontsize=8.5)
axes[0].axhline(1, color=MUTED, lw=1); axes[0].set_ylim(0, 4.1); axes[0].set_ylabel("speedup vs vLLM")
axes[0].bar([0], [0], color=C["cyan"], label="sliding-window layers"); axes[0].bar([0], [0], color=C["ours"], label="global layers")
axes[0].legend(loc="upper center", ncol=2, fontsize=9); axes[0].set_title("Decode attention kernels")
note(axes[0], "Label: sequences × cached tokens.")
e2e = pd.read_csv(DATA / "tg4dac_e2e.csv")
lab = [f"{s}×{c // 1024 if c >= 1024 else c}{'k' if c >= 1024 else ''}" for s, c in zip(e2e.sequences, e2e.context)]
axes[1].bar(np.arange(len(e2e)) - 0.2, e2e.vllm_tok_s, 0.4, color=C["ref"], label="vLLM (compiled)")
axes[1].bar(np.arange(len(e2e)) + 0.2, e2e.ours_tok_s, 0.4, color=C["ours"], label="with our kernels")
for i, (a, b) in enumerate(zip(e2e.vllm_tok_s, e2e.ours_tok_s)): axes[1].text(i + 0.2, b + 6, f"{b / a:.2f}×", ha="center", fontsize=9)
axes[1].set_xticks(range(len(e2e)), lab); axes[1].set_ylabel("tokens / s"); axes[1].set_ylim(0, 470)
axes[1].legend(loc="upper left", ncol=2, fontsize=9); axes[1].set_title("Whole-model decode in vLLM")
note(axes[1], "Label: sequences × context.")
save(fig, "tg4dac_decode"); plt.show()"""),
md(r"""**Why the end-to-end gain is smaller than the kernel gain.** Every decode step pays about 18 ms that attention
doesn't touch (the weights of the projections, the MLP, the adapters and the output layer). Attention grows with
context, so the speedup grows with context too: 1.06× at short context, 1.30× at 8 × 32k."""),
]


# ---------------------------------------------------------------------------------------------------------------------
NB["07_knowledge_graphs"] = [
md(r"""# Knowledge graphs: what we know, in a form a planner can read

Everything the search produces, and everything it designs against, is kept as a **typed graph with evidence**:

$$
G = (V, E), \qquad \tau_V : V \to \{\text{design}, \text{code move}, \text{submission}, \text{function}, \text{concept}, \text{instruction}\},
\qquad \tau_E : E \to \{\text{one-edit}, \text{touches}, \text{scored as}, \text{uses}, \text{is a}\}.
$$

Every measured node carries its values $J_\ell(v)$ per load $\ell$ **and how it was measured** (a 4-seed screen or a
12 × 2 confirm). Three queries do almost all the work:

1. **Neighbourhood** $N(v)$: what is one edit away from this design, and what did each edit do?
2. **Aggregate effect** of a field change $f: a \to b$, over all one-edit pairs that make it:
   $$\delta_\ell(f{:}\,a \to b) = \operatorname{median}_{(u, w)} \frac{J_\ell(w)}{J_\ell(u)} .$$
3. **Retrieval**: the $k$ nodes closest to a question, $\operatorname{top}_k\, \langle e(q), e(v) \rangle$ with unit
   embeddings $e$, read together with each node's evidence.

The picture below is a slice of the real graph around the shipped MoE design."""),
code(SETUP + r"""
from matplotlib.patches import FancyBboxPatch
TYPES = {"design": C["ours"], "code move": C["accent"], "submission": C["good"], "function": C["cyan"],
         "concept": C["pink"], "instruction": MUTED}
N = {"ship": ("shipped design", "design", 0, 0), "decA": ("neighbour: decoder A", "design", -3.5, 1.1),
     "dense": ("neighbour: reference dequant", "design", -3.4, -1.1), "judge": ("judge run: 2.407×", "submission", 3.7, 1.5),
     "hfma": ("grouped HFMA2", "code move", -0.6, 2.1), "hand": ("single-poller handoff", "code move", 3.4, -0.8),
     "dq4": ("dq4()", "function", -1.3, -2.1), "ring": ("skinny_ring_ws()", "function", 1.2, -2.4),
     "f2": ("layout over F2 (CuTe)", "concept", -3.8, -3.1), "ws": ("SmemResource (WarpSpeed)", "concept", 5.2, -1.6),
     "ptx1": ("fma.rn.bf16x2 (PTX)", "instruction", -3.4, 3.2), "ptx2": ("mbarrier.try_wait (PTX)", "instruction", 4.3, -3.3)}
EDGES = [("ship", "decA", "one-edit: decoder · d1 −1.3%"), ("ship", "dense", "one-edit: dequant · d1 −2.3%"),
         ("ship", "judge", "scored as"), ("ship", "hfma", "includes"), ("ship", "hand", "includes", (1.55, -0.12)), ("ship", "dq4", "touches"),
         ("hand", "ring", "patches"), ("dq4", "f2", "is a"), ("hfma", "ptx1", "uses"), ("ring", "ptx2", "uses"), ("ring", "ws", "modelled by", (3.05, -2.1))]
fig, ax = plt.subplots(figsize=(13, 6.6))
for e in EDGES:
    a, b, lab = e[:3]
    (_, _, xa, ya), (_, _, xb, yb) = N[a], N[b]
    lx, ly = e[3] if len(e) > 3 else ((xa + xb) / 2, (ya + yb) / 2)
    ax.plot([xa, xb], [ya, yb], color="#cfd3dd", lw=1.6, zorder=1)
    ax.text(lx, ly, lab, fontsize=8, color=MUTED, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1.2), zorder=2)
for k, (lab, t, x, y) in N.items():
    w = 0.13 * len(lab) + 0.5
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.28), w, 0.56, boxstyle="round,pad=0.05,rounding_size=0.18",
                                fc=TYPES[t], ec="white", lw=2, zorder=3, alpha=1 if k == "ship" else 0.9))
    ax.text(x, y, lab, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold" if k == "ship" else None, zorder=4)
for i, (t, cc) in enumerate(TYPES.items()):
    ax.add_patch(FancyBboxPatch((-6.4 + i * 2.15, -4.25), 0.3, 0.25, boxstyle="round,pad=0.02", fc=cc, ec="none"))
    ax.text(-6.0 + i * 2.15, -4.12, t, va="center", fontsize=9)
ax.set_xlim(-6.6, 7.2); ax.set_ylim(-4.5, 4.0); ax.axis("off")
ax.set_title("A slice of the knowledge graph around the shipped MoE design")
save(fig, "kg_slice"); plt.show()"""),
md(r"""## The design graph, built from the measured sweep

Each of the 514 screened designs becomes a node with three fields (dequant arithmetic, activation staging, decoder
graph). Edges join designs that differ in exactly one field. Decoder IDs are shortened here; the shipped decoder is
`B`, the one from notebook 04's interaction is `A`."""),
code(r"""sw = pd.read_csv(DATA / "moe_sweep.csv"); scr = sw[sw.grade == "screen_4x1"].reset_index(drop=True)
L = ("d1", "d16", "d32", "v1", "v16", "v32")
short = {"1246039045:0": "B", "1683507725:1": "A"}
for i, d in enumerate(sorted(set(scr.decoder) - set(short))): short[d] = f"d{i:03d}"
g = xc.DesignGraph(loads=L)
for i, r in scr.iterrows():
    g.add(xc.Node(id=f"n{i}", kind="design", fields={"compute": r.compute, "activation": r.activation, "decoder": short[r.decoder]},
                  loads={l: r[l] for l in L}, evidence="screen 4x1"))
edges = g.one_edit()
print(f"{len(g.nodes)} nodes, {len(edges):,} one-edit edges")
print("\nWhat a planner reads (the first lines of g.listing()):\n")
print("\n".join(g.listing().splitlines()[:5]))"""),
md(r"""## Query 2: what each field change does, per load

The median effect of every one-field change away from the shipped values. No one told the graph which build each
field reaches, yet the table shows it: the dequant arithmetic and the activation staging move **decode 1** and leave
the other loads within noise, except `sparse_small`, which also moves decode 16 and verify 1, the small builds it is
compiled into. **The graph rediscovered the reach map from data** (the physical formulation derives the
same map from the code)."""),
code(r"""eff = [e for e in g.effects() if (e["field"], e["from"]) in (("compute", "hfma2_group_d1"), ("activation", "register"))]
eff = sorted(eff, key=lambda e: e["median"]["d1"])
M = np.array([[(e["median"][l] - 1) * 100 for l in L] for e in eff])
fig, ax = plt.subplots(figsize=(10, 4.2))
cm = plt.matplotlib.colors.LinearSegmentedColormap.from_list("div", [C["bad"], "#ffffff", C["good"]])
ax.imshow(np.clip(M, -8, 8), cmap=cm, vmin=-8, vmax=8, aspect="auto")
for (a, b), v in np.ndenumerate(M):
    ax.text(b, a, f"{v:+.1f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 5 else INK)
ax.set_xticks(range(6), ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"])
ax.set_yticks(range(len(eff)), [f"{e['field']}: {e['from']} → {e['to']}  (n={e['pairs']})" for e in eff], fontsize=9)
ax.grid(False); ax.set_title("Median effect of each one-field change (%), from the measured graph")
save(fig, "kg_effects"); plt.show()"""),
md(r"""## Query 1: the shipped design's neighbourhood

All 393 designs one decoder edit away from the shipped design, placed around it by their six-load effect. None reads
more than half a percent above it; 18 sit within noise; the rest are slower. Read from the graph, that is the
evidence that this screened neighbourhood has no resolved improvement. It does not certify a local optimum or replace confirmation."""),
code(r"""ship = next(n for n in g.nodes if n.fields == {"compute": "hfma2_group_d1", "activation": "register", "decoder": "B"})
nb = []
for a, b, k, x, y, r in edges:
    if k == "decoder" and ship.id in (a, b):
        v = np.exp(np.mean(np.log(list(r.values()))))
        nb.append((v if a == ship.id else 1 / v) - 1)
nb = np.sort(np.array(nb) * 100)
th = np.linspace(0, 2 * np.pi, len(nb), endpoint=False)
fig = plt.figure(figsize=(6.6, 6.6)); ax = fig.add_subplot(projection="polar")
r = 12 + nb
col = np.where(nb > 0.5, C["good"], np.where(nb >= -0.5, C["accent"], C["ref"]))
for t, rr, c in zip(th, r, col): ax.plot([t, t], [0, rr], color=c, lw=0.6, alpha=0.6)
ax.scatter(th, r, s=10, c=col, zorder=3)
ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, 12), color=C["ours"], lw=1.5)
ax.scatter([0], [0], s=160, color=C["ours"], zorder=5)
ax.set_rticks([]); ax.set_xticks([]); ax.set_ylim(0, 14.5); ax.spines["polar"].set_visible(False)
ax.set_title(f"393 decoder neighbours: {int((nb > 0.5).sum())} faster, {int((abs(nb) <= 0.5).sum())} within ±0.5%, "
             f"{int((nb < -0.5).sum())} slower", fontsize=11)
fig.text(0.5, 0.04, "Centre: the shipped design. Ring: equal to it. Each spoke: one decoder edit, length = six-load effect.",
         ha="center", fontsize=8.5, color=MUTED)
save(fig, "kg_neighbourhood"); plt.show()"""),
md(r"""## Query 3: retrieval, and why evidence must ride along

Retrieval ranks nodes by similarity to a question. Similarity is not quality: the nearest nodes can be noisy screens
or designs for another build. So a context is built from **the compact listing** (the whole graph, one line per node),
**the top-$k$ nodes in full**, and **each node's evidence grade**, and exact checks still decide legality.

The embedding below is a hashed character-trigram stand-in so the notebook runs offline; in production we used
EmbeddingGemma with a code-retrieval prompt for queries (`xcalibur.embedding_gemma`)."""),
code(r"""import hashlib
def embed(texts, query=False, dim=512):
    out = []
    for t in texts:
        v = np.zeros(dim); t = f"  {t.lower()}  "
        for i in range(len(t) - 2): v[int(hashlib.md5(t[i:i + 3].encode()).hexdigest(), 16) % dim] += 1
        out.append(list(v / (np.linalg.norm(v) or 1)))
    return out
for n in g.nodes:
    n.text = (f"dequant {n.fields['compute']} activation {n.fields['activation']} decoder {n.fields['decoder']}; "
              f"one-token decode {100 * (n.loads['d1'] - 1):+.1f}%; measured as {n.evidence}")
ret = xc.Retriever(g.nodes, embed)
q = "faster one-token decode from a grouped hfma2 dequant with register activation"
rows = [{"node": n.id, "similarity": round(s, 3), **n.fields, "decode 1 (%)": round(100 * (n.loads["d1"] - 1), 1), "evidence": n.evidence}
        for n, s in ret.query(q, k=8)]
pd.DataFrame(rows)"""),
]


NB.update(research_notebooks(SETUP))

EVIDENCE = {
  "01": "CPU linear-map proof with recorded production-search counts; the CUDA enumerator is not run here.",
  "02": "CPU resource accounting; passing does not prove native synchronization or actual compiler allocation.",
  "03": "Recorded timing observations plus a statistical illustration; no hardware timing is performed here.",
  "04": "Archived judge and paired-study data; scores are historical snapshots, not live standings.",
  "05": "Synthetic seeded objective exercising the public search loop; no measured GPU speedup.",
  "06": "Recorded SM120 attention and end-to-end studies; operator latency and model throughput have different scopes.",
  "07": "Recorded design observations with offline hashed retrieval; neighbourhood screens do not certify an optimum.",
}


def build(only=None):
  if only and not any(name.startswith(only) for name in NB):
    raise SystemExit(f"No notebook matches prefix {only!r}")
  os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
  for name, cells in NB.items():
    if only and not name.startswith(only):
      continue
    cells = list(cells)
    if name[:2] in EVIDENCE:
      cells.insert(1, md("**Evidence and execution:** " + EVIDENCE[name[:2]]))
    cells.append(md("[Notebook guide](README.md) · [Documentation](../docs/README.md#reading-order) · [Data provenance](../data/README.md)"))
    for i, cell in enumerate(cells):
      cell["id"] = hashlib.sha256(f"{name}:{i}".encode()).hexdigest()[:12]
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
      NotebookClient(nb, timeout=900, startup_timeout=30, km=km, record_timing=False,
                     resources={"metadata": {"path": str(HERE)}}).execute()
    finally:
      if km.has_kernel:
        km.shutdown_kernel(now=True)
      km.cleanup_resources()
    nbf.write(nb, HERE / f"{name}.ipynb")
    print("built", name, flush=True)


if __name__ == "__main__":
  with TemporaryDirectory(prefix="xcalibur-notebooks-") as runtime:
    os.environ["IPYTHONDIR"] = str(Path(runtime) / "ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(Path(runtime) / "jupyter")
    build(sys.argv[1] if len(sys.argv) > 1 else None)
