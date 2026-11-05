# NVFP4 MoE on H200 (`muse_moe_w4a16`)

One routed-experts layer of a large MoE, as a single persistent CUDA kernel: 8 of 256 experts per token, 32 experts
owned per GPU, 4-bit NVFP4 weights, BF16 activations. Judged at **2.407×** over SGLang (geometric mean of six loads,
one H200) [M].

## The computation

For token $t$ with routes $s = 1..8$, expert $e_{t,s}$ and router weight $w_{t,s}$:

$$
h_{t,s} = \operatorname{SiLU}\!\big(x_t W^{g}_{e}\big) \odot \big(x_t W^{u}_{e}\big), \qquad
y_{t,s} = (h_{t,s} W^{d}_{e}) \odot (1 + \gamma), \qquad
\text{out}_t = \sum_s w_{t,s}\, \frac{y_{t,s}}{\operatorname{rms}(y_{t,s})} .
$$

Each weight is a 4-bit E2M1 code $c$ times a per-16-value scale: $W_{ij} = \operatorname{e2m1}(c_{ij})\,\sigma_{i,\lfloor j/16 \rfloor}$.

The kernel runs it in one launch:

1. **Route.** Every CTA builds the same per-expert token lists. Busy experts become *fat* jobs (up to 256 rows,
   streamed through shared memory); the rest become *skinny* jobs (16 rows, warps split K).
2. **Phase A** (gate + up + SwiGLU) and **phase B** (down + norm weight), balanced across CTAs by cost, with a small
   pool of tail units claimed atomically.
3. **Final pass**: the weighted, normalized sum per token, after a grid barrier.

The build is chosen by token count $M$:

| $M$ | Build | Serves |
|---|---|---|
| 1 | one-token program build (grouped HFMA2, below) | decode 1 |
| 2–4, 5–8 | two skinny-only builds | verify 1 (8 tokens) |
| 9–16 | mid build | decode 16 |
| 17–63 | **per-size build** (below) | decode 32 |
| 64+ | large-batch build (warpgroup MMA, stream-K) | verify 16, verify 32 |

## What we changed, and why each one is here

Every change was **selected by measurement** (paired, judge-method timing next to the incumbent), not by a model's
prediction. Changes 1 and 6 were in our starting kernel; 2–5 came out of the search described in the docs.

1. **Sign-folded SwiGLU.** In `prepare()`, the gate/up codes get their sign bits flipped (`w13 ^ 0x88`, a lossless
   XOR on both 4-bit codes in each byte). The kernel now holds $g' = -g$ and $u' = -u$, and
   $\operatorname{SiLU}(g)\,u = \dfrac{g\,u}{1 + e^{-g}} = \dfrac{g'\,u'}{1 + e^{g'}}$, so the exponent needs no negation.
2. **A new weight decoder with a proven packing.** The decode from packed codes to BF16 fields is a linear map over
   GF(2) (see [notebook 01](../../notebooks/01_decoders_over_gf2.ipynb) and the [decoder formulation](../../docs/formulation.md#gf2)).
   We enumerated all 1,816 legal decoder graphs and measured hundreds of them. The shipped decoder is

   ```c
   r0 = ((q << 2) ^ q) & 0x81C081C0;   r1 = (q << 6) & 0x81C081C0;
   r2 = rotl(q, 10)    & 0x81C081C0;   r3 = (rotl(q, 12) ^ q) & 0x81C081C0;
   ```

   It produces the same values as the original decode only after the weights are repacked by $T = D^{-1}P$, applied
   in `prepare()` as a 32-entry XOR table (`_XC_COLS`). $D\,T = P$ holds for all $2^{32}$ words, so the result is
   bit-identical. Unlike a pure bit shuffle, this $T$ mixes bits with XOR, which only an exact GF(2) view finds.
3. **Grouped HFMA2 for one token.** With $M = 1$, an `mma` tile with 8 token columns uses only one. The one-token build
   multiplies on CUDA cores instead: decoded weights and activations as packed BF16 pairs, `__hfma2` accumulation
   within each 16-value scale group, one conversion to FP32 per group, then the group scale:

   $$
   \text{acc} \mathrel{+}= \sigma_g \cdot \operatorname{sum}\Big(\textstyle\sum_{k \in g} \text{w}_{k} \cdot \text{x}_{k}\Big)_{\text{bf16x2}}
   $$

   Alone it gained +0.4% on decode 1, and the new decoder alone +0.9%; together, **+3.3%** (12 seeds × 2) [M]. The two
   only pay off together (see the [interaction figure](../../docs/figures/moe_interaction.png)).
4. **Single-poller FF1 → FF2 handoff.** Phase B waits until phase A has written the rows it reads. Before, lane 0
   of *every* warp spun on the ready counter. Now one thread per CTA spins and the other warps wait on a named
   barrier (`bar.sync 10`).
5. **A per-size build for decode 32.** The measured search bound three knobs of the large-batch build on decode 32:
   gather 4 activation rows per batch instead of 8 (`OKC_GATHB=4`), don't start phase A with the trailing partial
   unit (`OKC_WGSREORD=0`), and flush deferred stream-K work behind stage group 4 instead of 3 (`OKC_WGSFCI=4`).
   The same knobs hurt other loads, so they get their own build for $17 \le M \le 63$; every other load keeps
   identical code. On the judge, decode 32 went from 123.0 µs to 121.4 µs [M].
6. **Diagnostics removed.** The base file's development timing helpers are gone; nothing in the file synchronizes
   with the host or times itself outside `prepare()`.

Every build is regenerated from the base source by small rule functions (`_xc_dq4`, `_xc_handoff`, the program
rewrite), applied in the file's own compile hook, and was checked byte-identical to the design we measured. (This
public copy renames a few internal identifiers and macro names; the generated code is otherwise the same.)

## Results (judge, µs)

| Load | SGLang (last run) | Base + (1, 6) | + (2–4) | + (5), shipped |
|---|---:|---:|---:|---:|
| decode 1 | 76.6 | 25.41 | 24.27 | 24.39 |
| decode 16 | 185.9 | 91.42 | 91.53 | 89.25 |
| decode 32 | 204.6 | 123.75 | 123.03 | 121.39 |
| verify 1 (8 tok) | 149.9 | 62.58 | 63.08 | 67.22 |
| verify 16 (128 tok) | 457.1 | 174.34 | 172.17 | 172.52 |
| verify 32 (256 tok) | 622.5 | 210.28 | 206.21 | 208.25 |
| **Score** | | **2.381×** | **2.406×** | **2.407×** |

Each column is its own judge run, and the judge re-times SGLang in every run. Compare **ratios within a run**, not
microseconds across runs: verify 1 reads 67.22 µs in the last run on identical code because that whole case ran
slower (its SGLang time rose 5% too). Per-run data: [`data/moe_judge.csv`](../../data/moe_judge.csv).

## Running it

The file is a self-contained Open Kernel Challenge submission: `prepare(weights)` once, then
`kernel_fn(data, state)` per call. It needs an H200 (sm_90a), PyTorch and a CUDA toolkit; it compiles its builds
with `torch.utils.cpp_extension.load_inline` inside `prepare()`.

## Credits and license

This kernel is derived from **alygg77**'s submission, which builds on **vjeux**'s
[okc-2026](https://github.com/vjeux/okc-2026), with a helper that follows code posted by **Matsvei**. All are under
the Apache License 2.0; their notices are kept at the top of the file. Our changes are the six listed above.
The whole repository is Apache 2.0 ([LICENSE](../../LICENSE)).
