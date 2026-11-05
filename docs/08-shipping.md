# 8 · From pieces to one file: shipping

*How do measured pieces become one honest submission?*

> **Previously.** Decisions are made. Now they have to ship, exactly as measured.

| At a glance | |
|---|---|
| **Problem** | A design measured in a harness is not the file you submit, and per-load winners conflict. |
| **Idea** | Per-size builds, byte-identical capture of every build, syntax-tree edits, and gates. |
| **What it bought** | Decode 32 went from 123.0 µs to 121.4 µs on the judge while the other loads kept identical code [M]. |
| **Code** | `xcalibur.capture`, `xcalibur.gates` |
| **Notebook** | [04 · The MoE on H200](../notebooks/04_moe_on_h200.ipynb) |

---

## 1. Per-size builds

A kernel picks its compiled build by problem size. So when a change helps one load and hurts another, give the
winning load its own build for its size range. Compile it ahead of time in `prepare()`, never on the timed path. The
other loads keep identical code, so the gain costs them nothing. That's how we kept a winner for the load it helped
and dropped it for the load it hurt [M].

## 2. Byte-identical export (`xcalibur.Capture`)

A design measured inside a search harness is not the file you ship. Stand in for the compiler entry point, record
every build's sources and flags, rebuild the submission, and check each build byte-identical to the measured lowering
(`Capture.same`). Then time the exported file itself (see [chapter 4](04-measurement.md)).

Rule-based export keeps the submission readable. Each source change (a decoder rewrite, an architecture rewrite, a
code move, an extra build) is one small function, applied by the submission's own compile hook. The byte-identity
check then proves that the rules rebuild exactly the measured design.

## 3. Structural edits (`xcalibur.replace_assignment`)

Sometimes a component has to move into a larger file, for example a new MoE kernel into a submission that embeds
it. Replace that one module-level constant through Python's syntax tree, then prove nothing else moved: every other
line is identical, the file compiles, and the embedded component is exactly the one that was measured.

## 4. Gates (`xcalibur.gates`)

- **Rules gate.** The live rules version must equal the version read in full.
- **Lint.** The shipped file contains no host syncs, device-to-host reads, side streams, graph capture, timing code
  or measurement scaffolding in the timed path (one-time setup in `prepare()` is exempt).
- **Run card.** Parent, target, estimate from a measured analog, budget, risk, noise, what it teaches, falsifier;
  a person approves it.

## References

- <a id="okc-2026"></a>**[okc-2026]** vjeux. *okc-2026.* The public base the MoE kernel descends from, via alygg77 (Apache 2.0). [link](https://github.com/vjeux/okc-2026)

Full list: [references.md](references.md).

---

[← 7 · The OODA loop](07-OODA.md) · [Contents](README.md) · [9 · Anvil →](09-anvil.md)
