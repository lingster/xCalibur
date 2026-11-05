# TG4DAC × WarpSpeed: teaching a model to plan GPU kernels

October 24, 2026 · one NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, 188 SMs) on Colab · Gemma 4 31B

We trained a 31B Gemma model, called the **refiner**, to turn a rough kernel plan into a **WarpSpeed** design:
- which warps load data and which compute;
- how shared-memory buffers are staged and handed between them;
- how many registers each group gets.

A typed checker lets the model test each design against the GPU's real limits before it answers.

We then gave the system a problem it had never seen: Gemma 4 attention, starting from vLLM's own kernels. Built as written, the refiner's designs were the fastest kernels we measured:
- **4.6×** faster than vLLM for sliding-window prefill;
- **3.1×** for global-layer prefill;
- up to **3.6×** for decode.

Inside the full model in vLLM, the new attention kernels together with the user's `gemv8` MLP kernel decode up to **1.30×** faster than vLLM's fastest configuration (8 sequences × 32k tokens of context).

## How it works

Three roles work in sequence:

1. **Planner.** Gemma 4 31B with the `cake_scoped_c1` adapter from earlier work. It reads the starting kernel (source code, timings, compiler data) and proposes what to change. It can run layout-algebra checks (PyCuTe) and arithmetic tools while it thinks.
2. **Refiner.** Gemma 4 31B with a new adapter trained in this project. It turns the planner's proposal into a WarpSpeed plan:
   - squads of warps;
   - staged shared-memory buffers with phases;
   - barrier counts;
   - register budgets.

   It checks every plan with `warpspeed_check` and fixes what fails.
3. **Implementer (Claude).** Writes the kernels from the plans and measures everything on the same GPU, against the starting kernel, in the same run.

**WarpSpeed** is the warp-specialization library inside NVIDIA CCCL (`cub/detail/warpspeed`). A kernel is split into squads (groups of warps). The squads pass shared-memory buffers to each other through a ring of stages, and each stage moves through phases guarded by barriers.

**`warpspeed_check`** is a small tool we wrote. It follows CCCL's rules: at most 10 resources, 2–4 phases each, and barrier counts equal to the number of threads in the owning squads. It also enforces three facts we measured on this GPU:
- The register cap per thread is 65,536 / (128 × ⌈warps/4⌉). So a 9-warp kernel is capped at 168 registers, and a 7-warp kernel at 255.
- `setmaxnreg` can raise a squad's registers only if every warp in a warpgroup asks for the same value, and the total fits the block's budget.
- Shared memory is limited to 99 KB per block and 100 KB per SM.

The checker returns either a pass with every derived number, or the first violation as a counterexample.

**Training data:**
- 30 worked examples (29 for training, 1 held out), each with real tool runs, from 6 source families:
  - CCCL's own WarpSpeed scan kernel;
  - the user's MLP kernels (the warp-specialized GEMM, `gemv8`, and the dequant-ahead variant);
  - FlashInfer's warp-specialized MLP design;
  - small labeled drills.
- **Attention was deliberately left out, so the attention results are a test on unseen work.**

**Training run:**
- same recipe as the earlier `cake_scoped_c1` run: 2 passes over the data, 8 optimizer updates, about 7 minutes of GPU time;
- 92,236 supervised tokens;
- loss on the held-out example fell from 1.068 to 0.806.

## Results

Each design was timed against vLLM's kernel in the same run on the same GPU. Times are per layer. "SOL" is the speed limit of the hardware for that job. Numerical error matches vLLM's.

**Prefill / training: one causal sequence of M tokens**

| Layer | M | vLLM | Planner's best idea (vLLM kernel retuned) | Refiner's WarpSpeed plan (CUDA) |
|---|---|---|---|---|
| sliding (50 layers) | 7,131 | 4.51 ms | 1.80 ms (2.4×) | **0.99 ms (4.5×)** |
| sliding | 15,329 | 9.99 ms | 3.93 ms (2.4×) | **2.17 ms (4.6×)** |
| global (10 layers) | 7,131 | 23.3 ms | 14.7 ms (1.5×) | **7.95 ms (2.9×)** |
| global | 15,329 | 109.3 ms | 67.1 ms (1.5×) | **35.6 ms (3.1×)** |

The WarpSpeed prefill kernels run at 210–229 TFLOPS, about half of this GPU's measured matrix-unit peak (437 TFLOPS). vLLM's kernel runs at 52–77 TFLOPS.

**Decode: M sequences, one new token each, reading L cached tokens**

| Layer | M | L | vLLM | Planner's idea (Triton) | Refiner's WarpSpeed plan (CUDA) |
|---|---|---|---|---|---|
| sliding | 1 | 4k | 23.9 µs | 20.7 µs | **17.2 µs (1.4×)** |
| sliding | 1 | 32k–128k | 62.3 µs | 20.7 µs | **17.2 µs (3.6×)** |
| sliding | 8 | 4k–128k | 111–114 µs | 101 µs | **93.4–93.8 µs (1.2×)** |
| global | 1 | 4k | 36.0 µs | **23.8 µs** | 24.1 µs (1.5×) |
| global | 1 | 32k | 192.6 µs | 109.4 µs | **101.8 µs (1.9×)** |
| global | 1 | 128k | 717 µs | 542 µs | **527 µs (1.4×)** |
| global | 8 | 4k | 194.8 µs | **112.0 µs** | 116.8 µs (1.7×) |
| global | 8 | 32k | 1,435 µs | 747 µs | **738 µs (1.9×)** |
| global | 8 | 128k | 5,662 µs | 3,337 µs | **3,162 µs (1.8×)** |

**What made vLLM slow here.** vLLM falls back to a generic Triton kernel for this model on this GPU, because the model mixes head sizes 256 and 512. Three specific costs followed:
- **Sliding layers:** in decode, the work was split across the whole sequence. Only 1,024 tokens matter, so most of the GPU sat idle. That was 5× off the speed limit at long context.
- **Global layers:** these cache both K and V, but K can be rebuilt from V (K = RoPE(w ⊙ V)). Reading only V halves the memory traffic.
- **Prefill:** the tiles were too small (16 rows) for compute-heavy work.

## End to end: the whole model in vLLM

**Setup:**
- Gemma 4 31B (4-bit weights) with a rank-16 LoRA adapter, in vLLM 0.32, on the same GPU.
- Greedy decoding with CUDA graphs.
- Speed is new tokens per second across all sequences. Each run times 257 tokens and 1 token; the difference, divided by 256, gives the time per step, so the prompt's prefill drops out.

**Configurations:**
- **"vLLM"** is vLLM's fastest setup: torch.compile plus CUDA graphs.
- **"Ours"** is the WarpSpeed decode attention kernels plus the user's `gemv8` MLP kernel. Everything else in the model is stock vLLM.

**Decode speed (tokens/s)**

| Sequences × context | vLLM | Ours | Speedup |
|---|---|---|---|
| 1 × 556 | 52.2 | 55.2 | 1.06× |
| 8 × 556 | 365.3 | 384.9 | 1.05× |
| 1 × 32,768 | 42.1 | 52.0 | 1.24× |
| 8 × 8,192 | 291.1 | 334.8 | 1.15× |
| 8 × 32,768 | 209.8 | **273.7** | **1.30×** |

**Why the gain is smaller than the kernel gain.** Every new token pays a fixed cost of about 18 ms, whatever the context length. That time goes to reading the model's weights: the projections, the MLP, the adapters and the output layer. Attention comes on top of that and grows with context:

| | vLLM attention per step | Ours | Attention speedup | Whole step |
|---|---|---|---|---|
| 1 × 32k | about 5.0 ms | about 1.9 ms | 2.6× | 23.8 → 19.2 ms |
| 8 × 32k | about 20 ms | about 12 ms | 1.7× | 38.1 → 29.2 ms |

So the speedup grows with context, as attention becomes a bigger share of each step.

**Who contributed what.** Measured separately against the same vLLM setup without torch.compile:
- The attention kernels alone give 1.00–1.02× at short context and 1.11–1.17× at long context.
- `gemv8` adds another 6–9%.

The measured step times match the kernel benchmarks. Going from 8 × 8k to 8 × 32k adds 5.3 ms per step for us, against 5.4 ms predicted from the global-attention kernel.

**Run settings and checks:**
- The 8 × 32k run used smaller prefill chunks (1,024 tokens) and text-only mode, so that all eight long sequences fit in memory at once. Both columns used these settings.
- In every run, all sequences decoded together; this was checked from the scheduler.
- Prompts were random tokens. On those, greedy outputs cannot be compared token by token: even vLLM's own two configurations disagree within 3 tokens. Numerical accuracy is shown by the kernel-level checks above.

## What the models did

| | Planner | Refiner |
|---|---|---|
| Episodes finished | 16 / 16 | 16 / 16 |
| Tool calls that ran | 54 / 71 | 36 / 36 |
| Ended with a plan that passes the checker | — | 14 / 16 |
| Fixed at least one checker error | — | 10 / 16 |

- **The planner found all three problems above by itself.** It identified the wasted sliding-window work and proposed the exact fix, proposed the V-only cache, and proposed bigger prefill tiles. Two of its prefill proposals did not fit in shared memory (139,776 and 167,936 bytes, against a 101,376-byte limit).
- **The refiner fixed both of those over-budget proposals.** Its plans with a real split into squads (a loading squad and a computing squad) became the fastest kernels in the tables above. Seven of its 16 plans had a single squad doing everything: legal, but no specialization.
- **Who did what:**
  - The ideas came from the planner, and the warp structure from the refiner.
  - Claude wrote all the code.
  - Claude also made some coupled choices the plans did not specify:
    - splitting the 512-wide heads across warp pairs;
    - moving the RoPE angle table onto the loading squad;
    - giving each computing warp its own buffer stage in the sliding decode kernel.
  - Every result records where it came from.

## Earlier round: the MLP (Altair)

**Starting point.** Before attention, we ran the same loop on Gemma 4 31B's dense MLP (4-bit weights plus a rank-16 LoRA adapter), starting from the user's own kernels. Those kernels were already at about 80% of the GPU's matrix-unit peak, about as fast as decoding weights into BF16 and calling cuBLAS.

**Outcome.** None of the models' proposals beat them:
- deeper prefetch made things slower;
- smaller tiles made things slower;
- a "dequantize-ahead" design was exact but 9–21% slower, because it moved work through shared memory that the original kept in registers.

**What it produced:**
- **The register rule above.** It explains why every 9-warp kernel was capped at 168 registers and spilled. The original design assumed 224.
- **A real end-to-end gain from the user's `gemv8` kernel.** Inside vLLM decoding with Gemma 4 31B, it delivered 55.0 vs 50.4 tokens/s at batch 1 (1.09×), and 378.6 vs 353.1 tokens/s at batch 8 (1.07×). Against vLLM's fully compiled default it is 1.06× and 1.04×.

## What we learned

1. **Structure beats tuning.** The biggest wins came from warp-specialized designs: separate loading and computing squads with staged buffers. vLLM's Triton kernel cannot express that. Retuning its tile sizes gave 1.5–2.4×; the refiner's structure gave 2.9–4.6×.
2. **A checker turns guesses into fixes.** The models' own register and shared-memory estimates were often wrong. With a tool that says exactly what is over budget, the refiner corrected real errors instead of guessing.
3. **The starting point decides the headroom.** Near-peak MLP kernels left nothing to win. vLLM's generic attention kernel left a lot.
4. **The limits:**
   - The checker counts registers but does not compile. One plan passed the check, yet the compiled kernel still spilled.
   - Rebuilding K from V changes one rounding step; the measured error is equal to or lower than vLLM's own error against the exact K.
   - One decode shape still favors the simpler Triton version.
   - Test data was random. These kernels are specific to Gemma 4's head sizes.
   - End to end, the gain is 1.05–1.30×, far below the kernel gains. About 18 ms of every decode step goes to stock vLLM code we did not touch.
   - Prefill has not yet been measured end to end.

## Next

- Profile one decode step and attack the fixed ~18 ms (projections, LoRA adapters, output layer). It is now the largest share of every step.
- Measure prefill end to end, and decode at 128k context, where attention dominates.
- Feed the plans that worked back into training, and rerun the MLP round with the new refiner in the loop.
- Close the remaining prefill gap. It sits at about half of peak; the next step is likely deeper staging plus a dedicated loading warpgroup with `setmaxnreg`.

## What's in the package

| Path | What it is |
|---|---|
| `README.md` | This write-up |
| `deneb/TG4DAC_WarpSpeed.ipynb` | The executed experiment record: every table and figure, rendered from the two `results.csv` files (re-executes in place) |
| `altair/results.csv`, `deneb/results.csv` | Every model episode and every measurement, with origins |
| `deneb/attn.py` | All attention kernels (vLLM seed harness, Triton designs, the four WarpSpeed CUDA kernels) and the benchmark |
| `deneb/warpspeed_tool.py` | The `warpspeed_check` tool |
| `deneb/corpus.py`, `deneb/corpus.jsonl.gz` | Refiner training data builder and the 30 examples |
| `deneb/run.py` | Training, planner/refiner serving, packet building, end-to-end vLLM runs (`run.py e2e MODE`), CSV collection |
| `altair/bench.py`, `altair/run.py` | The user's MLP kernels plus Altair's variants; MLP and end-to-end decode drivers |
| `altair/PLAY.md`, `deneb/PLAY.md` | Detailed records of each round |
| `deneb/sources/` | Pinned CCCL WarpSpeed and vLLM attention sources, with SHA-256 manifests (PyCuTe is pinned by revision `cac71fe7`, not bundled) |
| `MANIFEST.txt` | SHA-256 of every file in the package |
| `refiner_ws1/` (separate archive) | The trained refiner adapter (PEFT, BF16) and its training receipts |

**To rerun on an RTX PRO 6000 (SM120):**
- Attention benchmarks: `python attn.py --designs --ws` (needs vLLM 0.32 for the seed).
- Training and serving use the Forge scripts in `../forge_ast`: `train.py` and `forge_model.py`.
