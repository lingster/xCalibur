# Kernels

Four workload-specific implementations are distributed here: Gemma 4 31B
attention and MLP kernels on SM120, the challenge's NVFP4 MoE on H200, and two
prefill GEMM studies for a hybrid decision model on sm_89 and sm_86 cards. Their
guides describe the computation, physical design, dependencies and recorded
measurements.

| Folder | What | Hardware | Result [M] |
|---|---|---|---|
| [`moe_w4a16_h200/`](moe_w4a16_h200/) | NVFP4 routed-experts MoE, one persistent kernel; a GF(2)-searched decoder, grouped HFMA2 for one token, per-size builds | H200 (sm_90a) | **2.407×** over SGLang, Open Kernel Challenge judge |
| [`fp8_block_sm89/`](fp8_block_sm89/) | Block-FP8 GEMM (128×128 weight scales, per-128-group activation quantisation): the stock Triton kernel's math with its tiling opened up, quantise-once, exact concatenation groups, a per-shape measured search, sm_89 SASS from any machine | RTX 6000 Ada (sm_89) | Probe only: stock reaches 150–162 TFLOPS against a ~210 TFLOPS practical ceiling at 28k tokens; tuned results pending ([run guide](fp8_block_sm89/RUN_ON_ADA.md)) |
| [`w8a8_sm86/`](w8a8_sm86/) | INT8 W8A8 linears (per-token activations, per-channel weights): Triton quantisers, a fused-dequant Triton GEMM and cuBLASLt `_int_mm`, SmoothQuant folds and per-128 activation scales, a model patch | RTX 3090 (sm_86) | **2.81×** on a state pass's projections, **1.84×** end to end over bf16 (**1.81×** smoothed, at half the drift); outside the answer gate, not adopted |
| [`tg4dac_sm120/`](tg4dac_sm120/) | Gemma 4 31B attention and dense W4A16 MLP + LoRA; TG4DAC planner/refiner designs | RTX PRO 6000 Blackwell (SM120) | **4.6×** / **3.1×** over vLLM on selected prefill cases, **1.30×** whole-model decode |

Each folder's README gives the math, the design choices, what changed against the starting point, and credits.
Results are archived hardware measurements. The Gemma 4 attention microbenchmarks
and whole-model vLLM integration have different scopes; the integration harness
is described in the TG4DAC guide and is not bundled with these kernel sources.
