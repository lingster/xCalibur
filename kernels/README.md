# Kernels

Two workload-specific implementations are distributed here: Gemma 4 31B
attention and MLP kernels on SM120, and the challenge's NVFP4 MoE on H200.
Their guides describe the computation, physical design, dependencies and
recorded measurements.

| Folder | What | Hardware | Result [M] |
|---|---|---|---|
| [`moe_w4a16_h200/`](moe_w4a16_h200/) | NVFP4 routed-experts MoE, one persistent kernel; a GF(2)-searched decoder, grouped HFMA2 for one token, per-size builds | H200 (sm_90a) | **2.407×** over SGLang, Open Kernel Challenge judge |
| [`tg4dac_sm120/`](tg4dac_sm120/) | Gemma 4 31B attention and dense W4A16 MLP + LoRA; TG4DAC planner/refiner designs | RTX PRO 6000 Blackwell (SM120) | **4.6×** / **3.1×** over vLLM on selected prefill cases, **1.30×** whole-model decode |

Each folder's README gives the math, the design choices, what changed against the starting point, and credits.
Results are archived hardware measurements. The Gemma 4 attention microbenchmarks
and whole-model vLLM integration have different scopes; the integration harness
is described in the TG4DAC guide and is not bundled with these kernel sources.
