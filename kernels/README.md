# Kernels

| Folder | What | Hardware | Result [M] |
|---|---|---|---|
| [`moe_w4a16_h200/`](moe_w4a16_h200/) | NVFP4 routed-experts MoE, one persistent kernel; a GF(2)-searched decoder, grouped HFMA2 for one token, per-size builds | H200 (sm_90a) | **2.407×** over SGLang, Open Kernel Challenge judge |
| [`tg4dac_sm120/`](tg4dac_sm120/) | Gemma 4 31B attention designed by language-model planners and WarpSpeed plans; the dense 4-bit MLP + LoRA they started on | RTX PRO 6000 Blackwell (SM120) | **4.6×** / **3.1×** over vLLM on prefill, **1.30×** end-to-end decode |

Each folder's README gives the math, the design choices, what changed against the starting point, and credits.
Every result was measured on real hardware against the named baseline in the same run; nothing here is a model's
prediction.
