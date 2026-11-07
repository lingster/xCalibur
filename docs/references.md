# References

Only sources this work actually used: papers we built on, references we turned into graphs, the baselines we
measured against, and the models, data and libraries we ran. Keys match the citations in the chapters.

## Layouts and GPU programming

- <a id="cecka-2026"></a>**[Cecka 2026]** C. Cecka. *CuTe Layout Representation and Algebra.* arXiv:2603.02298, 2026. The CuTe graph and the decoder-as-layout view. [link](https://arxiv.org/abs/2603.02298)
- <a id="colfax-2026"></a>**[Colfax 2026]** Colfax Research. *Categorical Foundations for CuTe Layouts.* arXiv:2601.05972, 2026. Also turned into a graph for retrieval. [link](https://arxiv.org/abs/2601.05972)
- <a id="zhou-2025"></a>**[Zhou 2025]** K. Zhou et al. *Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F₂.* arXiv:2505.23819, 2025. The same F₂ view, for data layouts. [link](https://arxiv.org/abs/2505.23819)
- <a id="bondhugula-2008"></a>**[Bondhugula 2008]** U. Bondhugula et al. *A Practical Automatic Polyhedral Parallelizer and Locality Optimizer.* PLDI 2008. The formulation behind our design enumeration. [link](https://www.csa.iisc.ac.in/~udayb/publications/uday-pldi08.pdf)
- <a id="cutlass"></a>**[CUTLASS]** NVIDIA. *CUTLASS and CuTe*, including PyCuTe. Layout algebra in the planner and the decoder layouts. [link](https://github.com/NVIDIA/cutlass)
- <a id="cccl"></a>**[CCCL]** NVIDIA. *CUDA Core Compute Libraries (CCCL)*, WarpSpeed in `cub/detail/warpspeed`. The plan language and our checker's semantics. [link](https://github.com/NVIDIA/cccl)
- <a id="ptx"></a>**[PTX]** NVIDIA. *Parallel Thread Execution ISA.* The PTX graph for retrieval. [link](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- <a id="cuda-pg"></a>**[CUDA-PG]** NVIDIA. *CUDA Programming Guide*, Compute Capabilities. Per-compute-capability limits in `warpspeed.ARCHS`. [link](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- <a id="ga102"></a>**[GA102]** NVIDIA. *NVIDIA Ampere GA102 GPU Architecture*, v2. RTX 3090 and RTX A6000 throughput. [link](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
- <a id="ada"></a>**[Ada]** NVIDIA. *NVIDIA Ada GPU Architecture.* RTX 4090 and L40 throughput. [link](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- <a id="flashinfer"></a>**[FlashInfer]** FlashInfer. Its warp-specialized MLP was one of the refiner's training examples. [link](https://github.com/flashinfer-ai/flashinfer)

## Baselines and starting points

- <a id="zheng-2024"></a>**[Zheng 2024]** L. Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS 2024. The challenge's baseline. [link](https://arxiv.org/abs/2312.07104)
- <a id="kwon-2023"></a>**[Kwon 2023]** W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP 2023. The TG4DAC baseline. [link](https://arxiv.org/abs/2309.06180)
- <a id="okc-2026"></a>**[okc-2026]** vjeux. *okc-2026.* The public base the MoE kernel descends from, via alygg77 (Apache 2.0). [link](https://github.com/vjeux/okc-2026)

## Models, data and libraries

- <a id="embeddinggemma-2025"></a>**[EmbeddingGemma 2025]** Google. *EmbeddingGemma: Powerful and Lightweight Text Representations.* arXiv:2509.20354, 2025. The embeddings for retrieval. [link](https://arxiv.org/abs/2509.20354)
- <a id="lightgbm"></a>**[LightGBM]** Microsoft. *LightGBM.* The ranker in the measured game. [link](https://github.com/microsoft/LightGBM)
- <a id="kev"></a>**[Kev]** J. Palmer. *Kev.* The Qwen3.5-hybrid decision classifier of the [sm_86/sm_89 prefill study](writeups/WRITEUP_SM86_SM89.md). [link](https://github.com/jaredpalmer/kev)
- <a id="kernelbot"></a>**[KernelBot]** GPU MODE. *KernelBot data.* The public submissions in our knowledge base. [link](https://huggingface.co/datasets/GPUMODE/kernelbot-data)

## Search and games

- <a id="rebel"></a>**ReBeL.** *Combining Deep Reinforcement Learning and Search for Imperfect-Information Games.* The search/self-play lineage adapted to the project's finite constructor/challenger game. [Paper](https://arxiv.org/abs/2007.13544) · [Official implementation](https://github.com/facebookresearch/rebel)
- <a id="uct"></a>**UCT.** L. Kocsis and C. Szepesvári. *Bandit Based Monte-Carlo Planning.* The tree-search allocation rule discussed in the structural formulation. [Paper](https://aima.cs.berkeley.edu/~russell/classes/cs294/s11/readings/Kocsis%2BSzepesvari%3A2006.pdf)
- <a id="cfr-plus"></a>**CFR+.** O. Tammelin et al. *Solving Heads-Up Limit Texas Hold'em.* Regret-based game solving; the chapter states the distinct utility and finite-game scope of our experiment. [Paper](https://ijcai.org/proceedings/2015/97)
- <a id="libratus"></a>**Libratus.** N. Brown and T. Sandholm. *Libratus: The Superhuman AI for No-Limit Poker.* The precedent for abstract search, nested subgames and abstraction repair. [Architecture](https://ijcai.org/proceedings/2017/4) · [Safe and nested subgame solving](https://arxiv.org/abs/1705.02955)

---

[← Contents](README.md)
