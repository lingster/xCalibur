# Running the FP8 kit on an RTX 6000 Ada

For running the kit on an RTX 6000 Ada. **Pin one physical GPU** (GPU1 below; substitute yours, and keep other
work off it). Every command below runs
with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, so inside the processes the card is `cuda:0`. Use the
production venv (torch 2.14.1+cu130, triton 3.8.0, transformers 5.17.0, kernels 0.16.0) and install nothing.
Everything writes into one results folder.

Order: **(a) first.** It alone answers "is the FP8 GEMM far from peak, and why". Then (b) search, (c) parity, (d) a
fresh confirm, (e) efficiency again with the tuned tables. If time is short, stop after any step and send what you
have.

## 0. Setup (2 min)

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export XC=/path/to/xCalibur              # a checkout of this repository
export PYTHONPATH=$XC
export PY=/path/to/production/venv/bin/python
export R=/path/to/fp8_results && mkdir -p $R
cd $XC/kernels/fp8_block_sm89

nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit,clocks.max.sm,temperature.gpu --format=csv | tee $R/gpu.txt
$PY -c "import torch, triton, transformers, kernels; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__, triton.__version__, transformers.__version__, kernels.__version__)" | tee $R/env.txt
# must say: NVIDIA RTX 6000 Ada Generation (8, 9). If it names another card, stop: CUDA_VISIBLE_DEVICES is wrong.

# which hub kernel revision production loads (its loader, its cache):
$PY -c "import sys; from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel as L; m = L().matmul; print(sys.modules[m.__module__].__file__)" | tee $R/hub_path.txt
cat "$(dirname "$(cat $R/hub_path.txt)")/metadata.json" > $R/hub_metadata.json

# power and clocks for the whole session, physical GPU1 (nvidia-smi ignores CUDA_VISIBLE_DEVICES):
nvidia-smi dmon -i 1 -s pucv -d 5 > $R/dmon.txt &
DMON=$!
```

Tests: CPU-only first, then the real FP8 path on the card. The second run is the hardware check that our kernel's
quantisation matches sm_89's cast bit for bit, and that retiling without changing the K tile is bitwise exact.

```bash
CUDA_VISIBLE_DEVICES= $PY -m pytest -q $XC/tests/test_fp8_block_plan.py $XC/tests/test_fp8_block_kernel.py $XC/tests/test_fp8_block_aot.py 2>&1 | tee $R/tests_cpu.txt
FP8_KERNEL_TEST_DEVICE=cuda $PY -m pytest -q $XC/tests/test_fp8_block_kernel.py 2>&1 | tee $R/tests_fp8_gpu.txt
```

Failures are results too: keep going and send the files.

## (a) Stock efficiency per shape (~20–30 min): run this first

```bash
$PY efficiency.py --groups --out $R/eff_stock.json 2>&1 | tee $R/eff_stock.log
```

Per shape and M = 64 / 512 / 4,096 / 28,672 it prints stock's **eager** time (the probe's method: CUDA events around
the Python call, so host dispatch is included) and its **device** time (CUDA-graph replay: kernels only). It also
prints TFLOPS against the 728.5 datasheet and against cuBLAS FP8 at the same shape (the ~210 practical ceiling), our
stock-tiling twin, the quant kernel alone, and bf16. `--groups` adds the concat-group lever: q|k|v, gate|up and
qkv|z as separate stock calls vs one concatenated stock GEMM (exact, fewer launches).

How to read the M = 64 rows: if `eager ≫ device`, the ~0.17 ms is host launch cost, and the cure is fewer launches
(concat groups) or CUDA graphs, not tiles.

Then the direct comparison with the earlier probe, in the same session:

```bash
$PY /path/to/fp8_gemm_probe.py --out $R/fp8_probe_rerun.json 2>&1 | tee $R/fp8_probe_rerun.log
```

And the machine code production actually ran: the hub kernels just compiled by (a) sit in Triton's cache.

```bash
$PY sass.py --triton-cache ${TRITON_CACHE_DIR:-~/.triton/cache} --out $R/sass_hub_cache.json 2>&1 | tee $R/sass.log
$PY sass.py --hub 128,8,2 --hub 64,4,3 \
  --config '{"quant": "once", "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "SPLIT_K": 1, "num_warps": 8, "num_stages": 4}' \
  2>&1 | tee -a $R/sass.log
```

**Send $R now** (tar it as below) before continuing, so the analysis can start.

## (b) Search (~2–4 h in total; each command saves after every shape)

Bitwise-exact builds first. These cannot change a single output bit, so no answer re-score is needed:

```bash
$PY search.py --bitwise-only --out $R/tuned_bitwise.json 2>&1 | tee $R/search_bitwise.log
$PY search.py --bitwise-only --shapes 14336x5120,34816x5120,16384x5120 --buckets m512,m64 \
  --out $R/tuned_concat.json 2>&1 | tee $R/search_concat.log
```

Then, if time allows, the full space for down_proj (K = 17,408: split-K and smaller K tiles). These builds differ
from stock by fp32 summation order only (≤ 1 bf16 ulp), but the [writeup's](../../docs/writeups/WRITEUP_SM86_SM89.md#fp8-on-ada) ADA2 run shows FP8 can amplify that, so they
would need the answer gate before shipping:

```bash
$PY search.py --shapes 5120x17408 --out $R/tuned_full_down.json 2>&1 | tee $R/search_full_down.log
```

Each search times every candidate against the **hub kernel itself**, ABBA-paired in the same session (CUDA graphs,
cold L2), with A/A per bucket. A winner must bind twice with no losing sample, then bind again in 3 fresh confirm
sessions next to stock (the winner's curse), then pass parity.

## (c) Parity (~10 min)

```bash
$PY parity.py --twin-check --reference hub --out $R/parity_twin.json 2>&1 | tee $R/parity.log
$PY parity.py --table $R/tuned_bitwise.json --reference hub --out $R/parity_bitwise.json 2>&1 | tee -a $R/parity.log
$PY parity.py --table $R/tuned_concat.json --reference hub --out $R/parity_concat.json 2>&1 | tee -a $R/parity.log
```

`--twin-check` compares our kernel at the hub's own tiling with the hub kernel; it should print `bitwise=True`
everywhere. Each table line prints `bitwise`, `max_abs`, `max_ulp`, and both kernels' error against the fp32
reference math.

## (d) Winner's curse: fresh confirm in a new process (~20–40 min)

The search already re-confirms its winners. This step does it once more in a separate process, later, so nothing
from the search session (clocks, warm caches) carries over. ≥ 3 fresh sessions, each ABBA next to stock with A/A
first, bound with `xcalibur.stats` (≥ 2 samples, mean above `aa_margin`, none negative). Entries that fail revert
to stock.

```bash
$PY search.py --confirm-table $R/tuned_bitwise.json --out $R/confirmed_bitwise.json 2>&1 | tee $R/confirm.log
$PY search.py --confirm-table $R/tuned_concat.json --out $R/confirmed_concat.json 2>&1 | tee -a $R/confirm.log
```

## (e) Efficiency with the tuned builds (~30 min)

```bash
$PY efficiency.py --groups --table $R/confirmed_bitwise.json,$R/confirmed_concat.json --out $R/eff_tuned.json 2>&1 | tee $R/eff_tuned.log
kill $DMON
```

This prints the same rows as (a) plus the tuned columns, and the per-request GEMM time (stock, tuned, cuBLAS) for
production's workload (8 unpacked row passes) and the packed-rows workload (B4).

## Collecting the results

```bash
tar czf fp8_results_$(date +%Y%m%d_%H%M).tgz -C "$(dirname $R)" "$(basename $R)"
```

That is everything in `$R`: `env.txt`, `gpu.txt`, `hub_path.txt`, `hub_metadata.json`, `dmon.txt`, `tests_*.txt`,
`eff_*.json/log`, `fp8_probe_rerun.*`, `sass*`, `tuned_*.json`, `search_*.log`, `parity_*.json`, `parity.log`,
`confirmed_*.json`, `confirm.log`. Add a line on anything unusual: other load on the card's neighbours, throttling, a step
you skipped.

## If something goes wrong

- `hub kernel unavailable` in a log: the hub kernel could not load, and the run fell back to our twin as the
  reference. Keep the log; the results are still useful but not against production's kernel.
- To pin the exact production revision, add `--hub-revision 8c178950e8710e26a2210e4e909cd02dc16c8715` to
  efficiency, search and parity. It needs network access to the Hub.
- Out of memory at M = 28,672 with K = 17,408: run that shape alone (`--shapes 5120x17408`) with nothing else on GPU1.
