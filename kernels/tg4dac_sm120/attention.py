"""Gemma 4 31B attention on SM120: incumbent (vLLM 0.32 TRITON_ATTN unified attention), SOL, implemented designs.

Contract gemma4-31b-attn-v1, per layer.  decode: B sequences x 1 query token over L cached tokens.  prefill/training:
one causal sequence of M query tokens (M as in the MLP contract: 7131, 15329).
  o_h = softmax(q_h . K_g(h)^T) V_g(h), scaling 1.0, no softcap, FP32 accumulation and softmax, BF16 out.
  sliding: 32 Q / 16 KV heads, d=256, last 1024 positions, paged blocks of 16 tokens
  global : 32 Q / 4 KV heads,  d=512, all positions,      paged blocks of 32 tokens (K and V both cached)
Inputs are what vLLM's attention layer receives: q after norm/RoPE, the paged BF16 KV cache in vLLM's layout
[num_blocks, num_kv_heads, block_size, 2*head_size] (each token's K row then V row), block tables, seq lens.
The incumbent call reproduces TritonAttentionMetadataBuilder's 2D/3D choice and segment count from vLLM's own constants.

  python attention.py [--B 1 8] [--L 4096 32768 131072] [--prefill-M 7131 15329] [--out attn.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch
import triton
import triton.language as tl

LAYERS = {"sliding": dict(hkv=16, d=256, window=1024, block=16), "global": dict(hkv=4, d=512, window=None, block=32)}
HQ, DRAM_GBS, L2_BYTES = 32, 1529.0, 134217728


def make_case(kind, B, L, seed=0):
    c = LAYERS[kind]
    g = torch.Generator(device="cuda").manual_seed(seed)
    blk, hkv, d = c["block"], c["hkv"], c["d"]
    nblk_seq = -(-L // blk)
    live = nblk_seq if c["window"] is None else min(nblk_seq, -(-c["window"] // blk) + 1)   # pages the window can touch
    nblocks = B * live + 1                                                     # block 0: shared dummy for evicted pages
    # vLLM's TRITON_ATTN allocation: [num_blocks, num_kv_heads, block_size, 2 * head_size], viewed as K | V
    kv = torch.randn(nblocks, hkv, blk, 2 * d, generator=g, device="cuda", dtype=torch.bfloat16)
    k, v = kv.transpose(1, 2).split(d, dim=-1)
    perm = torch.randperm(nblocks - 1, generator=g, device="cuda") + 1
    table = torch.zeros(B, nblk_seq, dtype=torch.int32, device="cuda")
    for b in range(B):
        mine = perm[b * live:(b + 1) * live].to(torch.int32)
        table[b, nblk_seq - live:] = mine                                      # out-of-window pages -> dummy block 0
    q = torch.randn(B, HQ, d, generator=g, device="cuda", dtype=torch.bfloat16)
    seqlens = torch.full((B,), L, dtype=torch.int32, device="cuda")
    cu = torch.arange(B + 1, dtype=torch.int32, device="cuda")
    return dict(kind=kind, phase="decode", B=B, L=L, q=q, k=k, v=v, table=table, seqlens=seqlens, cu=cu, max_q=1, **c)


def make_prefill(kind, M, seed=0):
    """One causal sequence of M query tokens whose K/V are already in the paged cache (vLLM's prefill call)."""
    c = LAYERS[kind]
    g = torch.Generator(device="cuda").manual_seed(seed)
    blk, hkv, d = c["block"], c["hkv"], c["d"]
    nblk = -(-M // blk)
    kv = torch.randn(nblk, hkv, blk, 2 * d, generator=g, device="cuda", dtype=torch.bfloat16)
    k, v = kv.transpose(1, 2).split(d, dim=-1)
    table = torch.randperm(nblk, generator=g, device="cuda").to(torch.int32)[None]
    q = torch.randn(M, HQ, d, generator=g, device="cuda", dtype=torch.bfloat16)
    return dict(kind=kind, phase="prefill", B=1, L=M, M=M, q=q, k=k, v=v, table=table,
                seqlens=torch.tensor([M], dtype=torch.int32, device="cuda"),
                cu=torch.tensor([0, M], dtype=torch.int32, device="cuda"), max_q=M, **c)


MMA_TFLOPS = 437.0   # measured mma.sync BF16 peak with FP32 accumulate on this G4 (TG4DAC record)


def prefill_sol(case):
    M, W = case["M"], case["window"] or case["M"]
    pairs = sum(min(i + 1, W) for i in range(M))
    flops = 4 * pairs * HQ * case["d"]
    return dict(score_pairs=pairs, flops=flops, sol_ms=flops / (MMA_TFLOPS * 1e12) * 1e3, sol_basis="mma.sync peak")


def prefill_reference(case, rows=256):
    """FP32 attention for a strided sample of query rows (the full M x M product is not needed to check)."""
    M, d, blk, grp = case["M"], case["d"], case["block"], HQ // case["hkv"]
    pos = torch.arange(M, device="cuda")
    blocks = case["table"][0, pos // blk].long()
    K, V = case["k"][blocks, pos % blk].float(), case["v"][blocks, pos % blk].float()
    idx = torch.linspace(0, M - 1, rows, device="cuda").long()
    out = torch.empty(rows, HQ, d, dtype=torch.float32, device="cuda")
    for r, i in enumerate(idx.tolist()):
        lo = 0 if case["window"] is None else max(0, i + 1 - case["window"])
        q = case["q"][i].float().view(case["hkv"], grp, d)
        s = torch.einsum("hgd,nhd->hgn", q, K[lo:i + 1])
        out[r] = torch.einsum("hgn,nhd->hgd", torch.softmax(s, -1), V[lo:i + 1]).reshape(HQ, d)
    return idx, out


def sol_ms(case):
    eff = case["L"] if case["window"] is None else min(case["L"], case["window"])
    kv = case["B"] * eff * case["hkv"] * case["d"] * 2 * 2
    io = case["B"] * HQ * case["d"] * 2 * 2
    return dict(kv_bytes=kv, sol_ms=(kv + io) / (DRAM_GBS * 1e9) * 1e3,
                sol_ms_k_from_v=(kv // 2 + io) / (DRAM_GBS * 1e9) * 1e3 if case["window"] is None else None)


def reference(case):
    """FP32 math over the gathered live keys of each sequence."""
    out = torch.empty(case["B"], HQ, case["d"], dtype=torch.float32, device="cuda")
    grp = HQ // case["hkv"]
    for b in range(case["B"]):
        L, blk = case["L"], case["block"]
        lo = 0 if case["window"] is None else max(0, L - case["window"])
        pos = torch.arange(lo, L, device="cuda")
        blocks = case["table"][b, pos // blk].long()
        K = case["k"][blocks, pos % blk].float()                               # [n, hkv, d]
        V = case["v"][blocks, pos % blk].float()
        q = case["q"][b].float().view(case["hkv"], grp, case["d"])
        s = torch.einsum("hgd,nhd->hgn", q, K)
        out[b] = torch.einsum("hgn,nhd->hgd", torch.softmax(s, -1), V).reshape(HQ, case["d"])
    return out


class Incumbent:
    """vLLM's unified_attention exactly as TritonAttentionMetadataBuilder sets it up for a decode step."""
    name = "vllm_triton_unified"

    def __init__(self, case, capture_sizes=(1, 2, 4, 8)):
        from vllm.v1.attention.ops import triton_unified_attention as tua
        from vllm.v1.attention.backends import triton_attn as ta
        self.f, self.c = tua.unified_attention, case
        hkv, d, B = case["hkv"], case["d"], case["B"]
        thr = ta.MIN_LAUNCH_GRID_SIZE_2D // hkv
        thr = min(capture_sizes, key=lambda x: abs(x - thr))
        seg = ta.NUM_PAR_SOFTMAX_SEGMENTS
        max_seqs_64 = (torch.cuda.get_device_properties(0).multi_processor_count - 1) // (seg * hkv)
        rows = thr
        if max_seqs_64 > 0:
            seg = 64
            rows = max(min(max_seqs_64, thr), -(-thr // (64 // ta.NUM_PAR_SOFTMAX_SEGMENTS)))
        if max_seqs_64 > 0 and B > max_seqs_64:                                # build(): >max_seqs_64 seqs use 16 segments
            seg = ta.NUM_PAR_SOFTMAX_SEGMENTS
        dp = 1 << (d - 1).bit_length()
        self.thr, self.seg = thr, seg
        self.segm = (torch.empty(max(rows, B), HQ, seg, dp, dtype=torch.float32, device="cuda"),
                     torch.empty(max(rows, B), HQ, seg, dtype=torch.float32, device="cuda"),
                     torch.empty(max(rows, B), HQ, seg, dtype=torch.float32, device="cuda"))
        self.out = torch.empty(case["q"].shape[0], HQ, d, dtype=torch.bfloat16, device="cuda")
        self.win = (-1, -1) if case["window"] is None else (case["window"] - 1, 0)

    def __call__(self):
        c = self.c
        self.f(q=c["q"], k=c["k"], v=c["v"], out=self.out, cu_seqlens_q=c["cu"], max_seqlen_q=c["max_q"], seqused_k=c["seqlens"],
               max_seqlen_k=c["L"], softmax_scale=1.0, causal=True, window_size=self.win, block_table=c["table"],
               softcap=0, q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=self.thr,
               num_par_softmax_segments=self.seg, softmax_segm_output=self.segm[0], softmax_segm_max=self.segm[1],
               softmax_segm_expsum=self.segm[2])
        return self.out


@triton.jit
def _trim_pages(table_ptr, seqlen_ptr, out_table, out_seqlen, table_stride, width, W: tl.constexpr, PAGE: tl.constexpr,
                NB: tl.constexpr, NBP: tl.constexpr):
    b = tl.program_id(0)
    L = tl.load(seqlen_ptr + b)
    first = tl.maximum(L - W, 0) // PAGE
    i = tl.arange(0, NBP)
    src = tl.minimum(first + i, width - 1)
    tl.store(out_table + b * NB + i, tl.load(table_ptr + b * table_stride + src, mask=i < NB, other=0), mask=i < NB)
    tl.store(out_seqlen + b, L - first * PAGE)


class WindowTrim:
    """Planner sliding_decode p1/p3: split-softmax segments should span the window, not the whole sequence.
    Claude's realization keeps vLLM's kernel unchanged: each sequence is re-based on the first page its window touches
    (cached K already carries RoPE, so only relative positions matter); the re-basing is one tiny kernel per step,
    inside the timed graph."""
    name = "window_trim"

    def __init__(self, case):
        W, blk, B = case["window"], case["block"], case["B"]
        self.src, self.W, self.blk = case, W, blk
        self.nb = min(case["table"].shape[1], -(-W // blk) + 1)
        self.ar = torch.arange(self.nb, device="cuda", dtype=torch.int64)
        self.t = dict(case, table=torch.empty(B, self.nb, dtype=torch.int32, device="cuda"),
                      seqlens=torch.empty_like(case["seqlens"]), L=self.nb * blk)
        self.inc = Incumbent(self.t)

    def __call__(self):
        src = self.src
        _trim_pages[(src["B"],)](src["table"], src["seqlens"], self.t["table"], self.t["seqlens"], src["table"].stride(0),
                                 src["table"].shape[1], W=self.W, PAGE=self.blk, NB=self.nb, NBP=1 << (self.nb - 1).bit_length())
        return self.inc()


# Planner prefill proposals, exactly as stated (unstated fields keep the seed's value: TILE 32, 3 stages).
PLANNER_PREFILL = {
    "sliding": [("p0", 64, 32, 8, 2), ("p1", 64, 64, 16, 3), ("p2", 32, 32, 8, 2), ("p3", 64, 32, 8, 3)],
    "global": [("p0", 32, 32, 8, 3), ("p1", 32, 64, 8, 2), ("p2", 32, 32, 8, 3), ("p3", 32, 32, 8, 3)],
}


# Refiner fixes that map onto the seed kernel's launch constants (Q tile bytes -> BLOCK_M rows, K/V tile bytes -> TILE
# tokens, squads -> warps, stages as checked).  Plans whose change is a squad split need a WarpSpeed kernel instead.
REFINER_PREFILL = {
    "sliding": [("p1", 32, 32, 16, 2)],                 # fixes planner p1 (64x64, 16 warps: 139,776 B smem)
    "global": [("p1", 32, 32, 8, 2), ("p2", 32, 32, 8, 2), ("p3", 16, 16, 8, 2)],   # p1 fixes 32x64 (167,936 B)
}


class SeedTiled:
    """vLLM's kernel_unified_attention, unchanged, launched exactly as its host does for the 2D (prefill) path except
    for the proposal's BLOCK_M / TILE_SIZE / num_warps / num_stages."""

    def __init__(self, case, block_m, tile, warps, stages):
        from vllm.v1.attention.ops import triton_unified_attention as tua
        from vllm.v1.kv_cache_interface import KVQuantMode
        self.kern, self.mode, self.c = tua.kernel_unified_attention, KVQuantMode.NONE, case
        self.bm, self.tile, self.warps, self.stages = block_m, tile, warps, stages
        self.out = torch.empty(case["q"].shape[0], HQ, case["d"], dtype=torch.bfloat16, device="cuda")

    def __call__(self):
        c, q, k, v, out = self.c, self.c["q"], self.c["k"], self.c["v"], self.out
        nq = HQ // c["hkv"]
        bq = self.bm // nq
        d, dp = c["d"], 1 << (c["d"] - 1).bit_length()
        grid = (q.shape[0] // bq + 1, c["hkv"])
        self.kern[grid](
            output_ptr=out, segm_output_ptr=None, segm_max_ptr=None, segm_expsum_ptr=None, query_ptr=q, key_cache_ptr=k,
            value_cache_ptr=v, sink_ptr=None, block_tables_ptr=c["table"], seq_lens_ptr=c["seqlens"], alibi_slopes_ptr=None,
            qq_bias_ptr=None, k_scale_cache_ptr=None, v_scale_cache_ptr=None, scale=1.0, q_scale=None, k_scale=None,
            v_scale=None, out_scale=1.0, softcap=0, num_query_heads=HQ, num_queries_per_kv=nq,
            block_table_stride=c["table"].stride(0), query_stride_0=q.stride(0), query_stride_1=q.stride(1),
            output_stride_0=out.stride(0), output_stride_1=out.stride(1), qq_bias_stride_0=0, BLOCK_SIZE=c["block"],
            TILE_SIZE=self.tile, HEAD_SIZE=d, HEAD_SIZE_PADDED=dp, USE_ALIBI_SLOPES=False, USE_ALIBI_SQRT=False,
            USE_QQ_BIAS=False, USE_SOFTCAP=False, USE_SINKS=False, SLIDING_WINDOW=c["window"] or 0, USE_CAUSAL=True,
            USE_PER_SEQ_CAUSAL=False, per_seq_causal_ptr=None, USE_MM_PREFIX=False, MAX_MM_RANGES=0,
            mm_prefix_range_ptr=None, rswa_prefix_lens_ptr=c["seqlens"], R_SWA_WINDOW=0, USE_R_SWA=False,
            stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3), stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3), stride_ks_blk=None, stride_ks_slot=None,
            stride_ks_head=None, stride_vs_blk=None, stride_vs_slot=None, stride_vs_head=None,
            query_start_len_ptr=c["cu"], BLOCK_Q=bq, num_seqs=1, BLOCK_M=self.bm, NUM_SEGMENTS_PER_SEQ=1, USE_FP8=False,
            IS_3D=False, KV_QUANT_MODE=self.mode, Q_IS_FP8=False, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1, USE_TD=False,
            USE_TD_QO=False, MM_PREFIX_CLAMP_SW=False, num_warps=self.warps, num_stages=self.stages)
        return out


def designs(a):
    """Every implemented design next to the seed, re-measured in the same process (same allocation, same inputs)."""
    rows = []
    if "sliding" in a.kinds:
        for B in a.B:
            for L in a.L:
                cases = [make_case("sliding", B, L)]
                n = max(1, min(16, -(-L2_BYTES * 4 // cases[0]["k"].untyped_storage().nbytes())))
                cases += [make_case("sliding", B, L, seed=i) for i in range(1, n)]
                ref = reference(cases[0])
                seed = [Incumbent(c) for c in cases]
                trim = [WindowTrim(c) for c in cases]
                ts = time_graph(lambda: [f() for f in seed]); tt = time_graph(lambda: [f() for f in trim])
                row = dict(kind="sliding", phase="decode", B=B, L=L, design="window_trim", origin="planner sliding_decode p1/p3",
                           implementation="Claude: per-sequence page re-basing; vLLM kernel unchanged", rotated_caches=n,
                           seed_ms=ts["median_ms"] / n, design_ms=tt["median_ms"] / n, **sol_ms(cases[0]), **check(trim[0](), ref))
                row.update(speedup=row["seed_ms"] / row["design_ms"], pct_of_sol=100 * row["sol_ms"] / row["design_ms"])
                rows.append(row); print(json.dumps(row), flush=True)
                del cases, seed, trim, ref
                torch.cuda.empty_cache()
    if a.ws:
        WS = {"sliding": (WSPrefillSliding, "refiner sliding_prefill p0: load squad 4 warps -> compute squad 4 warps; Q ring 2 x 32 KB, KV ring 2 x 16 KB",
                          "Claude: CUDA WarpSpeed realization (cp.async + mbarrier rings, mma.sync m16n8k16)"),
              "global": (WSPrefillGlobal, "refiner global_prefill p0 (8-warp MMA squad owning a cp.async KV ring) + p2 (S_tile exchange)",
                         "Claude: 4 row groups x 2 head-dim halves, partial Q.K^T summed through a double-buffered smem S buffer")}
        for kind in a.kinds:
          for M in a.prefill_M:
            case = make_prefill(kind, M)
            idx, ref = prefill_reference(case)
            ts = time_graph(Incumbent(case), warmup=3, samples=5)
            cls, origin, impl = WS[kind]
            f = cls(case)
            row = dict(kind=kind, phase="prefill", M=M, design=cls.name, origin=origin, implementation=impl,
                       BLOCK_M=64, TILE=32, warps=8, stages=2, regs=f.k.regs, local_bytes=f.k.local,
                       seed_ms=ts["median_ms"], **prefill_sol(case), **check(f()[idx], ref))
            row["design_ms"] = time_graph(f, warmup=3, samples=5)["median_ms"]
            row.update(speedup=row["seed_ms"] / row["design_ms"], tflops=row["flops"] / (row["design_ms"] * 1e9),
                       pct_of_sol=100 * row["sol_ms"] / row["design_ms"])
            rows.append(row); print(json.dumps(row), flush=True)
            del case, f, ref
            torch.cuda.empty_cache()
        for B in (a.B if "sliding" in a.kinds else []):
            for L in a.L:
                cases = [make_case("sliding", B, L)]
                n = max(1, min(16, -(-L2_BYTES * 4 // cases[0]["k"].untyped_storage().nbytes())))
                cases += [make_case("sliding", B, L, seed=i) for i in range(1, n)]
                ref = reference(cases[0])
                seed = [Incumbent(c) for c in cases]
                trim = [WindowTrim(c) for c in cases]
                tried = []
                for seg in (1, 2, 4, 8, 16):
                    f = WSDecodeSliding(cases[0], seg)
                    tried.append(dict(segments=seg, ms=time_graph(f, warmup=3, samples=3, target_ms=10)["median_ms"]))
                best = min(tried, key=lambda r: r["ms"])
                ws = [WSDecodeSliding(c, best["segments"]) for c in cases]
                ts = time_graph(lambda: [f() for f in seed]); tt = time_graph(lambda: [f() for f in trim])
                tw = time_graph(lambda: [f() for f in ws])
                row = dict(kind="sliding", phase="decode", B=B, L=L, design="ws_decode_sliding",
                           origin="refiner sliding_decode p1: load squad 1 warp -> compute squad 3 warps; KV page ring 2 x 16 KB; segment_partial drained by the load squad",
                           implementation="Claude: CUDA WarpSpeed realization; window bounds in-kernel, tiles round-robin over 3 compute warps",
                           rotated_caches=n, segments=best["segments"], configs_tried=tried, regs=ws[0].k.regs, local_bytes=ws[0].k.local,
                           seed_ms=ts["median_ms"] / n, triton_window_trim_ms=tt["median_ms"] / n, design_ms=tw["median_ms"] / n,
                           **sol_ms(cases[0]), **check(ws[0](), ref))
                row.update(speedup=row["seed_ms"] / row["design_ms"], speedup_vs_window_trim=row["triton_window_trim_ms"] / row["design_ms"],
                           pct_of_sol=100 * row["sol_ms"] / row["design_ms"])
                rows.append(row); print(json.dumps({k: v for k, v in row.items() if k != "configs_tried"}), flush=True)
                del cases, seed, trim, ws, ref
                torch.cuda.empty_cache()
        TRITON_CFG = {(1, 4096): (64, 16, 3), (1, 32768): (64, 16, 3), (1, 131072): (256, 16, 3),
                      (8, 4096): (32, 16, 3), (8, 32768): (32, 16, 3), (8, 131072): (64, 16, 3)}   # run-2 choices
        for B in (a.B if "global" in a.kinds else []):
            for L in a.L:
                cases = [make_case("global", B, L)]
                n = max(1, min(16, -(-L2_BYTES * 4 // cases[0]["k"].untyped_storage().nbytes())))
                cases += [make_case("global", B, L, seed=i) for i in range(1, n)]
                extras = [make_vonly(c, i) for i, c in enumerate(cases)]
                ref = reference_k(cases[0], extras[0]["k_true"])
                seed = [Incumbent(c) for c in cases]
                tseg, ttile, tst = TRITON_CFG.get((B, L), (64, 16, 3))
                tri = [VOnlyRebuild(c, x, segments=tseg, tile=ttile, stages=tst) for c, x in zip(cases, extras)]
                tried = []
                for seg in ((16, 32, 64, 128, 256) if B == 1 else (8, 16, 32, 64)):
                    f = WSDecodeGlobal(cases[0], extras[0], seg)
                    tried.append(dict(segments=seg, ms=time_graph(f, warmup=3, samples=3, target_ms=10)["median_ms"]))
                best = min(tried, key=lambda r: r["ms"])
                ws = [WSDecodeGlobal(c, x, best["segments"]) for c, x in zip(cases, extras)]
                ts = time_graph(lambda: [f() for f in seed]); tt = time_graph(lambda: [f() for f in tri])
                tw = time_graph(lambda: [f() for f in ws])
                row = dict(kind="global", phase="decode", B=B, L=L, design="ws_decode_global",
                           origin="refiner global_decode p1: load squad 4 warps -> compute squad 4 warps; V ring 3 x 16 KB (V-only cache)",
                           implementation="Claude: CUDA WarpSpeed realization; full score per compute warp, O split in 128-dim quarters",
                           rotated_caches=n, segments=best["segments"], configs_tried=tried, regs=ws[0].k.regs, local_bytes=ws[0].k.local,
                           seed_ms=ts["median_ms"] / n, triton_v_only_ms=tt["median_ms"] / n, design_ms=tw["median_ms"] / n,
                           **sol_ms(cases[0]), **check(ws[0](), ref), seed_check=check(seed[0](), ref))
                row.update(speedup=row["seed_ms"] / row["design_ms"], speedup_vs_triton_v_only=row["triton_v_only_ms"] / row["design_ms"],
                           pct_of_sol_v_only=100 * row["sol_ms_k_from_v"] / row["design_ms"])
                rows.append(row); print(json.dumps({k: v for k, v in row.items() if k != "configs_tried"}), flush=True)
                del cases, extras, seed, tri, ws, ref
                torch.cuda.empty_cache()
        json.dump(dict(device=torch.cuda.get_device_name(), designs=rows), open(a.out, "w"), indent=1)
        return
    if "global" in a.kinds:
        for B in a.B:
            for L in a.L:
                cases = [make_case("global", B, L)]
                n = max(1, min(16, -(-L2_BYTES * 4 // cases[0]["k"].untyped_storage().nbytes())))
                cases += [make_case("global", B, L, seed=i) for i in range(1, n)]
                extras = [make_vonly(c, i) for i, c in enumerate(cases)]
                ref = reference_k(cases[0], extras[0]["k_true"])
                seed = [Incumbent(c) for c in cases]
                # Claude's launch choice, measured: segments x tile x stages on this shape (every config is recorded)
                tried = []
                for seg in ((64, 128, 256) if B == 1 else (16, 32, 64)):
                    for tile in (16, 32):
                        for st in (2, 3):
                            try:
                                f = VOnlyRebuild(cases[0], extras[0], segments=seg, tile=tile, stages=st)
                                tried.append(dict(segments=seg, tile=tile, stages=st, ms=time_graph(f, warmup=3, samples=3, target_ms=10)["median_ms"]))
                            except Exception as e:
                                tried.append(dict(segments=seg, tile=tile, stages=st, error=f"{type(e).__name__}: {str(e)[:120]}"))
                cfg = min((t for t in tried if "ms" in t), key=lambda t: t["ms"])
                vo = [VOnlyRebuild(c, x, segments=cfg["segments"], tile=cfg["tile"], stages=cfg["stages"]) for c, x in zip(cases, extras)]
                ts = time_graph(lambda: [f() for f in seed]); tv = time_graph(lambda: [f() for f in vo])
                row = dict(kind="global", phase="decode", B=B, L=L, design="v_only_k_rebuild",
                           origin="planner global_decode p0/p1/p3 (V-only cache, K rebuilt)",
                           implementation="Claude: Triton split-softmax; w folded into q, RoPE correction on 64 pairs, one V read",
                           rotated_caches=n, segments=vo[0].seg, tile=vo[0].tile, stages=vo[0].stages, configs_tried=tried,
                           seed_ms=ts["median_ms"] / n, design_ms=tv["median_ms"] / n,
                           **sol_ms(cases[0]), **check(vo[0](), ref), seed_check=check(seed[0](), ref))
                row.update(speedup=row["seed_ms"] / row["design_ms"], pct_of_sol_kv=100 * row["sol_ms"] / row["design_ms"],
                           pct_of_sol_v_only=100 * row["sol_ms_k_from_v"] / row["design_ms"])
                rows.append(row); print(json.dumps(row), flush=True)
                del cases, extras, seed, vo, ref
                torch.cuda.empty_cache()
    for kind in a.kinds:
        for M in a.prefill_M:
            case = make_prefill(kind, M)
            idx, ref = prefill_reference(case)
            ts = time_graph(Incumbent(case), warmup=3, samples=5)
            points = REFINER_PREFILL if a.prefill_source == "refiner" else PLANNER_PREFILL
            for tag, bm, tile, warps, stages in points[kind]:
                row = dict(kind=kind, phase="prefill", M=M, design=f"seed_tiled_{bm}x{tile}_w{warps}_s{stages}",
                           origin=f"{a.prefill_source} {kind}_prefill {tag}", implementation="vLLM kernel unchanged; proposal's launch constants",
                           BLOCK_M=bm, TILE=tile, warps=warps, stages=stages, seed_ms=ts["median_ms"], **prefill_sol(case))
                try:
                    f = SeedTiled(case, bm, tile, warps, stages)
                    row.update(**check(f()[idx], ref), **{"design_ms": time_graph(f, warmup=3, samples=5)["median_ms"]})
                    row.update(speedup=row["seed_ms"] / row["design_ms"], pct_of_sol=100 * row["sol_ms"] / row["design_ms"],
                               tflops=row["flops"] / (row["design_ms"] * 1e9))
                except Exception as e:
                    row.update(error=f"{type(e).__name__}: {str(e)[:300]}")
                rows.append(row); print(json.dumps(row), flush=True)
            del case, ref
            torch.cuda.empty_cache()
    json.dump(dict(device=torch.cuda.get_device_name(), designs=rows, compiled=compiled_evidence()), open(a.out, "w"), indent=1)


# ---- global layers: cache V only, rebuild K = RoPE(w_k * V) in the kernel (planner global_decode p0/p1/p3) ----
# Claude's realization: for the 384 non-rotated dims K.q = V.(w*q), so w folds into q once per program (rotated dims
# zeroed) and that part of the score is one MMA on the raw V tile; the 64 rotated pairs (dims a, a+256; Gemma 4
# proportional RoPE, theta 1e6, inv_freq denominator 512) are rebuilt as K = RoPE(w*V) in FP32 and rounded once, as the
# seed's cached K is.  P.V reuses the same V tile, so every cached byte is read once.  Split-softmax + combine as the seed.
ROPE_PAIRS, ROPE_THETA = 64, 1e6


@triton.jit
def _vonly_decode(q_ptr, w_ptr, inv_ptr, v_ptr, table_ptr, seqlen_ptr, part_o, part_m, part_l,
                  stride_vb, stride_vh, stride_vt, table_stride, SEGMENTS: tl.constexpr, NQ: tl.constexpr,
                  HQ: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, R: tl.constexpr, PAGE: tl.constexpr,
                  TILE: tl.constexpr, BLOCK_M: tl.constexpr):
    b, kvh, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    L = tl.load(seqlen_ptr + b)
    tiles = tl.cdiv(L, TILE)
    per_seg = tl.cdiv(tiles, SEGMENTS)
    t_lo, t_hi = seg * per_seg, tl.minimum((seg + 1) * per_seg, tiles)
    rows = tl.arange(0, BLOCK_M)
    rmask = rows < NQ
    head = kvh * NQ + rows
    dd = tl.arange(0, D)
    q = tl.load(q_ptr + (b * HQ + head)[:, None] * D + dd[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    w = tl.load(w_ptr + dd).to(tl.float32)
    rot = (dd < R) | ((dd >= HALF) & (dd < HALF + R))
    qw = tl.where(rot[None, :], 0.0, q * w[None, :]).to(tl.bfloat16)        # [BLOCK_M, D], rotated dims zeroed
    ra = tl.arange(0, R)
    qa = tl.load(q_ptr + (b * HQ + head)[:, None] * D + ra[None, :], mask=rmask[:, None], other=0.0)
    qb = tl.load(q_ptr + (b * HQ + head)[:, None] * D + HALF + ra[None, :], mask=rmask[:, None], other=0.0)
    wa, wb = tl.load(w_ptr + ra).to(tl.float32), tl.load(w_ptr + HALF + ra).to(tl.float32)
    inv = tl.load(inv_ptr + ra)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    tt = tl.arange(0, TILE)
    for t in range(t_lo, t_hi):
        pos = t * TILE + tt
        pmask = pos < L
        page = tl.load(table_ptr + b * table_stride + pos // PAGE, mask=pmask, other=0).to(tl.int64)
        base = v_ptr + page * stride_vb + kvh * stride_vh + (pos % PAGE) * stride_vt
        V = tl.load(base[:, None] + dd[None, :], mask=pmask[:, None], other=0.0)          # [TILE, D] bf16
        va = tl.load(base[:, None] + ra[None, :], mask=pmask[:, None], other=0.0).to(tl.float32) * wa[None, :]
        vb = tl.load(base[:, None] + HALF + ra[None, :], mask=pmask[:, None], other=0.0).to(tl.float32) * wb[None, :]
        ang = pos.to(tl.float32)[:, None] * inv[None, :]
        c, sn = tl.cos(ang), tl.sin(ang)
        ka = (va * c - vb * sn).to(tl.bfloat16)                             # K = RoPE(w*V) on dims a: one BF16 cut
        kb = (vb * c + va * sn).to(tl.bfloat16)                             # ... and on dims a + 256
        S = tl.dot(qw, tl.trans(V)) + tl.dot(qa, tl.trans(ka)) + tl.dot(qb, tl.trans(kb))
        S = tl.where(pmask[None, :], S, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(S, 1))
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_i = l_i * alpha + tl.sum(P, 1)
        acc = acc * alpha[:, None] + tl.dot(P.to(tl.bfloat16), V)
        m_i = m_new
    o = (b * HQ + head) * SEGMENTS + seg
    tl.store(part_m + o, m_i, mask=rmask)
    tl.store(part_l + o, l_i, mask=rmask)
    tl.store(part_o + o[:, None] * D + dd[None, :], acc, mask=rmask[:, None])


@triton.jit
def _combine(part_o, part_m, part_l, out_ptr, SEGMENTS: tl.constexpr, D: tl.constexpr):
    bh = tl.program_id(0)
    sg = tl.arange(0, SEGMENTS)
    m = tl.load(part_m + bh * SEGMENTS + sg)
    l = tl.load(part_l + bh * SEGMENTS + sg)
    mx = tl.max(m, 0)
    wgt = tl.where(m == float("-inf"), 0.0, tl.exp(m - mx))
    dd = tl.arange(0, D)
    o = tl.load(part_o + (bh * SEGMENTS + sg)[:, None] * D + dd[None, :])
    tl.store(out_ptr + bh * D + dd, (tl.sum(o * wgt[:, None], 0) / tl.sum(l * wgt, 0)).to(tl.bfloat16))


def make_vonly(case, seed=0):
    """V-only page cache for a global-layer case plus the k_norm weight; K of the seed cache is rebuilt from it as
    vLLM would store it (BF16 after k_norm and RoPE), so both designs see the same model state."""
    g = torch.Generator(device="cuda").manual_seed(1000 + seed)
    d, half = case["d"], case["d"] // 2
    w = (1.0 + 0.1 * torch.randn(d, generator=g, device="cuda")).to(torch.bfloat16)
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, 2 * ROPE_PAIRS, 2, device="cuda", dtype=torch.float32) / d))
    vcache = torch.empty(case["k"].shape[0], case["hkv"], case["block"], d, dtype=torch.bfloat16, device="cuda")
    vcache.copy_(case["v"].transpose(1, 2))                                   # same V values, V-only layout
    nb, blk = case["table"].shape[1], case["block"]
    pos = torch.empty(case["k"].shape[0], blk, dtype=torch.float32, device="cuda").fill_(0)
    for b in range(case["B"]):                                                # logical position of every cached slot
        pages = case["table"][b].long()
        pos[pages] = (torch.arange(nb, device="cuda")[:, None] * blk + torch.arange(blk, device="cuda")[None, :]).float()
    kw = case["v"].float() * w.float()                                        # [blocks, blk, hkv, d]
    ang = pos[:, :, None, None] * inv[None, None, None, :]
    a, bb = kw[..., :ROPE_PAIRS].clone(), kw[..., half:half + ROPE_PAIRS].clone()
    kw[..., :ROPE_PAIRS] = a * ang.cos() - bb * ang.sin()
    kw[..., half:half + ROPE_PAIRS] = bb * ang.cos() + a * ang.sin()
    case["k"].copy_(kw.to(torch.bfloat16))                                    # the seed's K slot, as vLLM caches it
    return dict(vcache=vcache, w=w, inv=inv, k_true=kw)


class VOnlyRebuild:
    name = "v_only_k_rebuild"

    def __init__(self, case, extra, segments=None, tile=16, stages=2):
        self.c, self.x, self.tile, self.stages = case, extra, tile, stages
        B, hkv, d = case["B"], case["hkv"], case["d"]
        if segments is None:
            segments = 64 if B * 16 * hkv < torch.cuda.get_device_properties(0).multi_processor_count else 16
        self.seg = segments
        self.po = torch.empty(B * HQ * segments, d, dtype=torch.float32, device="cuda")
        self.pm = torch.empty(B * HQ * segments, dtype=torch.float32, device="cuda")
        self.pl = torch.empty(B * HQ * segments, dtype=torch.float32, device="cuda")
        self.out = torch.empty(B, HQ, d, dtype=torch.bfloat16, device="cuda")

    def __call__(self):
        c, x, vc = self.c, self.x, self.x["vcache"]
        _vonly_decode[(c["B"], c["hkv"], self.seg)](
            c["q"], x["w"], x["inv"], vc, c["table"], c["seqlens"], self.po, self.pm, self.pl,
            vc.stride(0), vc.stride(1), vc.stride(2), c["table"].stride(0), SEGMENTS=self.seg, NQ=HQ // c["hkv"], HQ=HQ,
            D=c["d"], HALF=c["d"] // 2, R=ROPE_PAIRS, PAGE=c["block"], TILE=self.tile, BLOCK_M=16, num_warps=4,
            num_stages=self.stages)
        _combine[(c["B"] * HQ,)](self.po, self.pm, self.pl, self.out, SEGMENTS=self.seg, D=c["d"], num_warps=4)
        return self.out


def reference_k(case, k_true):
    """FP32 attention with the exact K = RoPE(w * V) (no BF16 cut on K)."""
    out = torch.empty(case["B"], HQ, case["d"], dtype=torch.float32, device="cuda")
    grp = HQ // case["hkv"]
    for b in range(case["B"]):
        L, blk = case["L"], case["block"]
        pos = torch.arange(L, device="cuda")
        blocks = case["table"][b, pos // blk].long()
        K, V = k_true[blocks, pos % blk], case["v"][blocks, pos % blk].float()
        q = case["q"][b].float().view(case["hkv"], grp, case["d"])
        s = torch.einsum("hgd,nhd->hgn", q, K)
        out[b] = torch.einsum("hgn,nhd->hgd", torch.softmax(s, -1), V).reshape(HQ, case["d"])
    return out


# ---- WarpSpeed kernels realizing the refiner's checked plans (CUDA, NVRTC via (the MLP kernels' NVRTC driver) ----
WS_COMMON = r"""
typedef unsigned int u32; typedef unsigned short u16; typedef unsigned long long u64;
#define DEVI static __device__ __forceinline__
#define NEG_INF __int_as_float(0xff800000)
DEVI u32 smem_u32(const void* p) { u32 r; asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r; }
DEVI void cp16(u32 dst, const void* src, u32 bytes) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes) : "memory"); }
DEVI void cp_arrive(u32 bar) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(bar) : "memory"); }
DEVI void mbar_init(u32 a, u32 c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(c) : "memory"); }
DEVI void mbar_arrive(u32 a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
DEVI void mbar_wait(u32 a, u32 ph) {
  asm volatile("{ .reg .pred p; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra WAIT_%=; }" :: "r"(a), "r"(ph) : "memory");
}
DEVI void ldsm4(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEVI void ldsm4t(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEVI void mma(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI u32 pack_bf2(float lo, float hi) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI u32 swz(int row, int ch) { return (u32)(row * 512 + ((ch ^ (row & 7)) << 4)); }   // 512-B rows, 16-B chunks
"""

WS_PREFILL_SLIDING = WS_COMMON + r"""
// Refiner plan sliding_prefill_p0 (warpspeed_check pass, 98,480 B): squads load(4 warps) -> compute(4 warps);
// resources q_tile 2 x 32 KB and kv_tiles 2 x 16 KB, each a 2-phase ring (full: load squad, empty: compute squad),
// mbarrier arrival = owner threads (128).  Persistent CTAs; unit = (32 query positions, one KV head, its 2 Q heads).
#define D 256
#define HQ 32
#define HKV 16
#define PAGE 16
#define WIN 1024
#define BQ 32
#define TILE 32
#define QBYTES (64 * 512)
#define KVBYTES (TILE * 512)
extern "C" __global__ void __launch_bounds__(256, 1)
ws_prefill_sliding(const u16* __restrict__ q, const u16* __restrict__ kv, const int* __restrict__ table, u16* __restrict__ out,
                   int M, int nunits, int nblk) {
  extern __shared__ __align__(128) unsigned char sm[];
  const u32 qb0 = smem_u32(sm), kvb0 = qb0 + 2 * QBYTES, bar = kvb0 + 2 * KVBYTES;
  const u32 q_full = bar, q_empty = bar + 16, kv_full = bar + 32, kv_empty = bar + 48;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) {
      mbar_init(q_full + 8 * i, 128); mbar_init(q_empty + 8 * i, 128);
      mbar_init(kv_full + 8 * i, 128); mbar_init(kv_empty + 8 * i, 128);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  int qi = 0, it = 0;
  if (warp < 4) {                                            // ---------------- load squad
    for (int unit = blockIdx.x; unit < nunits; unit += gridDim.x) {
      const int qb = unit / HKV, kvh = unit % HKV, p0 = qb * BQ;
      const int j_lo = max(0, p0 - (WIN - 1)) / TILE, j_hi = min(p0 + BQ - 1, M - 1) / TILE;
      const int qs = qi & 1;
      mbar_wait(q_empty + 8 * qs, ((qi >> 1) & 1) ^ 1);
      for (int c = tid; c < 64 * 32; c += 128) {
        const int row = c >> 5, ch = c & 31, pos = p0 + (row & 31), head = 2 * kvh + (row >> 5);
        const int ps = min(pos, M - 1);
        cp16(qb0 + qs * QBYTES + swz(row, ch), q + ((u64)(ps * HQ + head) * D + ch * 8), pos < M ? 16u : 0u);
      }
      cp_arrive(q_full + 8 * qs);
      ++qi;
      for (int j = j_lo; j <= j_hi; ++j) {
        for (int half = 0; half < 2; ++half) {               // K tile, then V tile, through the same ring
          const int s = it & 1;
          mbar_wait(kv_empty + 8 * s, ((it >> 1) & 1) ^ 1);
          for (int c = tid; c < TILE * 32; c += 128) {
            const int row = c >> 5, ch = c & 31, tok = j * TILE + row;
            const int page = table[min(tok / PAGE, nblk - 1)];
            const u16* src = kv + ((u64)((page * HKV + kvh) * PAGE + (tok % PAGE)) * (2 * D) + half * D + ch * 8);
            cp16(kvb0 + s * KVBYTES + swz(row, ch), src, tok < M ? 16u : 0u);
          }
          cp_arrive(kv_full + 8 * s);
          ++it;
        }
      }
    }
    return;
  }
  // -------------------------------------------------------------- compute squad
  const int cw = warp - 4, g = lane >> 2, t = lane & 3;
  const float LOG2E = 1.4426950408889634f;
  for (int unit = blockIdx.x; unit < nunits; unit += gridDim.x) {
    const int qb = unit / HKV, kvh = unit % HKV, p0 = qb * BQ;
    const int j_lo = max(0, p0 - (WIN - 1)) / TILE, j_hi = min(p0 + BQ - 1, M - 1) / TILE;
    const int head = 2 * kvh + (cw >> 1), pb = p0 + (cw & 1) * 16;
    const int pa0 = pb + g, pa1 = pb + g + 8;
    const int qs = qi & 1;
    mbar_wait(q_full + 8 * qs, (qi >> 1) & 1);
    const u32 qbase = qb0 + qs * QBYTES;
    float o[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
    for (int j = j_lo; j <= j_hi; ++j) {
      int s = it & 1;
      mbar_wait(kv_full + 8 * s, (it >> 1) & 1);
      const u32 kb = kvb0 + s * KVBYTES;
      float sc[4][4];
#pragma unroll
      for (int n = 0; n < 4; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        u32 a0, a1, a2, a3;
        const int qr = cw * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
        ldsm4(qbase + swz(qr, ks * 2 + (lane >> 4)), a0, a1, a2, a3);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          u32 b0, b1, b2, b3;
          const int kr = np * 16 + (lane >> 4) * 8 + (lane & 7);
          ldsm4(kb + swz(kr, ks * 2 + ((lane >> 3) & 1)), b0, b1, b2, b3);
          mma(sc[2 * np], a0, a1, a2, a3, b0, b1);
          mma(sc[2 * np + 1], a0, a1, a2, a3, b2, b3);
        }
      }
      mbar_arrive(kv_empty + 8 * s);
      ++it;
      float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int tok = j * TILE + n * 8 + 2 * t + (e & 1), pos = e < 2 ? pa0 : pa1;
          const bool ok = tok <= pos && tok > pos - WIN && tok < M;
          sc[n][e] = ok ? sc[n][e] * LOG2E : NEG_INF;
          if (e < 2) mx0 = fmaxf(mx0, sc[n][e]); else mx1 = fmaxf(mx1, sc[n][e]);
        }
#pragma unroll
      for (int sh = 1; sh <= 2; sh <<= 1) {
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, sh));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, sh));
      }
      const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
      const float u0 = n0 == NEG_INF ? 0.f : n0, u1 = n1 == NEG_INF ? 0.f : n1;
      const float al0 = exp2f(m0 - u0), al1 = exp2f(m1 - u1);
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        sc[n][0] = exp2f(sc[n][0] - u0); sc[n][1] = exp2f(sc[n][1] - u0);
        sc[n][2] = exp2f(sc[n][2] - u1); sc[n][3] = exp2f(sc[n][3] - u1);
        s0 += sc[n][0] + sc[n][1]; s1 += sc[n][2] + sc[n][3];
      }
#pragma unroll
      for (int sh = 1; sh <= 2; sh <<= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, sh);
        s1 += __shfl_xor_sync(0xffffffffu, s1, sh);
      }
      l0 = l0 * al0 + s0; l1 = l1 * al1 + s1; m0 = n0; m1 = n1;
#pragma unroll
      for (int n = 0; n < 32; ++n) { o[n][0] *= al0; o[n][1] *= al0; o[n][2] *= al1; o[n][3] *= al1; }
      u32 pa[2][4];
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        pa[kk][0] = pack_bf2(sc[2 * kk][0], sc[2 * kk][1]); pa[kk][1] = pack_bf2(sc[2 * kk][2], sc[2 * kk][3]);
        pa[kk][2] = pack_bf2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]); pa[kk][3] = pack_bf2(sc[2 * kk + 1][2], sc[2 * kk + 1][3]);
      }
      s = it & 1;
      mbar_wait(kv_full + 8 * s, (it >> 1) & 1);
      const u32 vb = kvb0 + s * KVBYTES;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int dp = 0; dp < 16; ++dp) {
          u32 b0, b1, b2, b3;
          const int vr = kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
          ldsm4t(vb + swz(vr, dp * 2 + (lane >> 4)), b0, b1, b2, b3);
          mma(o[2 * dp], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b0, b1);
          mma(o[2 * dp + 1], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b2, b3);
        }
      mbar_arrive(kv_empty + 8 * s);
      ++it;
    }
    mbar_arrive(q_empty + 8 * qs);
    ++qi;
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int n = 0; n < 32; ++n) {
      const int col = n * 8 + 2 * t;
      if (pa0 < M) *(u32*)(out + ((u64)(pa0 * HQ + head) * D + col)) = pack_bf2(o[n][0] * i0, o[n][1] * i0);
      if (pa1 < M) *(u32*)(out + ((u64)(pa1 * HQ + head) * D + col)) = pack_bf2(o[n][2] * i1, o[n][3] * i1);
    }
  }
}
"""

WS_PREFILL_GLOBAL = WS_COMMON + r"""
// Refiner plans global_prefill p0 (one 8-warp MMA squad owning a cp.async KV ring) and p2 (an S_tile resource:
// partial scores exchanged through shared memory).  Realization: 8 warps = 4 row groups x 2 head-dim halves; each warp
// keeps Q and O for its 256 dims (64 + 128 registers), the pair sums partial Q.K^T through a double-buffered S buffer.
// CTA tile: 64 rows = 8 positions x the 8 Q heads of one KV head; KV ring = K tile and V tile (32 tokens x 512 dims).
#define D 512
#define HQ 32
#define HKV 4
#define PAGE 32
#define BQ 8
#define TILE 32
#define ROWB 1024
#define TBYTES (TILE * ROWB)
#define SXBYTES (8 * 32 * 16 * 4)
DEVI u32 swz1k(int row, int ch) { return (u32)(row * ROWB + ((ch ^ (row & 7)) << 4)); }  // 1 KB rows, 64 chunks
extern "C" __global__ void __launch_bounds__(256, 1)
ws_prefill_global(const u16* __restrict__ q, const u16* __restrict__ kv, const int* __restrict__ table, u16* __restrict__ out,
                  int M, int nunits, int nblk) {
  extern __shared__ __align__(128) unsigned char sm[];
  const u32 ring = smem_u32(sm), sx0 = ring + 2 * TBYTES, bar = sx0 + 2 * SXBYTES;
  const u32 full = bar, empty = bar + 16;
  float* sx = (float*)(sm + 2 * TBYTES);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rg = warp & 3, hh = warp >> 2;                  // row group (16 rows), head-dim half
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) { mbar_init(full + 8 * i, 256); mbar_init(empty + 8 * i, 256); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const float LOG2E = 1.4426950408889634f;
  const int nqb = (M + BQ - 1) / BQ;
  int it = 0, xp = 0;
  for (int u = blockIdx.x; u < nunits; u += gridDim.x) {
    const int qb = nqb - 1 - u / HKV, kvh = u % HKV, p0 = qb * BQ;   // heaviest causal blocks first
    const int j_hi = min(p0 + BQ - 1, M - 1) / TILE;
    // ---- Q (64 rows x 1 KB) staged through the ring once per unit, then held as A fragments of this warp's half
    __syncthreads();
    for (int c = tid; c < 64 * 64; c += 256) {
      const int row = c >> 6, ch = c & 63, pos = p0 + (row & 7), head = kvh * 8 + (row >> 3);
      const int ps = min(pos, M - 1);
      cp16(ring + swz1k(row, ch), q + ((u64)(ps * HQ + head) * D + ch * 8), pos < M ? 16u : 0u);
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    u32 qf[16][4];
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int r = rg * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
      ldsm4(ring + swz1k(r, hh * 32 + ks * 2 + (lane >> 4)), qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3]);
    }
    __syncthreads();
    auto load_tile = [&](int j, int half, int s) {          // half 0 = K rows, 1 = V rows (32 tokens x 512 dims)
      for (int c = tid; c < TILE * 64; c += 256) {
        const int row = c >> 6, ch = c & 63, tok = j * TILE + row;
        const int page = table[min(tok / PAGE, nblk - 1)];
        const u16* src = kv + ((u64)((page * HKV + kvh) * PAGE + (tok % PAGE)) * (2 * D) + half * D + ch * 8);
        cp16(ring + s * TBYTES + swz1k(row, ch), src, tok < M ? 16u : 0u);
      }
      cp_arrive(full + 8 * s);
    };
    const int r0 = rg * 16 + g, r1 = r0 + 8;
    const int pa0 = p0 + (r0 & 7), pa1 = p0 + (r1 & 7);
    const int head0 = kvh * 8 + (r0 >> 3), head1 = kvh * 8 + (r1 >> 3);
    float o[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
    // stage 0 holds K tiles, stage 1 V tiles; each is refilled as soon as every warp released it
    mbar_wait(empty + 0, ((it >> 1) & 1) ^ 1); load_tile(0, 0, 0);
    mbar_wait(empty + 8, ((it >> 1) & 1) ^ 1); load_tile(0, 1, 1);
    for (int j = 0; j <= j_hi; ++j) {
      const u32 par = (it >> 1) & 1;
      mbar_wait(full + 0, par);
      float sc[4][4];
#pragma unroll
      for (int n = 0; n < 4; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 16; ++ks)
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          u32 b0, b1, b2, b3;
          const int kr = np * 16 + (lane >> 4) * 8 + (lane & 7);
          ldsm4(ring + swz1k(kr, hh * 32 + ks * 2 + ((lane >> 3) & 1)), b0, b1, b2, b3);
          mma(sc[2 * np], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
          mma(sc[2 * np + 1], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b2, b3);
        }
      mbar_arrive(empty + 0);
      // ---- S exchange with the partner warp (other head-dim half), double-buffered by tile parity
      float* mine = sx + ((xp & 1) * 8 + warp) * 512 + lane * 16;
      const float* theirs = sx + ((xp & 1) * 8 + (warp ^ 4)) * 512 + lane * 16;
#pragma unroll
      for (int n = 0; n < 4; ++n) *(float4*)(mine + 4 * n) = make_float4(sc[n][0], sc[n][1], sc[n][2], sc[n][3]);
      asm volatile("bar.sync %0, 64;" :: "r"(1 + rg) : "memory");
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const float4 v = *(const float4*)(theirs + 4 * n);
        sc[n][0] += v.x; sc[n][1] += v.y; sc[n][2] += v.z; sc[n][3] += v.w;
      }
      ++xp;
      float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int tok = j * TILE + n * 8 + 2 * t + (e & 1), pos = e < 2 ? pa0 : pa1;
          sc[n][e] = (tok <= pos && tok < M) ? sc[n][e] * LOG2E : NEG_INF;
          if (e < 2) mx0 = fmaxf(mx0, sc[n][e]); else mx1 = fmaxf(mx1, sc[n][e]);
        }
#pragma unroll
      for (int sh = 1; sh <= 2; sh <<= 1) {
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, sh));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, sh));
      }
      const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
      const float u0 = n0 == NEG_INF ? 0.f : n0, u1 = n1 == NEG_INF ? 0.f : n1;
      const float al0 = exp2f(m0 - u0), al1 = exp2f(m1 - u1);
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        sc[n][0] = exp2f(sc[n][0] - u0); sc[n][1] = exp2f(sc[n][1] - u0);
        sc[n][2] = exp2f(sc[n][2] - u1); sc[n][3] = exp2f(sc[n][3] - u1);
        s0 += sc[n][0] + sc[n][1]; s1 += sc[n][2] + sc[n][3];
      }
#pragma unroll
      for (int sh = 1; sh <= 2; sh <<= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, sh);
        s1 += __shfl_xor_sync(0xffffffffu, s1, sh);
      }
      l0 = l0 * al0 + s0; l1 = l1 * al1 + s1; m0 = n0; m1 = n1;
#pragma unroll
      for (int n = 0; n < 32; ++n) { o[n][0] *= al0; o[n][1] *= al0; o[n][2] *= al1; o[n][3] *= al1; }
      u32 pa[2][4];
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        pa[kk][0] = pack_bf2(sc[2 * kk][0], sc[2 * kk][1]); pa[kk][1] = pack_bf2(sc[2 * kk][2], sc[2 * kk][3]);
        pa[kk][2] = pack_bf2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]); pa[kk][3] = pack_bf2(sc[2 * kk + 1][2], sc[2 * kk + 1][3]);
      }
      mbar_wait(full + 8, par);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int dp = 0; dp < 16; ++dp) {
          u32 b0, b1, b2, b3;
          const int vr = kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
          ldsm4t(ring + TBYTES + swz1k(vr, hh * 32 + dp * 2 + (lane >> 4)), b0, b1, b2, b3);
          mma(o[2 * dp], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b0, b1);
          mma(o[2 * dp + 1], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b2, b3);
        }
      mbar_arrive(empty + 8);
      it += 2;
      if (j < j_hi) {                                        // refill: next K into stage 0, next V into stage 1
        mbar_wait(empty + 0, ((it >> 1) & 1) ^ 1); load_tile(j + 1, 0, 0);
        mbar_wait(empty + 8, ((it >> 1) & 1) ^ 1); load_tile(j + 1, 1, 1);
      }
    }
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int n = 0; n < 32; ++n) {
      const int col = hh * 256 + n * 8 + 2 * t;
      if (pa0 < M) *(u32*)(out + ((u64)(pa0 * HQ + head0) * D + col)) = pack_bf2(o[n][0] * i0, o[n][1] * i0);
      if (pa1 < M) *(u32*)(out + ((u64)(pa1 * HQ + head1) * D + col)) = pack_bf2(o[n][2] * i1, o[n][3] * i1);
    }
  }
}
"""

WS_DECODE_GLOBAL = WS_COMMON + r"""
// Refiner plan global_decode_p1 (warpspeed_check pass on first call): squads load(4 warps) -> compute(4 warps),
// V_tile ring 3 x 16 KB (16 tokens x 512 dims of a V-only cache: K = RoPE(w*V) is rebuilt).  Unit = (sequence, KV
// head, segment); rows = the 8 Q heads (16-row MMA, 8 valid).  Each compute warp owns one 128-dim quarter: it scores
// that quarter (pass dims as (q*w).V, rotated dims as q.RoPE(w*V) built once) and the 4 partial scores are summed
// through shared memory; O is kept per quarter.  The load squad also writes each stage's cos/sin table.
#define D 512
#define HALF 256
#define HQ 32
#define HKV 4
#define PAGE 32
#define TILE 16
#define ROWB 1024
#define TB (TILE * ROWB)
#define RB (TILE * 64 * 8)
#define ST 3
DEVI u32 swz1k(int row, int ch) { return (u32)(row * ROWB + ((ch ^ (row & 7)) << 4)); }
DEVI float bfs(const unsigned char* base, int row, int col) {
  const u16 h = *(const u16*)(base + row * ROWB + (((col >> 3) ^ (row & 7)) << 4) + (col & 7) * 2);
  return __uint_as_float(((u32)h) << 16);
}
extern "C" __global__ void __launch_bounds__(256, 1)
ws_decode_global(const u16* __restrict__ q, const u16* __restrict__ w, const float* __restrict__ inv_freq,
                 const u16* __restrict__ vc, const int* __restrict__ table, const int* __restrict__ seqlens,
                 float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
                 int table_stride, int S, int vstride) {
  // vstride = elements between consecutive cached tokens: D for a V-only cache, 2*D when vc points at the V half of
  // vLLM's interleaved [K | V] rows (only V's sectors are then fetched).
  extern __shared__ __align__(128) unsigned char sm[];
  unsigned char* q_s = sm;                                   // 16 x 1 KB: q*w on pass dims, raw q on rotated dims
  unsigned char* v_s = sm + 16 * ROWB;                       // ST x 16 KB V ring
  float* rot = (float*)(v_s + ST * TB);                      // ST x [16 tokens][64 pairs][cos, sin]
  float* sx = rot + ST * TILE * 64 * 2;                      // 2 x [4 warps][32 lanes][8]
  const u32 qa = smem_u32(q_s), vring = smem_u32(v_s), bar = smem_u32(sx + 2 * 4 * 32 * 8);
  const u32 full = bar, empty = bar + 8 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int b = blockIdx.x / (HKV * S), kvh = (blockIdx.x / S) % HKV, seg = blockIdx.x % S;
  const int L = seqlens[b];
  const int tiles = (L + TILE - 1) / TILE, per = (tiles + S - 1) / S;
  const int t_lo = seg * per, t_hi = min(tiles, t_lo + per);
  if (tid == 0) {
    for (int i = 0; i < ST; ++i) { mbar_init(full + 8 * i, 256); mbar_init(empty + 8 * i, 128); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 16 * D; i += 256) {
    const int row = i / D, col = i % D;
    float v = row < 8 ? __uint_as_float(((u32)q[(b * HQ + kvh * 8 + row) * D + col]) << 16) : 0.f;
    const bool rotd = (col < 64) || (col >= HALF && col < HALF + 64);
    if (!rotd) v *= __uint_as_float(((u32)w[col]) << 16);
    *(u16*)(q_s + row * ROWB + (((col >> 3) ^ (row & 7)) << 4) + (col & 7) * 2) = (u16)(pack_bf2(v, 0.f) & 0xffffu);
  }
  __syncthreads();
  int it = 0;
  if (warp < 4) {                                            // ---------------- load squad
    for (int tt = t_lo; tt < t_hi; ++tt, ++it) {
      const int s = it % ST;
      mbar_wait(empty + 8 * s, ((it / ST) & 1) ^ 1);
      for (int c = tid; c < TILE * 64; c += 128) {
        const int row = c >> 6, ch = c & 63, tok = tt * TILE + row;
        const int page = table[b * table_stride + min(tok, L - 1) / PAGE];
        const u16* src = vc + ((u64)((page * HKV + kvh) * PAGE + (tok % PAGE)) * vstride + ch * 8);
        cp16(vring + s * TB + swz1k(row, ch), src, tok < L ? 16u : 0u);
      }
      cp_arrive(full + 8 * s);
      for (int c = tid; c < TILE * 64; c += 128) {          // rotation table of this stage (accurate sincosf)
        const int tok = c >> 6, a = c & 63;
        float sn, cs; sincosf((float)(tt * TILE + tok) * inv_freq[a], &sn, &cs);
        rot[(s * TILE + tok) * 128 + 2 * a] = cs; rot[(s * TILE + tok) * 128 + 2 * a + 1] = sn;
      }
      mbar_arrive(full + 8 * s);
    }
    return;
  }
  // -------------------------------------------------------------- compute squad
  const int cq = warp - 4;
  float wa[16], wb[16];                                     // quarters 0 and 2 rebuild K on rotated pairs
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int a = (i >> 2) * 16 + 2 * t + ((i >> 1) & 1) * 8 + (i & 1);
    wa[i] = __uint_as_float(((u32)w[a]) << 16); wb[i] = __uint_as_float(((u32)w[HALF + a]) << 16);
  }
  float o[16][2];
#pragma unroll
  for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = 0.f;
  float m0 = NEG_INF, l0 = 0.f;
  int xp = 0;
  for (int tt = t_lo; tt < t_hi; ++tt, ++it) {
    const int s = it % ST;
    mbar_wait(full + 8 * s, (it / ST) & 1);
    const u32 vb = vring + s * TB;
    const unsigned char* vbp = v_s + s * TB;
    const float* rt = rot + s * TILE * 128;
    float sc[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int ks = cq * 8 + kk;                          // 16-dim k-step of this warp's quarter
      u32 a0, a1, a2, a3;
      ldsm4(qa + swz1k(((lane >> 3) & 1) * 8 + (lane & 7), ks * 2 + (lane >> 4)), a0, a1, a2, a3);
      const bool rk = (cq == 0 || cq == 2) && kk < 4;       // dims 0-63 (K_a) or 256-319 (K_b)
      if (!rk) {
        u32 b0, b1, b2, b3;
        ldsm4(vb + swz1k((lane >> 4) * 8 + (lane & 7), ks * 2 + ((lane >> 3) & 1)), b0, b1, b2, b3);
        mma(sc[0], a0, a1, a2, a3, b0, b1);
        mma(sc[1], a0, a1, a2, a3, b2, b3);
      } else {
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const int tok = n * 8 + g;
          float kv4[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = kk * 4 + e, a = kk * 16 + 2 * t + (e >> 1) * 8 + (e & 1);
            const float cs = rt[tok * 128 + 2 * a], sn = rt[tok * 128 + 2 * a + 1];
            const float va = wa[i] * bfs(vbp, tok, a), vbv = wb[i] * bfs(vbp, tok, HALF + a);
            kv4[e] = cq == 0 ? va * cs - vbv * sn : vbv * cs + va * sn;
          }
          mma(sc[n], a0, a1, a2, a3, pack_bf2(kv4[0], kv4[1]), pack_bf2(kv4[2], kv4[3]));
        }
      }
    }
    float* mine = sx + ((xp & 1) * 4 + cq) * 256 + lane * 8;
#pragma unroll
    for (int n = 0; n < 2; ++n) *(float4*)(mine + 4 * n) = make_float4(sc[n][0], sc[n][1], sc[n][2], sc[n][3]);
    asm volatile("bar.sync 1, 128;" ::: "memory");
#pragma unroll
    for (int n = 0; n < 2; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < 4; ++w2) {
      const float* src = sx + ((xp & 1) * 4 + w2) * 256 + lane * 8;
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const float4 v = *(const float4*)(src + 4 * n);
        sc[n][0] += v.x; sc[n][1] += v.y; sc[n][2] += v.z; sc[n][3] += v.w;
      }
    }
    ++xp;
    float mx = NEG_INF;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int pos = tt * TILE + n * 8 + 2 * t + e;
        sc[n][e] = pos < L ? sc[n][e] * 1.4426950408889634f : NEG_INF;
        mx = fmaxf(mx, sc[n][e]);
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float nm = fmaxf(m0, mx), um = nm == NEG_INF ? 0.f : nm, al = exp2f(m0 - um);
    float sum = 0.f;
#pragma unroll
    for (int n = 0; n < 2; ++n) { sc[n][0] = exp2f(sc[n][0] - um); sc[n][1] = exp2f(sc[n][1] - um); sum += sc[n][0] + sc[n][1]; }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    l0 = l0 * al + sum; m0 = nm;
#pragma unroll
    for (int n = 0; n < 16; ++n) { o[n][0] *= al; o[n][1] *= al; }
    const u32 p0 = pack_bf2(sc[0][0], sc[0][1]), p2 = pack_bf2(sc[1][0], sc[1][1]);
#pragma unroll
    for (int dp = 0; dp < 8; ++dp) {
      u32 b0, b1, b2, b3;
      float c0[4] = {o[2 * dp][0], o[2 * dp][1], 0.f, 0.f}, c1[4] = {o[2 * dp + 1][0], o[2 * dp + 1][1], 0.f, 0.f};
      ldsm4t(vb + swz1k(((lane >> 3) & 1) * 8 + (lane & 7), cq * 16 + dp * 2 + (lane >> 4)), b0, b1, b2, b3);
      mma(c0, p0, 0u, p2, 0u, b0, b1);
      mma(c1, p0, 0u, p2, 0u, b2, b3);
      o[2 * dp][0] = c0[0]; o[2 * dp][1] = c0[1]; o[2 * dp + 1][0] = c1[0]; o[2 * dp + 1][1] = c1[1];
    }
    mbar_arrive(empty + 8 * s);
  }
  if (g < 8) {
    const int head = kvh * 8 + g, idx = (b * HQ + head) * S + seg;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      const int col = cq * 128 + n * 8 + 2 * t;
      part_o[(u64)idx * D + col] = o[n][0];
      part_o[(u64)idx * D + col + 1] = o[n][1];
    }
    if (cq == 0 && t == 0) { part_m[idx] = m0 == NEG_INF ? NEG_INF : m0 * 0.6931471805599453f; part_l[idx] = l0; }
  }
}
"""

WS_DECODE_SLIDING = WS_COMMON + r"""
// Refiner plan sliding_decode_p1 (warpspeed_check pass): squads load(1 warp) -> compute(3 warps);
// q_tile, k_v_tile ring of 16 KB stages (one vLLM page = 16 tokens x [K 256 | V 256], contiguous per KV head),
// segment_partial written by the compute squad and drained by the load squad (in-CTA combine of the 3 warps).
// Deviation from the plan's 2 stages: 3 stages, stage c owned by compute warp c (tiles round-robin), so no warp ever
// waits on a barrier two phases ahead (with 2 stages and 3 consumers, try_wait.parity aliases and deadlocks).
// Unit = (sequence, KV head, segment of the 1024-token window); rows = the 2 Q heads (16-row MMA, 2 valid).
#define D 256
#define HQ 32
#define HKV 16
#define PAGE 16
#define WIN 1024
#define TB (PAGE * 1024)
#define NC 3
DEVI u32 swz512(int row, int ch) { return (u32)(row * 512 + ((ch ^ (row & 7)) << 4)); }
extern "C" __global__ void __launch_bounds__(128, 1)
ws_decode_sliding(const u16* __restrict__ q, const u16* __restrict__ kv, const int* __restrict__ table,
                  const int* __restrict__ seqlens, u16* __restrict__ out, float* __restrict__ part_o,
                  float* __restrict__ part_m, float* __restrict__ part_l, int table_stride, int S) {
  extern __shared__ __align__(128) unsigned char sm[];
  const u32 qbuf = smem_u32(sm), kvr = qbuf + 16 * 512, bar = kvr + NC * TB;
  float* part = (float*)(sm + 16 * 512 + NC * TB + 64);             // [NC][2 rows][D + 2] (O, m, l)
  const u32 full = bar, empty = bar + 8 * NC, pdone = bar + 16 * NC;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int b = blockIdx.x / (HKV * S), kvh = (blockIdx.x / S) % HKV, seg = blockIdx.x % S;
  const int L = seqlens[b], first = max(0, L - WIN);
  const int p_lo = first / PAGE, p_n = (L - 1) / PAGE - p_lo + 1, per = (p_n + S - 1) / S;
  const int t_lo = p_lo + seg * per, t_hi = min(p_lo + p_n, t_lo + per);
  if (tid == 0) {
    for (int i = 0; i < NC; ++i) { mbar_init(full + 8 * i, 32); mbar_init(empty + 8 * i, 32); }
    mbar_init(pdone, 32 * NC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 16 * 32; i += 128) {                         // Q: rows 0,1 = the two heads, rest zero
    const int row = i >> 5, ch = i & 31;
    const u16* src = q + ((u64)(b * HQ + kvh * 2 + min(row, 1)) * D + ch * 8);
    cp16(qbuf + swz512(row, ch), src, row < 2 ? 16u : 0u);
  }
  asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  if (warp == 0) {                                                   // ---------------- load squad (1 warp)
    for (int tt = t_lo, i = 0; tt < t_hi; ++tt, ++i) {
      const int s = i % NC;
      mbar_wait(empty + 8 * s, ((i / NC) & 1) ^ 1);
      const int page = table[b * table_stride + tt];
      const u16* src = kv + (u64)(page * HKV + kvh) * PAGE * (2 * D);  // the whole page: 16 x [K | V]
      for (int c = lane; c < PAGE * 64; c += 32) {
        const int row = c >> 6, ch = c & 63;                         // row = token, ch over K(0-31) | V(32-63)
        cp16(kvr + s * TB + (ch < 32 ? 0 : PAGE * 512) + swz512(row, ch & 31), src + (u64)c * 8, 16u);
      }
      cp_arrive(full + 8 * s);
    }
    mbar_wait(pdone, 0);                                             // drain segment_partial: merge 3 warps
    for (int r = 0; r < 2; ++r) {
      float mx = NEG_INF;
      for (int w2 = 0; w2 < NC; ++w2) mx = fmaxf(mx, part[(w2 * 2 + r) * (D + 2) + D]);
      float wsum = 0.f, wt[NC];
      for (int w2 = 0; w2 < NC; ++w2) {
        const float m = part[(w2 * 2 + r) * (D + 2) + D];
        wt[w2] = m == NEG_INF ? 0.f : exp2f(m - mx);
        wsum += wt[w2] * part[(w2 * 2 + r) * (D + 2) + D + 1];
      }
      const int head = kvh * 2 + r;
      for (int c = lane; c < D; c += 32) {
        float acc = 0.f;
        for (int w2 = 0; w2 < NC; ++w2) acc += wt[w2] * part[(w2 * 2 + r) * (D + 2) + c];
        if (S == 1) out[(u64)(b * HQ + head) * D + c] = (u16)(pack_bf2(acc / wsum, 0.f) & 0xffffu);
        else part_o[(u64)((b * HQ + head) * S + seg) * D + c] = acc;
      }
      if (S > 1 && lane == 0) {
        const int idx = (b * HQ + head) * S + seg;
        part_m[idx] = mx == NEG_INF ? NEG_INF : mx * 0.6931471805599453f; part_l[idx] = wsum;
      }
    }
    return;
  }
  // -------------------------------------------------------------- compute squad (3 warps, tiles round-robin)
  const int cw = warp - 1;
  u32 qf[16][4];
#pragma unroll
  for (int ks = 0; ks < 16; ++ks)
    ldsm4(qbuf + swz512(((lane >> 3) & 1) * 8 + (lane & 7), ks * 2 + (lane >> 4)), qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3]);
  float o[32][2];
#pragma unroll
  for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = 0.f;
  float m0 = NEG_INF, l0 = 0.f;
  for (int tt = t_lo, i = 0; tt < t_hi; ++tt, ++i) {
    if (i % NC != cw) continue;
    const int s = cw;
    mbar_wait(full + 8 * s, (i / NC) & 1);
    const u32 kb = kvr + s * TB, vb = kb + PAGE * 512;
    float sc[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      u32 b0, b1, b2, b3;
      ldsm4(kb + swz512((lane >> 4) * 8 + (lane & 7), ks * 2 + ((lane >> 3) & 1)), b0, b1, b2, b3);
      mma(sc[0], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
      mma(sc[1], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b2, b3);
    }
    float mx = NEG_INF;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int tok = tt * PAGE + n * 8 + 2 * t + e;
        sc[n][e] = (tok >= first && tok < L) ? sc[n][e] * 1.4426950408889634f : NEG_INF;
        mx = fmaxf(mx, sc[n][e]);
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float nm = fmaxf(m0, mx), um = nm == NEG_INF ? 0.f : nm, al = exp2f(m0 - um);
    float sum = 0.f;
#pragma unroll
    for (int n = 0; n < 2; ++n) { sc[n][0] = exp2f(sc[n][0] - um); sc[n][1] = exp2f(sc[n][1] - um); sum += sc[n][0] + sc[n][1]; }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    l0 = l0 * al + sum; m0 = nm;
#pragma unroll
    for (int n = 0; n < 32; ++n) { o[n][0] *= al; o[n][1] *= al; }
    const u32 p0 = pack_bf2(sc[0][0], sc[0][1]), p2 = pack_bf2(sc[1][0], sc[1][1]);
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      u32 b0, b1, b2, b3;
      float c0[4] = {o[2 * dp][0], o[2 * dp][1], 0.f, 0.f}, c1[4] = {o[2 * dp + 1][0], o[2 * dp + 1][1], 0.f, 0.f};
      ldsm4t(vb + swz512(((lane >> 3) & 1) * 8 + (lane & 7), dp * 2 + (lane >> 4)), b0, b1, b2, b3);
      mma(c0, p0, 0u, p2, 0u, b0, b1);
      mma(c1, p0, 0u, p2, 0u, b2, b3);
      o[2 * dp][0] = c0[0]; o[2 * dp][1] = c0[1]; o[2 * dp + 1][0] = c1[0]; o[2 * dp + 1][1] = c1[1];
    }
    mbar_arrive(empty + 8 * s);
  }
  // publish this warp's partial (rows 0,1 live in lanes with g < 2) into segment_partial, then signal the load squad
  if (g < 2) {
    float* dst = part + (cw * 2 + g) * (D + 2);
#pragma unroll
    for (int n = 0; n < 32; ++n) { dst[n * 8 + 2 * t] = o[n][0]; dst[n * 8 + 2 * t + 1] = o[n][1]; }
    if (t == 0) { dst[D] = m0; dst[D + 1] = l0; }
  }
  mbar_arrive(pdone);
}
"""

_WS = {}


def ws_module(name, src):
    """Compile once per process with the MLP round bench's NVRTC/driver path (the user's toolchain)."""
    if name not in _WS:
        if __package__:
            from . import mlp as fb
        else:
            import mlp as fb
        fb._setup_torch()
        drv, nv = fb.Driver(), fb.NVRTC()
        cc = torch.cuda.get_device_capability()
        cubin, log = nv.compile(src, f"sm_{cc[0]}{cc[1]}")
        _WS[name] = (fb, drv, drv.load(cubin), log)
    return _WS[name]


class WSPrefillGlobal:
    name = "ws_prefill_global"
    SMEM = 2 * 32 * 1024 + 2 * 8 * 32 * 16 * 4 + 32

    def __init__(self, case):
        fb, drv, mod, _ = ws_module("prefill_global", WS_PREFILL_GLOBAL)
        self.k = fb.Kern(drv, mod, "ws_prefill_global", 256, self.SMEM)
        self.c = case
        self.out = torch.empty(case["M"], HQ, case["d"], dtype=torch.bfloat16, device="cuda")
        self.units = -(-case["M"] // 8) * case["hkv"]
        self.grid = min(self.units, torch.cuda.get_device_properties(0).multi_processor_count)

    def __call__(self):
        c = self.c
        self.k(self.grid, [c["q"], c["k"], c["table"], self.out, c["M"], self.units, c["table"].shape[1]])
        return self.out


class WSDecodeGlobal:
    """Refiner plan global_decode p1 realized: 4-warp load squad -> 4-warp compute squad over a V-only cache."""
    name = "ws_decode_global"
    SMEM = 16 * 1024 + 3 * 16 * 1024 + 3 * 16 * 64 * 8 + 2 * 4 * 32 * 8 * 4 + 48

    def __init__(self, case, extra, segments):
        fb, drv, mod, _ = ws_module("decode_global", WS_DECODE_GLOBAL)
        self.k = fb.Kern(drv, mod, "ws_decode_global", 256, self.SMEM)
        self.c, self.x, self.seg = case, extra, segments
        B, d = case["B"], case["d"]
        self.po = torch.empty(B * HQ * segments, d, dtype=torch.float32, device="cuda")
        self.pm = torch.empty(B * HQ * segments, dtype=torch.float32, device="cuda")
        self.pl = torch.empty(B * HQ * segments, dtype=torch.float32, device="cuda")
        self.out = torch.empty(B, HQ, d, dtype=torch.bfloat16, device="cuda")

    def __call__(self):
        c, x = self.c, self.x
        self.k(c["B"] * c["hkv"] * self.seg, [c["q"], x["w"], x["inv"], x["vcache"], c["table"], c["seqlens"],
                                              self.po, self.pm, self.pl, c["table"].stride(0), self.seg, c["d"]])
        _combine[(c["B"] * HQ,)](self.po, self.pm, self.pl, self.out, SEGMENTS=self.seg, D=c["d"], num_warps=4)
        return self.out


class WSDecodeSliding:
    """Refiner plan sliding_decode p1 realized: 1 load warp -> 3 compute warps, in-CTA combine by the load warp."""
    name = "ws_decode_sliding"
    SMEM = 16 * 512 + 3 * 16 * 1024 + 64 + 3 * 2 * 258 * 4

    def __init__(self, case, segments):
        fb, drv, mod, _ = ws_module("decode_sliding", WS_DECODE_SLIDING)
        self.k = fb.Kern(drv, mod, "ws_decode_sliding", 128, self.SMEM)
        self.c, self.seg = case, segments
        B, d = case["B"], case["d"]
        self.po = torch.empty(B * HQ * segments, d, dtype=torch.float32, device="cuda")
        self.pm = torch.empty(B * HQ * segments, dtype=torch.float32, device="cuda")
        self.pl = torch.empty(B * HQ * segments, dtype=torch.float32, device="cuda")
        self.out = torch.empty(B, HQ, d, dtype=torch.bfloat16, device="cuda")

    def __call__(self):
        c = self.c
        self.k(c["B"] * c["hkv"] * self.seg, [c["q"], c["k"], c["table"], c["seqlens"], self.out, self.po, self.pm, self.pl,
                                              c["table"].stride(0), self.seg])
        if self.seg > 1:
            _combine[(c["B"] * HQ,)](self.po, self.pm, self.pl, self.out, SEGMENTS=self.seg, D=c["d"], num_warps=4)
        return self.out


class WSPrefillSliding:
    name = "ws_prefill_sliding"
    SMEM = 2 * 64 * 512 + 2 * 32 * 512 + 64

    def __init__(self, case):
        fb, drv, mod, _ = ws_module("prefill_sliding", WS_PREFILL_SLIDING)
        self.k = fb.Kern(drv, mod, "ws_prefill_sliding", 256, self.SMEM)
        self.c = case
        self.out = torch.empty(case["M"], HQ, case["d"], dtype=torch.bfloat16, device="cuda")
        self.units = -(-case["M"] // 32) * case["hkv"]
        self.grid = min(self.units, torch.cuda.get_device_properties(0).multi_processor_count)

    def __call__(self):
        c = self.c
        self.k(self.grid, [c["q"], c["k"], c["table"], self.out, c["M"], self.units, c["table"].shape[1]])
        return self.out


def time_graph(fn, warmup=10, samples=7, target_ms=20.0):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        g.replay()
    e0.record(); g.replay(); e1.record(); e1.synchronize()
    reps = max(1, min(2000, int(target_ms / max(e0.elapsed_time(e1), 1e-3))))
    ts = []
    for _ in range(samples):
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1) / reps)
    ts.sort()
    return dict(median_ms=ts[len(ts) // 2], min_ms=ts[0], reps=reps)


def check(out, ref):
    o = out.float()
    err = (o - ref).norm() / ref.norm()
    return dict(rel_l2=float(err), max_abs=float((o - ref).abs().max()), finite=bool(torch.isfinite(o).all()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, nargs="*", default=[1, 8])
    ap.add_argument("--L", type=int, nargs="*", default=[4096, 32768, 131072])
    ap.add_argument("--kinds", nargs="+", default=["sliding", "global"])
    ap.add_argument("--prefill-M", type=int, nargs="*", default=[7131, 15329])
    ap.add_argument("--out", default="attn.json")
    ap.add_argument("--designs", action="store_true", help="measure the implemented designs next to the seed")
    ap.add_argument("--prefill-source", choices=("planner", "refiner"), default="planner")
    ap.add_argument("--ws", action="store_true", help="measure the WarpSpeed kernels (refiner plans)")
    a = ap.parse_args()
    if a.designs:
        return designs(a)
    rows = []
    for kind in a.kinds:
        for B in a.B:
            for L in a.L:
                # Consecutive layers never share KV: rotate through enough independent caches to exceed L2 4x,
                # as a decode step walks 50 sliding / 10 global layers.  Time is per layer.
                case = make_case(kind, B, L)
                n = max(1, min(16, -(-L2_BYTES * 4 // case["k"].untyped_storage().nbytes())))
                cases = [case] + [make_case(kind, B, L, seed=i) for i in range(1, n)]
                incs = [Incumbent(c) for c in cases]
                ref = reference(case)
                t = time_graph(lambda: [f() for f in incs])
                t = {k: v / n if k.endswith("_ms") else v for k, v in t.items()}
                row = dict(kind=kind, phase="decode", B=B, L=L, **sol_ms(case), rotated_caches=n, incumbent=dict(
                    path="3D" if B <= incs[0].thr else "2D", segments=incs[0].seg, **check(incs[0](), ref), **t))
                row["incumbent"]["pct_of_sol"] = 100 * row["sol_ms"] / row["incumbent"]["median_ms"]
                rows.append(row)
                print(json.dumps(row), flush=True)
                del cases, case, incs, ref
                torch.cuda.empty_cache()
    for kind in a.kinds:
        for M in a.prefill_M:
            case = make_prefill(kind, M)
            inc = Incumbent(case)
            idx, ref = prefill_reference(case)
            t = time_graph(inc, warmup=3, samples=5)
            row = dict(kind=kind, phase="prefill", M=M, **prefill_sol(case), incumbent=dict(
                path="2D", **check(inc()[idx], ref), **t))
            row["incumbent"]["pct_of_sol"] = 100 * row["sol_ms"] / row["incumbent"]["median_ms"]
            row["incumbent"]["tflops"] = row["flops"] / (row["incumbent"]["median_ms"] * 1e9)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del case, inc, ref
            torch.cuda.empty_cache()
    json.dump(dict(device=torch.cuda.get_device_name(), rows=rows, compiled=compiled_evidence()), open(a.out, "w"), indent=1)


def compiled_evidence():
    """Registers, spills, shared memory, warps and tile constants of every Triton kernel compiled in this process."""
    from vllm.v1.attention.ops import triton_unified_attention as tua
    keep = ("TILE_SIZE", "BLOCK_M", "BLOCK_Q", "HEAD_SIZE", "BLOCK_SIZE", "IS_3D", "NUM_SEGMENTS_PER_SEQ", "SLIDING_WINDOW")
    out = []
    for fn in (tua.kernel_unified_attention, tua.reduce_segments):
        for dev_cache in fn.device_caches.values():
            for kern in dev_cache[0].values():
                md = kern.metadata
                const = {}
                for key, val in (getattr(kern.src, "constants", None) or {}).items():
                    name = key if isinstance(key, str) else fn.arg_names[key[0] if isinstance(key, tuple) else key]
                    if name in keep:
                        const[name] = val
                out.append(dict(kernel=fn.__name__, n_regs=getattr(kern, "n_regs", None), n_spills=getattr(kern, "n_spills", None),
                                shared=md.shared, num_warps=md.num_warps, num_stages=md.num_stages, constants=const,
                                loaded=hasattr(kern, "n_regs")))
    return out


if __name__ == "__main__":
    main()
