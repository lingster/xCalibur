#!/usr/bin/env python3
"""
mlp.py -- SM120 (RTX PRO 6000 Blackwell SE / GCP G4) W4A16 dense MLP + rank-16 LoRA.

Contract: readme-sm120-dense-mlp-w4a16-lora16-v1
  H=5376  I=21504  r=16  alpha=32 (scale 2)  INT4 sym group-32 along K, BF16 scales, offset 8
  g   = bf(bf(X@Wg^T) + bf(2*bf(bf(X@Ag^T)@Bg^T)))
  u   = bf(bf(X@Wu^T) + bf(2*bf(bf(X@Au^T)@Bu^T)))
  mid = bf(bf(GELU_tanh(g)) * u)
  Y   = bf(bf(mid@Wd^T) + bf(2*bf(bf(mid@Ad^T)@Bd^T)))
  gate: finite, |Y-Yref| <= 1e-5 + 0.016*|Yref| elementwise.  Weights decoded inside every forward.

Designs
  ct_ref       oracle: torch decode + cuBLAS + eager pointwise (also timed as the baseline)
  forge_tc     M>=64: two fused INT4-dequant mma.sync GEMMs, LoRA-B + add + GELU*up fused in the GU epilogue,
               LoRA-B + add fused in the Down epilogue, rank-16 LoRA-A factors from small tensor-core kernels.
               One graph, four kernels, PDL-chained: lowk_gu -> GU (mainloop overlaps, waits before epilogue)
               -> lowk_d (waits) -> Down (mainloop overlaps lowk_d).  Per-stage autotune, then the top configs
               are re-timed as whole chains (with / without PDL) and the fastest real forward wins.  Over:
                 v1  cp.async multistage, 1 tile/CTA, 2 CTAs/SM
                 v3  CCCL-warpspeed squads: 2 MMA squads + 1 producer warp (TMA streams for both squads and the
                     CTA's single CLC issuer, non-blocking), 288 threads at 224 regs (no spills), mbarrier
                     stage x phase rings, Cluster-Launch-Control work stealing
                     (or static persistent), '+sk' = stream-K tail (last partial wave split along K,
                     deterministic in-order fp32 reduction).  '_w' configs: 128-row warps, so each INT4 word
                     is dequantized once per tile (not once per M-warp) and feeds 8 MMAs: ~1.0 dequant
                     ALU op per MMA vs ~2.0.  dq_* kernels skip the dequant to measure what it costs.
  forge_tc_f16 fp16 datapath (fp16 operands, fp16 MMA accumulation promoted to fp32 every stage). Runs only
               when this card's measured f16-acc MMA rate beats f32-acc (or --f16 on); numerics relaxed.
  forge_gemv8  M=8: operand-swapped tensor-core GEMV (weights = MMA A operand, the 8 tokens = N), deep
               register prefetch, 4 kernels chained with programmatic dependent launch inside one graph.
               Weights sit in one arena; the L2 is told what to keep resident across forwards (access-policy
               window with a persisting set-aside, or createpolicy evict_last/evict_first hints) -- autotuned.
  dec_cublas   M>=64 control: INT4->BF16 decode kernel + cuBLAS bases + fused LoRA-B/act epilogue kernels.

Usage:  python mlp.py                       (all M, all designs, real weights if findable)
        python mlp.py --M 8 15329 --synthetic
Needs: torch (CUDA), libcuda, libnvrtc (ships with torch CUDA wheels). No nvcc required.
"""
import argparse
import ctypes
import glob
import json
import math
import os
import struct
import sys
import time

torch = None
F = None

H, I, R, ALPHA = 5376, 21504, 16, 32
SEED = 20261022
REPO = "google/gemma-4-31B-it-qat-w4a16-ct"
REV = None  # pin a model revision for reproducible runs
TN = {
    "Qg": "model.language_model.layers.0.mlp.gate_proj.weight_packed",
    "Sg": "model.language_model.layers.0.mlp.gate_proj.weight_scale",
    "Qu": "model.language_model.layers.0.mlp.up_proj.weight_packed",
    "Su": "model.language_model.layers.0.mlp.up_proj.weight_scale",
    "Qd": "model.language_model.layers.0.mlp.down_proj.weight_packed",
    "Sd": "model.language_model.layers.0.mlp.down_proj.weight_scale",
}

# --------------------------------------------------------------------------------------------------
# CUDA source (compiled once with NVRTC). No CUDA headers: raw PTX for every non-trivial op.
# --------------------------------------------------------------------------------------------------
CUDA_COMMON = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
#define DEVI static __device__ __forceinline__
#define KS_H  (FG_H / 16)
#define KS_I  (FG_I / 16)

// ------------------------------------------------------------------ primitives
DEVI u32 smem_u32(const void* p) {
  u32 r;
  asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }\n" : "=r"(r) : "l"(p));
  return r;
}
DEVI void cp16(u32 dst, const void* src, u32 bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

DEVI void ldsm4(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEVI void lds128(u32 a, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEVI void lds64(u32 a, u32& r0, u32& r1) {
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
DEVI u32 lds32(u32 a) {
  u32 r;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(r) : "r"(a));
  return r;
}
DEVI void ldg_nc_v4(const void* p, u32 (&r)[4]) {
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}
DEVI void ldg_cg_v4(const void* p, u32 (&r)[4]) {
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}
// L2 eviction-priority policies (PTX createpolicy, sm_80+): weights we want resident across forwards are loaded
// evict_last, the remainder evict_first so the streamed bytes never displace the resident set.
DEVI unsigned long long l2_policy(bool keep) {
  unsigned long long pol;
  if (keep) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
  else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
template <int HINT> DEVI void ldw_v4(const void* p, u32 (&r)[4], unsigned long long pol) {
  if (HINT) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p), "l"(pol));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
  }
}
DEVI u32 ldg_nc32(const void* p) {
  u32 r;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}
DEVI u32 ldg_cg32(const void* p) {
  u32 r;
  asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}
DEVI float ldg_cg_f32(const void* p) {
  float r;
  asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(r) : "l"(p));
  return r;
}
template <int CG> DEVI u32 ldz(const u16* p) { return CG ? ldg_cg32(p) : ldg_nc32(p); }
DEVI void stg32(void* p, u32 v) { asm volatile("st.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }
DEVI void stg16(void* p, u32 v) {
  u16 h = (u16)v;
  asm volatile("st.global.u16 [%0], %1;\n" ::"l"(p), "h"(h) : "memory");
}
DEVI void stg128(void* p, u32 a, u32 b, u32 c, u32 d) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
DEVI u32 atom_add(u32* p, u32 v) {
  u32 r;
  asm volatile("atom.global.add.u32 %0, [%1], %2;\n" : "=r"(r) : "l"(p), "r"(v) : "memory");
  return r;
}
DEVI void pdl_wait() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
DEVI void pdl_launch() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

DEVI void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI u32 lop3_ea(u32 a, u32 b, u32 c) {  // (a & b) | c
  u32 d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
DEVI u32 bfma2(u32 a, u32 b, u32 c) {
  u32 d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
DEVI u32 prmt(u32 a, u32 sel) {
  u32 d;
  asm("prmt.b32 %0, %1, %1, %2;\n" : "=r"(d) : "r"(a), "r"(sel));
  return d;
}
// nibbles at bits [0,4) and [16,20) of w -> bf16x2 of bf((q-8)*s) (exact: one RN rounding of an exact product)
DEVI u32 dq2(u32 w, u32 s2) {
  const u32 p = lop3_ea(w, 0x000F000Fu, 0x43004300u);   // bf16(128+q)
  const u32 t = bfma2(p, 0x3F803F80u, 0xC308C308u);     // (128+q)*1 - 136 = q-8 (exact)
  return bfma2(t, s2, 0x80008000u);                      // (q-8)*s + (-0) -> RN
}
// ---- fp16 datapath (opt-in, numerics-relaxed): fp16 operands, fp16 MMA accumulation promoted to fp32 per stage
DEVI void mma16816_hacc(u32 (&c)[2], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI void mma16816_h(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {  // fp16 in, fp32 accumulate
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI u32 hfma2(u32 a, u32 b, u32 c) {
  u32 d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
// nibbles at bits [0,4) and [16,20) -> f16x2 of (q-8)*s (exact: 4-bit x 11-bit product fits the fp16 significand)
DEVI u32 dq2h(u32 w, u32 s2) {
  const u32 p = lop3_ea(w, 0x000F000Fu, 0x64006400u);   // fp16(1024+q)
  const u32 t = hfma2(p, 0x3C003C00u, 0xE408E408u);     // (1024+q)*1 - 1032 = q-8
  return hfma2(t, s2, 0x80008000u);
}
// one packed word (nibble i at bits 4i) -> the 4 f16x2 B registers of two n8 fragments; 1 shift instead of 3:
// nibbles 1,5,3,7 are extracted in place as 16q (fp16 1024+16q) and rescaled by the offset FMA (x 1/16 - 72).
DEVI void dq4h(u32 x, u32 s0, u32 s1, u32& b00, u32& b01, u32& b10, u32& b11) {
  const u32 y = x >> 8;
  b00 = hfma2(hfma2(lop3_ea(x, 0x000F000Fu, 0x64006400u), 0x3C003C00u, 0xE408E408u), s0, 0x80008000u);
  b01 = hfma2(hfma2(lop3_ea(x, 0x00F000F0u, 0x64006400u), 0x2C002C00u, 0xD480D480u), s0, 0x80008000u);
  b10 = hfma2(hfma2(lop3_ea(y, 0x000F000Fu, 0x64006400u), 0x3C003C00u, 0xE408E408u), s1, 0x80008000u);
  b11 = hfma2(hfma2(lop3_ea(y, 0x00F000F0u, 0x64006400u), 0x2C002C00u, 0xD480D480u), s1, 0x80008000u);
}
DEVI void h2f2(u32 v, float& lo, float& hi) {
  asm("{.reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h;}\n" : "=f"(lo), "=f"(hi) : "r"(v));
}
DEVI u32 pack_h2(float lo, float hi) {
  u32 d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
DEVI float u2f(u32 u) {
  float f;
  asm("mov.b32 %0, %1;\n" : "=f"(f) : "r"(u));
  return f;
}
DEVI float bf_lo(u32 v) { return u2f(v << 16); }
DEVI float bf_hi(u32 v) { return u2f(v & 0xFFFF0000u); }
DEVI u32 f2bf(float f) {
  u16 h;
  asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(h) : "f"(f));
  return (u32)h;
}
DEVI float rbf(float f) { return u2f(f2bf(f) << 16); }
DEVI u32 pack_bf2(float lo, float hi) { return f2bf(lo) | (f2bf(hi) << 16); }

// PyTorch's CUDA GELU(approximate='tanh') in opmath float, same expression tree (fmad contraction as nvcc).
DEVI float gelu_tanh_ref(float x) {
  const float kBeta = 0.7978845608028654f;
  const float kKappa = 0.044715f;
  float x_cube = x * x * x;
  float inner = kBeta * (x + kKappa * x_cube);
  return 0.5f * x * (1.0f + tanhf(inner));
}
// contract cuts for one (row, col) of the GU stage; returns un-rounded a*u (caller rounds = mid cut)
DEVI float gu_elem(float acc_g, float acc_u, float lg_acc, float lu_acc) {
  const float g0 = rbf(acc_g), u0 = rbf(acc_u);
  const float lg = rbf(2.0f * rbf(lg_acc)), lu = rbf(2.0f * rbf(lu_acc));
  const float g = rbf(g0 + lg), u = rbf(u0 + lu);
  const float a = rbf(gelu_tanh_ref(g));
  return a * u;
}
DEVI float dn_elem(float acc_d, float ld_acc) { return rbf(acc_d) + rbf(2.0f * rbf(ld_acc)); }

template <int BK> DEVI int swz_x(int r8) { return BK == 64 ? r8 : ((r8 >> 1) & 3); }

template <int WORDS> DEVI void lds_words(u32 a, u32 (&w)[4]) {
  if (WORDS == 4) lds128(a, w[0], w[1], w[2], w[3]);
  else if (WORDS == 2) lds64(a, w[0], w[1]);
  else w[0] = lds32(a);
}

// A-operand fragment of a rank-16 low-factor block: rows r_lo / r_hi, k = 2t.. and 8+2t.. (zero past M)
DEVI void ld_lowfrag(const u16* __restrict__ low, int ldl, int r_lo, int r_hi, int M, int t, u32 (&a)[4]) {
  const bool okl = r_lo < M, okh = r_hi < M;
  const u16* pl = low + (u64)(okl ? r_lo : 0) * (u64)ldl + 2 * t;
  const u16* ph = low + (u64)(okh ? r_hi : 0) * (u64)ldl + 2 * t;
  a[0] = okl ? ldg_cg32(pl) : 0u;
  a[1] = okh ? ldg_cg32(ph) : 0u;
  a[2] = okl ? ldg_cg32(pl + 8) : 0u;
  a[3] = okh ? ldg_cg32(ph + 8) : 0u;
}

DEVI void tile_coords(int bid, int ntm, int ntn, int& tm, int& tn) {  // grouped raster, GM=8 M-tiles per group
  const int GM = 8;
  const int per_group = GM * ntn;
  const int group = bid / per_group;
  const int first = group * GM;
  const int gsz = min(ntm - first, GM);
  const int local = bid - group * per_group;
  tm = first + local % gsz;
  tn = local / gsz;
}
"""

CUDA_V1 = r"""
// ------------------------------------------------------------------ large-M fused GEMM
// A [M,K] bf16 row-major (X or mid).  B packed "TC" layout: [N/WN][K/16][32 lanes][WN/16 u32]; scales
// [N/WN][K/32][8][WN/8] bf16.  Per lane (g=l/4,t=l%4) word w holds n-frags 2w,2w+1 of the k16 step:
// nibble i=4e+2jj+h  <->  (frag 2w+jj, row g, k=8h+2t+e).  Exactly the mma B fragment, no shuffles.
template <int BM, int BN, int WM, int WN, int BK, int ST, int EPI>
struct TcCfg {
  static constexpr int BMv = BM;
  static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, NT = WARPS_M * WARPS_N * 32;
  static constexpr int MFR = WM / 16, NFR = WN / 8, KSTEPS = BK / 16, WORDS = NFR / 2, NBLK = WARPS_N;
  static constexpr int CH = BK / 8;
  static constexpr int A_ST = BM * BK * 2;
  static constexpr int B_BLK = KSTEPS * 32 * WORDS * 4;
  static constexpr int B_ST = NBLK * B_BLK;
  static constexpr int S_BLK = (BK / 32) * 8 * NFR * 2;
  static constexpr int S_ST = NBLK * S_BLK;
  static constexpr int STAGE = A_ST + B_ST + S_ST;
  static constexpr int NCOL = EPI == 0 ? BN / 2 : BN;
  static constexpr int SMEM = ST * STAGE;
};


template <class C, int BK>
DEVI void tc_load_stage(u32 sa, const u16* __restrict__ A, const u32* __restrict__ Bq, const u16* __restrict__ Bs,
                        int tid, int m0, int M, int K, int kt, int nb0, int KS_TOT, int KG_TOT) {
  const u32 sb = sa + C::A_ST, ss = sb + C::B_ST;
  constexpr int AC = C::BMv * C::CH;
#pragma unroll
  for (int i = 0; i < (AC + C::NT - 1) / C::NT; ++i) {
    const int c = tid + i * C::NT;
    if ((AC % C::NT) == 0 || c < AC) {
      const int r = c / C::CH, ch = c % C::CH;
      const int gr = m0 + r;
      const bool ok = gr < M;
      const u16* src = A + (u64)(ok ? gr : 0) * (u64)K + (u64)(kt * BK + ch * 8);
      cp16(sa + r * (BK * 2) + ((ch ^ swz_x<BK>(r & 7)) << 4), src, ok ? 16u : 0u);
    }
  }
  constexpr int CPB = C::B_BLK / 16;
  constexpr int BC = C::NBLK * CPB;
#pragma unroll
  for (int i = 0; i < (BC + C::NT - 1) / C::NT; ++i) {
    const int c = tid + i * C::NT;
    if ((BC % C::NT) == 0 || c < BC) {
      const int b = c / CPB, o = c % CPB;
      const char* src = (const char*)Bq + ((u64)(nb0 + b) * (u64)KS_TOT + (u64)(kt * C::KSTEPS)) * (u64)(32 * C::WORDS * 4) + o * 16;
      cp16(sb + b * C::B_BLK + o * 16, src, 16u);
    }
  }
  constexpr int CPS = C::S_BLK / 16;
  constexpr int SC = C::NBLK * CPS;
#pragma unroll
  for (int i = 0; i < (SC + C::NT - 1) / C::NT; ++i) {
    const int c = tid + i * C::NT;
    if ((SC % C::NT) == 0 || c < SC) {
      const int b = c / CPS, o = c % CPS;
      const char* src = (const char*)Bs + ((u64)(nb0 + b) * (u64)KG_TOT + (u64)(kt * (BK / 32))) * (u64)(8 * C::NFR * 2) + o * 16;
      cp16(ss + b * C::S_BLK + o * 16, src, 16u);
    }
  }
}

template <int BM, int BN, int WM, int WN, int BK, int ST, int EPI>
DEVI void tc_gemm(const u16* __restrict__ A, const u32* __restrict__ Bq, const u16* __restrict__ Bs,
                  const u16* __restrict__ low0, const u16* __restrict__ low1,
                  const u16* __restrict__ lb0, const u16* __restrict__ lb1,
                  u16* __restrict__ out, int M, int K, int N, int ldo, int ldl, unsigned char* smem) {
  typedef TcCfg<BM, BN, WM, WN, BK, ST, EPI> C;
  static_assert(C::WORDS >= 1 && C::WORDS <= 4, "WN");
  static_assert(BK == 32 || BK == 64, "BK");
  static_assert(C::SMEM <= 101376, "smem");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wn = warp % C::WARPS_N, wm = warp / C::WARPS_N;
  const int g = lane >> 2, t = lane & 3;
  const int KT = K / BK, KS_TOT = K / 16, KG_TOT = K / 32;

  // grouped rasterization: GM M-tiles share the in-flight weight panel in L2
  const int ntm = (M + BM - 1) / BM, ntn = N / BN;
  const int GM = 8;
  const int bid = blockIdx.x;
  const int per_group = GM * ntn;
  const int group = bid / per_group;
  const int first = group * GM;
  const int gsz = min(ntm - first, GM);
  const int local = bid - group * per_group;
  const int tm = first + local % gsz, tn = local / gsz;
  const int m0 = tm * BM;
  const int nb0 = tn * C::NBLK;

  const u32 sbase = smem_u32(smem);
  pdl_launch();  // let the next kernel in the PDL chain get scheduled as our tail drains

  float acc[C::MFR][C::NFR][4];
#pragma unroll
  for (int i = 0; i < C::MFR; ++i)
#pragma unroll
    for (int j = 0; j < C::NFR; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < KT) tc_load_stage<C, BK>(sbase + s * C::STAGE, A, Bq, Bs, tid, m0, M, K, s, nb0, KS_TOT, KG_TOT);
    cp_commit();
  }

  const int q = lane >> 3, r8 = lane & 7;
  const int fsw = swz_x<BK>(r8);
  const int a_kh = q >> 1;
  u32 a_off[C::MFR];
#pragma unroll
  for (int i = 0; i < C::MFR; ++i) a_off[i] = (u32)((wm * WM + i * 16 + ((q & 1) << 3) + r8) * (BK * 2));
  const u32 b_off = (u32)(wn * C::B_BLK + lane * (C::WORDS * 4));
  const u32 s_off = (u32)(wn * C::S_BLK + g * (C::NFR * 2));

  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<ST - 2>();
    __syncthreads();
    {
      const int nk = kt + ST - 1;
      if (nk < KT) tc_load_stage<C, BK>(sbase + (nk % ST) * C::STAGE, A, Bq, Bs, tid, m0, M, K, nk, nb0, KS_TOT, KG_TOT);
      cp_commit();
    }
    const u32 sa = sbase + (kt % ST) * C::STAGE;
    const u32 sb = sa + C::A_ST;
    const u32 ss = sb + C::B_ST;

    u32 af[2][C::MFR][4];
    u32 bw[2][4];
    u32 sw[2][4];
#pragma unroll
    for (int i = 0; i < C::MFR; ++i) ldsm4(sa + a_off[i] + (((0 + a_kh) ^ fsw) << 4), af[0][i][0], af[0][i][1], af[0][i][2], af[0][i][3]);
    lds_words<C::WORDS>(sb + b_off, bw[0]);
    lds_words<C::WORDS>(ss + s_off, sw[0]);

#pragma unroll
    for (int ks = 0; ks < C::KSTEPS; ++ks) {
      const int cur = ks & 1;
      if (ks + 1 < C::KSTEPS) {
#pragma unroll
        for (int i = 0; i < C::MFR; ++i)
          ldsm4(sa + a_off[i] + ((((ks + 1) * 2 + a_kh) ^ fsw) << 4), af[cur ^ 1][i][0], af[cur ^ 1][i][1], af[cur ^ 1][i][2], af[cur ^ 1][i][3]);
        lds_words<C::WORDS>(sb + b_off + (ks + 1) * (32 * C::WORDS * 4), bw[cur ^ 1]);
        if (((ks + 1) & 1) == 0) lds_words<C::WORDS>(ss + s_off + ((ks + 1) >> 1) * (8 * C::NFR * 2), sw[((ks + 1) >> 1) & 1]);
      }
      const int si = (ks >> 1) & 1;
#pragma unroll
      for (int w = 0; w < C::WORDS; ++w) {
        const u32 x = bw[cur][w];
        const u32 sc = sw[si][w];
        const u32 s0 = prmt(sc, 0x1010u), s1 = prmt(sc, 0x3232u);
        const u32 b00 = dq2(x, s0), b01 = dq2(x >> 4, s0), b10 = dq2(x >> 8, s1), b11 = dq2(x >> 12, s1);
#pragma unroll
        for (int i = 0; i < C::MFR; ++i) {
          mma16816(acc[i][2 * w], af[cur][i][0], af[cur][i][1], af[cur][i][2], af[cur][i][3], b00, b01);
          mma16816(acc[i][2 * w + 1], af[cur][i][0], af[cur][i][1], af[cur][i][2], af[cur][i][3], b10, b11);
        }
      }
    }
  }
  cp_wait<0>();
  __syncthreads();

  // ---------------- epilogue: rank-16 LoRA-B on tensor cores (C=0, one k16 MMA) + contract cuts.
  // low factors come from the PDL-predecessor kernel -> wait, then read through L2 (.cg).
  pdl_wait();
#pragma unroll
  for (int i = 0; i < C::MFR; ++i) {
    const int rl = wm * WM + i * 16 + g;
    const int r_lo = m0 + rl, r_hi = r_lo + 8;
    u32 a0[4], a1[4];
    ld_lowfrag(low0, ldl, r_lo, r_hi, M, t, a0);
    if (EPI == 0) ld_lowfrag(low1, ldl, r_lo, r_hi, M, t, a1);
    if (EPI == 0) {
#pragma unroll
      for (int jj = 0; jj < C::NFR / 2; ++jj) {
        const int cl = wm * 0 + wn * (WN / 2) + jj * 8;
        const u16* gbr = lb0 + (u64)(tn * C::NCOL + cl + g) * 16u;
        const u16* ubr = lb1 + (u64)(tn * C::NCOL + cl + g) * 16u;
        float lg[4] = {0.f, 0.f, 0.f, 0.f}, lu[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(lg, a0[0], a0[1], a0[2], a0[3], ldg_nc32(gbr + 2 * t), ldg_nc32(gbr + 8 + 2 * t));
        mma16816(lu, a1[0], a1[1], a1[2], a1[3], ldg_nc32(ubr + 2 * t), ldg_nc32(ubr + 8 + 2 * t));
        const float* cg = acc[i][jj];
        const float* cu = acc[i][jj + C::NFR / 2];
        const int col = tn * C::NCOL + cl + 2 * t;
        if (r_lo < M)
          stg32(out + (u64)r_lo * (u64)ldo + col, pack_bf2(gu_elem(cg[0], cu[0], lg[0], lu[0]), gu_elem(cg[1], cu[1], lg[1], lu[1])));
        if (r_hi < M)
          stg32(out + (u64)r_hi * (u64)ldo + col, pack_bf2(gu_elem(cg[2], cu[2], lg[2], lu[2]), gu_elem(cg[3], cu[3], lg[3], lu[3])));
      }
    } else {
#pragma unroll
      for (int j = 0; j < C::NFR; ++j) {
        const int cl = wn * WN + j * 8;
        const u16* dbr = lb0 + (u64)(tn * BN + cl + g) * 16u;
        float l[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(l, a0[0], a0[1], a0[2], a0[3], ldg_nc32(dbr + 2 * t), ldg_nc32(dbr + 8 + 2 * t));
        const float* cd = acc[i][j];
        const int col = tn * BN + cl + 2 * t;
        if (r_lo < M) stg32(out + (u64)r_lo * (u64)ldo + col, pack_bf2(dn_elem(cd[0], l[0]), dn_elem(cd[1], l[1])));
        if (r_hi < M) stg32(out + (u64)r_hi * (u64)ldo + col, pack_bf2(dn_elem(cd[2], l[2]), dn_elem(cd[3], l[3])));
      }
    }
  }
}

#define TC_KERNEL(NAME, BM, BN, WM, WN, BK, ST, EPI, MINB)                                                   \
  extern "C" __global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, MINB)                              \
      NAME(const u16* A, const u32* Bq, const u16* Bs, const u16* low0, const u16* low1, const u16* lb0,     \
           const u16* lb1, u16* out, int M, int K, int N, int ldo, int ldl) {                                 \
    extern __shared__ __align__(128) unsigned char fg_smem[];                                                 \
    tc_gemm<BM, BN, WM, WN, BK, ST, EPI>(A, Bq, Bs, low0, low1, lb0, lb1, out, M, K, N, ldo, ldl, fg_smem);   \
  }

TC_KERNEL(fg_gu_c, 128, 128, 64, 64, 32, 3, 0, 2)
TC_KERNEL(fg_gu_d, 128, 128, 64, 32, 64, 2, 0, 2)
TC_KERNEL(fg_gu_e, 128, 128, 64, 32, 32, 4, 0, 2)
TC_KERNEL(fg_gu_f, 128, 128, 64, 64, 32, 4, 0, 2)
TC_KERNEL(fg_dn_c, 128, 128, 64, 32, 64, 4, 1, 1)
TC_KERNEL(fg_dn_d, 128, 128, 64, 32, 64, 2, 1, 2)
TC_KERNEL(fg_dn_e, 128, 128, 64, 32, 32, 4, 1, 2)

// ------------------------------------------------------------------ rank-16 low factors (large M), PDL-chained
// out[M,R] = bf(Z[M,K] @ A[R,K]^T).  One CTA per 16 rows; its KSPLIT warps each take a contiguous K slice (so the
// 16-row block streams with KSPLIT x the memory parallelism), then warp 0 adds the slices in fixed order.
template <int R, int ZCG, int U, int ZH, int W16>
DEVI void lowk_chunk(const u16* zl, const u16* zh, bool okl, bool okh, const u16* __restrict__ A, int K, int s0,
                     int g, int t, float (&acc)[R / 8][4], u16* __restrict__ z16, int r_lo, int r_hi) {
  u32 a[U][4];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int k = (s0 + u) * 16;
    a[u][0] = okl ? ldz<ZCG>(zl + k) : 0u;
    a[u][1] = okh ? ldz<ZCG>(zh + k) : 0u;
    a[u][2] = okl ? ldz<ZCG>(zl + k + 8) : 0u;
    a[u][3] = okh ? ldz<ZCG>(zh + k + 8) : 0u;
    if (W16) {  // fp16 copy of Z for the fp16 datapath (each element is loaded by exactly one lane)
      if (okl) {
        stg32(z16 + (u64)r_lo * (u64)K + k + 2 * t, pack_h2(bf_lo(a[u][0]), bf_hi(a[u][0])));
        stg32(z16 + (u64)r_lo * (u64)K + k + 8 + 2 * t, pack_h2(bf_lo(a[u][2]), bf_hi(a[u][2])));
      }
      if (okh) {
        stg32(z16 + (u64)r_hi * (u64)K + k + 2 * t, pack_h2(bf_lo(a[u][1]), bf_hi(a[u][1])));
        stg32(z16 + (u64)r_hi * (u64)K + k + 8 + 2 * t, pack_h2(bf_lo(a[u][3]), bf_hi(a[u][3])));
      }
    }
  }
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int k = (s0 + u) * 16 + 2 * t;
#pragma unroll
    for (int n = 0; n < R / 8; ++n) {
      const u16* ar = A + (u64)(n * 8 + g) * (u64)K + k;
      if (ZH) mma16816_h(acc[n], a[u][0], a[u][1], a[u][2], a[u][3], ldg_nc32(ar), ldg_nc32(ar + 8));
      else mma16816(acc[n], a[u][0], a[u][1], a[u][2], a[u][3], ldg_nc32(ar), ldg_nc32(ar + 8));
    }
  }
}

template <int R, int ZCG, int WAIT, int UNR, int KSPLIT, int ZH = 0, int W16 = 0>
DEVI void lowk(const u16* __restrict__ Z, const u16* __restrict__ A, u16* __restrict__ out, int M, int K,
               u16* __restrict__ z16 = nullptr) {
  if (WAIT) pdl_wait();
  pdl_launch();
  __shared__ float red[KSPLIT][R / 8][4][32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const int r_lo = r0 + g, r_hi = r0 + g + 8;
  const bool okl = r_lo < M, okh = r_hi < M;
  const u16* zl = Z + (u64)(okl ? r_lo : 0) * (u64)K + 2 * t;
  const u16* zh = Z + (u64)(okh ? r_hi : 0) * (u64)K + 2 * t;
  float acc[R / 8][4];
#pragma unroll
  for (int n = 0; n < R / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[n][e] = 0.f;
  const int KS = K / 16, s_beg = (warp * KS) / KSPLIT, s_end = ((warp + 1) * KS) / KSPLIT;
  int s0 = s_beg;
  for (; s0 + UNR <= s_end; s0 += UNR) lowk_chunk<R, ZCG, UNR, ZH, W16>(zl, zh, okl, okh, A, K, s0, g, t, acc, z16, r_lo, r_hi);
  for (; s0 < s_end; ++s0) lowk_chunk<R, ZCG, 1, ZH, W16>(zl, zh, okl, okh, A, K, s0, g, t, acc, z16, r_lo, r_hi);
#pragma unroll
  for (int n = 0; n < R / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) red[warp][n][e][lane] = acc[n][e];
  __syncthreads();
  if (warp != 0) return;
#pragma unroll
  for (int n = 0; n < R / 8; ++n) {
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[e] = red[0][n][e][lane];
#pragma unroll
      for (int w = 1; w < KSPLIT; ++w) v[e] += red[w][n][e][lane];
    }
    if (okl) stg32(out + (u64)r_lo * R + n * 8 + 2 * t, pack_bf2(v[0], v[1]));
    if (okh) stg32(out + (u64)r_hi * R + n * 8 + 2 * t, pack_bf2(v[2], v[3]));
  }
}
extern "C" __global__ void __launch_bounds__(128) fg_lowk_gu(const u16* Z, const u16* A, u16* out, int M, int K) {
  lowk<32, 0, 0, 4, 4>(Z, A, out, M, K);
}
extern "C" __global__ void __launch_bounds__(128) fg_lowk_d(const u16* Z, const u16* A, u16* out, int M, int K) {
  lowk<16, 1, 1, 8, 4>(Z, A, out, M, K);
}
// fp16 datapath: GU low factors from bf16 X + fp16 copy of X; Down low factors from the fp16 mid (fp32 accumulate)
extern "C" __global__ void __launch_bounds__(128) fg_lowk_gu16(const u16* Z, const u16* A, u16* out, int M, int K,
                                                               u16* z16) {
  lowk<32, 0, 0, 4, 4, 0, 1>(Z, A, out, M, K, z16);
}
extern "C" __global__ void __launch_bounds__(128) fg_lowk_d16(const u16* Z, const u16* A, u16* out, int M, int K) {
  lowk<16, 1, 1, 8, 4, 1, 0>(Z, A, out, M, K);
}

// ------------------------------------------------------------------ tensor-core throughput probe (f32 vs f16 accumulate)
extern "C" __global__ void __launch_bounds__(256) fg_mma_tput(float* sink, int iters, int f16acc) {
  u32 a0 = threadIdx.x * 0x00010001u, a1 = a0 ^ 0x3c003c00u, a2 = a0 + 7u, a3 = a1 + 3u, b0 = a0 ^ 0x11u, b1 = a1 ^ 0x22u;
  if (f16acc) {
    u32 c[8][2];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j][0] = c[j][1] = 0u;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                     : "+r"(c[j][0]), "+r"(c[j][1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    u32 x = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) x ^= c[j][0] ^ c[j][1];
    if (x == 0x12345678u) sink[0] = 1.f;
  } else {
    float c[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int j = 0; j < 8; ++j) mma16816(c[j], a0, a1, a2, a3, b0, b1);
    }
    float x = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) x += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    if (x == 1234.5f) sink[0] = x;
  }
}

// ------------------------------------------------------------------ M=8 operand-swapped GEMV
// "SW" layout: weights are the MMA A operand (16 rows x k16), tokens are N=8.  Per lane per k16 per quad
// (4 frags = 64 rows) one uint4; word f nibble i=4e+2h+rr <-> (row 16f+8rr+g, k=8h+2t+e).
// Scales [quads][K/32][8 g][4 f][2 rr] bf16.
DEVI void sw_step(float (&acc)[4][4], const u32 (&w)[4], const u32 (&sc)[4], u32 z0, u32 z1) {
#pragma unroll
  for (int f = 0; f < 4; ++f) {
    const u32 slo = prmt(sc[f], 0x1010u), shi = prmt(sc[f], 0x3232u);
    const u32 x = w[f];
    const u32 a0 = dq2(x, slo), a1 = dq2(x >> 4, shi), a2 = dq2(x >> 8, slo), a3 = dq2(x >> 12, shi);
    mma16816(acc[f], a0, a1, a2, a3, z0, z1);
  }
}

template <int WARPS, int DG, int EPI, int KS_TOTAL, int ZCG, int HINT = 0>
DEVI void gemv8(const u16* __restrict__ Z, const u32* __restrict__ Wq, const u16* __restrict__ Ws,
                const u16* __restrict__ lowf, const u16* __restrict__ lb0, const u16* __restrict__ lb1,
                u16* __restrict__ out, int ldo, float* red, u16* slow, int keep = 0) {
  constexpr int SPW = KS_TOTAL / WARPS, GPW = SPW / 2;
  static_assert(SPW * WARPS == KS_TOTAL && GPW * 2 == SPW && GPW % DG == 0, "gemv8 split");
  const int quad = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  if (EPI == 0) pdl_launch();
  const unsigned long long pol = HINT ? l2_policy(quad < keep) : 0ull;  // leading `keep` quads stay L2-resident
  const int s0 = warp * SPW;
  const char* wq = (const char*)Wq + ((u64)quad * KS_TOTAL + s0) * 512u + lane * 16;
  const char* ws = (const char*)Ws + ((u64)quad * (KS_TOTAL / 2) + (s0 >> 1)) * 128u + g * 16;
  const u16* zp = Z + (u64)g * (KS_TOTAL * 16) + s0 * 16 + 2 * t;

  float acc[4][4];
#pragma unroll
  for (int f = 0; f < 4; ++f)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[f][e] = 0.f;

  u32 wb[DG][2][4], sb[DG][4], zb[DG][4];
#pragma unroll
  for (int d = 0; d < DG; ++d) {
    ldw_v4<HINT>(wq + (u64)(2 * d) * 512u, wb[d][0], pol);
    ldw_v4<HINT>(wq + (u64)(2 * d + 1) * 512u, wb[d][1], pol);
    ldw_v4<HINT>(ws + (u64)d * 128u, sb[d], pol);
    const u16* zz = zp + d * 32;
    zb[d][0] = ldz<ZCG>(zz); zb[d][1] = ldz<ZCG>(zz + 8); zb[d][2] = ldz<ZCG>(zz + 16); zb[d][3] = ldz<ZCG>(zz + 24);
  }
  for (int gi0 = 0; gi0 < GPW; gi0 += DG) {
#pragma unroll
    for (int d = 0; d < DG; ++d) {
      u32 w0[4], w1[4], sc[4], z[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) { w0[k] = wb[d][0][k]; w1[k] = wb[d][1][k]; sc[k] = sb[d][k]; z[k] = zb[d][k]; }
      const int gn = gi0 + d + DG;
      if (gn < GPW) {
        ldw_v4<HINT>(wq + (u64)(2 * gn) * 512u, wb[d][0], pol);
        ldw_v4<HINT>(wq + (u64)(2 * gn + 1) * 512u, wb[d][1], pol);
        ldw_v4<HINT>(ws + (u64)gn * 128u, sb[d], pol);
        const u16* zz = zp + gn * 32;
        zb[d][0] = ldz<ZCG>(zz); zb[d][1] = ldz<ZCG>(zz + 8); zb[d][2] = ldz<ZCG>(zz + 16); zb[d][3] = ldz<ZCG>(zz + 24);
      }
      sw_step(acc, w0, sc, z[0], z[1]);
      sw_step(acc, w1, sc, z[2], z[3]);
    }
  }
  // CTA reduction over the K-split warps, fixed order (deterministic)
#pragma unroll
  for (int v = 0; v < 16; ++v) red[(warp * 16 + v) * 32 + lane] = acc[v >> 2][v & 3];
  __syncthreads();
  pdl_wait();  // low factors of the predecessor kernel are complete & visible past this point
  if (EPI == 0) {
    if (tid < 128) ((u32*)slow)[tid] = ldg_cg32(lowf + 2 * tid);  // [8 tokens][32] = 16 gate + 16 up ranks
    __syncthreads();
    const int token = tid >> 5, cl = tid & 31, f = cl >> 3, gg = cl & 7;
    const int ls = gg * 4 + (token >> 1), e1 = token & 1;
    float gs = 0.f, us = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      gs += red[(w * 16 + f * 4 + e1) * 32 + ls];
      us += red[(w * 16 + f * 4 + 2 + e1) * 32 + ls];
    }
    const int col = quad * 32 + cl;
    u32 gbw[8], ubw[8];
    {
      u32 tmp[4];
      ldg_nc_v4(lb0 + (u64)col * 16u, tmp); gbw[0] = tmp[0]; gbw[1] = tmp[1]; gbw[2] = tmp[2]; gbw[3] = tmp[3];
      ldg_nc_v4(lb0 + (u64)col * 16u + 8, tmp); gbw[4] = tmp[0]; gbw[5] = tmp[1]; gbw[6] = tmp[2]; gbw[7] = tmp[3];
      ldg_nc_v4(lb1 + (u64)col * 16u, tmp); ubw[0] = tmp[0]; ubw[1] = tmp[1]; ubw[2] = tmp[2]; ubw[3] = tmp[3];
      ldg_nc_v4(lb1 + (u64)col * 16u + 8, tmp); ubw[4] = tmp[0]; ubw[5] = tmp[1]; ubw[6] = tmp[2]; ubw[7] = tmp[3];
    }
    const u32* lw = (const u32*)(slow + token * 32);
    float lg = 0.f, lu = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const u32 xg = lw[p], xu = lw[8 + p];
      lg = fmaf(bf_lo(xg), bf_lo(gbw[p]), lg);
      lg = fmaf(bf_hi(xg), bf_hi(gbw[p]), lg);
      lu = fmaf(bf_lo(xu), bf_lo(ubw[p]), lu);
      lu = fmaf(bf_hi(xu), bf_hi(ubw[p]), lu);
    }
    stg16(out + (u64)token * (u64)ldo + col, f2bf(gu_elem(gs, us, lg, lu)));
  } else {
    if (tid < 64) ((u32*)slow)[tid] = ldg_cg32(lowf + 2 * tid);  // [8 tokens][16]
    __syncthreads();
    const int token = tid >> 6, cl = tid & 63, f = cl >> 4, rr = (cl >> 3) & 1, gg = cl & 7;
    const int ls = gg * 4 + (token >> 1), slot = f * 4 + rr * 2 + (token & 1);
    float ds = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) ds += red[(w * 16 + slot) * 32 + ls];
    const int col = quad * 64 + cl;
    u32 dbw[8];
    {
      u32 tmp[4];
      ldg_nc_v4(lb0 + (u64)col * 16u, tmp); dbw[0] = tmp[0]; dbw[1] = tmp[1]; dbw[2] = tmp[2]; dbw[3] = tmp[3];
      ldg_nc_v4(lb0 + (u64)col * 16u + 8, tmp); dbw[4] = tmp[0]; dbw[5] = tmp[1]; dbw[6] = tmp[2]; dbw[7] = tmp[3];
    }
    const u32* lw = (const u32*)(slow + token * 16);
    float ld = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const u32 x = lw[p];
      ld = fmaf(bf_lo(x), bf_lo(dbw[p]), ld);
      ld = fmaf(bf_hi(x), bf_hi(dbw[p]), ld);
    }
    stg16(out + (u64)token * (u64)ldo + col, f2bf(dn_elem(ds, ld)));
  }
}

// rank-16 low factors at M=8: low[token][r] = bf(sum_k Z[token,k] * A[r,k]); C CTAs split K, deterministic
// last-CTA reduction (fixed order over partials), counter self-resets -> graph-replay safe.
template <int NFRAG, int KS_TOTAL, int C, int WARPS, int WAIT>
DEVI void lowa8(const u16* __restrict__ Z, const u16* __restrict__ A, float* part, u32* counter,
                u16* low_out, float* red, int* flag) {
  constexpr int SPC = KS_TOTAL / C, SPW = SPC / WARPS, RR = NFRAG * 16, K = KS_TOTAL * 16, NT = WARPS * 32;
  static_assert(SPC * C == KS_TOTAL && SPW * WARPS == SPC, "lowa8 split");
  if (WAIT) pdl_wait();
  pdl_launch();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int sbeg = blockIdx.x * SPC + warp * SPW;
  float acc[NFRAG][4];
#pragma unroll
  for (int f = 0; f < NFRAG; ++f)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[f][e] = 0.f;
#pragma unroll 4
  for (int s = sbeg; s < sbeg + SPW; ++s) {
    const int k0 = s * 16 + 2 * t;
    const u32 z0 = ldz<WAIT>(Z + (u64)g * K + k0), z1 = ldz<WAIT>(Z + (u64)g * K + k0 + 8);
#pragma unroll
    for (int f = 0; f < NFRAG; ++f) {
      const u16* ar = A + (u64)(f * 16 + g) * K + k0;
      mma16816(acc[f], ldg_nc32(ar), ldg_nc32(ar + 8 * (u64)K), ldg_nc32(ar + 8), ldg_nc32(ar + 8 * (u64)K + 8), z0, z1);
    }
  }
#pragma unroll
  for (int v = 0; v < NFRAG * 4; ++v) red[(warp * NFRAG * 4 + v) * 32 + lane] = acc[v >> 2][v & 3];
  __syncthreads();
  for (int o = tid; o < RR * 8; o += NT) {
    const int r = o >> 3, token = o & 7;
    const int f = r >> 4, rl = r & 15, gg = rl & 7, hi = rl >> 3;
    const int ls = gg * 4 + (token >> 1), slot = f * 4 + hi * 2 + (token & 1);
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) v += red[(w * NFRAG * 4 + slot) * 32 + ls];
    part[(u64)blockIdx.x * (RR * 8) + o] = v;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) *flag = (atom_add(counter, 1u) == (u32)(C - 1)) ? 1 : 0;
  __syncthreads();
  if (*flag) {
    __threadfence();
    for (int o = tid; o < RR * 8; o += NT) {
      const int r = o >> 3, token = o & 7;
      float v = 0.f;
      for (int c = 0; c < C; ++c) v += ldg_cg_f32(part + (u64)c * (RR * 8) + o);
      low_out[token * RR + r] = (u16)f2bf(v);
    }
    if (tid == 0) stg32(counter, 0u);
  }
}

extern "C" __global__ void __launch_bounds__(256, 2)
fg_gu8(const u16* Z, const u32* Wq, const u16* Ws, const u16* lowf, const u16* lb0, const u16* lb1, u16* out, int ldo) {
  __shared__ float red[8 * 16 * 32];
  __shared__ __align__(16) u16 slow[8 * 32];
  gemv8<8, 3, 0, KS_H, 0>(Z, Wq, Ws, lowf, lb0, lb1, out, ldo, red, slow);
}
extern "C" __global__ void __launch_bounds__(512, 1)
fg_dn8(const u16* Z, const u32* Wq, const u16* Ws, const u16* lowf, const u16* lb0, const u16* lb1, u16* out, int ldo) {
  __shared__ float red[16 * 16 * 32];
  __shared__ __align__(16) u16 slow[8 * 16];
  gemv8<16, 3, 1, KS_I, 1>(Z, Wq, Ws, lowf, lb0, lb1, out, ldo, red, slow);
}
#ifdef FG_L2HINT
extern "C" __global__ void __launch_bounds__(256, 2)
fg_gu8h(const u16* Z, const u32* Wq, const u16* Ws, const u16* lowf, const u16* lb0, const u16* lb1, u16* out, int ldo,
        int keep) {
  __shared__ float red[8 * 16 * 32];
  __shared__ __align__(16) u16 slow[8 * 32];
  gemv8<8, 3, 0, KS_H, 0, 1>(Z, Wq, Ws, lowf, lb0, lb1, out, ldo, red, slow, keep);
}
extern "C" __global__ void __launch_bounds__(512, 1)
fg_dn8h(const u16* Z, const u32* Wq, const u16* Ws, const u16* lowf, const u16* lb0, const u16* lb1, u16* out, int ldo,
        int keep) {
  __shared__ float red[16 * 16 * 32];
  __shared__ __align__(16) u16 slow[8 * 16];
  gemv8<16, 3, 1, KS_I, 1, 1>(Z, Wq, Ws, lowf, lb0, lb1, out, ldo, red, slow, keep);
}
// MLP round: deeper register prefetch (31B planner + 12B refiner proposal "DG 3 -> 6").  The epilogues fix
// GU at 256 threads (8 tokens x 32 cols) and Down at 512 (8 x 64), so legal DG must divide GPW (GU 21, Down 42) and
// Down cannot exceed 128 registers per thread.  GU DG=7 is the nearest legal depth, tried at 1 and 2 CTAs/SM.
#define GU8H(NAME, DGV, MINB)                                                                                          \
  extern "C" __global__ void __launch_bounds__(256, MINB)                                                           \
      NAME(const u16* Z, const u32* Wq, const u16* Ws, const u16* lowf, const u16* lb0, const u16* lb1, u16* out,     \
           int ldo, int keep) {                                                                                      \
    __shared__ float red[8 * 16 * 32];                                                                              \
    __shared__ __align__(16) u16 slow[8 * 32];                                                                      \
    gemv8<8, DGV, 0, KS_H, 0, 1>(Z, Wq, Ws, lowf, lb0, lb1, out, ldo, red, slow, keep);                             \
  }
#define DN8H(NAME, DGV)                                                                                               \
  extern "C" __global__ void __launch_bounds__(512, 1)                                                              \
      NAME(const u16* Z, const u32* Wq, const u16* Ws, const u16* lowf, const u16* lb0, const u16* lb1, u16* out,     \
           int ldo, int keep) {                                                                                      \
    __shared__ float red[16 * 16 * 32];                                                                             \
    __shared__ __align__(16) u16 slow[8 * 16];                                                                      \
    gemv8<16, DGV, 1, KS_I, 1, 1>(Z, Wq, Ws, lowf, lb0, lb1, out, ldo, red, slow, keep);                            \
  }
GU8H(fg_gu8h_d7a, 7, 1)
GU8H(fg_gu8h_d7b, 7, 2)
DN8H(fg_dn8h_d6, 6)
DN8H(fg_dn8h_d7, 7)
#endif
// memory-system probes: grid-stride 16 B reads that bypass L1 (DRAM pass over a buffer >> L2, or many passes
// over a buffer << L2 for the L2 rate)
extern "C" __global__ void __launch_bounds__(256) fg_bw_read(const u32* p, long long nvec, int passes, u32* sink) {
  u32 x = 0u;
  for (int ps = 0; ps < passes; ++ps)
    for (long long base = (long long)blockIdx.x * 1024; base < nvec; base += (long long)gridDim.x * 1024) {
      u32 r[4][4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const long long idx = base + u * 256 + threadIdx.x;
        if (idx < nvec) ldg_cg_v4(p + idx * 4, r[u]);
        else r[u][0] = r[u][1] = r[u][2] = r[u][3] = 0u;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) x ^= r[u][0] ^ r[u][1] ^ r[u][2] ^ r[u][3];
    }
  if (x == 0x9E3779B9u) sink[0] = x;
}
extern "C" __global__ void __launch_bounds__(256, 1)
fg_lowa_gu(const u16* Z, const u16* A, float* part, u32* counter, u16* low_out) {
  __shared__ float red[8 * 8 * 32];
  __shared__ int flag;
  lowa8<2, KS_H, 6, 8, 0>(Z, A, part, counter, low_out, red, &flag);
}
extern "C" __global__ void __launch_bounds__(256, 1)
fg_lowa_d(const u16* Z, const u16* A, float* part, u32* counter, u16* low_out) {
  __shared__ float red[8 * 4 * 32];
  __shared__ int flag;
  lowa8<1, KS_I, 21, 8, 1>(Z, A, part, counter, low_out, red, &flag);
}

// ------------------------------------------------------------------ control-design helpers
// original packing -> BF16 (exact contract decode); one thread per INT32 word (8 nibbles, 16B store)
extern "C" __global__ void __launch_bounds__(256)
fg_decode(const u32* Q, const u16* S, u16* W, long long nwords, int kw, int kg) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nwords) return;
  const long long n = i / kw;
  const int w = (int)(i - n * kw);
  const u32 x = Q[i];
  const float s = u2f(((u32)S[n * kg + (w >> 2)]) << 16);
  u32 o[4];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const float v0 = (float)((int)((x >> (8 * p)) & 15u) - 8) * s;
    const float v1 = (float)((int)((x >> (8 * p + 4)) & 15u) - 8) * s;
    o[p] = pack_bf2(v0, v1);
  }
  stg128(W + i * 8, o[0], o[1], o[2], o[3]);
}

// fused LoRA-B (tensor core) + contract cuts on library base outputs. CTA 64x64, 8 warps (4 x 2).
template <int EPI>
DEVI void epi_lib(const u16* __restrict__ P0, const u16* __restrict__ P1, const u16* __restrict__ low0,
                  const u16* __restrict__ low1, const u16* __restrict__ lb0, const u16* __restrict__ lb1,
                  u16* __restrict__ out, int M, int N, int ldp, int ldl, int ldb) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int wm = warp & 3, wn = warp >> 2;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  const int r_lo = m0 + wm * 16 + g, r_hi = r_lo + 8;
  const bool ok_lo = r_lo < M, ok_hi = r_hi < M;
  u32 a0[4], a1[4];
  a0[0] = ok_lo ? ldg_nc32(low0 + (u64)r_lo * ldl + 2 * t) : 0u;
  a0[1] = ok_hi ? ldg_nc32(low0 + (u64)r_hi * ldl + 2 * t) : 0u;
  a0[2] = ok_lo ? ldg_nc32(low0 + (u64)r_lo * ldl + 8 + 2 * t) : 0u;
  a0[3] = ok_hi ? ldg_nc32(low0 + (u64)r_hi * ldl + 8 + 2 * t) : 0u;
  if (EPI == 0) {
    a1[0] = ok_lo ? ldg_nc32(low1 + (u64)r_lo * ldl + 2 * t) : 0u;
    a1[1] = ok_hi ? ldg_nc32(low1 + (u64)r_hi * ldl + 2 * t) : 0u;
    a1[2] = ok_lo ? ldg_nc32(low1 + (u64)r_lo * ldl + 8 + 2 * t) : 0u;
    a1[3] = ok_hi ? ldg_nc32(low1 + (u64)r_hi * ldl + 8 + 2 * t) : 0u;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = n0 + wn * 32 + j * 8;
    const int col = c + 2 * t;
    float l0[4] = {0.f, 0.f, 0.f, 0.f};
    mma16816(l0, a0[0], a0[1], a0[2], a0[3], ldg_nc32(lb0 + (u64)(c + g) * ldb + 2 * t), ldg_nc32(lb0 + (u64)(c + g) * ldb + 8 + 2 * t));
    if (EPI == 0) {
      float l1[4] = {0.f, 0.f, 0.f, 0.f};
      mma16816(l1, a1[0], a1[1], a1[2], a1[3], ldg_nc32(lb1 + (u64)(c + g) * ldb + 2 * t), ldg_nc32(lb1 + (u64)(c + g) * ldb + 8 + 2 * t));
      if (ok_lo) {
        const u32 gv = ldg_nc32(P0 + (u64)r_lo * ldp + col), uv = ldg_nc32(P1 + (u64)r_lo * ldp + col);
        stg32(out + (u64)r_lo * N + col, pack_bf2(gu_elem(bf_lo(gv), bf_lo(uv), l0[0], l1[0]), gu_elem(bf_hi(gv), bf_hi(uv), l0[1], l1[1])));
      }
      if (ok_hi) {
        const u32 gv = ldg_nc32(P0 + (u64)r_hi * ldp + col), uv = ldg_nc32(P1 + (u64)r_hi * ldp + col);
        stg32(out + (u64)r_hi * N + col, pack_bf2(gu_elem(bf_lo(gv), bf_lo(uv), l0[2], l1[2]), gu_elem(bf_hi(gv), bf_hi(uv), l0[3], l1[3])));
      }
    } else {
      if (ok_lo) {
        const u32 dv = ldg_nc32(P0 + (u64)r_lo * ldp + col);
        stg32(out + (u64)r_lo * N + col, pack_bf2(dn_elem(bf_lo(dv), l0[0]), dn_elem(bf_hi(dv), l0[1])));
      }
      if (ok_hi) {
        const u32 dv = ldg_nc32(P0 + (u64)r_hi * ldp + col);
        stg32(out + (u64)r_hi * N + col, pack_bf2(dn_elem(bf_lo(dv), l0[2]), dn_elem(bf_hi(dv), l0[3])));
      }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256)
fg_epi_gu(const u16* P0, const u16* P1, const u16* low0, const u16* low1, const u16* lb0, const u16* lb1, u16* out, int M, int N,
          int ldp, int ldl, int ldb) {
  epi_lib<0>(P0, P1, low0, low1, lb0, lb1, out, M, N, ldp, ldl, ldb);
}
extern "C" __global__ void __launch_bounds__(256)
fg_epi_dn(const u16* P0, const u16* P1, const u16* low0, const u16* low1, const u16* lb0, const u16* lb1, u16* out, int M, int N,
          int ldp, int ldl, int ldb) {
  epi_lib<1>(P0, P1, low0, low1, lb0, lb1, out, M, N, ldp, ldl, ldb);
}

extern "C" __global__ void fg_gelu_table(u16* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 65536) out[i] = (u16)f2bf(gelu_tanh_ref(u2f(((u32)i) << 16)));
}
"""

CUDA_V3 = r"""
// ================= V3: warpspeed-style warp-specialized fused INT4 GEMM =================
// Squads (CCCL warpspeed formulation):
//   warps 0-3, 4-7 = MMA squads : independent tile streams (each 4 warps, 64xWN or 128xWN warp tiles)
//   warp 8 lane 0  = producer   : TMA/bulk streams for both squads + the CTA's single CLC issuer (as in CCCL's
//                                 lookahead kernels: CLC folded into the load squad, a new try_cancel only after
//                                 a success), driven as a non-blocking state machine over mbarrier.test_wait.
// Resources per MMA squad (stage x phase rings on mbarriers):
//   stages : phase L (loader: 2-D TMA of the A tile + 1-D bulk of packed B words / scales -> releaseTx)
//            phase C (MMA squad: ldmatrix + in-register dequant + mma -> release, one arrive per warp)
//   tileid : phase W (loader writes the next tile id or -1) / phase R (MMA squad reads it)
// Scheduling: CLC work stealing (grid = #tiles; the scheduler steals into a 1-deep queue per loader, filling
// whichever queue frees first, so the two squads balance dynamically) or a static stride (CLC=0).  PDL: trigger at start, wait before
// the first epilogue (low-rank factors come from the PDL predecessor).
struct __align__(64) TmaDesc { u64 v[16]; };

#ifndef FG_WATCHDOG
#define FG_WATCHDOG 0x04000000
#endif

DEVI void mbar_init(u32 a, u32 c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(c) : "memory"); }
DEVI void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
DEVI void mbar_expect_tx(u32 a, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory");
}
DEVI void mbar_arrive(u32 a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory"); }
DEVI void mbar_wait(u32 a, u32 parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      ".reg .u32 C1;\n"
      "mov.u32 C1, 0;\n"
      "FG_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra FG_DONE;\n"
      "add.u32 C1, C1, 1;\n"
      "setp.ge.u32 P1, C1, %2;\n"
      "@P1 trap;\n"
      "bra FG_WAIT;\n"
      "FG_DONE:\n"
      "}\n" ::"r"(a), "r"(parity), "n"(FG_WATCHDOG) : "memory");
}
DEVI void tma_2d(u32 dst, const void* tmap, int c0, int c1, u32 mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      ::"r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(mbar) : "memory");
}
DEVI void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
DEVI bool mbar_test(u32 a, u32 parity) {
  u32 ok;
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
      "selp.b32 %0, 1, 0, P1;\n"
      "}\n" : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  return ok != 0;
}
DEVI void sts32(u32 a, u32 v) { asm volatile("st.shared.u32 [%0], %1;\n" ::"r"(a), "r"(v) : "memory"); }
DEVI u32 lds32m(u32 a) {
  u32 v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a) : "memory");
  return v;
}
DEVI void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
template <int N> DEVI void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N)); }
template <int N> DEVI void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N)); }
DEVI void clc_try_cancel(u32 resp, u32 mbar) {
  asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n"
               ::"r"(resp), "r"(mbar) : "memory");
}
DEVI int clc_result(u32 resp) {  // first ctaid.x of the canceled cluster, or -1 when nothing is left to steal
  u64 lo, hi;
  asm volatile("ld.shared.v2.u64 {%0, %1}, [%2];\n" : "=l"(lo), "=l"(hi) : "r"(resp) : "memory");
  u32 ok, id;
  asm volatile(
      "{\n"
      ".reg .pred P_OUT;\n"
      ".reg .b128 B128_R;\n"
      "mov.b128 B128_R, {%2, %3};\n"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P_OUT, B128_R;\n"
      "selp.b32 %0, 1, 0, P_OUT;\n"
      "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, B128_R;\n"
      "}\n" : "=r"(ok), "=r"(id) : "l"(lo), "l"(hi));
  return ok ? (int)id : -1;
}

template <int BN, int WN, int BK, int ST, int EPI, int CLC, int F16, int WMT>
struct WsCfg {
  static constexpr int BM = 128, WM = WMT;
  static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, MW = WARPS_M * WARPS_N;   // MMA warps per squad
  static constexpr int MFR = WM / 16, NFR = WN / 8, KSTEPS = BK / 16, WORDS = NFR / 2, NBLK = WARPS_N;
  static constexpr int A_ST = BM * BK * 2;
  static constexpr int B_BLK = KSTEPS * 32 * WORDS * 4, B_ST = NBLK * B_BLK;
  static constexpr int S_BLK = (BK / 32) * 8 * NFR * 2, S_ST = NBLK * S_BLK;
  static constexpr int ATOM = BK == 64 ? 1024 : 512;
  static constexpr int STAGE = ((A_ST + B_ST + S_ST + ATOM - 1) / ATOM) * ATOM;
  static constexpr int TX = A_ST + B_ST + S_ST;
  static constexpr int RING = ((ST * STAGE + 1023) / 1024) * 1024;
  static constexpr int NCOL = EPI == 0 ? BN / 2 : BN;
  static constexpr int NACC = MFR * NFR * 4;                       // fp32 accumulators per lane
  // B (weights) fragments are dequantized by every warp that owns them: BM/WM warps per column block.
  // WM=128 -> dequant once per tile, each dequantized fragment feeds MFR=8 MMAs; A fragments single-buffered.
  static constexpr int ABUF = MFR > 4 ? 1 : 2;
  // per-squad aux: full[ST], empty[ST], tfull[2], tempty[2], unit ids[2]
  static constexpr int OFF_TID = 8 * (2 * ST + 4);
  static constexpr int AUX = ((OFF_TID + 8 + 15) / 16) * 16;
  // CTA aux (producer): +0 clc barrier, +16 clc response (16B)
  static constexpr int GAUX = 32;
  static constexpr int SMEM = 2 * RING + 2 * AUX + GAUX + 1024;
};

// Work units: tiles [0, dpu) are whole (data-parallel); every tile >= dpu (the last partial wave) is split into S
// K-ranges ("stream-K tail") so the final wave keeps every squad busy.  Partials go to an fp32 workspace; the last
// arriving warp of each 64-row x WN sub-tile reduces them in split order (deterministic) and runs the epilogue.
DEVI void decode_unit(int u, int dpu, int S, int KT, int& tile, int& k0, int& k1, int& sp) {
  if (S <= 1 || u < dpu) {
    tile = u; k0 = 0; k1 = KT; sp = -1;
    return;
  }
  const int v = u - dpu;
  tile = dpu + v / S;
  sp = v % S;
  k0 = (sp * KT) / S;
  k1 = ((sp + 1) * KT) / S;
}

// Producer bookkeeping for one MMA squad (lives in the producer thread's registers).
struct SqState {
  int unit, kt, k1, tm, tn, it, ti, st;  // st: 0 needs a unit, 1 publish unit id, 2 streaming stages, 3 done
};

// One non-blocking step for one squad: hand it a unit, publish the unit id, or issue one stage of loads.
// Returns true if anything happened.  Never blocks: every barrier is polled with mbarrier.test_wait.
template <class C, int BK, int ST, int CLC, int NSQ = 2>
DEVI bool prod_step(SqState& q, const int sq, int& snext, int& stolen, int& clc_state, bool& first0,
                    const u32 aux0, const u32 sbase, const u32 clc_bar, const u32 resp, const TmaDesc* tmA,
                    const char* bq, const char* bs, const int units, const int dpu, const int S, const int KT,
                    const int ntm, const int ntn, const int KS_TOT, const int KG_TOT) {
  const u32 ax = aux0 + sq * C::AUX;
  bool prog = false;
  if (q.st == 0) {
    int u = -2;
    if (!CLC) {
      u = snext;
      snext += NSQ * (int)gridDim.x;
    } else if (sq == 0 && first0) {
      u = (int)blockIdx.x;
      first0 = false;
    } else if (stolen >= 0) {
      u = stolen;
      stolen = -2;
      mbar_expect_tx(clc_bar, 16u);  // only after a success: never try_cancel again after a failure
      clc_try_cancel(resp, clc_bar);
      clc_state = 1;
    } else if (clc_state == 2) {
      u = -1;
    }
    if (u != -2) {
      q.unit = u;
      q.st = 1;
      prog = true;
    }
  }
  if (q.st == 1) {
    const int slot = q.ti & 1;
    if (mbar_test(ax + 8 * (2 * ST + 2 + slot), (u32)(((q.ti >> 1) & 1) ^ 1))) {
      const bool valid = q.unit >= 0 && q.unit < units;
      sts32(ax + C::OFF_TID + 4 * slot, valid ? (u32)q.unit : 0xFFFFFFFFu);
      mbar_arrive(ax + 8 * (2 * ST + slot));
      ++q.ti;
      if (valid) {
        int tile, k0, sp;
        decode_unit(q.unit, dpu, S, KT, tile, k0, q.k1, sp);
        tile_coords(tile, ntm, ntn, q.tm, q.tn);
        q.kt = k0;
        q.st = 2;
      } else {
        q.st = 3;
      }
      prog = true;
    }
  }
  if (q.st == 2) {
    const int s = q.it % ST, use = q.it / ST;
    if (mbar_test(ax + 8 * (ST + s), (u32)((use & 1) ^ 1))) {
      const u32 full = ax + 8 * s;
      const u32 sa = sbase + sq * C::RING + s * C::STAGE, sb = sa + C::A_ST, ss = sb + C::B_ST;
      const int nb0 = q.tn * C::NBLK, kt = q.kt;
      mbar_expect_tx(full, (u32)C::TX);
      tma_2d(sa, tmA, kt * BK, q.tm * C::BM, full);
#pragma unroll
      for (int b = 0; b < C::NBLK; ++b)
        bulk_g2s(sb + b * C::B_BLK, bq + ((u64)(nb0 + b) * (u64)KS_TOT + (u64)(kt * C::KSTEPS)) * (u64)(32 * C::WORDS * 4),
                 (u32)C::B_BLK, full);
#pragma unroll
      for (int b = 0; b < C::NBLK; ++b)
        bulk_g2s(ss + b * C::S_BLK, bs + ((u64)(nb0 + b) * (u64)KG_TOT + (u64)(kt * (BK / 32))) * (u64)(8 * C::NFR * 2),
                 (u32)C::S_BLK, full);
      ++q.it;
      if (++q.kt == q.k1) q.st = 0;
      prog = true;
    }
  }
  return prog;
}

template <int BN, int WN, int BK, int ST, int EPI, int CLC, int F16, int WMT, int NODQ>
DEVI void ws_gemm(const TmaDesc* tmA, const u32* __restrict__ Bq, const u16* __restrict__ Bs,
                  const u16* __restrict__ low0, const u16* __restrict__ low1, int ldl,
                  const u16* __restrict__ lb0, const u16* __restrict__ lb1, u16* __restrict__ out,
                  int M, int K, int N, int ldo, int dpu, int S, float* __restrict__ wsp, int* __restrict__ cnt,
                  unsigned char* smem_raw) {
  // 288 threads: warps 0-3 = MMA squad 0, warps 4-7 = MMA squad 1, warp 8 lane 0 = producer (both squads' TMA
  // streams + the CTA's single CLC issuer, non-blocking).  No setmaxnreg: every thread gets the 224-register
  // launch budget, which the 128-accumulator MMA warps need (the 40/232 split compiled to 168 and spilled).
  typedef WsCfg<BN, WN, BK, ST, EPI, CLC, F16, WMT> C;
  static_assert(C::MW == 4, "4 MMA warps per squad");
  static_assert(C::SMEM <= 101376, "smem");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const u32 sbase = (smem_u32(smem_raw) + 1023u) & ~1023u;
  const u32 aux0 = sbase + 2 * C::RING;
  if (tid == 0) {
    for (int sq = 0; sq < 2; ++sq) {
      const u32 ax = aux0 + sq * C::AUX;
      for (int s = 0; s < ST; ++s) {
        mbar_init(ax + 8 * s, 1);
        mbar_init(ax + 8 * (ST + s), C::MW);
      }
      mbar_init(ax + 8 * (2 * ST), 1);
      mbar_init(ax + 8 * (2 * ST + 1), 1);
      mbar_init(ax + 8 * (2 * ST + 2), C::MW);
      mbar_init(ax + 8 * (2 * ST + 3), C::MW);
    }
    mbar_init(aux0 + 2 * C::AUX, 1);
    mbar_fence_init();
  }
  __syncthreads();
  pdl_launch();

  const int KT = K / BK, KS_TOT = K / 16, KG_TOT = K / 32;
  const int ntm = (M + C::BM - 1) / C::BM, ntn = N / BN, total = ntm * ntn;
  const int units = S > 1 ? dpu + (total - dpu) * S : total;

  if (warp == 8) {
    // ===================== producer =====================
    if (lane != 0) return;
    if (F16 && EPI == 0) pdl_wait();  // the fp16 copy of X is produced by the PDL predecessor
    const u32 clc_bar = aux0 + 2 * C::AUX, resp = clc_bar + 16;
    const char* bq = (const char*)Bq;
    const char* bs = (const char*)Bs;
    SqState q0, q1;
    q0.st = q1.st = 0;
    q0.it = q1.it = q0.ti = q1.ti = 0;
    q0.unit = q1.unit = -1;
    q0.kt = q1.kt = q0.k1 = q1.k1 = q0.tm = q1.tm = q0.tn = q1.tn = 0;
    int sn0 = 2 * (int)blockIdx.x, sn1 = sn0 + 1;
    int stolen = -2, clc_state = 0;  // clc_state: 0 idle, 1 request in flight, 2 failed (no more work)
    bool first0 = true;
    u32 cph = 0;
    if (CLC) {
      mbar_expect_tx(clc_bar, 16u);
      clc_try_cancel(resp, clc_bar);
      clc_state = 1;
    }
    while (q0.st != 3 || q1.st != 3) {
      bool prog = false;
      if (CLC && clc_state == 1 && mbar_test(clc_bar, cph & 1u)) {
        ++cph;
        fence_proxy_async_smem();
        const int r = clc_result(resp);
        fence_proxy_async_smem();
        if (r >= 0) { stolen = r; clc_state = 0; } else { clc_state = 2; }
        prog = true;
      }
      prog |= prod_step<C, BK, ST, CLC>(q0, 0, sn0, stolen, clc_state, first0, aux0, sbase, clc_bar, resp, tmA, bq, bs,
                                        units, dpu, S, KT, ntm, ntn, KS_TOT, KG_TOT);
      prog |= prod_step<C, BK, ST, CLC>(q1, 1, sn1, stolen, clc_state, first0, aux0, sbase, clc_bar, resp, tmA, bq, bs,
                                        units, dpu, S, KT, ntm, ntn, KS_TOT, KG_TOT);
      if (!prog) asm volatile("nanosleep.u32 32;\n");
    }
    return;
  }

  // ===================== MMA squads =====================
  const int sq = warp >> 2;
  const int lw = warp & 3;
  const int wn = lw % C::WARPS_N, wm = lw / C::WARPS_N;
  const int g = lane >> 2, t = lane & 3;
  const u32 ax = aux0 + sq * C::AUX;
  const u32 ring = sbase + sq * C::RING;
  const int q = lane >> 3, r8 = lane & 7;
  const int fsw = swz_x<BK>(r8);
  const int a_kh = q >> 1;
  u32 a_off[C::MFR];
#pragma unroll
  for (int i = 0; i < C::MFR; ++i) a_off[i] = (u32)((wm * C::WM + i * 16 + ((q & 1) << 3) + r8) * (BK * 2));
  const u32 b_off = (u32)(wn * C::B_BLK + lane * (C::WORDS * 4));
  const u32 s_off = (u32)(wn * C::S_BLK + g * (C::NFR * 2));
  bool waited = false;
  int it = 0, ti = 0;
  while (true) {
    int unit;
    {
      const int slot = ti & 1;
      mbar_wait(ax + 8 * (2 * ST + slot), (u32)((ti >> 1) & 1));
      unit = (int)lds32m(ax + C::OFF_TID + 4 * slot);
      __syncwarp();
      if (lane == 0) mbar_arrive(ax + 8 * (2 * ST + 2 + slot));
      ++ti;
    }
    if (unit < 0) break;
    int tile, k0, k1, sp;
    decode_unit(unit, dpu, S, KT, tile, k0, k1, sp);
    int tm, tn;
    tile_coords(tile, ntm, ntn, tm, tn);
    const int m0 = tm * C::BM;
    float acc[C::MFR][C::NFR][4];
#pragma unroll
    for (int i = 0; i < C::MFR; ++i)
#pragma unroll
      for (int jj = 0; jj < C::NFR; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[i][jj][e] = 0.f;

    for (int kt = k0; kt < k1; ++kt, ++it) {
      const int s = it % ST;
      mbar_wait(ax + 8 * s, (u32)((it / ST) & 1));
      const u32 sa = ring + s * C::STAGE, sb = sa + C::A_ST, ss = sb + C::B_ST;
      u32 af[2][C::MFR][4];
      u32 bw[2][4];
      u32 sw[2][4];
      u32 hacc[C::MFR][C::NFR][2];
      if (F16) {
#pragma unroll
        for (int i = 0; i < C::MFR; ++i)
#pragma unroll
          for (int jj = 0; jj < C::NFR; ++jj) hacc[i][jj][0] = hacc[i][jj][1] = 0u;
      }
      if (C::ABUF == 2) {
#pragma unroll
        for (int i = 0; i < C::MFR; ++i) ldsm4(sa + a_off[i] + ((a_kh ^ fsw) << 4), af[0][i][0], af[0][i][1], af[0][i][2], af[0][i][3]);
        lds_words<C::WORDS>(sb + b_off, bw[0]);
        lds_words<C::WORDS>(ss + s_off, sw[0]);
      }
#pragma unroll
      for (int ks = 0; ks < C::KSTEPS; ++ks) {
        const int cur = C::ABUF == 2 ? (ks & 1) : 0;
        if (C::ABUF == 1) {  // load this k-step's operands; the dequant ALU work below covers their latency
#pragma unroll
          for (int i = 0; i < C::MFR; ++i)
            ldsm4(sa + a_off[i] + (((ks * 2 + a_kh) ^ fsw) << 4), af[0][i][0], af[0][i][1], af[0][i][2], af[0][i][3]);
          lds_words<C::WORDS>(sb + b_off + ks * (32 * C::WORDS * 4), bw[0]);
          if ((ks & 1) == 0) lds_words<C::WORDS>(ss + s_off + (ks >> 1) * (8 * C::NFR * 2), sw[(ks >> 1) & 1]);
        } else if (ks + 1 < C::KSTEPS) {
#pragma unroll
          for (int i = 0; i < C::MFR; ++i)
            ldsm4(sa + a_off[i] + ((((ks + 1) * 2 + a_kh) ^ fsw) << 4), af[cur ^ 1][i][0], af[cur ^ 1][i][1], af[cur ^ 1][i][2], af[cur ^ 1][i][3]);
          lds_words<C::WORDS>(sb + b_off + (ks + 1) * (32 * C::WORDS * 4), bw[cur ^ 1]);
          if (((ks + 1) & 1) == 0) lds_words<C::WORDS>(ss + s_off + ((ks + 1) >> 1) * (8 * C::NFR * 2), sw[((ks + 1) >> 1) & 1]);
        }
        const int si = (ks >> 1) & 1;
#pragma unroll
        for (int w = 0; w < C::WORDS; ++w) {
          const u32 x = bw[cur][w];
          const u32 sc = sw[si][w];
          const u32 s0 = prmt(sc, 0x1010u), s1 = prmt(sc, 0x3232u);
          u32 b00, b01, b10, b11;
          if (NODQ) {  // diagnostic only: raw words straight into the MMA (garbage values, zero dequant ALU)
            b00 = b01 = b10 = b11 = x;
          } else if (F16) {
            dq4h(x, s0, s1, b00, b01, b10, b11);
          } else {
            b00 = dq2(x, s0); b01 = dq2(x >> 4, s0); b10 = dq2(x >> 8, s1); b11 = dq2(x >> 12, s1);
          }
#pragma unroll
          for (int i = 0; i < C::MFR; ++i) {
            if (F16) {
              mma16816_hacc(hacc[i][2 * w], af[cur][i][0], af[cur][i][1], af[cur][i][2], af[cur][i][3], b00, b01);
              mma16816_hacc(hacc[i][2 * w + 1], af[cur][i][0], af[cur][i][1], af[cur][i][2], af[cur][i][3], b10, b11);
            } else {
              mma16816(acc[i][2 * w], af[cur][i][0], af[cur][i][1], af[cur][i][2], af[cur][i][3], b00, b01);
              mma16816(acc[i][2 * w + 1], af[cur][i][0], af[cur][i][1], af[cur][i][2], af[cur][i][3], b10, b11);
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(ax + 8 * (ST + s));
      if (F16) {  // promote the stage's fp16 partial sums (BK products each) into fp32
#pragma unroll
        for (int i = 0; i < C::MFR; ++i)
#pragma unroll
          for (int jj = 0; jj < C::NFR; ++jj) {
            float x0, x1, x2, x3;
            h2f2(hacc[i][jj][0], x0, x1);
            h2f2(hacc[i][jj][1], x2, x3);
            acc[i][jj][0] += x0; acc[i][jj][1] += x1; acc[i][jj][2] += x2; acc[i][jj][3] += x3;
          }
      }
    }

    if (sp >= 0) {
      // ---------------- stream-K tail: publish the partial, last arriver reduces in split order
      const int tt = tile - dpu;
      float* mine = wsp + ((u64)(tt * S + sp) * 4u + (u64)lw) * (u64)(C::NACC * 32) + lane;
#pragma unroll
      for (int i = 0; i < C::MFR; ++i)
#pragma unroll
        for (int jj = 0; jj < C::NFR; ++jj)
#pragma unroll
          for (int e = 0; e < 4; ++e) mine[((i * C::NFR + jj) * 4 + e) * 32] = acc[i][jj][e];
      __threadfence();
      __syncwarp();
      u32 old = 0u;
      if (lane == 0) old = atom_add((u32*)(cnt + tt * 4 + lw), 1u);
      asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1f, 0xffffffff;\n" : "+r"(old));
      if ((int)old != S - 1) continue;
      __threadfence();
      const float* base = wsp + ((u64)(tt * S) * 4u + (u64)lw) * (u64)(C::NACC * 32) + lane;
      for (int s2 = 0; s2 < S; ++s2) {
        const float* src = base + (u64)s2 * 4u * (u64)(C::NACC * 32);
#pragma unroll
        for (int i = 0; i < C::MFR; ++i)
#pragma unroll
          for (int jj = 0; jj < C::NFR; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float v = ldg_cg_f32(src + ((i * C::NFR + jj) * 4 + e) * 32);
              acc[i][jj][e] = s2 == 0 ? v : acc[i][jj][e] + v;
            }
      }
      if (lane == 0) cnt[tt * 4 + lw] = 0;  // re-arm for the next launch (graph replay)
    }

    // ---------------- epilogue (the other squad keeps the tensor cores busy meanwhile)
    if (!waited) {
      pdl_wait();
      waited = true;
    }
#pragma unroll
    for (int i = 0; i < C::MFR; ++i) {
      const int rl = wm * C::WM + i * 16 + g;
      const int r_lo = m0 + rl, r_hi = r_lo + 8;
      u32 a0[4], a1[4];
      ld_lowfrag(low0, ldl, r_lo, r_hi, M, t, a0);
      if (EPI == 0) ld_lowfrag(low1, ldl, r_lo, r_hi, M, t, a1);
      if (EPI == 0) {
#pragma unroll
        for (int jj = 0; jj < C::NFR / 2; ++jj) {
          const int cl = wn * (WN / 2) + jj * 8;
          const u16* gbr = lb0 + (u64)(tn * C::NCOL + cl + g) * 16u;
          const u16* ubr = lb1 + (u64)(tn * C::NCOL + cl + g) * 16u;
          float lg[4] = {0.f, 0.f, 0.f, 0.f}, lu[4] = {0.f, 0.f, 0.f, 0.f};
          mma16816(lg, a0[0], a0[1], a0[2], a0[3], ldg_nc32(gbr + 2 * t), ldg_nc32(gbr + 8 + 2 * t));
          mma16816(lu, a1[0], a1[1], a1[2], a1[3], ldg_nc32(ubr + 2 * t), ldg_nc32(ubr + 8 + 2 * t));
          const float* cg = acc[i][jj];
          const float* cu = acc[i][jj + C::NFR / 2];
          const int col = tn * C::NCOL + cl + 2 * t;
          const float v0 = gu_elem(cg[0], cu[0], lg[0], lu[0]), v1 = gu_elem(cg[1], cu[1], lg[1], lu[1]);
          const float v2 = gu_elem(cg[2], cu[2], lg[2], lu[2]), v3 = gu_elem(cg[3], cu[3], lg[3], lu[3]);
          // mid cut = bf16 rounding; the fp16 datapath stores that bf16 value as fp16 for the fp16 Down GEMM
          const u32 o01 = F16 ? pack_h2(rbf(v0), rbf(v1)) : pack_bf2(v0, v1);
          const u32 o23 = F16 ? pack_h2(rbf(v2), rbf(v3)) : pack_bf2(v2, v3);
          if (r_lo < M) stg32(out + (u64)r_lo * (u64)ldo + col, o01);
          if (r_hi < M) stg32(out + (u64)r_hi * (u64)ldo + col, o23);
        }
      } else {
#pragma unroll
        for (int jj = 0; jj < C::NFR; ++jj) {
          const int cl = wn * WN + jj * 8;
          const u16* dbr = lb0 + (u64)(tn * BN + cl + g) * 16u;
          float l[4] = {0.f, 0.f, 0.f, 0.f};
          mma16816(l, a0[0], a0[1], a0[2], a0[3], ldg_nc32(dbr + 2 * t), ldg_nc32(dbr + 8 + 2 * t));
          const float* cd = acc[i][jj];
          const int col = tn * BN + cl + 2 * t;
          if (r_lo < M) stg32(out + (u64)r_lo * (u64)ldo + col, pack_bf2(dn_elem(cd[0], l[0]), dn_elem(cd[1], l[1])));
          if (r_hi < M) stg32(out + (u64)r_hi * (u64)ldo + col, pack_bf2(dn_elem(cd[2], l[2]), dn_elem(cd[3], l[3])));
        }
      }
    }
  }
}


// ================= MLP round: dequant-ahead warp specialization =================
// Idea: 31B planner proposal dense_M7131_p2 ("producer/specialized warps dequantize INT4 into BF16
// shared memory; MMA squads run pure BF16").  The 12B refiner did not deliver a refinement of it.  Physical
// realization below is Claude's: one 3-phase ring per stage (WarpSpeed SmemResource with phases L -> D -> C):
//   full[s]   producer arrive.expect_tx  (TMA A tile + 1-D bulk packed words + scales)        -> phase L done
//   dqfull[s] DQW dequant warps          (INT4+group32 scale -> BF16 B fragments, MMA reg order) -> phase D done
//   empty[s]  4 MMA warps                (ldmatrix A + lds128 B, no dequant ALU)              -> slot reusable
// Warps 0..MW-1: MMA squad (WARPS_M x WARPS_N warps of WM x WN), then DQW dequant warps (NBLK/DQW WN blocks each),
// last warp lane 0: producer (+CLC issuer).  Register cap = 16384 / (32 * warps on the busiest SM sub-partition):
// 7 warps -> 255, 9..12 -> 168, 13..16 -> 128.  WM=64 (8 MMA warps) is the coupled change dequant-ahead enables:
// smaller warp tiles no longer dequantize the same B twice, and 64 accumulators fit the 168/128 caps.
// Dequantized values and the k-order of every MMA are those of v1/v3, so outputs are bitwise identical.
DEVI void sts128(u32 a, u32 x, u32 y, u32 z, u32 w) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

template <int BN, int WN, int BK, int ST, int EPI, int CLC, int DQW, int WMT>
struct WsdCfg : WsCfg<BN, WN, BK, ST, EPI, CLC, 0, WMT> {
  typedef WsCfg<BN, WN, BK, ST, EPI, CLC, 0, WMT> B0;
  static constexpr int BD_BLK = B0::KSTEPS * B0::WORDS * 32 * 16;   // [ks][word][lane] x 16 B of bf16x2 fragments
  static constexpr int BD_ST = B0::NBLK * BD_BLK;
  static constexpr int BD_OFF = B0::A_ST + B0::B_ST + B0::S_ST;
  static constexpr int STAGE = ((BD_OFF + BD_ST + B0::ATOM - 1) / B0::ATOM) * B0::ATOM;
  static constexpr int RING = ((ST * STAGE + 1023) / 1024) * 1024;
  static constexpr int OFF_DQ = B0::OFF_TID + 16;
  static constexpr int AUX = ((OFF_DQ + 8 * ST + 15) / 16) * 16;
  static constexpr int SMEM = RING + AUX + B0::GAUX + 1024;
};

template <int BN, int WN, int BK, int ST, int EPI, int CLC, int DQW, int WMT>
DEVI void wsd_gemm(const TmaDesc* tmA, const u32* __restrict__ Bq, const u16* __restrict__ Bs,
                   const u16* __restrict__ low0, const u16* __restrict__ low1, int ldl,
                   const u16* __restrict__ lb0, const u16* __restrict__ lb1, u16* __restrict__ out,
                   int M, int K, int N, int ldo, int dpu, int S, float* __restrict__ wsp, int* __restrict__ cnt,
                   unsigned char* smem_raw) {
  typedef WsdCfg<BN, WN, BK, ST, EPI, CLC, DQW, WMT> C;
  static_assert(C::MW + DQW + 1 <= 16, "warps");
  static_assert(C::NBLK % DQW == 0, "dequant warps own whole WN blocks");
  static_assert(C::SMEM <= 101376, "smem");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const u32 sbase = (smem_u32(smem_raw) + 1023u) & ~1023u;
  const u32 ax = sbase + C::RING;
  const u32 clc_bar = ax + C::AUX, resp = clc_bar + 16;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(ax + 8 * s, 1);
      mbar_init(ax + 8 * (ST + s), C::MW);
      mbar_init(ax + C::OFF_DQ + 8 * s, DQW);
    }
    mbar_init(ax + 8 * (2 * ST), 1);
    mbar_init(ax + 8 * (2 * ST + 1), 1);
    mbar_init(ax + 8 * (2 * ST + 2), C::MW + DQW);
    mbar_init(ax + 8 * (2 * ST + 3), C::MW + DQW);
    mbar_init(clc_bar, 1);
    mbar_fence_init();
  }
  __syncthreads();
  pdl_launch();

  const int KT = K / BK, KS_TOT = K / 16, KG_TOT = K / 32;
  const int ntm = (M + C::BM - 1) / C::BM, ntn = N / BN, total = ntm * ntn;
  const int units = S > 1 ? dpu + (total - dpu) * S : total;

  if (warp == C::MW + DQW) {
    // ===================== producer (one squad's stream + CLC) =====================
    if (lane != 0) return;
    const char* bq = (const char*)Bq;
    const char* bs = (const char*)Bs;
    SqState q;
    q.st = 0;
    q.it = q.ti = 0;
    q.unit = -1;
    q.kt = q.k1 = q.tm = q.tn = 0;
    int sn = (int)blockIdx.x, stolen = -2, clc_state = 0;
    bool first0 = true;
    u32 cph = 0;
    if (CLC) {
      mbar_expect_tx(clc_bar, 16u);
      clc_try_cancel(resp, clc_bar);
      clc_state = 1;
    }
    while (q.st != 3) {
      bool prog = false;
      if (CLC && clc_state == 1 && mbar_test(clc_bar, cph & 1u)) {
        ++cph;
        fence_proxy_async_smem();
        const int r = clc_result(resp);
        fence_proxy_async_smem();
        if (r >= 0) { stolen = r; clc_state = 0; } else { clc_state = 2; }
        prog = true;
      }
      prog |= prod_step<C, BK, ST, CLC, 1>(q, 0, sn, stolen, clc_state, first0, ax, sbase, clc_bar, resp, tmA, bq, bs,
                                           units, dpu, S, KT, ntm, ntn, KS_TOT, KG_TOT);
      if (!prog) asm volatile("nanosleep.u32 32;\n");
    }
    return;
  }

  const int g = lane >> 2, t = lane & 3;
  int ti = 0, it = 0;
  if (warp >= C::MW) {
    // ===================== dequant squad: phase D =====================
    const int dw = warp - C::MW;
    while (true) {
      const int slot = ti & 1;
      mbar_wait(ax + 8 * (2 * ST + slot), (u32)((ti >> 1) & 1));
      const int unit = (int)lds32m(ax + C::OFF_TID + 4 * slot);
      __syncwarp();
      if (lane == 0) mbar_arrive(ax + 8 * (2 * ST + 2 + slot));
      ++ti;
      if (unit < 0) break;
      int tile, k0, k1, sp;
      decode_unit(unit, dpu, S, KT, tile, k0, k1, sp);
      for (int kt = k0; kt < k1; ++kt, ++it) {
        const int s = it % ST;
        mbar_wait(ax + 8 * s, (u32)((it / ST) & 1));
        const u32 sa = sbase + s * C::STAGE, sb = sa + C::A_ST, ss = sb + C::B_ST, sd = sa + C::BD_OFF;
#pragma unroll
        for (int j = 0; j < C::NBLK / DQW; ++j) {
          const int wn = dw * (C::NBLK / DQW) + j;
          const u32 b_off = (u32)(wn * C::B_BLK + lane * (C::WORDS * 4));
          const u32 s_off = (u32)(wn * C::S_BLK + g * (C::NFR * 2));
          u32 sw[4];
#pragma unroll
          for (int ks = 0; ks < C::KSTEPS; ++ks) {
            u32 bw[4];
            lds_words<C::WORDS>(sb + b_off + ks * (32 * C::WORDS * 4), bw);
            if ((ks & 1) == 0) lds_words<C::WORDS>(ss + s_off + (ks >> 1) * (8 * C::NFR * 2), sw);
#pragma unroll
            for (int w = 0; w < C::WORDS; ++w) {
              const u32 x = bw[w], sc = sw[w];
              const u32 s0 = prmt(sc, 0x1010u), s1 = prmt(sc, 0x3232u);
              sts128(sd + wn * C::BD_BLK + (ks * C::WORDS + w) * 512 + lane * 16,
                     dq2(x, s0), dq2(x >> 4, s0), dq2(x >> 8, s1), dq2(x >> 12, s1));
            }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(ax + C::OFF_DQ + 8 * s);
      }
    }
    return;
  }

  // ===================== MMA squad: phase C =====================
  const int lw = warp, wn = warp % C::WARPS_N, wm = warp / C::WARPS_N;
  const int q = lane >> 3, r8 = lane & 7;
  const int fsw = swz_x<BK>(r8);
  const int a_kh = q >> 1;
  u32 a_off[C::MFR];
#pragma unroll
  for (int i = 0; i < C::MFR; ++i) a_off[i] = (u32)((wm * C::WM + i * 16 + ((q & 1) << 3) + r8) * (BK * 2));
  bool waited = false;
  while (true) {
    int unit;
    {
      const int slot = ti & 1;
      mbar_wait(ax + 8 * (2 * ST + slot), (u32)((ti >> 1) & 1));
      unit = (int)lds32m(ax + C::OFF_TID + 4 * slot);
      __syncwarp();
      if (lane == 0) mbar_arrive(ax + 8 * (2 * ST + 2 + slot));
      ++ti;
    }
    if (unit < 0) break;
    int tile, k0, k1, sp;
    decode_unit(unit, dpu, S, KT, tile, k0, k1, sp);
    int tm, tn;
    tile_coords(tile, ntm, ntn, tm, tn);
    const int m0 = tm * C::BM;
    float acc[C::MFR][C::NFR][4];
#pragma unroll
    for (int i = 0; i < C::MFR; ++i)
#pragma unroll
      for (int jj = 0; jj < C::NFR; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[i][jj][e] = 0.f;

    for (int kt = k0; kt < k1; ++kt, ++it) {
      const int s = it % ST;
      const u32 par = (u32)((it / ST) & 1);
      mbar_wait(ax + 8 * s, par);                  // A tile landed (async proxy, complete_tx)
      mbar_wait(ax + C::OFF_DQ + 8 * s, par);      // BF16 B fragments published by the dequant squad
      const u32 sa = sbase + s * C::STAGE, sd = sa + C::BD_OFF + wn * C::BD_BLK;
#pragma unroll
      for (int ks = 0; ks < C::KSTEPS; ++ks) {
        // B fragments of this k-step first (WORDS x 4 regs), then one A fragment per M-row: 4 live A regs, not 32
        u32 bf[C::WORDS][4];
#pragma unroll
        for (int w = 0; w < C::WORDS; ++w)
          lds128(sd + (ks * C::WORDS + w) * 512 + lane * 16, bf[w][0], bf[w][1], bf[w][2], bf[w][3]);
#pragma unroll
        for (int i = 0; i < C::MFR; ++i) {
          u32 a0, a1, a2, a3;
          ldsm4(sa + a_off[i] + (((ks * 2 + a_kh) ^ fsw) << 4), a0, a1, a2, a3);
#pragma unroll
          for (int w = 0; w < C::WORDS; ++w) {
            mma16816(acc[i][2 * w], a0, a1, a2, a3, bf[w][0], bf[w][1]);
            mma16816(acc[i][2 * w + 1], a0, a1, a2, a3, bf[w][2], bf[w][3]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(ax + 8 * (ST + s));
    }

    if (sp >= 0) {
      const int tt = tile - dpu;
      float* mine = wsp + ((u64)(tt * S + sp) * (u64)C::MW + (u64)lw) * (u64)(C::NACC * 32) + lane;
#pragma unroll
      for (int i = 0; i < C::MFR; ++i)
#pragma unroll
        for (int jj = 0; jj < C::NFR; ++jj)
#pragma unroll
          for (int e = 0; e < 4; ++e) mine[((i * C::NFR + jj) * 4 + e) * 32] = acc[i][jj][e];
      __threadfence();
      __syncwarp();
      u32 old = 0u;
      if (lane == 0) old = atom_add((u32*)(cnt + tt * C::MW + lw), 1u);
      asm volatile("shfl.sync.idx.b32 %0, %0, 0, 0x1f, 0xffffffff;\n" : "+r"(old));
      if ((int)old != S - 1) continue;
      __threadfence();
      const float* base = wsp + ((u64)(tt * S) * (u64)C::MW + (u64)lw) * (u64)(C::NACC * 32) + lane;
      for (int s2 = 0; s2 < S; ++s2) {
        const float* src = base + (u64)s2 * (u64)C::MW * (u64)(C::NACC * 32);
#pragma unroll
        for (int i = 0; i < C::MFR; ++i)
#pragma unroll
          for (int jj = 0; jj < C::NFR; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float v = ldg_cg_f32(src + ((i * C::NFR + jj) * 4 + e) * 32);
              acc[i][jj][e] = s2 == 0 ? v : acc[i][jj][e] + v;
            }
      }
      if (lane == 0) cnt[tt * C::MW + lw] = 0;
    }

    if (!waited) {
      pdl_wait();
      waited = true;
    }
#pragma unroll
    for (int i = 0; i < C::MFR; ++i) {
      const int rl = wm * C::WM + i * 16 + g;
      const int r_lo = m0 + rl, r_hi = r_lo + 8;
      u32 a0[4], a1[4];
      ld_lowfrag(low0, ldl, r_lo, r_hi, M, t, a0);
      if (EPI == 0) ld_lowfrag(low1, ldl, r_lo, r_hi, M, t, a1);
      if (EPI == 0) {
#pragma unroll
        for (int jj = 0; jj < C::NFR / 2; ++jj) {
          const int cl = wn * (WN / 2) + jj * 8;
          const u16* gbr = lb0 + (u64)(tn * C::NCOL + cl + g) * 16u;
          const u16* ubr = lb1 + (u64)(tn * C::NCOL + cl + g) * 16u;
          float lg[4] = {0.f, 0.f, 0.f, 0.f}, lu[4] = {0.f, 0.f, 0.f, 0.f};
          mma16816(lg, a0[0], a0[1], a0[2], a0[3], ldg_nc32(gbr + 2 * t), ldg_nc32(gbr + 8 + 2 * t));
          mma16816(lu, a1[0], a1[1], a1[2], a1[3], ldg_nc32(ubr + 2 * t), ldg_nc32(ubr + 8 + 2 * t));
          const float* cg = acc[i][jj];
          const float* cu = acc[i][jj + C::NFR / 2];
          const int col = tn * C::NCOL + cl + 2 * t;
          if (r_lo < M) stg32(out + (u64)r_lo * (u64)ldo + col, pack_bf2(gu_elem(cg[0], cu[0], lg[0], lu[0]), gu_elem(cg[1], cu[1], lg[1], lu[1])));
          if (r_hi < M) stg32(out + (u64)r_hi * (u64)ldo + col, pack_bf2(gu_elem(cg[2], cu[2], lg[2], lu[2]), gu_elem(cg[3], cu[3], lg[3], lu[3])));
        }
      } else {
#pragma unroll
        for (int jj = 0; jj < C::NFR; ++jj) {
          const int cl = wn * WN + jj * 8;
          const u16* dbr = lb0 + (u64)(tn * BN + cl + g) * 16u;
          float l[4] = {0.f, 0.f, 0.f, 0.f};
          mma16816(l, a0[0], a0[1], a0[2], a0[3], ldg_nc32(dbr + 2 * t), ldg_nc32(dbr + 8 + 2 * t));
          const float* cd = acc[i][jj];
          const int col = tn * BN + cl + 2 * t;
          if (r_lo < M) stg32(out + (u64)r_lo * (u64)ldo + col, pack_bf2(dn_elem(cd[0], l[0]), dn_elem(cd[1], l[1])));
          if (r_hi < M) stg32(out + (u64)r_hi * (u64)ldo + col, pack_bf2(dn_elem(cd[2], l[2]), dn_elem(cd[3], l[3])));
        }
      }
    }
  }
}

#define WSD_KERNEL(NAME, BN, WN, BK, ST, EPI, CLC, DQW, WMT)                                                         \
  extern "C" __global__ void __launch_bounds__(((128 / WMT) * (BN / WN) + DQW + 1) * 32, 1)                         \
      NAME(const __grid_constant__ TmaDesc tmA, const u32* Bq, const u16* Bs, const u16* low0, const u16* low1,     \
           int ldl, const u16* lb0, const u16* lb1, u16* out, int M, int K, int N, int ldo, int dpu, int S,         \
           float* wsp, int* cnt) {                                                                                  \
    extern __shared__ __align__(1024) unsigned char fg4_smem[];                                                     \
    wsd_gemm<BN, WN, BK, ST, EPI, CLC, DQW, WMT>(&tmA, Bq, Bs, low0, low1, ldl, lb0, lb1, out, M, K, N, ldo, dpu, S,      \
                                            wsp, cnt, fg4_smem);                                                    \
  }

#define WS_KERNEL(NAME, BN, WN, BK, ST, EPI, CLC, F16, WMT, NODQ)                                                           \
  extern "C" __global__ void __launch_bounds__(288, 1)                                                              \
      NAME(const __grid_constant__ TmaDesc tmA, const u32* Bq, const u16* Bs, const u16* low0, const u16* low1,     \
           int ldl, const u16* lb0, const u16* lb1, u16* out, int M, int K, int N, int ldo, int dpu, int S,         \
           float* wsp, int* cnt) {                                                                                  \
    extern __shared__ __align__(1024) unsigned char fg3_smem[];                                                     \
    ws_gemm<BN, WN, BK, ST, EPI, CLC, F16, WMT, NODQ>(&tmA, Bq, Bs, low0, low1, ldl, lb0, lb1, out, M, K, N, ldo,  \
                                                      dpu, S, wsp, cnt, fg3_smem);                                  \
  }

#ifndef FG_NO_CLC
WS_KERNEL(ws_gu_a, 128, 64, 64, 2, 0, 1, 0, 64, 0)
WS_KERNEL(ws_gu_b, 128, 64, 32, 4, 0, 1, 0, 64, 0)
WS_KERNEL(ws_dn_a, 128, 64, 64, 2, 1, 1, 0, 64, 0)
WS_KERNEL(ws_dn_b, 128, 64, 32, 4, 1, 1, 0, 64, 0)
WS_KERNEL(ws_gu_w, 128, 32, 64, 2, 0, 1, 0, 128, 0)
WS_KERNEL(ws_gu_wb, 128, 32, 32, 4, 0, 1, 0, 128, 0)
WS_KERNEL(ws_dn_w, 128, 32, 64, 2, 1, 1, 0, 128, 0)
WS_KERNEL(ws_dn_wb, 128, 32, 32, 4, 1, 1, 0, 128, 0)
WS_KERNEL(ws_gu_h, 64, 32, 64, 2, 0, 1, 1, 64, 0)
WS_KERNEL(ws_gu_hb, 64, 32, 32, 4, 0, 1, 1, 64, 0)
WS_KERNEL(ws_dn_h, 64, 32, 64, 2, 1, 1, 1, 64, 0)
WS_KERNEL(ws_dn_hb, 64, 32, 32, 4, 1, 1, 1, 64, 0)
WS_KERNEL(ws_gu_hw, 64, 16, 64, 2, 0, 1, 1, 128, 0)
WS_KERNEL(ws_dn_hw, 64, 16, 64, 2, 1, 1, 1, 128, 0)
WS_KERNEL(dq_gu_a, 128, 64, 64, 2, 0, 1, 0, 64, 1)
WS_KERNEL(dq_gu_w, 128, 32, 64, 2, 0, 1, 0, 128, 1)
WS_KERNEL(dq_dn_w, 128, 32, 64, 2, 1, 1, 0, 128, 1)
#endif
WS_KERNEL(ws_gu_s, 128, 64, 64, 2, 0, 0, 0, 64, 0)
WS_KERNEL(ws_dn_s, 128, 64, 64, 2, 1, 0, 0, 64, 0)
WS_KERNEL(ws_gu_ws, 128, 32, 64, 2, 0, 0, 0, 128, 0)
WS_KERNEL(ws_dn_ws, 128, 32, 64, 2, 1, 0, 0, 128, 0)
WS_KERNEL(ws_gu_hs, 64, 32, 64, 2, 0, 0, 1, 64, 0)
WS_KERNEL(ws_dn_hs, 64, 32, 64, 2, 1, 0, 1, 64, 0)
#ifndef FG_NO_CLC
WSD_KERNEL(wsd_gu_a, 128, 32, 64, 2, 0, 1, 2, 128)
WSD_KERNEL(wsd_gu_b, 128, 32, 64, 2, 0, 1, 4, 128)
WSD_KERNEL(wsd_gu_c, 128, 32, 32, 4, 0, 1, 2, 128)
WSD_KERNEL(wsd_dn_a, 128, 32, 64, 2, 1, 1, 2, 128)
WSD_KERNEL(wsd_dn_b, 128, 32, 64, 2, 1, 1, 4, 128)
WSD_KERNEL(wsd_dn_c, 128, 32, 32, 4, 1, 1, 2, 128)
WSD_KERNEL(wsd_gu_m, 128, 32, 64, 2, 0, 1, 2, 64)
WSD_KERNEL(wsd_gu_m4, 128, 32, 64, 2, 0, 1, 4, 64)
WSD_KERNEL(wsd_dn_m, 128, 32, 64, 2, 1, 1, 2, 64)
WSD_KERNEL(wsd_dn_m4, 128, 32, 64, 2, 1, 1, 4, 64)
#endif
WSD_KERNEL(wsd_gu_s, 128, 32, 64, 2, 0, 0, 2, 128)
WSD_KERNEL(wsd_dn_s, 128, 32, 64, 2, 1, 0, 2, 128)
WSD_KERNEL(wsd_gu_ms, 128, 32, 64, 2, 0, 0, 2, 64)
WSD_KERNEL(wsd_dn_ms, 128, 32, 64, 2, 1, 0, 2, 64)
"""


# name -> (BM, BN, WM, WN, BK, ST, EPI, MINB)   must mirror the TC_KERNEL instantiations
TC_CFGS = {  # v1: cp.async multistage, one tile per CTA, 2 CTAs/SM (epilogue operands from L2 after the PDL wait)
    "fg_gu_c": (128, 128, 64, 64, 32, 3, 0, 2),
    "fg_gu_d": (128, 128, 64, 32, 64, 2, 0, 2),
    "fg_gu_e": (128, 128, 64, 32, 32, 4, 0, 2),
    "fg_gu_f": (128, 128, 64, 64, 32, 4, 0, 2),
    "fg_dn_c": (128, 128, 64, 32, 64, 4, 1, 1),
    "fg_dn_d": (128, 128, 64, 32, 64, 2, 1, 2),
    "fg_dn_e": (128, 128, 64, 32, 32, 4, 1, 2),
}
# name -> (BN, WN, BK, ST, EPI, CLC, F16, WM, NODQ)  must mirror the WS_KERNEL instantiations (v3, 384 thr, 1 CTA/SM)
#   WM=64 : 2x2 warps of 64xWN   -> each weight word dequantized by 2 warps (once per M-half of the tile)
#   WM=128: 1x4 warps of 128xWN  -> each weight word dequantized once per tile, its fragment feeds 8 MMAs
#   dq_*  : NODQ diagnostics (raw words into the MMA, wrong results) -> measures what the dequant costs
WS_CFGS = {
    "ws_gu_a": (128, 64, 64, 2, 0, 1, 0, 64, 0),
    "ws_gu_b": (128, 64, 32, 4, 0, 1, 0, 64, 0),
    "ws_gu_s": (128, 64, 64, 2, 0, 0, 0, 64, 0),
    "ws_dn_a": (128, 64, 64, 2, 1, 1, 0, 64, 0),
    "ws_dn_b": (128, 64, 32, 4, 1, 1, 0, 64, 0),
    "ws_dn_s": (128, 64, 64, 2, 1, 0, 0, 64, 0),
    "ws_gu_w": (128, 32, 64, 2, 0, 1, 0, 128, 0),
    "ws_gu_wb": (128, 32, 32, 4, 0, 1, 0, 128, 0),
    "ws_gu_ws": (128, 32, 64, 2, 0, 0, 0, 128, 0),
    "ws_dn_w": (128, 32, 64, 2, 1, 1, 0, 128, 0),
    "ws_dn_wb": (128, 32, 32, 4, 1, 1, 0, 128, 0),
    "ws_dn_ws": (128, 32, 64, 2, 1, 0, 0, 128, 0),
    "ws_gu_h": (64, 32, 64, 2, 0, 1, 1, 64, 0),
    "ws_gu_hb": (64, 32, 32, 4, 0, 1, 1, 64, 0),
    "ws_gu_hs": (64, 32, 64, 2, 0, 0, 1, 64, 0),
    "ws_gu_hw": (64, 16, 64, 2, 0, 1, 1, 128, 0),
    "ws_dn_h": (64, 32, 64, 2, 1, 1, 1, 64, 0),
    "ws_dn_hb": (64, 32, 32, 4, 1, 1, 1, 64, 0),
    "ws_dn_hs": (64, 32, 64, 2, 1, 0, 1, 64, 0),
    "ws_dn_hw": (64, 16, 64, 2, 1, 1, 1, 128, 0),
    "dq_gu_a": (128, 64, 64, 2, 0, 1, 0, 64, 1),
    "dq_gu_w": (128, 32, 64, 2, 0, 1, 0, 128, 1),
    "dq_dn_w": (128, 32, 64, 2, 1, 1, 0, 128, 1),
}
# MLP round dequant-ahead kernels (wsd_*): name -> DQW; their tuples join WS_CFGS so probe/tune/chain treat them alike
WSD_DQW = {"wsd_gu_a": 2, "wsd_gu_b": 4, "wsd_gu_c": 2, "wsd_dn_a": 2, "wsd_dn_b": 4, "wsd_dn_c": 2,
           "wsd_gu_s": 2, "wsd_dn_s": 2, "wsd_gu_m": 2, "wsd_gu_m4": 4, "wsd_dn_m": 2, "wsd_dn_m4": 4,
           "wsd_gu_ms": 2, "wsd_dn_ms": 2}
WS_CFGS.update({
    "wsd_gu_a": (128, 32, 64, 2, 0, 1, 0, 128, 0), "wsd_gu_b": (128, 32, 64, 2, 0, 1, 0, 128, 0),
    "wsd_gu_c": (128, 32, 32, 4, 0, 1, 0, 128, 0), "wsd_dn_a": (128, 32, 64, 2, 1, 1, 0, 128, 0),
    "wsd_dn_b": (128, 32, 64, 2, 1, 1, 0, 128, 0), "wsd_dn_c": (128, 32, 32, 4, 1, 1, 0, 128, 0),
    "wsd_gu_s": (128, 32, 64, 2, 0, 0, 0, 128, 0), "wsd_dn_s": (128, 32, 64, 2, 1, 0, 0, 128, 0),
    "wsd_gu_m": (128, 32, 64, 2, 0, 1, 0, 64, 0), "wsd_gu_m4": (128, 32, 64, 2, 0, 1, 0, 64, 0),
    "wsd_dn_m": (128, 32, 64, 2, 1, 1, 0, 64, 0), "wsd_dn_m4": (128, 32, 64, 2, 1, 1, 0, 64, 0),
    "wsd_gu_ms": (128, 32, 64, 2, 0, 0, 0, 64, 0), "wsd_dn_ms": (128, 32, 64, 2, 1, 0, 0, 64, 0)})
DQ_PAIRS = [("ws_gu_a", "dq_gu_a"), ("ws_gu_w", "dq_gu_w"), ("ws_dn_w", "dq_dn_w")]
V1_GU = ["fg_gu_c", "fg_gu_d", "fg_gu_e", "fg_gu_f"]
# M=8 hint-GEMV kernels: incumbent first, then the MLP round prefetch-depth variants
GEMV8_VARIANTS = {"gu": [("fg_gu8h", 256), ("fg_gu8h_d7a", 256), ("fg_gu8h_d7b", 256)],
                  "dn": [("fg_dn8h", 512), ("fg_dn8h_d6", 512), ("fg_dn8h_d7", 512)]}
V1_DN = ["fg_dn_d", "fg_dn_c", "fg_dn_e"]
REF_GU, REF_DN = "fg_gu_c", "fg_dn_d"   # bitwise references for the v3 probe (validated on G4)


def cfg_base(name):
    return name.split("+")[0]


def cfg_meta(name):
    """'+sk' suffix = same kernel with the stream-K tail enabled (runtime arguments only)."""
    name = cfg_base(name)
    if name in WSD_DQW:
        BN, WN, BK, ST, EPI, CLC, F16, WM, NODQ = WS_CFGS[name]
        BM, NFR, KSTEPS, MW = 128, WN // 8, BK // 16, (128 // WM) * (BN // WN)
        WORDS, NBLK = NFR // 2, BN // WN
        a_st, b_st, s_st = BM * BK * 2, NBLK * KSTEPS * 32 * WORDS * 4, NBLK * (BK // 32) * 8 * NFR * 2
        bd_st = NBLK * KSTEPS * WORDS * 32 * 16
        atom = 1024 if BK == 64 else 512
        stage = -(-(a_st + b_st + s_st + bd_st) // atom) * atom
        ring = -(-(ST * stage) // 1024) * 1024
        aux = -(-(8 * (2 * ST + 4) + 16 + 8 * ST) // 16) * 16
        return dict(BM=BM, BN=BN, WN=WN, BK=BK, ST=ST, EPI=EPI, MINB=1, CLC=CLC, F16=0, WM=WM, NODQ=0,
                    smem=ring + aux + 32 + 1024, threads=(MW + WSD_DQW[name] + 1) * 32, kind="wsd")
    if name in WS_CFGS:
        BN, WN, BK, ST, EPI, CLC, F16, WM, NODQ = WS_CFGS[name]
        BM = 128
        NFR, KSTEPS = WN // 8, BK // 16
        WORDS, NBLK = NFR // 2, BN // WN
        a_st, b_st, s_st = BM * BK * 2, NBLK * KSTEPS * 32 * WORDS * 4, NBLK * (BK // 32) * 8 * NFR * 2
        atom = 1024 if BK == 64 else 512
        stage = -(-(a_st + b_st + s_st) // atom) * atom
        ring = -(-(ST * stage) // 1024) * 1024
        off_tid = 8 * (2 * ST + 4)
        aux = (-(-(off_tid + 8) // 16)) * 16
        return dict(BM=BM, BN=BN, WN=WN, BK=BK, ST=ST, EPI=EPI, MINB=1, CLC=CLC, F16=F16, WM=WM, NODQ=NODQ,
                    smem=2 * ring + 2 * aux + 32 + 1024, threads=288, kind="ws")
    BM, BN, WM, WN, BK, ST, EPI, MINB = TC_CFGS[name]
    NFR, KSTEPS = WN // 8, BK // 16
    WORDS, NBLK = NFR // 2, BN // WN
    a_st, b_st, s_st = BM * BK * 2, NBLK * KSTEPS * 32 * WORDS * 4, NBLK * (BK // 32) * 8 * NFR * 2
    return dict(BM=BM, BN=BN, WN=WN, BK=BK, ST=ST, EPI=EPI, MINB=MINB, CLC=0, F16=0, smem=ST * (a_st + b_st + s_st),
                threads=(BM // WM) * (BN // WN) * 32, kind="v1")


# --------------------------------------------------------------------------------------------------
# Layout transforms (backend-agnostic: works on numpy arrays and torch tensors)
# --------------------------------------------------------------------------------------------------
def _perm(x, p):
    return x.permute(*p) if hasattr(x, "permute") else x.transpose(p)


def merge_gu_tc(qg, qu, WN):
    """rows: block b of WN = [gate cols b*WN/2 .. +WN/2) ++ [up cols same]. works for nibbles [I,K] or scales."""
    half = WN // 2
    n, k = qg.shape
    a = qg.reshape(n // half, 1, half, k)
    b = qu.reshape(n // half, 1, half, k)
    if hasattr(qg, "permute"):
        return torch.cat([a, b], dim=1).reshape(2 * n, k)
    import numpy as np
    return np.concatenate([a, b], axis=1).reshape(2 * n, k)


def merge_gu_sw(qg, qu):
    """frag F (16 rows) = [gate cols 8F..8F+7] ++ [up cols 8F..8F+7]."""
    n, k = qg.shape
    a = qg.reshape(n // 8, 1, 8, k)
    b = qu.reshape(n // 8, 1, 8, k)
    if hasattr(qg, "permute"):
        return torch.cat([a, b], dim=1).reshape(2 * n, k)
    import numpy as np
    return np.concatenate([a, b], axis=1).reshape(2 * n, k)


def layout_tc_nibbles(q, WN):
    """q: nibbles [N, K] -> [N/WN, K/16, 32 lanes, WN/16 words, 8 nibbles]; nibble i = 4e + 2jj + h."""
    N, K = q.shape
    NB, KS, WORDS = N // WN, K // 16, WN // 16
    x = q.reshape(NB, WORDS, 2, 8, KS, 2, 4, 2)          # nb, w, jj, g, ks, h, t, e
    x = _perm(x, (0, 4, 3, 6, 1, 7, 2, 5))               # nb, ks, g, t, w, e, jj, h
    return x.reshape(NB, KS, 32, WORDS, 8)


def layout_tc_scales(s, WN):
    """s: [N, K/32] -> [N/WN, K/32, 8 g, WN/8 j]."""
    N, KG = s.shape
    x = s.reshape(N // WN, WN // 8, 8, KG)               # nb, j, g, kg
    return _perm(x, (0, 3, 2, 1))


def layout_sw_nibbles(q):
    """q: [N, K] -> [N/64, K/16, 32, 4 frags, 8]; nibble i = 4e + 2h + rr; row = 16f + 8rr + g."""
    N, K = q.shape
    NQ, KS = N // 64, K // 16
    x = q.reshape(NQ, 4, 2, 8, KS, 2, 4, 2)              # nq, f, rr, g, ks, h, t, e
    x = _perm(x, (0, 4, 3, 6, 1, 7, 5, 2))               # nq, ks, g, t, f, e, h, rr
    return x.reshape(NQ, KS, 32, 4, 8)


def layout_sw_scales(s):
    """s: [N, K/32] -> [N/64, K/32, 8 g, 4 f, 2 rr]."""
    N, KG = s.shape
    x = s.reshape(N // 64, 4, 2, 8, KG)                  # nq, f, rr, g, kg
    return _perm(x, (0, 4, 3, 1, 2))


# --------------------------------------------------------------------------------------------------
# CUDA driver + NVRTC via ctypes (no nvcc, no cuda-python API-version coupling)
# --------------------------------------------------------------------------------------------------
c_void_p, c_int, c_uint, c_size_t = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_size_t


class _APW(ctypes.Structure):  # CUaccessPolicyWindow (32 B)
    _fields_ = [("base_ptr", ctypes.c_void_p), ("num_bytes", ctypes.c_size_t), ("hitRatio", ctypes.c_float),
                ("hitProp", ctypes.c_int), ("missProp", ctypes.c_int)]


class _AttrVal(ctypes.Union):
    _fields_ = [("pad", ctypes.c_char * 64), ("i", ctypes.c_int), ("u64", ctypes.c_uint64), ("apw", _APW)]


class _LaunchAttr(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("pad", ctypes.c_char * 4), ("value", _AttrVal)]


class _LaunchCfg(ctypes.Structure):
    _fields_ = [("gridDimX", c_uint), ("gridDimY", c_uint), ("gridDimZ", c_uint),
                ("blockDimX", c_uint), ("blockDimY", c_uint), ("blockDimZ", c_uint),
                ("sharedMemBytes", c_uint), ("hStream", c_void_p),
                ("attrs", ctypes.POINTER(_LaunchAttr)), ("numAttrs", c_uint)]


class Driver:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        L = self.lib
        L.cuGetErrorString.argtypes = [c_int, ctypes.POINTER(ctypes.c_char_p)]
        L.cuModuleLoadData.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_char_p]
        L.cuModuleGetFunction.argtypes = [ctypes.POINTER(c_void_p), c_void_p, ctypes.c_char_p]
        L.cuFuncSetAttribute.argtypes = [c_void_p, c_int, c_int]
        L.cuFuncGetAttribute.argtypes = [ctypes.POINTER(c_int), c_int, c_void_p]
        L.cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                                     c_void_p, ctypes.POINTER(c_void_p), ctypes.POINTER(c_void_p)]
        L.cuLaunchKernelEx.argtypes = [ctypes.POINTER(_LaunchCfg), c_void_p, ctypes.POINTER(c_void_p),
                                       ctypes.POINTER(c_void_p)]
        L.cuCtxGetCurrent.argtypes = [ctypes.POINTER(c_void_p)]
        L.cuCtxGetDevice.argtypes = [ctypes.POINTER(c_int)]
        L.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(c_int), c_int, c_int]
        L.cuCtxSetLimit.argtypes = [c_int, c_size_t]
        L.cuCtxGetLimit.argtypes = [ctypes.POINTER(c_size_t), c_int]
        P64, P32 = ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)
        L.cuTensorMapEncodeTiled.argtypes = [c_void_p, c_int, c_uint, c_void_p, P64, P64, P32, P32, c_int, c_int, c_int, c_int]
        self.chk(L.cuInit(0), "cuInit")
        ctx = c_void_p()
        self.chk(L.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            raise RuntimeError("no current CUDA context (torch must initialise the device first)")

    def chk(self, rc, what):
        if rc != 0:
            s = ctypes.c_char_p()
            self.lib.cuGetErrorString(rc, ctypes.byref(s))
            raise RuntimeError(f"{what}: CUDA driver error {rc} ({(s.value or b'?').decode()})")

    def attr(self, a):
        dev, v = c_int(), c_int()
        self.chk(self.lib.cuCtxGetDevice(ctypes.byref(dev)), "cuCtxGetDevice")
        self.chk(self.lib.cuDeviceGetAttribute(ctypes.byref(v), a, dev.value), f"attr {a}")
        return v.value

    def set_persisting_l2(self, nbytes):
        """CU_LIMIT_PERSISTING_L2_CACHE_SIZE set-aside; returns what the driver granted. 0 releases it."""
        torch.cuda.synchronize()
        self.lib.cuCtxResetPersistingL2Cache()   # drop persisting lines left by a previous mode
        self.chk(self.lib.cuCtxSetLimit(6, nbytes), "cuCtxSetLimit(persisting L2)")
        v = c_size_t()
        self.lib.cuCtxGetLimit(ctypes.byref(v), 6)
        return v.value

    def load(self, cubin):
        mod = c_void_p()
        self.chk(self.lib.cuModuleLoadData(ctypes.byref(mod), cubin), "cuModuleLoadData")
        return mod


class NVRTC:
    def __init__(self):
        self.lib, self.path = self._find()
        L = self.lib
        L.nvrtcGetErrorString.restype = ctypes.c_char_p
        L.nvrtcCreateProgram.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_char_p, ctypes.c_char_p, c_int,
                                         c_void_p, c_void_p]
        L.nvrtcCompileProgram.argtypes = [c_void_p, c_int, ctypes.POINTER(ctypes.c_char_p)]
        L.nvrtcGetProgramLogSize.argtypes = [c_void_p, ctypes.POINTER(c_size_t)]
        L.nvrtcGetProgramLog.argtypes = [c_void_p, ctypes.c_char_p]
        L.nvrtcGetCUBINSize.argtypes = [c_void_p, ctypes.POINTER(c_size_t)]
        L.nvrtcGetCUBIN.argtypes = [c_void_p, ctypes.c_char_p]
        L.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(c_void_p)]
        a, b = c_int(), c_int()
        L.nvrtcVersion(ctypes.byref(a), ctypes.byref(b))
        self.version = (a.value, b.value)

    @staticmethod
    def _find():
        cands = []
        roots = list(sys.path)
        try:
            roots.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
        except Exception:
            pass
        for r in roots:
            if r and os.path.isdir(r):
                cands += glob.glob(os.path.join(r, "nvidia", "**", "libnvrtc.so*"), recursive=True)
                cands += glob.glob(os.path.join(r, "libnvrtc.so*"))
        for env in ("CUDA_HOME", "CUDA_PATH"):
            if os.environ.get(env):
                cands += glob.glob(os.path.join(os.environ[env], "lib64", "libnvrtc.so*"))
        cands += glob.glob("/usr/local/cuda*/lib64/libnvrtc.so*")
        cands = [c for c in cands if "builtins" not in os.path.basename(c) and "alt" not in os.path.basename(c)]
        best = None
        for p in sorted(set(cands)) + ["libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"]:
            try:
                lib = ctypes.CDLL(p)
                a, b = c_int(), c_int()
                if lib.nvrtcVersion(ctypes.byref(a), ctypes.byref(b)) != 0:
                    continue
                v = (a.value, b.value)
                if best is None or v > best[0]:
                    best = (v, lib, p)
            except OSError:
                continue
        if best is None:
            raise RuntimeError("libnvrtc not found (pip install nvidia-cuda-nvrtc matching your torch CUDA)")
        return best[1], best[2]

    def compile(self, src, arch, defines=()):
        L = self.lib
        prog = c_void_p()
        rc = L.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"forge.cu", 0, None, None)
        if rc:
            raise RuntimeError(f"nvrtcCreateProgram: {L.nvrtcGetErrorString(rc)}")
        opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17", b"--fmad=true"]
        opts += [f"-D{d}".encode() for d in defines]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        rc = L.nvrtcCompileProgram(prog, len(opts), arr)
        n = c_size_t()
        L.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(max(1, n.value))
        L.nvrtcGetProgramLog(prog, log)
        logs = log.value.decode(errors="replace")
        if rc:
            raise RuntimeError(f"NVRTC compile failed ({L.nvrtcGetErrorString(rc)}):\n{logs}")
        L.nvrtcGetCUBINSize(prog, ctypes.byref(n))
        buf = ctypes.create_string_buffer(n.value)
        L.nvrtcGetCUBIN(prog, buf)
        L.nvrtcDestroyProgram(ctypes.byref(prog))
        return buf.raw, logs


class TmaArg:
    """CUtensorMap (128 B, 64-B aligned) for a row-major bf16 [rows, K] tensor, box [BM, BK], hardware swizzle =
    the kernel's ldmatrix XOR layout (128B for BK=64, 64B for BK=32), OOB rows zero-filled."""

    def __init__(self, drv, t, BK, BM):
        rows, K = t.shape
        self.raw = (ctypes.c_uint8 * 256)()
        self.addr = (ctypes.addressof(self.raw) + 63) & ~63
        dims = (ctypes.c_uint64 * 2)(K, rows)
        strides = (ctypes.c_uint64 * 1)(t.stride(0) * 2)
        box = (ctypes.c_uint32 * 2)(BK, BM)
        es = (ctypes.c_uint32 * 2)(1, 1)
        swz = 3 if BK * 2 == 128 else 2
        drv.chk(drv.lib.cuTensorMapEncodeTiled(c_void_p(self.addr), 9, 2, c_void_p(t.data_ptr()), dims, strides,
                                               box, es, 0, swz, 2, 0), "cuTensorMapEncodeTiled")


class Kern:
    def __init__(self, drv, mod, name, threads, smem=0):
        self.drv, self.name, self.threads, self.smem = drv, name, threads, smem
        fn = c_void_p()
        drv.chk(drv.lib.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()), f"get {name}")
        self.fn = fn
        if smem > 48 * 1024:
            drv.chk(drv.lib.cuFuncSetAttribute(fn, 8, smem), f"smem attr {name}")  # MAX_DYNAMIC_SHARED_SIZE
        v = c_int()
        drv.lib.cuFuncGetAttribute(ctypes.byref(v), 4, fn); self.regs = v.value          # NUM_REGS
        drv.lib.cuFuncGetAttribute(ctypes.byref(v), 3, fn); self.local = v.value         # LOCAL_SIZE_BYTES
        drv.lib.cuFuncGetAttribute(ctypes.byref(v), 1, fn); self.static_smem = v.value   # SHARED_SIZE_BYTES

    @staticmethod
    def _arg(a):
        if a is None:
            return c_void_p(0)
        if isinstance(a, bool):
            return c_int(int(a))
        if isinstance(a, int):
            return c_int(a)
        if isinstance(a, tuple) and a[0] == "i64":
            return ctypes.c_longlong(a[1])
        return c_void_p(a.data_ptr())

    def __call__(self, grid, args, pdl=False, apw=None):
        """apw = (base_ptr, num_bytes, hitRatio): L2 access-policy window, hits persisting / misses streaming."""
        if isinstance(grid, int):
            grid = (grid, 1, 1)
        grid = tuple(grid) + (1,) * (3 - len(grid))
        cargs = [a if isinstance(a, TmaArg) else self._arg(a) for a in args]
        params = (c_void_p * len(cargs))(*[a.addr if isinstance(a, TmaArg) else ctypes.addressof(a) for a in cargs])
        stream = c_void_p(torch.cuda.current_stream().cuda_stream)
        if pdl or apw:
            attrs = (_LaunchAttr * 2)()
            na = 0
            if pdl:
                attrs[na].id = 6  # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
                attrs[na].value.i = 1
                na += 1
            if apw:
                attrs[na].id = 1  # CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW
                w = attrs[na].value.apw
                w.base_ptr, w.num_bytes, w.hitRatio = c_void_p(apw[0]), apw[1], apw[2]
                w.hitProp, w.missProp = 2, 1  # CU_ACCESS_PROPERTY_PERSISTING, CU_ACCESS_PROPERTY_STREAMING
                na += 1
            cfg = _LaunchCfg(grid[0], grid[1], grid[2], self.threads, 1, 1, self.smem, stream,
                             ctypes.cast(attrs, ctypes.POINTER(_LaunchAttr)), na)
            rc = self.drv.lib.cuLaunchKernelEx(ctypes.byref(cfg), self.fn, params, None)
        else:
            rc = self.drv.lib.cuLaunchKernel(self.fn, grid[0], grid[1], grid[2], self.threads, 1, 1,
                                             self.smem, stream, params, None)
        self.drv.chk(rc, f"launch {self.name}")


def _compile_cached(nv, src, arch, tag, extra=()):  # noqa: E302
    import hashlib
    key = hashlib.sha256(f"{src}|{arch}|{nv.version}|{H}|{I}|{extra}".encode()).hexdigest()[:24]
    cdir = os.environ.get("FORGE_CACHE", os.path.expanduser("~/.cache/forge_mlp"))
    cpath = os.path.join(cdir, f"{tag}_{key}.cubin")
    if os.path.isfile(cpath):
        return open(cpath, "rb").read(), "", True
    cubin, log = nv.compile(src, arch, defines=(f"FG_H={H}", f"FG_I={I}") + tuple(extra))
    try:
        os.makedirs(cdir, exist_ok=True)
        open(cpath, "wb").write(cubin)
    except OSError:
        pass
    return cubin, log, False


def build_kernels(drv, nv, arch, quiet=False):
    t0 = time.time()
    K = {"_drv": drv}
    cubin, log, hit = _compile_cached(nv, CUDA_COMMON + CUDA_V1, arch, "v1")
    mod = drv.load(cubin)
    for n in TC_CFGS:
        m = cfg_meta(n)
        K[n] = Kern(drv, mod, n, m["threads"], m["smem"])
    for n, th in [("fg_gu8", 256), ("fg_dn8", 512), ("fg_lowa_gu", 256), ("fg_lowa_d", 256), ("fg_decode", 256),
                  ("fg_epi_gu", 256), ("fg_epi_dn", 256), ("fg_gelu_table", 256), ("fg_lowk_gu", 128),
                  ("fg_lowk_d", 128), ("fg_lowk_gu16", 128), ("fg_lowk_d16", 128), ("fg_mma_tput", 256),
                  ("fg_bw_read", 256)]:
        K[n] = Kern(drv, mod, n, th)
    try:  # M=8 GEMVs with L2 eviction-priority hints, own module so a PTX rejection only drops this variant
        cubin_h, _, _ = _compile_cached(nv, CUDA_COMMON + CUDA_V1, arch, "v1h", ("FG_L2HINT",))
        mod_h = drv.load(cubin_h)
        K["fg_gu8h"], K["fg_dn8h"] = Kern(drv, mod_h, "fg_gu8h", 256), Kern(drv, mod_h, "fg_dn8h", 512)
        for n, th in GEMV8_VARIANTS["gu"][1:] + GEMV8_VARIANTS["dn"][1:]:
            try:
                K[n] = Kern(drv, mod_h, n, th)
            except Exception as e:
                if not quiet:
                    print(f"[nvrtc] {n} unavailable: {e}")
    except Exception as e:
        if not quiet:
            print(f"[nvrtc] L2-hint GEMVs unavailable: {str(e)[-800:]}")
    v3_ok, log3, hit3 = False, "", False
    if int(arch[3:]) >= 90:
        for extra in ((), ("FG_NO_CLC",)):  # CLC needs PTX ISA 8.6 (sm_100+); fall back to static persistent
            try:
                cubin3, log3, hit3 = _compile_cached(nv, CUDA_COMMON + CUDA_V3, arch, "v3", extra)
                mod3 = drv.load(cubin3)
                for n in WS_CFGS:
                    if extra and WS_CFGS[n][5]:
                        continue
                    m = cfg_meta(n)
                    K[n] = Kern(drv, mod3, n, m["threads"], m["smem"])
                v3_ok = any(n in K for n in WS_CFGS)
                break
            except Exception as e:
                log3 = f"{e}"
                if not quiet:
                    print(f"[nvrtc] v3 (warpspeed{'' if not extra else ', no CLC'}) build failed:\n{str(e)[-3000:]}")
    K["_v3_ok"] = v3_ok
    if not quiet:
        print(f"[nvrtc] {nv.path} v{nv.version[0]}.{nv.version[1]} arch={arch} build {time.time() - t0:.1f}s "
              f"(cache v1={hit} v3={hit3})")
        for n, k in K.items():
            if isinstance(k, Kern):
                flag = "  <-- SPILL" if k.local > 0 else ""
                print(f"   {n:14s} threads={k.threads:4d} dyn_smem={k.smem:6d} regs={k.regs:3d} local={k.local}B{flag}")
        for lg in (log, log3 if v3_ok else ""):
            if lg.strip():
                print("[nvrtc log]\n" + lg)
    return K


# --------------------------------------------------------------------------------------------------
# Weights: real Gemma-4-31B layer-0 MLP tensors (local cache / HF range download) or synthetic
# --------------------------------------------------------------------------------------------------
def _st_dtype(s):
    return {"I32": torch.int32, "BF16": torch.bfloat16, "I64": torch.int64, "F32": torch.float32,
            "F16": torch.float16, "U8": torch.uint8}[s]


def _read_local(path, names):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for nm in names:
            if nm not in hdr:
                continue
            info = hdr[nm]
            s, e = info["data_offsets"]
            f.seek(base + s)
            buf = bytearray(f.read(e - s))
            out[nm] = torch.frombuffer(buf, dtype=_st_dtype(info["dtype"])).reshape(info["shape"]).clone()
        return out


def _find_local(args):
    paths = []
    if args.weights:
        paths.append(args.weights)
    if os.environ.get("FORGE_WEIGHTS"):
        paths.append(os.environ["FORGE_WEIGHTS"])
    roots = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HUGGINGFACE_HUB_CACHE"),
             os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
             os.path.expanduser("~/.cache/huggingface/hub"), "/content/model_cache",
             "/content/model_cache", "./model_cache"]
    rd = "models--" + REPO.replace("/", "--")
    for r in roots:
        if r and os.path.isdir(os.path.join(r, rd)):
            snaps = sorted(glob.glob(os.path.join(r, rd, "snapshots", "*")),
                           key=lambda p: (os.path.basename(p) != REV, p))
            paths += snaps
    for p in paths:
        if os.path.isfile(p):
            yield p
        elif os.path.isdir(p):
            idx = os.path.join(p, "model.safetensors.index.json")
            if os.path.isfile(idx):
                wm = json.load(open(idx))["weight_map"]
                files = sorted({wm[TN[k]] for k in TN if TN[k] in wm})
                for fn in files:
                    yield os.path.join(p, fn)
            elif os.path.isfile(os.path.join(p, "model.safetensors")):
                yield os.path.join(p, "model.safetensors")


def _http_range(url, a, b, token):
    hdr = {"Range": f"bytes={a}-{b}"}
    try:
        import requests  # strips Authorization on cross-host redirects (CDN signed URLs)
        h = dict(hdr)
        if token:
            h["Authorization"] = f"Bearer {token}"
        r = requests.get(url, headers=h, timeout=120, allow_redirects=True)
        r.raise_for_status()
        return r.content
    except ImportError:
        import urllib.request
        req = urllib.request.Request(url, headers=dict(hdr, **({"Authorization": f"Bearer {token}"} if token else {})))
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()


def _download(names):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors"
    n = struct.unpack("<Q", _http_range(url, 0, 7, token))[0]
    hdr = json.loads(_http_range(url, 8, 8 + n - 1, token))
    base = 8 + n
    out = {}
    for nm in names:
        info = hdr[nm]
        s, e = info["data_offsets"]
        buf = bytearray(_http_range(url, base + s, base + e - 1, token))
        assert len(buf) == e - s, f"short read for {nm}"
        out[nm] = torch.frombuffer(buf, dtype=_st_dtype(info["dtype"])).reshape(info["shape"]).clone()
    return out


def _synthetic(Hd=None):
    Hd = Hd or H
    g = torch.Generator(device="cuda")
    g.manual_seed(1234)

    def quant(n, k, std):
        Q = torch.empty(n, k // 8, dtype=torch.int32, device="cuda")
        S = torch.empty(n, k // 32, dtype=torch.bfloat16, device="cuda")
        for r0 in range(0, n, 4096):
            r1 = min(n, r0 + 4096)
            w = torch.randn(r1 - r0, k, generator=g, device="cuda") * std
            wg = w.view(r1 - r0, k // 32, 32)
            s = (wg.abs().amax(-1) / 7.0).clamp_min(1e-8).to(torch.bfloat16)
            q = torch.clamp(torch.round(wg / s.float().unsqueeze(-1)), -8, 7).to(torch.int64) + 8
            q = q.view(r1 - r0, k // 8, 8)
            sh = torch.arange(0, 32, 4, device="cuda", dtype=torch.int64)
            v = (q << sh).sum(-1)
            v = torch.where(v >= 2 ** 31, v - 2 ** 32, v)
            Q[r0:r1] = v.to(torch.int32)
            S[r0:r1] = s
        return Q, S

    Qg, Sg = quant(I, Hd, 0.015)
    Qu, Su = quant(I, Hd, 0.015)
    Qd, Sd = quant(Hd, I, 0.01)
    return dict(Qg=Qg, Sg=Sg, Qu=Qu, Su=Su, Qd=Qd, Sd=Sd), "synthetic (seeded int4 sym g32)"


def load_weights(args):
    if not args.synthetic:
        names = list(TN.values())
        parts, used = {}, []
        for p in _find_local(args):
            try:
                got = _read_local(p, [n for n in names if n not in parts])
                if got:
                    parts.update(got)
                    used.append(p)
                if all(n in parts for n in names):
                    return {k: parts[TN[k]].cuda() for k in TN}, "local " + ", ".join(used)
            except Exception as e:
                print(f"[weights] {p}: {e}")
        if not args.no_download:
            try:
                t0 = time.time()
                got = _download(names)
                print(f"[weights] downloaded {sum(v.numel() * v.element_size() for v in got.values()) / 1e6:.0f} MB "
                      f"by HTTP range in {time.time() - t0:.1f}s")
                return {k: got[TN[k]].cuda() for k in TN}, f"hf {REPO}@{REV[:8]}"
            except Exception as e:
                print(f"[weights] HF range download failed ({type(e).__name__}: {e}); using synthetic")
    return _synthetic()


# --------------------------------------------------------------------------------------------------
# contract reference
# --------------------------------------------------------------------------------------------------
def unpack_nibbles(Q):
    sh = torch.arange(0, 32, 4, device=Q.device, dtype=torch.int32)
    return ((Q.unsqueeze(-1) >> sh) & 15).to(torch.uint8).reshape(Q.shape[0], -1)


def pack_nibbles(nib):
    sh = torch.arange(0, 32, 4, device=nib.device, dtype=torch.int64)
    v = (nib.to(torch.int64) << sh).sum(-1)
    v = torch.where(v >= 2 ** 31, v - 2 ** 32, v)
    return v.to(torch.int32).contiguous()


def decode_torch(Q, S):
    q = unpack_nibbles(Q).to(torch.int32)
    return (q - 8).to(torch.bfloat16) * S.repeat_interleave(32, dim=1)


def ref_forward(X, W, A, want_mid=False):
    Wg = decode_torch(W["Qg"], W["Sg"])
    Wu = decode_torch(W["Qu"], W["Su"])
    Wd = decode_torch(W["Qd"], W["Sd"])
    g = torch.mm(X, Wg.t()) + torch.mm(torch.mm(X, A["gA"].t()), A["gB"].t()) * 2
    u = torch.mm(X, Wu.t()) + torch.mm(torch.mm(X, A["uA"].t()), A["uB"].t()) * 2
    mid = F.gelu(g, approximate="tanh") * u
    Y = torch.mm(mid, Wd.t()) + torch.mm(torch.mm(mid, A["dA"].t()), A["dB"].t()) * 2
    return (Y, mid) if want_mid else Y


def make_adapters_and_x(M):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED)  # per-shape reset, order gA,gB,uA,uB,dA,dB,X

    def U(shape, bound):
        return ((torch.rand(shape, generator=gen, device="cuda") * 2 - 1) * bound).to(torch.bfloat16)

    def Nrm(shape, std):
        return (torch.randn(shape, generator=gen, device="cuda") * std).to(torch.bfloat16)

    A = {}
    A["gA"] = U((R, H), 1 / math.sqrt(H)); A["gB"] = Nrm((I, R), 0.02)
    A["uA"] = U((R, H), 1 / math.sqrt(H)); A["uB"] = Nrm((I, R), 0.02)
    A["dA"] = U((R, I), 1 / math.sqrt(I)); A["dB"] = Nrm((H, R), 0.02)
    X = Nrm((M, H), 1.0)
    A["guA"] = torch.cat([A["gA"], A["uA"]], 0).contiguous()        # [32, H]  lossless adapter concat
    A["guB"] = torch.cat([A["gB"], A["uB"]], 1).contiguous()        # [I, 32]
    return A, X


# --------------------------------------------------------------------------------------------------
# packing (cold, once)
# --------------------------------------------------------------------------------------------------
def prepare_packed(W, gemv=True, quiet=False):
    t0 = time.time()
    P = {}
    qg, qu, qd = unpack_nibbles(W["Qg"]), unpack_nibbles(W["Qu"]), unpack_nibbles(W["Qd"])
    # large-M TC layouts
    for wn in (64, 32):
        P[f"gu{wn}_q"] = pack_nibbles(layout_tc_nibbles(merge_gu_tc(qg, qu, wn), wn))
        P[f"gu{wn}_s"] = layout_tc_scales(merge_gu_tc(W["Sg"], W["Su"], wn), wn).contiguous()
    P["dn64_q"] = pack_nibbles(layout_tc_nibbles(qd, 64))
    P["dn64_s"] = layout_tc_scales(W["Sd"], 64).contiguous()
    P["dn32_q"] = pack_nibbles(layout_tc_nibbles(qd, 32))
    P["dn32_s"] = layout_tc_scales(W["Sd"], 32).contiguous()
    P["gu32_s16"] = P["gu32_s"].to(torch.float16)     # fp16 datapath scales (same layout)
    P["dn32_s16"] = P["dn32_s"].to(torch.float16)
    P["gu16_q"] = pack_nibbles(layout_tc_nibbles(merge_gu_tc(qg, qu, 16), 16))   # fp16 128x16-warp configs
    P["gu16_s16"] = layout_tc_scales(merge_gu_tc(W["Sg"], W["Su"], 16), 16).contiguous().to(torch.float16)
    P["dn16_q"] = pack_nibbles(layout_tc_nibbles(qd, 16))
    P["dn16_s16"] = layout_tc_scales(W["Sd"], 16).contiguous().to(torch.float16)
    # M=8 SW layouts
    if gemv:
        P["gu8_q"] = pack_nibbles(layout_sw_nibbles(merge_gu_sw(qg, qu)))
        P["gu8_s"] = layout_sw_scales(merge_gu_sw(W["Sg"], W["Su"])).contiguous()
        P["dn8_q"] = pack_nibbles(layout_sw_nibbles(qd))
        P["dn8_s"] = layout_sw_scales(W["Sd"]).contiguous()
    del qg, qu, qd
    torch.cuda.synchronize()
    nbytes = sum(v.numel() * v.element_size() for v in P.values())
    if not quiet:
        print(f"[prepare] lossless INT4 repacks: {nbytes / 1e6:.0f} MB in {time.time() - t0:.2f}s (cold, once)")
    return P


# --------------------------------------------------------------------------------------------------
# designs: forward() -> Y (fixed output buffer); .mid for diagnostics; anchor=True marks non-contract timings
# --------------------------------------------------------------------------------------------------
def _decode_fresh(K, Q, S):
    n, kw = Q.shape
    out = torch.empty(n, kw * 8, dtype=torch.bfloat16, device="cuda")
    K["fg_decode"]((n * kw + 255) // 256, [Q, S, out, ("i64", n * kw), kw, kw // 4])
    return out


class CtRef:
    """contract-literal baseline: exact INT4 decode every forward + cuBLAS + eager torch pointwise (CT-style)."""
    name, anchor = "ct_ref", False

    def __init__(self, K, W, A, X):
        self.K, self.W, self.A, self.X = K, W, A, X
        self.mid = None

    def forward(self):
        K, W, A, X = self.K, self.W, self.A, self.X
        Wg, Wu, Wd = _decode_fresh(K, W["Qg"], W["Sg"]), _decode_fresh(K, W["Qu"], W["Su"]), _decode_fresh(K, W["Qd"], W["Sd"])
        g = torch.mm(X, Wg.t()) + torch.mm(torch.mm(X, A["gA"].t()), A["gB"].t()) * 2
        u = torch.mm(X, Wu.t()) + torch.mm(torch.mm(X, A["uA"].t()), A["uB"].t()) * 2
        mid = F.gelu(g, approximate="tanh") * u
        return torch.mm(mid, Wd.t()) + torch.mm(torch.mm(mid, A["dA"].t()), A["dB"].t()) * 2


class AnchorCublas:
    """same-session ceiling, NOT a design: only the two base GEMMs (N=2I joined, Down) on pre-decoded weights."""
    name, anchor = "anchor_cublas", True

    def __init__(self, K, W, A, X, mid):
        self.Wgu = torch.cat([_decode_fresh(K, W["Qg"], W["Sg"]), _decode_fresh(K, W["Qu"], W["Su"])], 0)
        self.Wd = _decode_fresh(K, W["Qd"], W["Sd"])
        self.X, self.mid_in, self.mid = X, mid, None

    def forward(self):
        torch.mm(self.X, self.Wgu.t())
        return torch.mm(self.mid_in, self.Wd.t())


class ForgeTC:
    """fused INT4 GEMMs. Default chain (all PDL-linked inside one graph):
         lowk_gu(X) -> GU kernel [mainloop overlaps lowk_gu; waits only before its epilogue]
                    -> lowk_d(mid) [waits for GU, then releases Down] -> Down kernel [mainloop overlaps lowk_d]
       f16=True : fp16 datapath (fp16 X copy written by lowk_gu16, fp16 MMA accumulation promoted to fp32 every
                  stage, mid stored as fp16 of its bf16 value) -- numerics-relaxed, reported as its own design.
       lib_lowrank=True uses the oracle's cuBLAS calls for the rank-16 factors instead (bitwise path, no PDL)."""
    anchor = False

    def __init__(self, K, P, A, X, nsm, Hd=H, lib_lowrank=False, f16=False):
        self.K, self.P, self.A, self.X, self.nsm, self.Hd = K, P, A, X, nsm, Hd
        self.f16, self.lib = f16, (lib_lowrank and not f16)
        self.name = "forge_tc_f16" if f16 else "forge_tc"
        self.M = M = X.shape[0]
        dev = X.device
        if f16:
            self.X16 = torch.empty(M, Hd, dtype=torch.float16, device=dev)
            self.mid16 = torch.empty(M, I, dtype=torch.float16, device=dev)
            self.dA16 = A["dA"].to(torch.float16)
            self.mid = None
        else:
            self.mid = torch.empty(M, I, dtype=torch.bfloat16, device=dev)
        self.Y = torch.empty(M, Hd, dtype=torch.bfloat16, device=dev)
        self.lowgu = torch.empty(M, 32, dtype=torch.bfloat16, device=dev)
        self.lowd = torch.empty(M, 16, dtype=torch.bfloat16, device=dev)
        P2 = 2 * nsm                                       # stream-K workspace: <= P2 partial 128x128 fp32 tiles
        self.wsp = {e: torch.empty(P2 * 4 * 128 * 32, dtype=torch.float32, device=dev) for e in (0, 1)}
        self.cnt = {e: torch.zeros(P2 * 8, dtype=torch.int32, device=dev) for e in (0, 1)}   # x8: wsd WM=64 squads
        self.gu, self.dn = (None, None) if f16 else (V1_GU[0], V1_DN[0])
        self.pdl = True
        self._tm = {}

    def set_cfg(self, gu, dn):
        self.gu, self.dn = gu, dn

    def _tmap(self, t, name):
        m = cfg_meta(name)
        key = (t.data_ptr(), tuple(t.shape), m["BK"], m["BM"])
        if key not in self._tm:
            self._tm[key] = TmaArg(self.K["_drv"], t, m["BK"], m["BM"])
        return self._tm[key]

    def plan(self, name, N, K):
        """-> (dpu, S, units, grid). Whole tiles for full waves; the last partial wave is split S ways along K."""
        m = cfg_meta(name)
        T = ((self.M + m["BM"] - 1) // m["BM"]) * (N // m["BN"])
        dpu, S = T, 1
        if name.endswith("+sk"):
            P2, KT = 2 * self.nsm, K // m["BK"]
            R = T % P2
            if R:
                S = min(P2 // R, KT // 2, 16)
                if S > 1:
                    dpu = T - R
                else:
                    S = 1
        units = dpu + (T - dpu) * S if S > 1 else T
        if m["kind"] == "ws":
            grid = units if m["CLC"] else min((units + 1) // 2, self.nsm)
        elif m["kind"] == "wsd":                           # one MMA squad per CTA
            grid = units if m["CLC"] else min(units, self.nsm)
        else:
            grid = T
        return dpu, S, units, grid

    def low_gu(self, out, pdl=False):
        if self.lib:
            out.copy_(torch.mm(self.X, self.A["guA"].t()))
        elif self.f16:
            self.K["fg_lowk_gu16"]((self.M + 15) // 16, [self.X, self.A["guA"], out, self.M, self.Hd, self.X16], pdl=pdl)
        else:
            self.K["fg_lowk_gu"]((self.M + 15) // 16, [self.X, self.A["guA"], out, self.M, self.Hd], pdl=pdl)

    def low_d(self, mid, out, pdl=False):
        if self.lib:
            out.copy_(torch.mm(mid, self.A["dA"].t()))
        elif self.f16:
            self.K["fg_lowk_d16"]((self.M + 15) // 16, [mid, self.dA16, out, self.M, I], pdl=pdl)
        else:
            self.K["fg_lowk_d"]((self.M + 15) // 16, [mid, self.A["dA"], out, self.M, I], pdl=pdl)

    def run_gu(self, name, lowgu, out, pdl=False):
        m, P, A = cfg_meta(name), self.P, self.A
        wn, base = m["WN"], cfg_base(name)
        sc = P[f"gu{wn}_s16"] if self.f16 else P[f"gu{wn}_s"]
        dpu, S, units, grid = self.plan(name, 2 * I, self.Hd)
        if m["kind"] in ("ws", "wsd"):
            xin = self.X16 if self.f16 else self.X
            self.K[base](grid, [self._tmap(xin, name), P[f"gu{wn}_q"], sc, lowgu, lowgu[:, 16:], 32, A["gB"], A["uB"],
                                out, self.M, self.Hd, 2 * I, I, dpu, S, self.wsp[0], self.cnt[0]], pdl=pdl)
        else:
            self.K[base](grid, [self.X, P[f"gu{wn}_q"], sc, lowgu, lowgu[:, 16:],
                                A["gB"], A["uB"], out, self.M, self.Hd, 2 * I, I, 32], pdl=pdl)

    def run_dn(self, name, mid, lowd, out, pdl=False):
        m, P, A = cfg_meta(name), self.P, self.A
        wn, base = m["WN"], cfg_base(name)
        sc = P[f"dn{wn}_s16"] if self.f16 else P[f"dn{wn}_s"]
        dpu, S, units, grid = self.plan(name, self.Hd, I)
        if m["kind"] in ("ws", "wsd"):
            self.K[base](grid, [self._tmap(mid, name), P[f"dn{wn}_q"], sc, lowd, None, 16, A["dB"], None,
                                out, self.M, I, self.Hd, self.Hd, dpu, S, self.wsp[1], self.cnt[1]], pdl=pdl)
        else:
            self.K[base](grid, [mid, P[f"dn{wn}_q"], sc, lowd, None, A["dB"], None,
                                out, self.M, I, self.Hd, self.Hd, 16], pdl=pdl)

    def forward(self):
        chain = self.pdl and not self.lib
        mid = self.mid16 if self.f16 else self.mid
        self.low_gu(self.lowgu)
        self.run_gu(self.gu, self.lowgu, mid, pdl=chain)
        self.low_d(mid, self.lowd, pdl=chain)
        self.run_dn(self.dn, mid, self.lowd, self.Y, pdl=chain)
        return self.Y


def measure_memory(K, drv, nsm):
    """DRAM stream rate (one pass over 2 GiB), L2 rate (passes over L2/4), L2 size and persisting limits."""
    out = dict(l2_bytes=drv.attr(38), persist_max=drv.attr(108), window_max=drv.attr(109))
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    grid = nsm * 8
    for name, nbytes, passes in (("dram", 2 << 30, 1), ("l2", max(1 << 20, out["l2_bytes"] // 4), 40)):
        buf = torch.empty(nbytes // 4, dtype=torch.int32, device="cuda")
        buf.fill_(1)
        nvec = nbytes // 16
        K["fg_bw_read"](grid, [buf, ("i64", nvec), passes, sink])
        torch.cuda.synchronize()
        best = 1e30
        for _ in range(3):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            K["fg_bw_read"](grid, [buf, ("i64", nvec), passes, sink])
            e1.record(); e1.synchronize()
            best = min(best, e0.elapsed_time(e1))
        out[name + "_GBs"] = nbytes * passes / (best * 1e-3) / 1e9
        del buf
    torch.cuda.empty_cache()
    return out


class ForgeGemv8:
    """M=8: operand-swapped tensor-core GEMVs.  All M=8 weights live in ONE contiguous arena (Down first, then GU
    scales, then GU words) so the L2 can be told which part to keep resident across forwards:
      none    plain loads (every forward streams 198 MB from DRAM)
      window  access-policy window over the whole arena, hitRatio = set-aside / arena (hardware picks the lines)
      prefix  access-policy window over the first set-aside bytes (all of Down + the start of GU), hitRatio 1
      hintNN  no set-aside: createpolicy evict_last on the leading quads worth NN% of L2, evict_first on the rest"""
    name, anchor = "forge_gemv8", False

    def __init__(self, K, P, A, X, drv=None, mem=None):
        assert X.shape[0] == 8
        self.K, self.P, self.A, self.X, self.drv, self.mem = K, P, A, X, drv, mem
        al = lambda n: (n + 255) // 256 * 256
        parts = [("dn8_q", P["dn8_q"]), ("dn8_s", P["dn8_s"]), ("gu8_s", P["gu8_s"]), ("gu8_q", P["gu8_q"])]
        self.arena = torch.empty(sum(al(t.numel() * t.element_size()) for _, t in parts), dtype=torch.uint8,
                                 device="cuda")
        self.Wt, off = {}, 0
        for n, t in parts:
            nb = t.numel() * t.element_size()
            v = self.arena[off:off + nb].view(t.dtype).view(t.shape)
            v.copy_(t)
            self.Wt[n] = v
            off += al(nb)
        self.mode, self.apw, self.keep = "none", None, (0, 0)
        self.part_gu = torch.empty(6 * 32 * 8, dtype=torch.float32, device="cuda")
        self.part_d = torch.empty(21 * 16 * 8, dtype=torch.float32, device="cuda")
        self.cnt = torch.zeros(2, dtype=torch.int32, device="cuda")
        self.low_gu = torch.empty(8, 32, dtype=torch.bfloat16, device="cuda")
        self.low_d = torch.empty(8, 16, dtype=torch.bfloat16, device="cuda")
        self.mid = torch.empty(8, I, dtype=torch.bfloat16, device="cuda")
        self.Y = torch.empty(8, H, dtype=torch.bfloat16, device="cuda")
        self.c0, self.c1 = self.cnt[0:1], self.cnt[1:2]
        self.pdl = True
        self.gu_k, self.dn_k = "fg_gu8h", "fg_dn8h"

    def modes(self):
        m = ["none"]
        if self.mem and self.mem["persist_max"] > 0 and self.mem["window_max"] > 0:
            m += ["window", "prefix"]
        if "fg_gu8h" in self.K and self.mem:
            m += ["hint60", "hint80", "hint95"]
        return m

    def set_mode(self, mode):
        """configures the L2 policy for `mode`; resets persisting lines so modes do not inherit each other's state."""
        self.mode, self.apw, self.keep = mode, None, (0, 0)
        if self.drv is None:
            return
        total = self.arena.numel()
        if mode in ("window", "prefix"):
            carve = self.drv.set_persisting_l2(self.mem["persist_max"])
            if mode == "window":
                nb = min(total, self.mem["window_max"])
                self.apw = (self.arena.data_ptr(), nb, min(1.0, carve / nb))
            else:
                self.apw = (self.arena.data_ptr(), min(carve, self.mem["window_max"], total), 1.0)
        else:
            self.drv.set_persisting_l2(0)
            if mode.startswith("hint"):
                budget = self.mem["l2_bytes"] * int(mode[4:]) // 100
                dn_bytes = self.Wt["dn8_q"].numel() * 4 + self.Wt["dn8_s"].numel() * 2
                dn_quad = dn_bytes / (H // 64)
                gu_quad = (self.Wt["gu8_q"].numel() * 4 + self.Wt["gu8_s"].numel() * 2) / (2 * I // 64)
                kd = min(H // 64, int(budget // dn_quad))
                kg = min(2 * I // 64, max(0, int((budget - kd * dn_quad) // gu_quad)))
                self.keep = (kg, kd)

    def release(self):
        if self.drv is not None:
            try:
                self.drv.set_persisting_l2(0)
            except Exception:
                pass

    def forward(self):
        K, Wt, A, pdl, apw = self.K, self.Wt, self.A, self.pdl, self.apw
        hint = self.mode.startswith("hint")
        K["fg_lowa_gu"](6, [self.X, A["guA"], self.part_gu, self.c0, self.low_gu])
        gu_args = [self.X, Wt["gu8_q"], Wt["gu8_s"], self.low_gu, A["gB"], A["uB"], self.mid, I]
        if hint:
            K[self.gu_k](2 * I // 64, gu_args + [self.keep[0]], pdl=pdl)
        else:
            K["fg_gu8"](2 * I // 64, gu_args, pdl=pdl, apw=apw)
        K["fg_lowa_d"](21, [self.mid, A["dA"], self.part_d, self.c1, self.low_d], pdl=pdl)
        dn_args = [self.mid, Wt["dn8_q"], Wt["dn8_s"], self.low_d, A["dB"], None, self.Y, H]
        if hint:
            K[self.dn_k](H // 64, dn_args + [self.keep[1]], pdl=pdl)
        else:
            K["fg_dn8"](H // 64, dn_args, pdl=pdl, apw=apw)
        return self.Y


def tune_gemv8(design):
    """MLP round: every (GU, Down) prefetch-depth variant under the best resident mode and under pure streaming
    ('hint0': evict_first everywhere, i.e. a layer inside a 60-layer model). Keeps the fastest replay-exact chain."""
    best_mode = design.mode if design.mode.startswith("hint") else "hint95"
    rows = []
    for mode in dict.fromkeys([best_mode, "hint0"]):
        design.set_mode(mode)
        for gk, _ in GEMV8_VARIANTS["gu"]:
            for dk, _ in GEMV8_VARIANTS["dn"]:
                if gk not in design.K or dk not in design.K:
                    continue
                design.gu_k, design.dn_k = gk, dk
                try:
                    r = bench(design.forward, 20, 5, 30.0)
                    rows.append((r["forward_ms"], mode, gk, dk, r["replay_matches_eager"]))
                except Exception as e:
                    torch.cuda.synchronize()
                    rows.append((float("inf"), mode, gk, dk, f"{type(e).__name__}: {e}"))
    ok = [r for r in rows if r[4] is True]
    _, mode, gk, dk, _ = min(ok) if ok else (0, best_mode, "fg_gu8h", "fg_dn8h", True)
    design.set_mode(mode)
    design.gu_k, design.dn_k = gk, dk
    return rows


def tune_l2(design):
    """times every L2 residency mode as a sustained graph replay (steady state across forwards) and keeps the best."""
    rows = []
    for m in design.modes():
        try:
            design.set_mode(m)
            r = bench(design.forward, 20, 5, 30.0)
            rows.append((r["forward_ms"], m, r["replay_matches_eager"]))
        except Exception as e:
            torch.cuda.synchronize()
            rows.append((float("inf"), m, f"{type(e).__name__}: {e}"))
    ok = [r for r in rows if r[2] is True]
    design.set_mode(min(ok)[1] if ok else "none")
    return rows


class DecCublas:
    """control: exact decode every forward into one [2I,H] buffer + ONE base GU GEMM + fused LoRA-B/act epilogues."""
    name, anchor = "dec_cublas", False

    def __init__(self, K, W, A, X):
        self.K, self.W, self.A, self.X = K, W, A, X
        self.M = X.shape[0]
        self.Wgu = torch.empty(2 * I, H, dtype=torch.bfloat16, device="cuda")
        self.Wd = torch.empty(H, I, dtype=torch.bfloat16, device="cuda")
        self.mid = torch.empty(self.M, I, dtype=torch.bfloat16, device="cuda")
        self.Y = torch.empty(self.M, H, dtype=torch.bfloat16, device="cuda")

    def _dec(self, Q, S, Wout):
        n, kw = Q.shape
        self.K["fg_decode"]((n * kw + 255) // 256, [Q, S, Wout, ("i64", n * kw), kw, kw // 4])

    def forward(self):
        W, A, X, M, K = self.W, self.A, self.X, self.M, self.K
        self._dec(W["Qg"], W["Sg"], self.Wgu[:I])
        self._dec(W["Qu"], W["Su"], self.Wgu[I:])
        self._dec(W["Qd"], W["Sd"], self.Wd)
        gu0 = torch.mm(X, self.Wgu.t())
        lowgu = torch.mm(X, A["guA"].t())
        K["fg_epi_gu"]((I // 64, (M + 63) // 64), [gu0, gu0[:, I:], lowgu, lowgu[:, 16:], A["guB"], A["guB"][:, 16:],
                                                   self.mid, M, I, 2 * I, 32, 32])
        d0 = torch.mm(self.mid, self.Wd.t())
        lowd = torch.mm(self.mid, A["dA"].t())
        K["fg_epi_dn"]((H // 64, (M + 63) // 64), [d0, None, lowd, None, A["dB"], None, self.Y, M, H, H, 16, 16])
        return self.Y


# --------------------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------------------
def numerics(y, yref):
    yf, rf = y.float(), yref.float()
    finite = bool(torch.isfinite(yf).all().item())
    diff = (yf - rf).abs()
    tol = 1e-5 + 0.016 * rf.abs()
    failed = int((~(diff <= tol)).sum().item())
    out = dict(elements=y.numel(), failed_elements=failed, finite=finite,
               max_abs=float(diff.max().item()),
               relative_l2_pct=float(100 * torch.linalg.vector_norm(yf - rf).item() /
                                     max(1e-30, torch.linalg.vector_norm(rf).item())),
               bitwise_equal_frac=float((y.view(torch.int16) == yref.view(torch.int16)).float().mean().item()))
    out["strict_bf16_pass"] = finite and failed == 0
    return out


def bench(fwd, warmup, samples, target_ms):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y_eager = fwd().clone()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_static = fwd()
    g.replay()
    torch.cuda.synchronize()
    replay_ok = bool(torch.equal(y_static.view(torch.int16), y_eager.view(torch.int16)))
    for _ in range(warmup):
        g.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); g.replay(); e1.record(); e1.synchronize()
    one = max(1e-3, e0.elapsed_time(e1))
    reps = max(1, min(200, int(target_ms / one)))
    ts = []
    for _ in range(samples):
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) / reps)
    g.replay()
    torch.cuda.synchronize()
    replay_ok = replay_ok and bool(torch.equal(y_static.view(torch.int16), y_eager.view(torch.int16)))
    del g
    ts.sort()
    return dict(forward_ms=ts[len(ts) // 2], minimum_ms=ts[0], maximum_ms=ts[-1], reps=reps,
                samples=samples, replay_matches_eager=replay_ok)


def _time_eager(fn, reps):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) / reps


def _consensus(outs):
    names = list(outs)
    good = []
    for a in names:
        agree = sum(torch.equal(outs[a].view(torch.int16), outs[b].view(torch.int16)) for b in names)
        if agree * 2 > len(names):
            good.append(a)
    return good


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return float(torch.linalg.vector_norm(a - b).item() / max(1e-30, torch.linalg.vector_norm(b).item()))


def _accept(outs, f16):
    """exact candidates must agree bitwise with the majority; '+sk' (fp32 re-association in the tail tiles only) must
    sit within 1e-3 relative L2 of that majority; fp16-datapath candidates were range-checked by the probe."""
    if f16:
        return list(outs)
    exact = [n for n in outs if not n.endswith("+sk")]
    ok = _consensus({n: outs[n] for n in exact}) if exact else []
    if not ok:
        return list(outs)
    ref = outs[ok[0]]
    return ok + [n for n in outs if n.endswith("+sk") and _rel_l2(outs[n], ref) < 1e-3]


def tune_tc(design, cfgs_gu, cfgs_dn, reps=3):
    """per-stage timing of every candidate (eager, standalone), then pick the fastest accepted config per stage."""
    M = design.M
    dt = torch.float16 if design.f16 else torch.bfloat16
    lowgu = torch.empty(M, 32, dtype=torch.bfloat16, device="cuda")
    design.low_gu(lowgu)
    res = {"gu": {}, "dn": {}}
    outs = {}
    for name in cfgs_gu:
        o = torch.empty(M, I, dtype=dt, device="cuda")
        res["gu"][name] = _time_eager(lambda: design.run_gu(name, lowgu, o), reps)
        outs[name] = o
    ok_gu = _accept(outs, design.f16)
    best_gu = min(ok_gu or cfgs_gu, key=lambda n: res["gu"][n])
    mid = outs[best_gu]
    lowd = torch.empty(M, 16, dtype=torch.bfloat16, device="cuda")
    design.low_d(mid, lowd)
    outs_d = {}
    for name in cfgs_dn:
        o = torch.empty(M, design.Hd, dtype=torch.bfloat16, device="cuda")
        res["dn"][name] = _time_eager(lambda: design.run_dn(name, mid, lowd, o), reps)
        outs_d[name] = o
    ok_dn = _accept(outs_d, design.f16)
    best_dn = min(ok_dn or cfgs_dn, key=lambda n: res["dn"][n])
    if not design.f16:  # what the dequant costs: same kernel with raw words fed to the MMA (wrong values, same traffic)
        res["dequant"] = {}
        for good, diag in DQ_PAIRS:
            if diag not in design.K or (good not in ok_gu and good not in ok_dn):
                continue
            epi = WS_CFGS[diag][4]
            o = torch.empty(M, I if epi == 0 else design.Hd, dtype=torch.bfloat16, device="cuda")
            if epi == 0:
                t_diag = _time_eager(lambda: design.run_gu(diag, lowgu, o), reps)
                t_good = res["gu"][good]
            else:
                t_diag = _time_eager(lambda: design.run_dn(diag, mid, lowd, o), reps)
                t_good = res["dn"][good]
            res["dequant"][good] = dict(with_ms=t_good, without_ms=t_diag, share=1.0 - t_diag / t_good)
            del o
    res["lowk_gu_ms"] = _time_eager(lambda: design.low_gu(lowgu), reps)
    res["lowk_d_ms"] = _time_eager(lambda: design.low_d(mid, lowd), reps)
    res["accepted_gu"], res["accepted_dn"] = ok_gu, ok_dn
    res["best"] = (best_gu, best_dn)
    del outs, outs_d
    return res


def flops(M):
    return 2.0 * M * (H * 2 * I + I * H)


WEIGHT_BYTES = 3 * H * I // 2 + 3 * H * (I // 32) * 2 + 2 * R * (H + I) * 2 + R * (I + H) * 2


def sol_ms(M, tput, mem):
    """roofline floor for one forward: tensor rate, or weights streamed from DRAM beyond what L2 can hold
    (L2 residency across forwards counts), or every weight byte through L2 -- whichever binds."""
    act = 2 * M * (H + H)  # X in, Y out
    t_mma = flops(M) / (tput["f32acc"] * 1e12)
    t_dram = (max(0, WEIGHT_BYTES - mem["l2_bytes"]) + act) / (mem["dram_GBs"] * 1e9)
    t_l2 = WEIGHT_BYTES / (mem["l2_GBs"] * 1e9)
    return 1e3 * max(t_mma, t_dram, t_l2)


def tune_chain(design, tr, top=3):
    """the forward is what counts: time real PDL chains (CUDA graph, sustained) for the top standalone configs,
    with and without PDL, and keep the fastest.  Standalone stage sums missed 2.7-3.8 ms of chain effects."""
    gu = sorted(tr["accepted_gu"], key=lambda n: tr["gu"][n])[:top]
    dn = sorted(tr["accepted_dn"], key=lambda n: tr["dn"][n])[:top]
    rows = []
    for g in gu:
        for d in dn:
            for pdl in ((False,) if design.lib else (True, False)):
                design.set_cfg(g, d)
                design.pdl = pdl
                try:
                    rows.append((bench(design.forward, 3, 3, 20.0)["forward_ms"], g, d, pdl))
                except Exception:
                    torch.cuda.synchronize()
    best = min(rows)
    design.set_cfg(best[1], best[2])
    design.pdl = best[3]
    return rows


def stage_breakdown(design):
    """each stage of the chosen chain on its own, as a sustained CUDA-graph replay (same clocks as the forward)."""
    mid = design.mid16 if design.f16 else design.mid
    st = {}
    st["lowk_gu"] = bench(lambda: (design.low_gu(design.lowgu), design.lowgu)[1], 3, 3, 20.0)["forward_ms"]
    st["gu"] = bench(lambda: (design.run_gu(design.gu, design.lowgu, mid), mid)[1], 3, 3, 20.0)["forward_ms"]
    st["lowk_d"] = bench(lambda: (design.low_d(mid, design.lowd), design.lowd)[1], 3, 3, 20.0)["forward_ms"]
    st["dn"] = bench(lambda: (design.run_dn(design.dn, mid, design.lowd, design.Y), design.Y)[1], 3, 3, 20.0)["forward_ms"]
    return st


# --------------------------------------------------------------------------------------------------
# v2 probe: run the new TMA/mbarrier kernels in a child process first (bitwise vs v1, watchdog-trapped), so a
# pipeline bug can only kill the child, never the benchmark session.
# --------------------------------------------------------------------------------------------------
def _setup_torch():
    global torch, F
    import torch as _torch
    import torch.nn.functional as _F
    torch, F = _torch, _F
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    cc = torch.cuda.get_device_capability()
    return torch.cuda.get_device_properties(0), f"sm_{cc[0]}{cc[1]}"


def probe_main(args):
    """child process: run every v3 config (+ its stream-K variant) once on a small problem; exact configs must be
    bitwise equal to the validated v1 kernels, '+sk' within 1e-3 rel-L2, fp16-datapath configs within 2e-2."""
    props, arch = _setup_torch()
    drv, nv = Driver(), NVRTC()
    K = build_kernels(drv, nv, arch, quiet=True)
    res = {}
    if not K["_v3_ok"]:
        json.dump({"_v3_compiled": False}, open(args.probe_out, "w"))
        return
    skip = set(filter(None, (args.probe_skip or "").split(",")))
    nsm = props.multi_processor_count
    Hd, M = 512, 777                      # K=512 for GU (thousands of stolen tiles), K=21504 for Down (long pipeline)
    W, _ = _synthetic(Hd)
    P = prepare_packed(W, gemv=False, quiet=True)
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    A = {"guA": (torch.randn(32, Hd, generator=g, device="cuda") / math.sqrt(Hd)).to(torch.bfloat16),
         "guB": (torch.randn(I, 32, generator=g, device="cuda") * 0.02).to(torch.bfloat16),
         "dA": (torch.randn(16, I, generator=g, device="cuda") / math.sqrt(I)).to(torch.bfloat16),
         "dB": (torch.randn(Hd, 16, generator=g, device="cuda") * 0.02).to(torch.bfloat16)}
    A["gB"], A["uB"] = A["guB"][:, :16].contiguous(), A["guB"][:, 16:].contiguous()
    X = torch.randn(M, Hd, generator=g, device="cuda").to(torch.bfloat16)
    d = ForgeTC(K, P, A, X, nsm, Hd=Hd)
    dh = ForgeTC(K, P, A, X, nsm, Hd=Hd, f16=True)
    lowgu = torch.empty(M, 32, dtype=torch.bfloat16, device="cuda")
    d.low_gu(lowgu)
    lowgu_h = torch.empty(M, 32, dtype=torch.bfloat16, device="cuda")
    dh.low_gu(lowgu_h)
    ref = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
    d.run_gu(REF_GU, lowgu, ref)
    lowd = torch.empty(M, 16, dtype=torch.bfloat16, device="cuda")
    d.low_d(ref, lowd)
    ref16 = ref.to(torch.float16)
    lowd_h = torch.empty(M, 16, dtype=torch.bfloat16, device="cuda")
    dh.low_d(ref16, lowd_h)
    yref = torch.empty(M, Hd, dtype=torch.bfloat16, device="cuda")
    d.run_dn(REF_DN, ref, lowd, yref)
    torch.cuda.synchronize()

    def judge(o, r, name, f16):
        if f16:
            e = _rel_l2(o, r)
            return "ok" if e < 2e-2 else f"relL2:{e:.3g}"
        if name.endswith("+sk"):
            e = _rel_l2(o, r)
            return "ok" if e < 1e-3 else f"relL2:{e:.3g}"
        bad = int((o.view(torch.int16) != r.view(torch.int16)).sum().item())
        return "ok" if bad == 0 else f"mismatch:{bad}"

    for epi in (0, 1):
        for n, c in WS_CFGS.items():
            if c[4] != epi or c[8] or n not in K:
                continue
            f16 = bool(c[6])
            for v in (n, n + "+sk"):
                if v in skip:
                    continue
                print(f"PROBE {v}", flush=True)
                des = dh if f16 else d
                dt = torch.float16 if (f16 and epi == 0) else torch.bfloat16
                o = torch.full((M, I if epi == 0 else Hd), float("nan"), dtype=dt, device="cuda")
                for rep in range(2):          # 2nd launch checks the stream-K counters re-armed themselves
                    o.fill_(float("nan"))
                    if epi == 0:
                        des.run_gu(v, lowgu_h if f16 else lowgu, o)
                    else:
                        des.run_dn(v, ref16 if f16 else ref, lowd_h if f16 else lowd, o)
                torch.cuda.synchronize()
                res[v] = judge(o, ref if epi == 0 else yref, v, f16)
                print(f"RESULT {v} {res[v]}", flush=True)
    json.dump(res, open(args.probe_out, "w"))


def run_probe(args, pre_skip=()):
    import subprocess
    import tempfile
    status = {}
    for _ in range(2 * len(WS_CFGS) + 1):
        fd, outp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(outp)
        cmd = [sys.executable, os.path.abspath(__file__), "--_probe", "--probe-out", outp]
        sk = list(status) + [v for n in pre_skip for v in (n, n + "+sk")]
        if sk:
            cmd += ["--probe-skip", ",".join(sk)]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=args.probe_timeout)
            text, err, rc = r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace"), r.returncode
        except subprocess.TimeoutExpired as e:
            text = (e.stdout or b"").decode(errors="replace")
            err, rc = "timeout", -1
        for line in text.splitlines():
            if line.startswith("RESULT "):
                _, n, v = line.split(" ", 2)
                status[n] = v
        if rc == 0 and os.path.isfile(outp):
            data = json.load(open(outp))
            if data.get("_v3_compiled") is False:
                print("[probe] v3 kernels did not compile; v1 only")
                return {}
            status.update(data)
            break
        inflight = [l.split()[1] for l in text.splitlines() if l.startswith("PROBE ")]
        inflight = [n for n in inflight if n not in status]
        if not inflight:
            print(f"[probe] child failed before any v3 config (rc={rc}); v1 only\n{err[-2000:]}")
            break
        status[inflight[-1]] = "crash" if rc != -1 else "hang"
        print(f"[probe] {inflight[-1]} -> {status[inflight[-1]]} ({time.time() - t0:.1f}s); retrying the rest\n{err[-600:]}")
    print("[probe] v3 bitwise vs v1: " + "  ".join(f"{n}={v}" for n, v in status.items()))
    return status


def mma_throughput(K, nsm):
    """tensor-core rate on this card: bf16 inputs + f32 accumulate vs f16 inputs + f16 accumulate."""
    sink = torch.zeros(1, dtype=torch.float32, device="cuda")
    out = {}
    for f16 in (0, 1):
        iters = 4096
        grid = nsm * 4
        K["fg_mma_tput"](grid, [sink, iters, f16])
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        K["fg_mma_tput"](grid, [sink, iters, f16])
        e1.record(); e1.synchronize()
        ms = e0.elapsed_time(e1)
        out["f16acc" if f16 else "f32acc"] = grid * 8 * iters * 8 * 4096 / (ms * 1e-3) / 1e12
    return out


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[8, 7131, 15329])
    ap.add_argument("--designs", nargs="+",
                    default=["ct_ref", "anchor_cublas", "forge_tc", "forge_tc_f16", "dec_cublas", "forge_gemv8"])
    ap.add_argument("--weights", default=None, help="model.safetensors (or snapshot dir) containing layer-0 MLP")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--target-ms", type=float, default=30.0, help="replays per sample ~= target/forward")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--no-v3", action="store_true", help="skip the warpspeed kernels entirely")
    ap.add_argument("--f16", choices=["auto", "on", "off"], default="auto",
                    help="fp16-accumulate datapath design: auto = only if this card's f16-acc MMA rate beats f32-acc")
    ap.add_argument("--lib-lowrank", action="store_true",
                    help="forge_tc: rank-16 factors via the oracle's cuBLAS calls (bitwise path, no PDL chain)")
    ap.add_argument("--probe-timeout", type=float, default=300.0)
    ap.add_argument("--out", default="forge_results.json")
    ap.add_argument("--_probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--probe-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--probe-skip", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._probe:
        return probe_main(args)

    props, arch = _setup_torch()
    dev = torch.device("cuda")
    nsm = props.multi_processor_count
    print(f"[device] {props.name} cc={arch} SMs={nsm} torch={torch.__version__} cuda={torch.version.cuda}")

    drv = Driver()
    nv = NVRTC()
    K = build_kernels(drv, nv, arch)
    tput = mma_throughput(K, nsm)
    mem = measure_memory(K, drv, nsm)
    print(f"[mem] DRAM {mem['dram_GBs']:.0f} GB/s  L2 {mem['l2_GBs']:.0f} GB/s  L2 {mem['l2_bytes'] / 2**20:.0f} MiB  "
          f"persisting set-aside max {mem['persist_max'] / 2**20:.0f} MiB  window max {mem['window_max'] / 2**20:.0f} MiB")
    print(f"[mma] dense tensor throughput on this card: bf16->f32 acc {tput['f32acc']:.1f} TFLOPS, "
          f"f16->f16 acc {tput['f16acc']:.1f} TFLOPS (x{tput['f16acc'] / tput['f32acc']:.2f})")
    ratio = tput["f16acc"] / tput["f32acc"]
    use_f16 = args.f16 == "on" or (args.f16 == "auto" and ratio >= 1.15)
    f16_names = [n for n in WS_CFGS if WS_CFGS[n][6]] if not use_f16 else []
    v3_status = run_probe(args, f16_names) if (K["_v3_ok"] and not args.no_v3) else {}
    ok_v3 = [n for n, v in v3_status.items() if v == "ok"]

    def cands(epi, f16):
        return [n for n in ok_v3 if WS_CFGS[cfg_base(n)][4] == epi and WS_CFGS[cfg_base(n)][6] == f16]
    GU_C, DN_C = V1_GU + cands(0, 0), V1_DN + cands(1, 0)
    GU_H, DN_H = cands(0, 1), cands(1, 1)
    use_f16 = use_f16 and bool(GU_H) and bool(DN_H)
    print(f"[tune] GU candidates {GU_C}\n[tune] DN candidates {DN_C}")
    print(f"[tune] fp16-datapath design: {'ON' if use_f16 else 'off'} (f16/f32 MMA rate x{ratio:.2f}, --f16 {args.f16})"
          + (f"  GU {GU_H}  DN {DN_H}" if use_f16 else ""))

    W, wsrc = load_weights(args)
    for k, shp in dict(Qg=(I, H // 8), Sg=(I, H // 32), Qu=(I, H // 8), Su=(I, H // 32),
                       Qd=(H, I // 8), Sd=(H, I // 32)).items():
        assert tuple(W[k].shape) == shp, f"{k} shape {tuple(W[k].shape)} != {shp}"
        W[k] = W[k].contiguous()
    print(f"[weights] {wsrc}")
    P = prepare_packed(W)

    import numpy as np
    tab = torch.empty(65536, dtype=torch.int16, device=dev)
    K["fg_gelu_table"](256, [tab])
    allbf = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.int16)).to(dev).view(torch.bfloat16)
    tref = F.gelu(allbf, approximate="tanh").view(torch.int16)
    gelu_mis = int(((tab != tref) & torch.isfinite(allbf)).sum().item())
    print(f"[gelu] device GELU_tanh vs torch: {gelu_mis} mismatching finite BF16 patterns of 65536")

    results = dict(device=props.name, torch=torch.__version__, cuda=torch.version.cuda, weights=wsrc,
                   nvrtc=f"{nv.version}", gelu_mismatches=gelu_mis, v3_probe=v3_status, mma_tflops=tput, mem=mem,
                   kernels={n: dict(regs=k.regs, local=k.local, smem=k.smem) for n, k in K.items() if isinstance(k, Kern)},
                   runs=[])

    def dump():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=str)

    for M in args.M:
        print(f"\n======================= M={M} =======================")
        A, X = make_adapters_and_x(M)
        t0 = time.time()
        Yref, midref = ref_forward(X, W, A, want_mid=True)
        torch.cuda.synchronize()
        print(f"[ref] oracle computed in {time.time() - t0:.2f}s")
        todo = []
        for d in args.designs:
            if d == "ct_ref":
                todo.append(lambda: CtRef(K, W, A, X))
            elif d == "anchor_cublas" and M >= 64:
                todo.append(lambda: AnchorCublas(K, W, A, X, midref))
            elif d == "forge_gemv8" and M == 8:
                todo.append(lambda: ForgeGemv8(K, P, A, X, drv, mem))
            elif d == "forge_tc" and M >= 64:
                todo.append(lambda: ForgeTC(K, P, A, X, nsm, lib_lowrank=args.lib_lowrank))
            elif d == "forge_tc_f16" and M >= 64 and use_f16:
                todo.append(lambda: ForgeTC(K, P, A, X, nsm, f16=True))
            elif d == "dec_cublas" and M >= 64:
                todo.append(lambda: DecCublas(K, W, A, X))
        base_ms = None
        while todo:
            des = todo.pop(0)()
            rec = dict(M=M, design=des.name, anchor=des.anchor)
            try:
                if isinstance(des, ForgeGemv8):
                    l2rows = tune_l2(des)
                    rec["l2_modes"] = l2rows
                    print("[l2] " + "  ".join(f"{m}={t:.4f}ms" + ("" if ok is True else f"({ok})") for t, m, ok in l2rows)
                          + f"  -> {des.mode} keep(gu,dn quads)={des.keep}")
                    rec["config"] = [des.mode]
                    if des.drv is not None and "fg_gu8h" in K:
                        vrows = tune_gemv8(des)
                        rec["gemv8_variants"] = vrows
                        for t, m, gk, dk, ok in sorted(vrows):
                            print(f"[gemv8] {m:7s} {gk:13s} {dk:12s} {t:.4f} ms" + ("" if ok is True else f" ({ok})"))
                        rec["config"] = [des.mode, des.gu_k, des.dn_k]
                if isinstance(des, ForgeTC):
                    cg, cd = (GU_H, DN_H) if des.f16 else (GU_C, DN_C)
                    if des.f16:
                        des.set_cfg(cg[0], cd[0])
                    if not args.no_tune:
                        tr = tune_tc(des, cg, cd)
                        des.set_cfg(*tr["best"])
                        rec["tune"] = tr
                        print("[tune] GU " + "  ".join(f"{n}={v:.3f}ms" for n, v in tr["gu"].items())
                              + f"  accepted={tr['accepted_gu']}")
                        print("[tune] DN " + "  ".join(f"{n}={v:.3f}ms" for n, v in tr["dn"].items())
                              + f"  accepted={tr['accepted_dn']}  -> {tr['best']}")
                        for gname, dd in tr.get("dequant", {}).items():
                            print(f"[dequant] {gname}: {dd['with_ms']:.3f} ms with dequant, {dd['without_ms']:.3f} ms with "
                                  f"raw words -> dequant = {100 * dd['share']:.1f}% of that GEMM")
                        print(f"[tune] low-rank kernels (standalone): gu {tr['lowk_gu_ms']:.3f}ms  d {tr['lowk_d_ms']:.3f}ms"
                              + ("  (cuBLAS)" if des.lib else "  (overlapped via PDL in the chain)"))
                        ch = tune_chain(des, tr)
                        rec["chain_tune"] = ch
                        for t_, g_, d_, p_ in sorted(ch):
                            print(f"[chain] {g_:12s} {d_:12s} pdl={str(p_):5s} {t_:8.3f} ms")
                        stb = stage_breakdown(des)
                        rec["stages"] = stb
                        pdl_on = [r[0] for r in ch if r[1:3] == (des.gu, des.dn) and r[3]]
                        pdl_off = [r[0] for r in ch if r[1:3] == (des.gu, des.dn) and not r[3]]
                        print(f"[chain] sustained per stage: lowk_gu {stb['lowk_gu']:.3f}  GU {stb['gu']:.3f}  lowk_d "
                              f"{stb['lowk_d']:.3f}  DN {stb['dn']:.3f}  = {sum(stb.values()):.3f} ms | chained: "
                              f"PDL {pdl_on[0] if pdl_on else float('nan'):.3f}  no-PDL "
                              f"{pdl_off[0] if pdl_off else float('nan'):.3f} ms -> using pdl={des.pdl}")
                    rec["config"] = [des.gu, des.dn] + ([] if des.pdl else ["no-pdl"])
                y = des.forward()
                torch.cuda.synchronize()
                if not des.anchor:
                    rec["numerical"] = numerics(y, Yref)
                    if getattr(des, "mid", None) is not None:
                        rec["mid_bitwise_mismatch"] = int((des.mid.view(torch.int16) != midref.view(torch.int16)).sum().item())
                try:
                    rec.update(bench(des.forward, args.warmup, args.samples, args.target_ms))
                except Exception as e:
                    if not getattr(des, "pdl", False):
                        raise
                    print(f"[{des.name}] graph capture with PDL failed ({type(e).__name__}: {e}); retrying without PDL")
                    torch.cuda.synchronize()
                    des.pdl = False
                    rec["pdl"] = False
                    rec.update(bench(des.forward, args.warmup, args.samples, args.target_ms))
                rec["TFLOPS"] = flops(M) / (rec["forward_ms"] * 1e-3) / 1e12
                rec["sol_ms"] = sol_ms(M, tput, mem)
                rec["sol_pct"] = 100.0 * rec["sol_ms"] / rec["forward_ms"]
                if des.name == "ct_ref":
                    base_ms = rec["forward_ms"]
                if base_ms:
                    rec["speedup_vs_ct_ref"] = base_ms / rec["forward_ms"]
                line = f"[{des.name:13s}] {rec['forward_ms']:9.4f} ms (min {rec['minimum_ms']:.4f})  {rec['TFLOPS']:6.1f} TFLOPS"
                if des.anchor:
                    line += "  (anchor: base GEMMs only, pre-decoded; not a design)"
                else:
                    n = rec["numerical"]
                    line += (f"  pass={n['strict_bf16_pass']} fails={n['failed_elements']} max_abs={n['max_abs']:.3g} "
                             f"relL2={n['relative_l2_pct']:.5f}% bitwise={100 * n['bitwise_equal_frac']:.3f}% "
                             + (f"mid_mismatch={rec['mid_bitwise_mismatch']} " if "mid_bitwise_mismatch" in rec else "")
                             + f"replay_ok={rec['replay_matches_eager']}"
                             + (f"  x{rec['speedup_vs_ct_ref']:.3f} vs ct_ref" if "speedup_vs_ct_ref" in rec else ""))
                print(line)
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"[{des.name}] FAILED: {rec['error']}")
            results["runs"].append(rec)
            dump()
            if isinstance(des, ForgeGemv8):
                des.release()
            del des
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        del Yref, midref, A, X
        torch.cuda.empty_cache()

    print("\n======================= summary (CUDA-graph median ms) =======================")
    print(f"SOL = max(6*M*H*I / {tput['f32acc']:.1f} TFLOPS mma.sync, weight bytes beyond the {mem['l2_bytes'] / 2**20:.0f} MiB "
          f"L2 / {mem['dram_GBs']:.0f} GB/s DRAM, weight bytes / {mem['l2_GBs']:.0f} GB/s L2)  -- measured on this card")
    print(f"{'M':>6s}  {'design':14s} {'ms':>10s} {'TFLOPS':>8s} {'%SOL':>6s} {'pass':>5s} {'fails':>7s} {'vs ref':>7s}  config")
    for r in results["runs"]:
        if "forward_ms" not in r:
            print(f"{r['M']:6d}  {r['design']:14s} {'ERROR':>10s}  {r.get('error', '')[:80]}")
            continue
        npass = "-" if r["anchor"] else str(r["numerical"]["strict_bf16_pass"])
        nfail = -1 if r["anchor"] else r["numerical"]["failed_elements"]
        print(f"{r['M']:6d}  {r['design']:14s} {r['forward_ms']:10.4f} {r['TFLOPS']:8.1f} {r['sol_pct']:6.1f} {npass:>5s} {nfail:7d} "
              f"{r.get('speedup_vs_ct_ref', float('nan')):7.3f}  {r.get('config', '')}")
    print("\nfastest design per M (anchors excluded):")
    for M in args.M:
        ok = [r for r in results["runs"] if r["M"] == M and "forward_ms" in r and not r["anchor"]
              and r.get("replay_matches_eager", False)]
        if ok:
            b = min(ok, key=lambda r: r["forward_ms"])
            print(f"  M={M:6d}: {b['design']:12s} {b['forward_ms']:.4f} ms  ({b['TFLOPS']:.1f} TFLOPS, "
                  f"fails={b['numerical']['failed_elements']}) {b.get('config', '')}")
    dump()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
