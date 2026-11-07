# Copied verbatim from kernels-community/finegrained-fp8 @ 8c178950e8710e26a2210e4e909cd02dc16c8715,
# build/torch-cuda/matmul.py (w8a8_block_dynamic_fp8_matmul_kernel, _swizzle_offsets, _store_masked) and
# build/torch-cuda/utils.py (fp8_act_quant_inline, decode_ue8m0_scale).
# Copyright 2025-2026 The HuggingFace Inc. team. Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Modified by xArmory: the @triton.autotune decorator is removed (aot.py compiles each config explicitly), docstrings
# are dropped, and the helpers are gathered into this one file. Function signatures and bodies are unchanged (checked
# AST-equal against the revision's matmul.py / utils.py).
"""The stock kernel's source, so its sm_89 machine code can be compiled and read on any box (aot.py, sass.py).

This file is for inspection only; timing and parity always use the hub kernel itself (stock.load_hub) on Ada.
"""
import triton
import triton.language as tl


@triton.jit
def fp8_act_quant_inline(a_raw):
    a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0
    a_fp8 = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)
    return a_fp8, a_s


@triton.jit
def decode_ue8m0_scale(scale):
    if scale.dtype == tl.uint8:
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    return scale


@triton.jit
def _swizzle_offsets(
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    return pid_m, pid_n, offs_am, offs_bn, offs_k


@triton.jit
def _store_masked(
    C,
    accumulator,
    pid_m,
    pid_n,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def w8a8_block_dynamic_fp8_matmul_kernel(
    A,  # (M, K) raw BF16/FP16 activations
    B,  # (N, K) FP8 weights
    C,  # (M, N) output
    Bs,  # (N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (fp32 or uint8/UE8M0)
    # Shape
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bs_k,
    stride_bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_m, pid_n, offs_am, offs_bn, offs_k = _swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // BLOCK_SIZE_N
    bs_ptrs = Bs + offs_bsn * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a_raw = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0).to(
            tl.float32
        )
        a, a_s = fp8_act_quant_inline(a_raw)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        b_s = decode_ue8m0_scale(tl.load(bs_ptrs))
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    _store_masked(
        C,
        accumulator,
        pid_m,
        pid_n,
        M,
        N,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
