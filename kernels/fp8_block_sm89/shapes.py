"""The Kev-27B FP8 GEMM inventory: every FP8 Linear of one forward pass, with call counts per request.

Kev-27B is a Qwen3.8-27B hybrid (64 layers: 48 Gated DeltaNet, 16 full attention, every 4th) served prefill-only on
RTX 6000 Ada with transformers' FineGrainedFP8Config. Every Linear except DeltaNet in_proj_a / in_proj_b is FP8 e4m3
with 128x128 block weight scales and dynamic per-token-group(128) activation quantisation.

  hidden 5120, intermediate 17408; attention 24 q heads (with output gate) / 4 kv heads, head_dim 256;
  DeltaNet 16 key heads x 128, 48 value heads x 128.

A GEMM computes C[M, N] = A[M, K] @ W[N, K]^T. M is tokens: ~28k for the state pass, ~40 per question row pass
(each question its own pass at >= 16k context), ~320-600 when the rows are packed into one pass (B4).

  >>> gemm_flops_per_token() / 1e9                                # doctest: +ELLIPSIS
  48.6...
"""
from __future__ import annotations

from dataclasses import dataclass

HIDDEN, INTERMEDIATE = 5120, 17408
LAYERS, DELTANET_LAYERS, ATTENTION_LAYERS = 64, 48, 16
SCALE_BLOCK = 128                                   # weight scale block (N and K) == activation quant group (K)


@dataclass(frozen=True)
class Gemm:
  name: str
  N: int
  K: int
  per_pass: int                                     # calls per forward pass (one per layer of its kind)
  group: str | None = None                          # GEMMs reading the same input tensor (quantise-once candidates)


GEMMS: tuple[Gemm, ...] = (
  Gemm("deltanet.in_proj_qkv", 10240, HIDDEN, DELTANET_LAYERS, "deltanet_in"),   # q, k: 16x128 each; v: 48x128
  Gemm("deltanet.in_proj_z", 6144, HIDDEN, DELTANET_LAYERS, "deltanet_in"),
  Gemm("deltanet.out_proj", HIDDEN, 6144, DELTANET_LAYERS),
  Gemm("attn.q_proj", 12288, HIDDEN, ATTENTION_LAYERS, "attn_in"),               # 24 x 256 x 2 (query + gate)
  Gemm("attn.k_proj", 1024, HIDDEN, ATTENTION_LAYERS, "attn_in"),
  Gemm("attn.v_proj", 1024, HIDDEN, ATTENTION_LAYERS, "attn_in"),
  Gemm("attn.o_proj", HIDDEN, 6144, ATTENTION_LAYERS),
  Gemm("mlp.gate_proj", INTERMEDIATE, HIDDEN, LAYERS, "mlp_in"),
  Gemm("mlp.up_proj", INTERMEDIATE, HIDDEN, LAYERS, "mlp_in"),
  Gemm("mlp.down_proj", HIDDEN, INTERMEDIATE, LAYERS),
)


@dataclass(frozen=True)
class Bucket:
  """A range of M served by one build. The hub kernel keys its autotuning on (N, K, BLOCK_M), with
  BLOCK_M = min(max(next_pow2(M), 16), 128), so every M > 64 shares one stock config; we split finer."""
  name: str
  lo: int
  hi: int                                           # inclusive
  M: int                                            # the representative size we tune and time at


BUCKETS: tuple[Bucket, ...] = (                    # representatives = the Ada probe's token counts (fp8_gemm_probe.py)
  Bucket("m64", 1, 64, 64),                         # one question row pass (~40 tokens)
  Bucket("m256", 65, 256, 256),
  Bucket("m512", 257, 1024, 512),                   # packed question rows (B4)
  Bucket("m4k", 1025, 8192, 4096),                  # chunked or short state passes
  Bucket("m28k", 8193, 1 << 31, 28672),             # the state pass
)
DEFAULT_BUCKETS = ("m28k", "m4k", "m512", "m64")

# Passes per request: (M, passes). Almost every request is a ~28k-token state plus <= 10 questions.
WORKLOADS: dict[str, tuple[tuple[int, int], ...]] = {
  "unpacked_rows": ((28672, 1), (40, 8)),           # production today: each question its own row pass
  "packed_rows": ((28672, 1), (320, 1)),            # B4: all rows in one pass on the shared state
}


def bucket_of(M: int) -> Bucket:
  return next(b for b in BUCKETS if b.lo <= M <= b.hi)


def bucket_named(name: str) -> Bucket:
  return next(b for b in BUCKETS if b.name == name)


def unique_shapes() -> list[tuple[int, int]]:
  """Distinct (N, K) in inventory order: each is its own autotune key (and its own build) per bucket."""
  return list(dict.fromkeys((g.N, g.K) for g in GEMMS))


def input_groups() -> dict[str, list[Gemm]]:
  out: dict[str, list[Gemm]] = {}
  for g in GEMMS:
    if g.group:
      out.setdefault(g.group, []).append(g)
  return out


def concat_shapes() -> dict[str, tuple[int, int]]:
  """The single GEMM each input group becomes when its FP8 weights are concatenated along N (exact; concat.py)."""
  return {name: (sum(g.N for g in gs), gs[0].K) for name, gs in input_groups().items()}


def launches_per_pass(concat: bool) -> int:
  """FP8 GEMM launches per forward pass, with or without the concat groups."""
  if not concat:
    return sum(g.per_pass for g in GEMMS)
  return sum(g.per_pass for g in GEMMS if not g.group) + sum(gs[0].per_pass for gs in input_groups().values())


def gemm_flops_per_token() -> int:
  return sum(2 * g.N * g.K * g.per_pass for g in GEMMS)


def request_calls(workload: tuple[tuple[int, int], ...]) -> dict[tuple[str, int], int]:
  """{(gemm name, M): calls per request}: the weights for splitting a request's GEMM time by shape."""
  out: dict[tuple[str, int], int] = {}
  for M, passes in workload:
    for g in GEMMS:
      out[(g.name, M)] = out.get((g.name, M), 0) + g.per_pass * passes
  return out


def parse_shapes(spec: str | None) -> list[tuple[int, int]]:
  """'10240x5120,1024x5120' -> [(10240, 5120), (1024, 5120)]; None or 'all' -> every unique shape."""
  if not spec or spec == "all":
    return unique_shapes()
  return [tuple(int(v) for v in s.lower().split("x")) for s in spec.split(",")]  # type: ignore[misc]


def gemm_names(N: int, K: int) -> list[str]:
  return [g.name for g in GEMMS if (g.N, g.K) == (N, K)]
