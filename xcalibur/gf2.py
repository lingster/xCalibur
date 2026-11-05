"""Exact GF(2) maps for weight decoders: bit layouts as linear maps, proofs by composition.

Packed low-bit weights are decoded with bit operations (shifts, masks, LOP3, PRMT) before the math. Every such
decoder that only moves and selects bits is a linear map over GF(2) from a 32-bit packed word to its output bit
fields, so the whole design space of "how are the bits packed, and which bit tricks read them" becomes linear
algebra:

* a decoder is a tuple of rows (one per output bit; each row is the mask of input bits XOR-ed into it);
* two decoders are equivalent if their maps are equal, which composition checks exactly for all 2^32 words;
* for any invertible decoder D and the reference decode P, the weight-packing transform that makes D correct is
  exactly T = D^-1 P, so a new decoder comes with a lossless repacking for prepare(), proven, not tested.

In CuTe terms a map is ``Layout((2,)*32, F2-strides)``: a 32-mode layout over GF(2) whose stride for input bit i is
the image of basis vector e_i (``columns``). Evaluating the layout at a word is applying the map.

  >>> P = tuple(1 << i for i in range(32))          # identity decode as a stand-in reference
  >>> D = compose(rotation(5), P)                   # a decoder that reads the word rotated by 5
  >>> T = packing_transform(D, P)                    # repack weights so that D(T(w)) == P(w)
  >>> all(apply(D, apply(T, w)) == apply(P, w) for w in (0, 1, 0x12345678, 0xffffffff))
  True
"""
from __future__ import annotations

U32 = (1 << 32) - 1


def rank(rows) -> int:
  pivots = {}
  for x in rows:
    while x:
      p = x.bit_length() - 1
      if p not in pivots:
        pivots[p] = x
        break
      x ^= pivots[p]
  return len(pivots)


def apply(rows, word: int) -> int:
  """Map a word: output bit i is the parity of (row_i AND word)."""
  return sum(((r & word).bit_count() & 1) << i for i, r in enumerate(rows))


def compose(a, b):
  """Rows of a(b(x))."""
  out = []
  for row in a:
    acc = 0
    while row:
      bit = row & -row
      acc ^= b[bit.bit_length() - 1]
      row ^= bit
    out.append(acc)
  return tuple(out)


def inverse(rows):
  n = len(rows)
  a = [r | (1 << (n + i)) for i, r in enumerate(rows)]
  for i in range(n):
    piv = next((j for j in range(i, n) if (a[j] >> i) & 1), None)
    if piv is None:
      raise ValueError("singular map")
    a[i], a[piv] = a[piv], a[i]
    for j in range(n):
      if j != i and (a[j] >> i) & 1:
        a[j] ^= a[i]
  return tuple(x >> n for x in a)


def columns(rows):
  """Images of the basis vectors: the F2 strides of the equivalent CuTe layout."""
  return tuple(apply(rows, 1 << i) for i in range(len(rows)))


def from_function(fn, bits: int = 32):
  """The rows of a linear bit function fn(word) -> word (checked linear on the basis only; verify with equal())."""
  images = [fn(1 << i) for i in range(bits)]
  return tuple(sum(((images[i] >> r) & 1) << i for i in range(bits)) for r in range(bits))


def rotation(k: int, bits: int = 32):
  k %= bits
  return from_function(lambda q: ((q << k) | (q >> ((bits - k) % bits))) & ((1 << bits) - 1), bits)


def equal(a, b) -> bool:
  """Exact equality of two maps for every input word (compare basis images)."""
  return columns(a) == columns(b)


def packing_transform(decoder, reference):
  """T with decoder o T == reference: the lossless repacking that makes a new decoder produce the reference values."""
  t = compose(inverse(decoder), reference)
  assert compose(decoder, t) == tuple(reference), "packing proof failed"
  assert rank(t) == len(t)
  return t


def as_cute_layout(rows):
  """The map as a CuTe layout over GF(2), if NVIDIA's pycute (CUTLASS python) is installed."""
  from pycute import Layout                    # noqa: PLC0415
  from pycute.swizzle import F2                # noqa: PLC0415
  return Layout((2,) * len(rows), tuple(F2(c) for c in columns(rows)))


def compatible(a_rows, b_rows) -> bool:
  """Two partial decoders (row sets for disjoint output fields) can be joined if their rows stay independent."""
  return rank(tuple(a_rows) + tuple(b_rows)) == len(a_rows) + len(b_rows)


def join_all(atoms, target_rank: int):
  """Enumerate every set of pairwise-compatible atoms whose rows reach full rank. Each atom: (name, rows).

  This is the exact subgame idea at toy scale: a pairwise-compatibility bitset prunes the product before any
  combination is materialized; the production version runs the same join over bitsets on the GPU.
  """
  n = len(atoms)
  ok = [[i != j and compatible(atoms[i][1], atoms[j][1]) for j in range(n)] for i in range(n)]
  out = []

  def grow(chosen, rows, start):
    if len(rows) == target_rank and rank(rows) == target_rank:
      out.append(tuple(atoms[i][0] for i in chosen))
      return
    for j in range(start, n):
      if all(ok[i][j] for i in chosen) and len(rows) + len(atoms[j][1]) <= target_rank:
        grow(chosen + [j], rows + tuple(atoms[j][1]), j + 1)
  grow([], (), 0)
  return out
