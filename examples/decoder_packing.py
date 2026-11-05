"""A toy decoder grammar over GF(2): enumerate legal decoders by exact joins, then prove each one's weight packing.

Real decoders read packed 4-bit codes with shifts, masks and permutes; here the "reference decode" is a nibble swap
and the atoms are small bit movements, so the whole space fits on a screen.
"""

import xcalibur as xc

BITS = 8
ref = xc.from_function(lambda q: ((q << 4) | (q >> 4)) & 0xFF, BITS)        # reference decode: swap nibbles




# Atoms: 4-row blocks of a decoder (each row = the input bits XOR-ed into one output bit). A decoder stacks two
# blocks into 8 rows; the join keeps only stacks whose rows are independent (full rank), i.e. invertible decoders.
e = [1 << i for i in range(BITS)]
atoms = [
  ("read_lo", tuple(e[i] for i in range(4))),                  # rows read input bits 0..3
  ("read_hi", tuple(e[i + 4] for i in range(4))),              # rows read input bits 4..7
  ("read_lo^hi", tuple(e[i] ^ e[i + 4] for i in range(4))),    # rows read the XOR of both nibbles
  ("read_lo_rot", tuple(e[(i + 1) % 4] for i in range(4))),    # rows read the low nibble, rotated
]
joins = xc.join_all(atoms, BITS)
print(f"{len(atoms)} atoms -> {len(joins)} full-rank joins:", joins)

for names in joins:
  rows = tuple(r for n in names for r in dict(atoms)[n])
  D = rows[:BITS]
  T = xc.packing_transform(D, ref)                             # D(T(w)) == ref(w) for every byte
  ok = all(xc.apply(D, xc.apply(T, w)) == xc.apply(ref, w) for w in range(256))
  print(f"  decoder {' | '.join(names):24s} packing proven on all 256 words: {ok}")

try:
  print("as a CuTe layout:", xc.as_cute_layout(ref))
except ImportError:
  print("(install pycute / nvidia-cutlass to see the same map as a CuTe Layout over GF(2))")
