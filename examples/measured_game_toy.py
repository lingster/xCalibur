"""The measured game on a synthetic problem (CPU only): a learned ranker proposes, noisy "measurements" decide,
and binding needs pooled evidence. The hidden truth has one real gain on one load and many look-alikes."""

import itertools
import math
import random

import xcalibur as xc

LOADS = ("m1", "m16", "m32")
SPACE = {"cutoff": [8, 17], "tile": [64, 128, 256], "stages": [2, 3, 4], "unroll": [1, 2, 4], "decoder": list(range(16))}
INC = {"cutoff": 17, "tile": 128, "stages": 3, "unroll": 2, "decoder": 0}
REACH = xc.ReachMap(loads={"m1": 1, "m16": 16, "m32": 32}, cutoff_field="cutoff",
                    fields={"large": ("tile", "stages"), "small": ("unroll",), "all": ("decoder",)})
rng = random.Random(0)


def truth(d, load):
  """Hidden speedup vs the reference: decoder 11 helps m16 by 2%; nothing else beats the incumbent by much."""
  g = 1.0
  if load == "m16" and d["decoder"] == 11:
    g *= 1.020
  if load == "m32" and REACH.build_of(load, d) == "large":
    g *= 1 + 0.003 * (d["stages"] - 3) - 0.004 * abs(math.log2(d["tile"] / 128))
  if load == "m1":
    g *= 1 - 0.01 * abs(d["unroll"] - 2)
  return g


def measure(designs, grade):
  sd = 0.010 if grade == "screen" else 0.003
  return [{L: truth(d, L) * math.exp(rng.gauss(0, sd)) for L in LOADS} for d in designs]


def featurize(d, compiled):
  return [d["cutoff"], d["tile"], d["stages"], d["unroll"]] + xc.bits(d["decoder"], 4)


def frontier(history):
  seen = {repr(sorted(h["design"].items())) for h in history}
  everything = [dict(zip(SPACE, v)) for v in itertools.product(*SPACE.values())]
  return [d for d in everything if repr(sorted(d.items())) not in seen]


game = xc.Game(loads=LOADS, reach=REACH, incumbent=INC, frontier=frontier, featurize=featurize,
               compile=lambda ds: [{"ok": True} for _ in ds], measure=measure, per_load=6, compile_n=18, screen_n=12,
               margin=0.008, min_samples=3)
seed = [INC] + rng.sample(frontier([]), 40)                                  # round 0: a small random sweep
for d, loads in zip(seed, measure(seed, "screen")):
  game.history.append({"design": d, "loads": loads, "weight": 1.0})
for rnd in range(1, 9):
  print(f"round {rnd}:", game.round())
bound, _ = game.evidence.bind(3, 0.008)
for L, b in bound.items():
  print(f"bound {L}: {b['identity']}  +{b['gain'] * 100:.2f}% over the incumbent ({b['samples']} samples)")
