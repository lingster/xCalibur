"""The measured game for one GEMM shape, wired to this design space (GPU-free; ``search.py`` adds the hardware).

Loads are M buckets of one (N, K). Each load binds its own build, so a design that only wins for question rows
ships for question rows only, and the state pass keeps stock or its own winner (per-size builds, as in kernels/moe_w4a16_h200).

The incumbent is the stock hub kernel itself (``space.STOCK``). Every measurement is a speedup over stock, timed
in the same session; at confirm grade stock is timed against itself first (an A/A), and a design's gain is its
ratio over that A/A ratio, so binding is "beats stock by more than the session's noise, twice, never losing".
"""
from __future__ import annotations

import xcalibur as xc

import space


class PerShapeReach:
  """``xcalibur.ReachMap``'s interface for per-size builds: each load is its own build, so any change reaches
  every load it is applicable to, and a design's identity is the design itself (no field is dead for a load)."""

  def __init__(self, loads: dict[str, int], K: int | None = None):
    self.loads, self.K = loads, K

  def reaches(self, load: str, design: dict, base: dict) -> bool:
    return space.key(design) != space.key(base) and space.applicable(design, self.loads[load], self.K)

  def identity(self, load: str, design: dict) -> str:
    return space.key(design)


def aa_margin(aa: dict, floor: float = 0.005) -> float:
  """The binding margin (as a gain) for a session: ``xcalibur.aa_margin`` over the A/A median and its CI ends
  (1 + 2 max|r - 1|), never below ``floor``. A gain under it is indistinguishable from timing stock against itself."""
  return max(floor, xc.aa_margin([aa["ratio"], *aa["ci"]]) - 1.0)


def make_game(loads: dict[str, int], K: int, measure, compile, *, surrogate_backend: str = "auto",
              limit: int = space.SMEM_LIMIT_SM89, standin_bytes: int = 1, keep=None, **kw) -> xc.Game:
  """A ``xcalibur.Game`` over the legal sm_89 designs with stock as the incumbent.

  measure(designs, grade) -> [{load: speedup over stock}] (skip loads where ``space.applicable`` is False);
  compile(designs) -> [{"ok": bool, "n_regs", "n_spills", "shared"}];
  keep(design) -> bool restricts the frontier (e.g. ``space.bitwise_safe``).
  """
  front = lambda h: [d for d in space.frontier(h, limit, standin_bytes) if keep is None or keep(d)]   # noqa: E731
  return xc.Game(loads=tuple(loads), reach=PerShapeReach(loads, K), incumbent=dict(space.STOCK),
                 frontier=front, featurize=space.featurize,
                 compile=compile, measure=measure, surrogate={"backend": surrogate_backend}, **kw)


def seed_designs(n: int, rng, limit: int = space.SMEM_LIMIT_SM89, standin_bytes: int = 1,
                 extra: list[dict] = ()) -> list[dict]:
  """Round 0: stock, ``extra`` (e.g. public.seeds: vLLM's L40S picks), a few hand anchors (stock's tile with
  quantise-once and more stages, smaller N tiles, split-K for skinny M), then a random sample of the legal space.
  The surrogate needs >= 12 measured rows per load before it ranks."""
  legal = {space.key(d) for d in space.designs(limit, standin_bytes)}
  base = dict(space.STOCK, BLOCK_M=64, num_warps=4, num_stages=3)
  anchors = list(extra) + [
    dict(base, quant="once"), dict(base, BLOCK_M=128, num_warps=8, num_stages=2),
    dict(base, quant="once", BLOCK_M=128, num_warps=8, num_stages=4), dict(base, quant="once", BLOCK_N=64),
    dict(base, quant="once", SPLIT_K=2), dict(base, quant="once", BLOCK_M=32, SPLIT_K=4),
    dict(base, quant="once", BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=4)]
  anchors = list({space.key(a): a for a in anchors if space.key(a) in legal}.values())
  rest = [d for d in space.designs(limit, standin_bytes) if space.key(d) not in {space.key(a) for a in anchors}]
  return [dict(space.STOCK)] + anchors + rng.sample(rest, max(0, n - len(anchors)))


def play(game: xc.Game, seeds: list[dict], rounds: int, log=print, confirm_top: int = 4) -> dict[str, dict]:
  """Screen the seeds, confirm the best seed per load that beats stock by ``game.nominate`` (a measured design
  never re-enters the frontier, so seed leaders need this explicit confirm), then play ``rounds`` rounds."""
  results = game.measure(seeds, "screen")
  for d, loads in zip(seeds, results):
    game.history.append({"design": d, "loads": loads, "weight": 1.0})
  leaders: list[dict] = []
  for L in game.loads:
    ranked = sorted(((r[L], i) for i, r in enumerate(results) if L in r and not space.is_stock(seeds[i])),
                    reverse=True)
    stock = [r[L] for i, r in enumerate(results) if L in r and space.is_stock(seeds[i])] or [1.0]
    for ratio, i in ranked[:confirm_top]:
      if ratio > stock[0] * (1 + game.nominate) and seeds[i] not in leaders:
        leaders.append(seeds[i])
  log(f"seed: {len(seeds)} screened, {len(leaders)} leaders to confirm")
  game.confirm(leaders)
  for r in range(1, rounds + 1):
    log(f"round {r}: {game.round()}")
  return bound_per_load(game, game.min_samples, game.margin)


def fresh_confirm(measure, bound: dict[str, dict], runs: int = 3, margin: float = 0.005,
                  min_samples: int = 2) -> dict[str, dict]:
  """Winner's curse (xcalibur.stats): the best of many noisy reads reads high, so a search winner is
  re-measured in ``runs`` fresh confirm sessions, each timing stock against itself first and then the winners, and
  bound again on that fresh evidence alone (>= min_samples, mean gain above the margin, no negative sample).
  ``measure`` is the game's measure callback; ``bound`` is {load: {"design": ...}} from the search."""
  designs = list({space.key(b["design"]): b["design"] for b in bound.values()}.values())
  ev = xc.Evidence()
  for _ in range(runs):
    rows = measure([dict(space.STOCK)] + designs, "confirm")
    ctl = rows[0]
    for d, loads in zip(designs, rows[1:]):
      for L, b in bound.items():
        if space.key(b["design"]) == space.key(d) and L in loads and L in ctl:
          ev.add(L, space.key(d), loads[L] / ctl[L] - 1.0, d)
  final, _ = ev.bind(min_samples, margin)
  return final


def bound_per_load(game: xc.Game, min_samples: int, margin: float) -> dict[str, dict]:
  """The pooled-evidence winner per load (``xcalibur.Evidence.bind``): {load: {design, gain, samples}}."""
  bound, _ = game.evidence.bind(min_samples, margin)
  return bound
