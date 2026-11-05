"""
The measured game: a search loop in which models rank and hardware decides.

One round:
  1. fit a surrogate per load on every measured design (features -> log speedup over the incumbent);
  2. rank the frontier of new designs by UCB, per load, keeping only designs that reach that load's build;
  3. compile the top candidates (cheap, parallel, exact: a compile error or identical code is decided here);
  4. re-rank with the compiled features and screen the best on the GPU;
  5. re-time screen leaders, and earlier single-sample leaders, next to the incumbent at confirm grade;
  6. pool confirm samples by build identity and bind (>= 2 samples, mean above a margin, none negative).

Everything problem-specific is a callback, so the same loop runs over compile-time knobs, a design grammar, or
any other finite space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .stats import Evidence
from .surrogate import Surrogate


@dataclass
class Game:
  """
  Callbacks:
    frontier(history)          -> list of designs (dicts) not yet measured
    featurize(design, compiled)-> feature vector; compiled is the compile record or None
    compile(designs)           -> list of compile records ({"ok": bool, ...}), in order
    measure(designs, grade)    -> list of {load: speedup vs the reference}, grade in {"screen", "confirm"};
                                  at "confirm" the first design is the incumbent, timed in the same session
    reach                      -> a ReachMap (reaches / identity per load)

  Examples:
    g = Game(loads, reach, incumbent, frontier, featurize, compile, measure)
    for _ in range(rounds): report = g.round()
  """
  loads: tuple
  reach: object
  incumbent: dict
  frontier: object
  featurize: object
  compile: object
  measure: object
  per_load: int = 8
  compile_n: int = 48
  screen_n: int = 24
  kappa: float = 1.0
  nominate: float = 0.006
  margin: float = 0.005
  min_samples: int = 2
  history: list = field(default_factory=list)       # [{"design", "loads": {load: speedup}, "weight"}]
  evidence: Evidence = field(default_factory=Evidence)
  compiled: dict = field(default_factory=dict)

  def _key(self, d):
    return repr(sorted(d.items()))

  def _ref(self):
    inc = [h for h in self.history if self._key(h["design"]) == self._key(self.incumbent)]
    return {L: (sum(math.log(h["loads"][L]) for h in inc if L in h["loads"]) /
                max(1, sum(1 for h in inc if L in h["loads"]))) for L in self.loads}

  def round(self) -> dict:
    ref = self._ref()
    cands = self.frontier(self.history)
    pick = {}
    for L in self.loads:
      rows = [h for h in self.history if L in h["loads"]]
      if len(rows) < 12:
        continue
      sur = Surrogate().fit([self.featurize(h["design"], self.compiled.get(self._key(h["design"]))) for h in rows],
                            [math.log(h["loads"][L]) for h in rows], [h.get("weight", 1.0) for h in rows])
      cl = [c for c in cands if self.reach.reaches(L, c, self.incumbent)]
      if not cl:
        continue
      for i, mu, sd in sur.top([self.featurize(c, None) for c in cl], self.per_load, self.kappa, ref[L]):
        pick.setdefault(self._key(cl[i]), (cl[i], L, sur))
    stage1 = list(pick.values())[: self.compile_n]
    records = self.compile([d for d, _, _ in stage1])
    ok = []
    for (d, L, sur), rec in zip(stage1, records):
      self.compiled[self._key(d)] = rec
      if rec.get("ok"):
        score, _, _ = sur.ucb([self.featurize(d, rec)], self.kappa, ref[L])
        ok.append((float(score[0]), d, L))
    ok.sort(key=lambda x: -x[0])
    screen = [(d, L) for _, d, L in ok[: self.screen_n]]
    results = self.measure([d for d, _ in screen], "screen") if screen else []
    for (d, L), loads in zip(screen, results):
      self.history.append({"design": d, "loads": loads, "weight": 1.0})
    _, again = self.evidence.bind(self.min_samples, self.margin)
    confirm = [(d, L) for L, _, d in again][:4]
    for (d, L), loads in zip(screen, results):
      if L in loads and math.log(loads[L]) > ref[L] + self.nominate and (d, L) not in confirm:
        confirm.append((d, L))
    if confirm:
      rows = self.measure([self.incumbent] + [d for d, _ in confirm], "confirm")
      ctl = rows[0]
      self.history.append({"design": self.incumbent, "loads": ctl, "weight": 6.0})
      for (d, _), loads in zip(confirm, rows[1:]):
        self.history.append({"design": d, "loads": loads, "weight": 6.0})
        for L in self.loads:
          if L in loads and L in ctl and self.reach.reaches(L, d, self.incumbent):
            self.evidence.add(L, self.reach.identity(L, d), loads[L] / ctl[L] - 1, d)
    bound, _ = self.evidence.bind(self.min_samples, self.margin)
    return {"frontier": len(cands), "compiled": sum(1 for r in records if r.get("ok")), "screened": len(screen),
            "confirmed": len(confirm), "bound": {L: round(b["gain"] * 100, 2) for L, b in bound.items()}}
