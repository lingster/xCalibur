"""A callback-based search loop against a fixed reference.

Compile and screen a bounded frontier, then confirm promising designs alongside
the reference. Once a load has twelve observations a surrogate ranks its frontier;
before that, candidates are explored in frontier order. Confirmation contributes
at most one observation per (load, declared build identity) per round.

Callbacks own construction, numerical qualification and actual hardware timing.
The returned bindings use Evidence's repeat-and-margin heuristic; Game neither
proves statistical significance nor automatically replaces the reference.
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field

from .stats import Evidence
from .surrogate import Surrogate


@dataclass
class Game:
  """
  Callbacks:
    frontier(history)          -> list of designs (dicts) not yet measured
    featurize(design, compiled)-> feature vector; compiled is the compile record or None
    compile(designs)           -> one compile record ({"ok": bool, ...}) per design, in order
    measure(designs, grade)    -> one {load: positive finite speedup} per design, covering every configured load;
                                  all ratios use the same external baseline, grade in {"screen", "confirm"};
                                  at "confirm" the first design is the incumbent, timed in the same session
    reach                      -> a validated ReachMap (reaches / declared build identity per load)

  The incumbent is frozen for the life of this game. Start fresh evidence and
  history when changing the reference, hardware, compiler or timing protocol.

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
  _reference_key: str = field(init=False, repr=False)

  def __post_init__(self):
    self.incumbent = copy.deepcopy(self.incumbent)
    self._reference_key = self._key(self.incumbent)
    if not self.loads or len(set(self.loads)) != len(self.loads):
      raise ValueError("loads must be nonempty and unique")
    for name in ("per_load", "compile_n", "screen_n"):
      value = getattr(self, name)
      if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    self.evidence.bind(self.min_samples, self.margin)

  def _key(self, d):
    return repr(sorted(d.items()))

  def _ref(self):
    inc = [h for h in self.history if self._key(h["design"]) == self._key(self.incumbent)]
    if any(not any(L in h["loads"] for h in inc) for L in self.loads):
      loads = self._measure([self.incumbent], "screen")[0]
      self._record(self.incumbent, loads, 1.0)
      inc.append(self.history[-1])
    return {L: (math.fsum(math.log(h["loads"][L]) for h in inc if L in h["loads"]) /
                sum(1 for h in inc if L in h["loads"])) for L in self.loads}

  def _batch(self, callback, designs, *args):
    rows = list(callback(copy.deepcopy(designs), *args))
    if len(rows) != len(designs):
      raise ValueError(f"callback returned {len(rows)} records for {len(designs)} designs")
    return rows

  def _check_loads(self, loads):
    if not isinstance(loads, dict):
      raise ValueError("measurement must be a mapping from load to speedup")
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in loads.values()):
      raise ValueError("measurement ratios must be finite and positive")

  def _measure(self, designs, grade):
    rows = self._batch(self.measure, designs, grade)
    for loads in rows:
      self._check_loads(loads)
      if not set(self.loads).issubset(loads):
        raise ValueError("measurement must cover every configured load")
    return [{L: float(v) for L, v in loads.items()} for loads in rows]

  def _record(self, design, loads, weight):
    self.history.append(copy.deepcopy({"design": design, "loads": loads, "weight": weight}))

  def round(self) -> dict:
    if self._key(self.incumbent) != self._reference_key:
      raise ValueError("incumbent reference changed; start a new Game with fresh history and evidence")
    for row in self.history:
      self._check_loads(row["loads"])
    ref = self._ref()
    cands = list(self.frontier(self.history))
    pick = {}
    for L in self.loads:
      cl = [c for c in cands if self.reach.reaches(L, c, self.incumbent)
            and self.compiled.get(self._key(c), {}).get("ok") is not False]
      if not cl:
        continue
      rows = [h for h in self.history if L in h["loads"]]
      sur = None
      chosen = cl[:self.per_load]
      if len(rows) >= 12:
        sur = Surrogate().fit([self.featurize(h["design"], self.compiled.get(self._key(h["design"]))) for h in rows],
                            [math.log(h["loads"][L]) for h in rows], [h.get("weight", 1.0) for h in rows])
        chosen = [cl[i] for i, _, _ in sur.top([self.featurize(c, None) for c in cl], self.per_load, self.kappa, ref[L])]
      for design in chosen:
        entry = pick.setdefault(self._key(design), {"design": design, "rankers": {}})
        entry["rankers"][L] = sur
    stage1 = list(pick.values())[: self.compile_n]
    records = self._batch(self.compile, [e["design"] for e in stage1]) if stage1 else []
    if any(not isinstance(rec, dict) or not isinstance(rec.get("ok"), bool) for rec in records):
      raise ValueError("compile must return a record with a boolean 'ok' for every design")
    ok = []
    for entry, rec in zip(stage1, records):
      d = entry["design"]
      self.compiled[self._key(d)] = copy.deepcopy(rec)
      if rec.get("ok"):
        scores = [float(sur.ucb([self.featurize(d, rec)], self.kappa, ref[L])[0][0]) if sur else 0.
                  for L, sur in entry["rankers"].items()]
        ok.append((max(scores), entry))
    ok.sort(key=lambda x: -x[0])
    screen = [entry for _, entry in ok[: self.screen_n]]
    results = self._measure([e["design"] for e in screen], "screen") if screen else []
    for entry, loads in zip(screen, results):
      self._record(entry["design"], loads, 1.0)
    _, again = self.evidence.bind(self.min_samples, self.margin)
    confirm = {}
    for _, _, d in again:
      if d is not None and len(confirm) < 4:
        confirm.setdefault(self._key(d), d)
    for entry, loads in zip(screen, results):
      d = entry["design"]
      if any(math.log(loads[L]) > ref[L] + self.nominate for L in entry["rankers"]):
        confirm.setdefault(self._key(d), d)
    if confirm:
      rows = self._measure([self.incumbent] + list(confirm.values()), "confirm")
      ctl = rows[0]
      observations = {}
      for d, loads in zip(confirm.values(), rows[1:]):
        for L in self.loads:
          if self.reach.reaches(L, d, self.incumbent):
            gain = loads[L] / ctl[L] - 1
            if not math.isfinite(gain) or gain <= -1:
              raise ValueError("confirmation ratio must be finite and positive")
            # Aliases sharing a build in this round are correlated observations.
            # Keep the first, never the most favorable, for that load and identity.
            observations.setdefault((L, self.reach.identity(L, d)), (gain, d))
      self._record(self.incumbent, ctl, 6.0)
      for d, loads in zip(confirm.values(), rows[1:]):
        self._record(d, loads, 6.0)
      for (L, identity), (gain, d) in observations.items():
        self.evidence.add(L, identity, gain, d)
    bound, _ = self.evidence.bind(self.min_samples, self.margin)
    return {"frontier": len(cands), "compiled": sum(1 for r in records if r.get("ok")), "screened": len(screen),
            "confirmed": len(confirm), "bound": {L: round(b["gain"] * 100, 2) for L, b in bound.items()}}
