"""
xCalibur: exact-first, measurement-driven GPU kernel co-design.

  exact layer    gf2 (decoder bit maps, CuTe F2 layouts), warpspeed (CTA plan accounting), reach (what touches which build)
  measurement    bench (judge-faithful paired timing), stats (intervals, pooling by build identity, binding)
  proposers      surrogate (LightGBM bootstrap UCB), game (the measured search loop), planner (LM planner + refiner)
  memory         graph (the design graph, compact listing, retrieval)
  shipping       capture (byte-identical builds, structural edits), gates (rules, lint, run cards)
"""

from .gf2 import *
from .warpspeed import *
from .reach import *
from .stats import *
from .bench import *
from .surrogate import *
from .game import *
from .graph import *
from .planner import *
from .capture import *
from .gates import *

# Public API: every name re-exported above, minus submodules and incidental names pulled in by the star imports.
import types as _types
_INCIDENTAL = {"annotations", "dataclass", "field", "Path", "Any"}
__all__ = sorted(
  _n for _n, _v in globals().items()
  if not _n.startswith("_")
  and _n not in _INCIDENTAL
  and not isinstance(_v, _types.ModuleType)
)
del _types, _INCIDENTAL
