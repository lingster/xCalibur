"""Reach: does a design change the build that serves a given problem size?

A kernel that ships several compiled variants ("builds") and picks one by problem size M has a simple exact
question with large consequences: an edit to a field that does not reach the build serving load L cannot change
L's time, so a measured difference there is noise. Declaring the map once lets a search

* reject no-op candidates before measuring (``reaches``), and
* pool measurements of designs that are identical for a load (``identity``), so binding uses all of them.

The map is derived from the lowering (which field is read when compiling which build) and can be cross-checked
against compiled SASS: designs with the same identity for a load must produce identical code for its build.

  >>> rm = ReachMap(loads={"m1": 1, "m32": 32}, cutoff_field="cutoff",
  ...               fields={"large": ("tile", "stages"), "small": ("unroll",), "all": ("decoder",)})
  >>> base = {"cutoff": 17, "tile": 64, "stages": 3, "unroll": 2, "decoder": "a"}
  >>> rm.reaches("m1", dict(base, tile=128), base), rm.reaches("m32", dict(base, tile=128), base)
  (False, True)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ReachMap:
  loads: dict                      # load name -> problem size M
  cutoff_field: str                # sizes M >= design[cutoff_field] run the "large" build
  fields: dict = field(default_factory=dict)   # "large" / "small" / per-load name / "all" -> fields read by that build

  def build_of(self, load, design):
    return "large" if self.loads[load] >= int(design[self.cutoff_field]) else "small"

  def fields_for(self, load, design):
    b = self.build_of(load, design)
    return tuple(self.fields.get(b, ())) + tuple(self.fields.get(load, ())) + tuple(self.fields.get("all", ()))

  def reaches(self, load, design, base) -> bool:
    """True when design and base differ in anything that reaches load's build (including its routing)."""
    if self.build_of(load, design) != self.build_of(load, base):
      return True
    return any(design.get(f) != base.get(f) for f in self.fields_for(load, design))

  def identity(self, load, design) -> str:
    """Declared lowering signature, assuming fields cover every build-affecting choice.

    Validate that assumption against captured builds or compiled code with a fixed
    compiler context before pooling measurements. This function does not inspect code.
    """
    b = self.build_of(load, design)
    return json.dumps({"build": b, **{f: design.get(f) for f in self.fields_for(load, design)}}, sort_keys=True, default=str)
