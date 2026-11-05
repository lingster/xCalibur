"""
Export without drift: capture exactly what a measured design compiles, and edit submission files structurally.

A design measured inside a search harness is not the same object as the file you ship. Two tools close the gap:
  * ``Capture`` stands in for the compiler entry point (e.g. torch's ``load_inline``) and records every build's
    inputs instead of compiling, so exported source units and build options can be compared to those measured;
  * ``replace_assignment`` swaps one module-level value through the AST and preserves the surrounding bytes.
"""

import ast
import copy
import json


class Capture:
  """
  Record builds instead of compiling them.

  Examples:
    cap = Capture(); module.load_inline = cap; module.prepare(weights)
    cap.builds[0]["flags"] == ["-O3", "-DTILE=64"]
    Capture.same(cap_a, cap_b) == True
  """

  def __init__(self):
    self.builds = []

  def __call__(self, **kw):
    # Keep source units, flag order, binding declarations and every other option.
    # In particular, repeated builds with the same CUDA flags need not be identical.
    self.builds.append(copy.deepcopy({
      "flags": kw.get("extra_cuda_cflags"), "sources": kw.get("cuda_sources"),
      "cpp": kw.get("cpp_sources"), "functions": kw.get("functions"),
      "options": {k: v for k, v in kw.items()
                  if k not in {"extra_cuda_cflags", "cuda_sources", "cpp_sources", "functions"}},
    }))
    return _Stub()

  def keyed(self):
    """Full build records grouped by content, retaining repeated calls."""
    groups = {}
    for build in self.builds:
      key = json.dumps(build, sort_keys=True, ensure_ascii=False, allow_nan=False)
      groups.setdefault(key, []).append(build)
    return groups

  @staticmethod
  def same(a, b) -> bool:
    """Compare build multisets, including options and multiplicity, ignoring call order.

    This checks captured inputs, not compiler versions, binaries or runtime behavior.
    Options must be JSON-serializable, as with ordinary load_inline arguments.
    """
    ka, kb = a.keyed(), b.keyed()
    return ka.keys() == kb.keys() and all(len(ka[k]) == len(kb[k]) for k in ka)


class _Stub:
  def __getattr__(self, name):
    return lambda *a, **k: 0


def replace_assignment(source: str, name: str, value) -> str:
  """
  Replace one unambiguous module-level assignment's value, preserving all surrounding bytes.
  Annotated assignments are supported; chained and repeated assignments are rejected.

  Examples:
    replace_assignment("A = 1\\nB = 'x'\\n", "B", "y") == "A = 1\\nB = 'y'\\n"
  """
  tree = ast.parse(source)
  matches = []
  for node in tree.body:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    if any(isinstance(t, ast.Name) and t.id == name for t in targets):
      if len(targets) != 1 or node.value is None:
        raise ValueError(f"{name!r} must have a single assignment with a value")
      matches.append(node.value)
  if len(matches) != 1:
    raise ValueError(f"expected one module-level assignment to {name!r}, found {len(matches)}")
  node = matches[0]
  # AST columns are UTF-8 byte offsets, not Python string indices.
  encoded = source.encode("utf-8")
  lines = encoded.splitlines(keepends=True)
  a = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
  b = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
  out = (encoded[:a] + repr(value).encode("utf-8") + encoded[b:]).decode("utf-8")
  compile(out, "<replaced>", "exec")
  return out
