"""
Export without drift: capture exactly what a measured design compiles, and edit submission files structurally.

A design measured inside a search harness is not the same object as the file you ship. Two tools close the gap:
  * ``Capture`` stands in for the compiler entry point (e.g. torch's ``load_inline``) and records every build's
    sources and flags instead of compiling, so a rebuilt file can be checked byte-identical to the measured one;
  * ``replace_assignment`` swaps one module-level string constant through the AST and proves nothing else moved.
"""

import ast
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
    srcs = kw.get("cuda_sources")
    self.builds.append({"flags": list(kw.get("extra_cuda_cflags") or []),
                        "sources": srcs if isinstance(srcs, str) else "\n".join(srcs or []),
                        "cpp": kw.get("cpp_sources") or "", "functions": list(kw.get("functions") or [])})
    return _Stub()

  def keyed(self):
    return {json.dumps(sorted(b["flags"])): b for b in self.builds}

  @staticmethod
  def same(a, b) -> bool:
    """True when both captures hold the same builds with byte-identical sources."""
    ka, kb = a.keyed(), b.keyed()
    return ka.keys() == kb.keys() and all(ka[k]["sources"] == kb[k]["sources"] and ka[k]["cpp"] == kb[k]["cpp"] for k in ka)


class _Stub:
  def __getattr__(self, name):
    return lambda *a, **k: 0


def replace_assignment(source: str, name: str, value) -> str:
  """
  Replace the value of the module-level assignment ``name = ...`` and check that every other line is unchanged.

  Examples:
    replace_assignment("A = 1\\nB = 'x'\\n", "B", "y") == "A = 1\\nB = 'y'\\n"
  """
  tree = ast.parse(source)
  node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
  lines = source.splitlines(keepends=True)
  a = sum(map(len, lines[:node.value.lineno - 1])) + node.value.col_offset
  b = sum(map(len, lines[:node.value.end_lineno - 1])) + node.value.end_col_offset
  out = source[:a] + repr(value) + source[b:]
  keep = lambda s: [l for i, l in enumerate(s.splitlines()) if not (node.lineno - 1 <= i <= node.end_lineno - 1)]
  assert keep(source)[: node.lineno - 1] == keep(out)[: node.lineno - 1], "lines before the assignment changed"
  assert keep(source)[node.lineno - 1:] == [l for l in out.splitlines()][node.lineno:], "lines after the assignment changed"
  compile(out, "<replaced>", "exec")
  return out
