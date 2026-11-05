"""
The design graph: every measured design as a node, one-field differences as edges, and retrieval over it.

A planner (person or model) that can see everything that was tried, with its measured result per load, builds on
evidence instead of repeating it. Two things make the graph usable in a model's context:
  * a compact listing (one line per node, per-load results as signed percents), so the whole graph fits;
  * evidence weights on every row (how it was measured), so a lucky screen is not read as a result.
Retrieval adds the nearest nodes in full (code included) using any embedding function; EmbeddingGemma with a
code-retrieval prompt for queries worked well for us.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field


def _pct(r):
  return "." if r is None else f"{(r - 1) * 100:+.1f}"


@dataclass
class Node:
  id: str
  kind: str                        # e.g. "design", "code_move", "submission", "code"
  fields: dict = field(default_factory=dict)
  loads: dict = field(default_factory=dict)       # load -> speedup vs the reference
  evidence: str = ""               # how it was measured, e.g. "screen 4x1", "confirm 12x2"
  text: str = ""                   # long form for retrieval (rationale, code)


@dataclass
class DesignGraph:
  loads: tuple
  nodes: list = field(default_factory=list)

  def add(self, node: Node) -> Node:
    self.nodes.append(node)
    return node

  def one_edit(self):
    """
    Pairs of nodes that differ in exactly one field, with the per-load ratio b / a.

    Examples:
      g.one_edit()[0] == ("n3", "n7", "tile", 64, 128, {"m16": 1.012})
    """
    out = []
    ds = [n for n in self.nodes if n.fields]
    for i, a in enumerate(ds):
      for b in ds[i + 1:]:
        diff = [k for k in a.fields if a.fields.get(k) != b.fields.get(k)]
        if len(diff) == 1 and set(a.fields) == set(b.fields):
          k = diff[0]
          r = {L: b.loads[L] / a.loads[L] for L in self.loads if a.loads.get(L) and b.loads.get(L)}
          out.append((a.id, b.id, k, a.fields[k], b.fields[k], r))
    return out

  def effects(self):
    """Median measured effect of each field change (field, from, to) over its one-edit pairs."""
    agg = {}
    for _, _, k, x, y, r in self.one_edit():
      for f, t, rr in ((x, y, r), (y, x, {L: 1 / v for L, v in r.items()})):
        agg.setdefault((k, json.dumps(f), json.dumps(t)), []).append(rr)
    out = []
    for (k, f, t), rs in agg.items():
      med = {}
      for L in self.loads:
        xs = sorted(r[L] for r in rs if L in r)
        if xs:
          med[L] = xs[len(xs) // 2]
      out.append({"field": k, "from": json.loads(f), "to": json.loads(t), "pairs": len(rs), "median": med})
    return sorted(out, key=lambda e: -e["pairs"])

  def listing(self) -> str:
    """The whole graph as compact text: one line per node, then the measured one-field effects."""
    head = "/".join(self.loads)
    lines = [f"# nodes [{len(self.nodes)}] (per-load % vs reference, order {head})"]
    for n in self.nodes:
      f = ",".join(f"{k}={v}" for k, v in n.fields.items())
      lines.append(f"{n.id} {n.kind} {f} [{'/'.join(_pct(n.loads.get(L)) for L in self.loads)}] {n.evidence}")
    eff = self.effects()
    lines.append(f"# one-field effects [{len(eff)}]")
    for e in eff:
      lines.append(f"{e['field']} {e['from']} -> {e['to']} (n={e['pairs']}): " +
                   " ".join(f"{L} {_pct(v)}" for L, v in e["median"].items()))
    return "\n".join(lines)


class Retriever:
  """
  Nearest nodes by embedding similarity. ``embed(texts, query=False) -> list of unit vectors``.

  Examples:
    r = Retriever(graph.nodes, embed); r.query("faster m16 decode", k=8)
  """

  def __init__(self, nodes, embed):
    self.nodes, self.embed = list(nodes), embed
    self.vecs = embed([f"title: {n.id} | text: {n.text or n.id}" for n in self.nodes], query=False)

  def query(self, q: str, k: int = 8):
    qv = self.embed([q], query=True)[0]
    sims = [sum(a * b for a, b in zip(v, qv)) for v in self.vecs]
    order = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
    return [(self.nodes[i], sims[i]) for i in order]


def embedding_gemma(model: str = "google/embeddinggemma-300m", device: str = "cpu"):
  """An ``embed`` function backed by EmbeddingGemma through sentence-transformers (documents vs code-retrieval queries)."""
  from sentence_transformers import SentenceTransformer       # noqa: PLC0415
  m = SentenceTransformer(model, device=device)

  def embed(texts, query=False):
    kw = {"prompt_name": "CodeRetrieval"} if query and "CodeRetrieval" in (m.prompts or {}) else {}
    return [list(map(float, v)) for v in m.encode(list(texts), normalize_embeddings=True, **kw)]
  return embed
