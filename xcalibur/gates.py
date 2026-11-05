"""
Gates: the checks that stand between a candidate and anything public.

  * ``rules_gate``   the live rules version must match the version you read in full;
  * ``lint``         the shipped file contains no timing-path patterns a judge would reject or that only help a
                     benchmark (host syncs, side streams, graph capture in the timed call, measurement scaffolding);
  * ``Card``         the one-screen summary a person approves before a scarce run: the parent, the measured loss it
                     targets, an estimate only from a measured analog, the budget, the risk, the noise band, what the
                     result would teach, and one falsifier.
"""

import re
from dataclasses import dataclass

LINT = (
  (r"torch\.cuda\.synchronize\(", "host synchronization"),
  (r"\.item\(\)|\.cpu\(\)|\.tolist\(\)", "device-to-host read"),
  (r"torch\.cuda\.Stream\(|cudaStreamCreate", "side stream"),
  (r"CUDAGraph\(|cudaGraphLaunch|torch\.cuda\.graph\(", "graph capture or replay in the shipped file"),
  (r"time\.perf_counter|time\.time\(|cudaEventElapsedTime", "timing code"),
  (r"#\s*(probe|rider|flush)\b", "measurement scaffolding"),
)


def rules_gate(live_version: str, read_version: str) -> None:
  """
  Examples:
    rules_gate("2026-10-22.1", "2026-10-22.1")  # passes
    rules_gate("2026-11-01.1", "2026-10-22.1")  # raises: re-read the rules in full
  """
  if live_version != read_version:
    raise RuntimeError(f"rules changed ({read_version} -> {live_version}): read the new version in full before submitting")


def lint(source: str, patterns=LINT, allow_in=("def prepare",)):
  """
  Findings as (line number, reason, text). Lines inside functions named in ``allow_in`` are skipped
  (setup that runs once, untimed).

  Examples:
    lint("def kernel_fn(x):\\n  torch.cuda.synchronize()\\n") == [(2, "host synchronization", "torch.cuda.synchronize()")]
  """
  out, skip_indent = [], None
  for i, line in enumerate(source.splitlines(), 1):
    indent = len(line) - len(line.lstrip())
    if skip_indent is not None and line.strip() and indent <= skip_indent:
      skip_indent = None
    if any(line.lstrip().startswith(a) for a in allow_in):
      skip_indent = indent
      continue
    if skip_indent is not None:
      continue
    for pat, why in patterns:
      if re.search(pat, line):
        out.append((i, why, line.strip()[:110]))
  return out


@dataclass
class Card:
  parent: str
  target: str
  estimate: str = "unknown"          # only from a measured analog
  budget: str = ""
  risk: str = ""
  noise: str = ""
  teaches: str = ""
  falsifier: str = ""

  def render(self) -> str:
    rows = [("parent", self.parent), ("target", self.target), ("estimate", self.estimate), ("budget", self.budget),
            ("risk", self.risk), ("noise", self.noise), ("teaches", self.teaches), ("falsifier", self.falsifier)]
    w = max(len(k) for k, _ in rows)
    return "\n".join(f"{k:<{w}}  {v}" for k, v in rows)
