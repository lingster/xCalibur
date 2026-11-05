"""
The planner loop: a language-model planner proposes, an exact layer checks, a refiner repairs.

  context  -> the whole design graph (compact), retrieved nodes in full, the incumbent for this problem, the fields
              that reach this problem's build, and this planner's own past plans with their verdicts and results
  planner  -> one plan per problem, as a typed tool call
  check    -> exact diagnostics: grammar contracts, legal decoders, WarpSpeed accounting, reach (no-op) checks
  refiner  -> the plan repaired against those diagnostics (re-checked; a plan that still fails is dropped)

The models only propose. The plan that comes out is measured like any other candidate. Sampling settings live in a
control file that is re-read every round, so exploration can be turned up or down without a restart.

Works against any OpenAI-compatible endpoint (for example vLLM serving Gemma 4 with planner and refiner LoRAs).
"""

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def chat(base_url, model, messages, tool, temperature=0.2, max_tokens=6000, timeout=600, **extra):
  """
  One forced tool call; returns (arguments dict, seconds). Recovers a JSON object from the content if the server
  returns the call as text.

  Examples:
    args, secs = chat("http://127.0.0.1:8000", "planner", msgs, tool_schema)
  """
  body = {"model": model, "messages": messages, "tools": [tool], "temperature": temperature, "top_p": 0.95,
          "tool_choice": {"type": "function", "function": {"name": tool["function"]["name"]}},
          "max_tokens": max_tokens, **extra}
  req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
  t = time.perf_counter()
  with urllib.request.urlopen(req, timeout=timeout) as r:
    msg = json.load(r)["choices"][0]["message"]
  calls = msg.get("tool_calls") or []
  raw = calls[0]["function"]["arguments"] if calls else None
  if raw is None:
    m = re.search(r"\{.*\}", msg.get("content") or "", re.S)
    raw = m.group(0) if m else "{}"
  return (json.loads(raw) if isinstance(raw, str) else raw), round(time.perf_counter() - t, 1)


def tool(name, description, parameters):
  return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


@dataclass
class PlannerLoop:
  """
  Callbacks:
    check(problem, plan)        -> list of exact diagnostics (empty = passes)
    retrieve(problem, k)        -> list of {"id", "text"} for the retrieval block
  Fields:
    plan_schema                 -> JSON schema of one plan (the tool's parameters)
    reach_note(problem)         -> text: which fields reach this problem's build

  Examples:
    loop = PlannerLoop(url, "planner", "refiner", schema, check, retrieve, reach_note, ledger)
    plans = loop.round(problems, graph_listing, incumbents)
  """
  base_url: str
  planner_model: str
  refiner_model: str
  plan_schema: dict
  check: object
  retrieve: object
  reach_note: object
  ledger: str = ""
  control: Path | None = None
  memory: dict = field(default_factory=dict)       # problem -> [past plan records]

  def _temperature(self):
    try:
      return float(json.loads(Path(self.control).read_text()).get("planner_temperature", 0.2))
    except Exception:
      return 0.2

  def plan(self, problem, listing, incumbent):
    ctx = ("DESIGN GRAPH\n" + listing + "\n\n" + json.dumps({
      "ledger": self.ledger, "problem": problem, "incumbent": incumbent,
      "reaches_this_build": self.reach_note(problem),
      "your_past_plans_for_this_problem": self.memory.get(problem, [])[-6:],
      "retrieved": self.retrieve(problem, 14),
      "task": f"One plan that beats the incumbent on {problem}. It must change something that reaches this "
              "problem's build; anything else is rejected before measuring."}))
    sys_msg = "You plan GPU kernel designs. Propose one concrete, legal plan via the tool. Cite graph node ids."
    p, secs = chat(self.base_url, self.planner_model, [{"role": "system", "content": sys_msg},
                   {"role": "user", "content": ctx}], tool("propose_plan", "Propose one plan.", self.plan_schema),
                   temperature=self._temperature())
    return p, secs

  def refine(self, problem, plan, diagnostics):
    ctx = json.dumps({"ledger": self.ledger, "problem": problem, "plan": plan,
                      "exact_diagnostics": diagnostics or ["all exact checks pass"],
                      "task": "Return the plan repaired so every exact check passes, keeping its idea."})
    q, _ = chat(self.base_url, self.refiner_model, [{"role": "system", "content": "You repair plans against exact checker feedback."},
                {"role": "user", "content": ctx}], tool("refine_plan", "Return one repaired plan.", self.plan_schema), temperature=0.2)
    return q

  def round(self, problems, listing, incumbents):
    """One plan per problem; returns {problem: plan or None} with every verdict kept in memory."""
    out = {}
    for prob in problems:
      p, secs = self.plan(prob, listing, incumbents[prob])
      d = self.check(prob, p)
      q = self.refine(prob, p, d) if d else p
      d2 = self.check(prob, q)
      self.memory.setdefault(prob, []).append({"planned": p, "verdict": d or "pass", "refined": q if d else None,
                                               "after_refiner": d2 or "pass", "seconds": secs})
      out[prob] = q if not d2 else None
    return out

  def record_result(self, problem, plan, result):
    """Attach a measured result to the latest memory entry, so the next round's planner sees it."""
    for m in reversed(self.memory.get(problem, [])):
      if m.get("refined") == plan or m.get("planned") == plan:
        m["measured"] = result
        return
