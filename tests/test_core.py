"""CPU tests for the exact layer, statistics, the design graph and the shipping tools (no GPU needed)."""

import xcalibur as xc


def test_gf2_packing_proof():
  P = tuple(1 << i for i in range(32))
  D = xc.compose(xc.rotation(5), P)
  T = xc.packing_transform(D, P)
  for w in (0, 1, 0x12345678, 0xFFFFFFFF, 0x0F0F0F0F):
    assert xc.apply(D, xc.apply(T, w)) == xc.apply(P, w)
  assert xc.rank(T) == 32
  assert xc.equal(xc.compose(xc.inverse(D), D), P)


def test_gf2_join():
  e = [1 << i for i in range(4)]
  atoms = [("a", (e[0], e[1])), ("b", (e[2], e[3])), ("c", (e[0] ^ e[2], e[1])), ("d", (e[2] ^ e[3],))]
  joins = xc.join_all(atoms, 4)
  assert ("a", "b") in joins and all(xc.rank(sum((dict(atoms)[n] for n in j), ())) == 4 for j in joins)


def test_warpspeed_plan_and_counterexample():
  plan = xc.plan_of([("mma", 16), ("w_load", 1), ("x_load", 1)],
                    [("w_ring", 3, 36864, [("w_load", 36864), ("mma", None)]),
                     ("x_ring", 3, 32768, [("x_load", 32768), ("mma", None)])])
  r = xc.warpspeed_check(**plan)
  assert r["pass"] and r["registers"]["launch_cap"] == 96 and r["smem"]["total_bytes"] == 208992 and r["mbarriers"] == 12
  plan["resources"][0]["stages"] = 6
  bad = xc.warpspeed_check(**plan)
  assert not bad["pass"] and bad["counterexample"]["code"] == "smem_cta"
  assert xc.execute({"squads": [{"name": "a", "warps": 4}], "resources": []})["ok"]


def test_reach_and_identity():
  rm = xc.ReachMap(loads={"m1": 1, "m32": 32}, cutoff_field="cutoff",
                   fields={"large": ("tile",), "small": ("unroll",), "all": ("decoder",)})
  base = {"cutoff": 17, "tile": 64, "unroll": 2, "decoder": "a"}
  assert not xc.ReachMap.reaches(rm, "m1", dict(base, tile=128), base)
  assert rm.reaches("m32", dict(base, tile=128), base)
  assert rm.reaches("m1", dict(base, cutoff=1), base)                  # re-routed to the large build
  assert rm.identity("m1", dict(base, tile=128)) == rm.identity("m1", base)


def test_evidence_binding():
  ev = xc.Evidence()
  for g in (0.009, 0.006):
    ev.add("m16", "dec=x", g, {"dec": "x"})
  ev.add("m16", "dec=y", 0.02, {"dec": "y"})
  ev.add("m32", "tile=128", 0.012, {"tile": 128})
  ev.add("m32", "tile=128", -0.001, {"tile": 128})
  bound, again = ev.bind()
  assert bound["m16"]["identity"] == "dec=x" and bound["m16"]["samples"] == 2
  assert "m32" not in bound
  assert ("m16", "dec=y", {"dec": "y"}) in again


def test_graph_effects_and_listing():
  g = xc.DesignGraph(loads=("m1", "m16"))
  g.add(xc.Node("n1", "design", {"tile": 64, "dec": "a"}, {"m1": 1.00, "m16": 1.00}, "confirm 12x2"))
  g.add(xc.Node("n2", "design", {"tile": 128, "dec": "a"}, {"m1": 1.00, "m16": 1.02}, "screen 4x1"))
  eff = g.effects()
  assert eff[0]["field"] == "tile" and abs(eff[0]["median"]["m16"] - 1.02) < 1e-9 or abs(eff[0]["median"]["m16"] - 1 / 1.02) < 1e-9
  assert "n2 design tile=128,dec=a [+0.0/+2.0] screen 4x1" in g.listing()


def test_capture_and_replace():
  a, b = xc.Capture(), xc.Capture()
  for cap in (a, b):
    cap(cuda_sources="k1", extra_cuda_cflags=["-O3", "-DX=1"])
  assert xc.Capture.same(a, b)
  src = "A = 1\nB = 'x'\nC = [1, 2]\n"
  assert xc.replace_assignment(src, "B", "y") == "A = 1\nB = 'y'\nC = [1, 2]\n"


def test_gates():
  assert xc.lint("def kernel_fn(x):\n  torch.cuda.synchronize()\n")[0][1] == "host synchronization"
  assert xc.lint("def prepare(w):\n  torch.cuda.synchronize()\n  return w\n") == []
  xc.rules_gate("v1", "v1")
  assert "falsifier" in xc.Card("p", "t", falsifier="f").render()
