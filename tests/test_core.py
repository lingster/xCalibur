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


def test_arch_tables():
  assert set(xc.ARCHS) >= {"h200", "rtx3090", "a6000", "rtx6000ada"}
  h, g, a, ada = (xc.ARCHS[k] for k in ("h200", "rtx3090", "a6000", "rtx6000ada"))
  assert h.wgmma and h.tma and h.setmaxnreg and h.max_threads_sm == 2048 and h.fp8_tflops_dense == 1979
  for arch in (g, a, ada):
    assert not (arch.wgmma or arch.tma or arch.setmaxnreg)
    assert (arch.max_threads_sm, arch.regs_per_sm, arch.smem_per_sm, arch.smem_max_cta) == (1536, 65536, 102400, 101376)
  assert (g.sms, a.sms, ada.sms) == (82, 84, 142) and (g.max_ctas_sm, ada.max_ctas_sm) == (16, 24)
  assert g.bf16_tflops_dense == 71 and a.bf16_tflops_dense == 154.8      # GeForce: half rate with fp32 accumulate
  assert g.fp8_tflops_dense is None and ada.fp8_tflops_dense == 728.5
  assert xc.launch_register_cap(g, 48) == 40                              # 1,536 threads: 42 registers, floored to 8


def test_warpspeed_pre_hopper_counterexamples():
  moe = xc.plan_of([("mma", 16), ("w_load", 1), ("x_load", 1)],
                   [("w_ring", 3, 36864, [("w_load", 36864), ("mma", None)]),
                    ("x_ring", 3, 32768, [("x_load", 32768), ("mma", None)])])
  first = lambda plan, arch="h200": (xc.warpspeed_check(**plan, arch=arch)["counterexample"] or {}).get("code")
  assert xc.warpspeed_check(**moe, arch="h200") == xc.warpspeed_check(**moe)
  r = xc.warpspeed_check(**moe, arch="rtx3090")
  assert not r["pass"] and {"tma_arch", "smem_cta", "smem_sm"} <= {v["code"] for v in r["violations"]}
  regs = {"squads": [{"name": "mma", "warps": 8, "regs": 232}, {"name": "load", "warps": 4, "regs": 40}],
          "resources": []}
  assert first(regs) is None and first(regs, "rtx6000ada") == "setmaxnreg_arch"
  wg = {"squads": [{"name": "load", "warps": 1}, {"name": "mma", "warps": 4, "wgmma": True}], "resources": []}
  assert first(wg) == "wgmma_alignment"
  wg["squads"].reverse()
  assert first(wg) is None and first(wg, "a6000") == "wgmma_arch"
  big = {"squads": [{"name": "a", "warps": 32}], "resources": [], "ctas_per_sm": 2}
  assert first(big) is None and first(big, "rtx3090") == "sm_threads"
  ring = {"squads": [{"name": "a", "warps": 4}], "resources": [{"name": "r", "stages": 2, "stride_bytes": 32768,
          "phases": [{"owners": ["a"]}, {"owners": ["a"]}]}], "ctas_per_sm": 2}
  assert first(ring) is None and first(ring, "rtx3090") == "smem_sm"
  tiny = {"squads": [{"name": "a", "warps": 1}], "resources": [], "ctas_per_sm": 20}
  assert first(tiny, "rtx6000ada") is None and first(tiny, "rtx3090") == "sm_ctas"
  assert xc.execute({**wg, "arch": "rtx3090"})["result"]["counterexample"]["code"] == "wgmma_arch"
  assert not xc.execute({**wg, "arch": "rtx9999"})["ok"]


def test_bootstrap_default_unchanged_and_clusters():
  import random
  rng = random.Random(1)
  xs = [1 + rng.gauss(0, 0.01) for _ in range(23)]
  assert xc.bootstrap_ci(xs) == (0.9985529769183508, 1.0053799717866103)
  assert xc.bootstrap_ci(xs, level=0.9, n=500, seed=3) == (0.9993525823962731, 1.0032071110099885)
  assert xc.bootstrap_ci(xs, clusters=range(23)) == xc.bootstrap_ci(xs)    # singleton clusters: the iid bootstrap
  # 6 inputs, each timed 8 times: repeats agree to 0.02%, inputs disagree by up to 1.8%. Repeats are not new evidence:
  # resampled one by one they "clear" a 1.002 margin; resampled by input, the interval includes no gain at all.
  means = (1.010, 1.007, 1.005, 1.003, 0.998, 0.992)
  ratios = [m + rng.gauss(0, 0.0002) for m in means for _ in range(8)]
  inputs = [i for i in range(6) for _ in range(8)]
  iid, clustered = xc.bootstrap_ci(ratios), xc.bootstrap_ci(ratios, clusters=inputs)
  assert iid[0] > 1.002 and clustered[0] < 1.0 and clustered[1] - clustered[0] > 2 * (iid[1] - iid[0])
  try:
    xc.bootstrap_ci(ratios, clusters=inputs[:-1])
    assert False
  except ValueError:
    pass


def test_aa_margin():
  assert abs(xc.aa_margin([1.0002, 1.0009]) - 1.0018) < 1e-12               # K1a's A/A CI ends, same process
  assert xc.aa_margin([1.0002, 1.0009], step=0.001) == 1.002
  assert xc.aa_margin([0.9987, 0.9993], step=0.001) == 1.003                # K1b, with reloads
  assert xc.aa_margin([1.0, 1.0]) == 1.0


def test_long_bench_abba_warmup_and_snapshots():
  calls, snaps = [], []
  def clock(fn):
    ms = fn()
    return ms, ms + 1.0
  cand = lambda: calls.append("c") or 90.0
  ref = lambda: calls.append("r") or 100.0
  small = {"candidate": lambda: calls.append("c_small"), "reference": lambda: calls.append("r_small")}
  lb = xc.LongBench(clock=clock, snapshot=lambda: snaps.append(1) or {"clocks.sm": "1695"},
                    warmup=xc.small_first(small))
  r = lb.paired(cand, ref, pairs=5, block=2)
  assert calls[:4] == ["c_small", "c", "r_small", "r"]                      # each side: small input, then full size
  assert calls[4:] == ["r", "c", "c", "r", "r", "c", "c", "r", "r", "c"]  # ABBA
  assert abs(r["ratio"] - 100 / 90) < 1e-12 and r["candidate_ms"] == 90.0 and r["reference_wall_ms"] == 101.0
  assert r["ci"][0] == r["ci"][1] and r["pairs"] == 5
  assert len(r["snapshots"]) == 4 and [s["pair"] for s in r["snapshots"]] == [0, 2, 4, 5]
  assert xc.LongBench(clock=clock, snapshot=dict).aa(ref, pairs=2)["ratio"] == 1.0


def test_game_confirm_and_unreached_loads_may_be_omitted():
  reach = xc.ReachMap(loads={"m1": 1, "m32": 32}, cutoff_field="cutoff", fields={"large": ("tile",)})
  base, cand = {"cutoff": 17, "tile": 64}, {"cutoff": 17, "tile": 128}       # tile reaches m32 only
  def measure(designs, grade):
    return [{"m1": 1.0, "m32": 1.0} if d == base else {"m32": 1.03} for d in designs]
  g = xc.Game(("m1", "m32"), reach, base, lambda h: [], lambda d, c: [d["tile"]],
              lambda ds: [{"ok": True}] * len(ds), measure, surrogate={"backend": "ridge"})
  g.confirm([cand])
  gains = g.evidence.rows["m32"][reach.identity("m32", cand)]["gains"]
  assert set(g.evidence.rows) == {"m32"} and abs(gains[0] - 0.03) < 1e-12
  g.measure = lambda designs, grade: [{"m32": 1.0} for _ in designs]           # the incumbent must cover every load
  try:
    g.confirm([cand])
    assert False
  except ValueError:
    pass
