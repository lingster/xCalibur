"""Counterexamples for exactness, evidence accounting and captured-build identity."""
import importlib
import math

import numpy as np
import pytest

import xcalibur as xc


def test_capture_retains_every_build_with_the_same_flags():
    a, b = xc.Capture(), xc.Capture()
    a(cuda_sources="first", extra_cuda_cflags=["-O3"])
    a(cuda_sources="shared last", extra_cuda_cflags=["-O3"])
    b(cuda_sources="different first", extra_cuda_cflags=["-O3"])
    b(cuda_sources="shared last", extra_cuda_cflags=["-O3"])
    assert not xc.Capture.same(a, b)


@pytest.mark.parametrize("change", [
    {"extra_cuda_cflags": ["-DX=2", "-DX=1"]},
    {"extra_cflags": ["-DHOST=2"]},
    {"extra_ldflags": ["-lother"]},
    {"functions": ["other_entry"]},
    {"cuda_sources": ["a", "b"]},
])
def test_capture_compares_ordered_flags_source_units_and_build_options(change):
    options = dict(cuda_sources="a\nb", extra_cuda_cflags=["-DX=1", "-DX=2"],
                   extra_cflags=["-DHOST=1"], extra_ldflags=["-lbase"], functions=["entry"])
    a, b = xc.Capture(), xc.Capture()
    a(**options); b(**(options | change))
    assert not xc.Capture.same(a, b)


def test_capture_compares_build_multisets_without_discarding_multiplicity():
    a, b = xc.Capture(), xc.Capture()
    for source in ["one", "two", "one"]:
        a(cuda_sources=source)
    for source in ["two", "one", "one"]:
        b(cuda_sources=source)
    assert xc.Capture.same(a, b)
    b(cuda_sources="one")
    assert not xc.Capture.same(a, b)


def test_capture_freezes_mutable_compiler_arguments():
    flags, sources = ["-O3"], ["original"]
    a, b = xc.Capture(), xc.Capture()
    a(cuda_sources=sources, extra_cuda_cflags=flags)
    b(cuda_sources=["original"], extra_cuda_cflags=["-O3"])
    flags.append("-DCHANGED=1")
    sources[0] = "changed"
    assert xc.Capture.same(a, b)


@pytest.mark.parametrize("source,name,value,expected", [
    ("A = '''first\nsecond'''\nB = 2\n", "A", "new", "A = 'new'\nB = 2\n"),
    ("label = 'λ'; A = 1; B = 2\n", "A", 3, "label = 'λ'; A = 3; B = 2\n"),
    ("A: str = 'old'  # preserve comment\r\nB = 2\r\n", "A", "new", "A: str = 'new'  # preserve comment\r\nB = 2\r\n"),
])
def test_assignment_replacement_preserves_surrounding_source(source, name, value, expected):
    assert xc.replace_assignment(source, name, value) == expected


@pytest.mark.parametrize("source", ["A = B = 'old'\n", "A = 1\nA = 2\n", "B = 1\n"])
def test_assignment_replacement_rejects_ambiguous_or_absent_targets(source):
    with pytest.raises(ValueError):
        xc.replace_assignment(source, "A", "new")


@pytest.mark.parametrize("values", [[], [1, 0], [1, -1], [1, math.nan], [1, math.inf]])
def test_geomean_cannot_silently_drop_a_failed_load(values):
    with pytest.raises(ValueError):
        xc.geomean(values)


def test_geomean_preserves_the_complete_denominator():
    assert xc.geomean([4, .25]) == pytest.approx(1)


def test_bootstrap_uses_the_actual_even_sample_median(monkeypatch):
    stats = importlib.import_module("xcalibur.stats")
    monkeypatch.setattr(stats.random.Random, "choices", lambda self, population, k: list(population))
    assert xc.bootstrap_ci(np.array([1., 9.]), n=8) == (5., 5.)


@pytest.mark.parametrize("value", [math.inf, math.nan, -1., -2.])
def test_invalid_relative_gain_cannot_enter_confirmation_evidence(value):
    evidence = xc.Evidence()
    with pytest.raises(ValueError):
        evidence.add("load", "build", value)
    assert not evidence.rows


def test_graph_effects_use_the_even_sample_median():
    graph = xc.DesignGraph(loads=("load",))
    for group, ratio in [(0, 1.), (1, 3.)]:
        graph.add(xc.Node(f"a{group}", "design", {"group": group, "choice": 0}, {"load": 1.}))
        graph.add(xc.Node(f"b{group}", "design", {"group": group, "choice": 1}, {"load": ratio}))
    effect = next(e for e in graph.effects() if e["field"] == "choice" and e["from"] == 0)
    assert effect["median"]["load"] == 2.


def test_gf2_equality_does_not_ignore_high_input_bits():
    assert not xc.equal((1,), (3,))


@pytest.mark.parametrize("rows", [(3,), (-1,), (1, 4)])
def test_gf2_inverse_rejects_invalid_square_maps(rows):
    with pytest.raises(ValueError):
        xc.inverse(rows)


def test_basis_extraction_rejects_affine_offsets_and_out_of_range_outputs():
    with pytest.raises(ValueError):
        xc.from_function(lambda x: x ^ 1, bits=4)
    with pytest.raises(ValueError):
        xc.from_function(lambda x: x << 1, bits=4)


@pytest.mark.parametrize("change", [
    {"ctas_per_sm": 0}, {"ctas_per_sm": -1}, {"extra_smem_bytes": -1},
    {"squads": [{"name": "mma", "warps": -4}]},
    {"squads": [{"name": "mma", "warps": 33}]},
    {"squads": [{"name": "mma", "warps": 4, "regs": 0}]},
    {"squads": [{"name": "mma", "warps": 4, "regs": 264}]},
    {"resources": [{"name": "ring", "stages": -1, "stride_bytes": 64,
                    "phases": [{"owners": ["mma"]}, {"owners": ["mma"]}]}]},
    {"resources": [{"name": "ring", "stages": 1, "stride_bytes": 64,
                    "phases": [{"owners": []}, {"owners": ["mma"]}]}]},
])
def test_resource_checker_cannot_pass_malformed_plans(change):
    result = xc.execute(dict(squads=[{"name": "mma", "warps": 4}], resources=[]) | change)
    assert not result["ok"] or not result["result"]["pass"]


def test_register_cap_accepts_numpy_integer_sweeps():
    assert xc.launch_register_cap(xc.H200, np.int64(18), np.int64(1)) == 96


class Ranker:
    def fit(self, *args):
        return self

    def top(self, xs, *args):
        return [(i, .1, 0.) for i in range(len(xs))]

    def ucb(self, xs, *args):
        return np.ones(len(xs)), np.ones(len(xs)), np.zeros(len(xs))


def make_game(monkeypatch, history=True):
    module = importlib.import_module("xcalibur.game")
    monkeypatch.setattr(module, "Surrogate", Ranker)
    base, candidate = {"cutoff": 8, "choice": 0}, {"cutoff": 8, "choice": 1}
    reach = xc.ReachMap(loads={"a": 1, "b": 2}, cutoff_field="cutoff", fields={"all": ("choice",)})
    calls = []

    def measure(designs, grade):
        calls.append((grade, [dict(d) for d in designs]))
        return [{load: 1. + .05 * d["choice"] for load in ("a", "b")} for d in designs]

    game = xc.Game(("a", "b"), reach, base, lambda h: [candidate], lambda d, c: [d["choice"]],
                   lambda ds: [{"ok": True} for _ in ds], measure, min_samples=3)
    if history:
        game.history = [{"design": dict(base), "loads": {"a": 1., "b": 1.}} for _ in range(12)]
    return game, candidate, calls


def test_one_confirmation_round_does_not_count_the_same_design_twice(monkeypatch):
    game, candidate, calls = make_game(monkeypatch)
    for load in game.loads:
        game.evidence.add(load, game.reach.identity(load, candidate), .05, candidate)
    result = game.round()
    confirm = next(ds for grade, ds in calls if grade == "confirm")
    assert len(confirm) == 2  # one reference, one candidate serving both loads
    assert not result["bound"]  # only two independent rounds of evidence, not three


def test_game_can_start_without_manually_seeded_history(monkeypatch):
    game, _, _ = make_game(monkeypatch, history=False)
    assert game.round()["screened"] == 1


def test_game_keeps_nominations_from_every_load(monkeypatch):
    game, _, _ = make_game(monkeypatch)
    game.measure = lambda designs, grade: [
        {"a": 1., "b": 1. + .05 * d["choice"]} for d in designs
    ]
    assert game.round()["confirmed"] == 1
    assert len(next(iter(game.evidence.rows["b"].values()))["gains"]) == 1


def test_cold_start_advances_past_failed_compilations(monkeypatch):
    game, candidate, _ = make_game(monkeypatch, history=False)
    game.per_load = 1
    game.frontier = lambda history: [candidate | {"choice": -1}, candidate]
    game.compile = lambda designs: [{"ok": d["choice"] > 0} for d in designs]
    assert game.round()["screened"] == 0
    assert game.round()["screened"] == 1


def test_build_aliases_need_separate_confirmation_rounds(monkeypatch):
    game, candidate, calls = make_game(monkeypatch)
    game.frontier = lambda history: [candidate, candidate | {"unrelated": 1}]
    for repeat in range(1, 4):
        result = game.round()
        for load in game.loads:
            observations = next(iter(game.evidence.rows[load].values()))["gains"]
            assert len(observations) == repeat
        assert bool(result["bound"]) == (repeat == 3)


@pytest.mark.parametrize("bad", [{"a": 1.}, {"a": 1., "b": 0.}, {"a": 1., "b": math.nan}])
def test_game_rejects_incomplete_or_invalid_measurements_before_recording(monkeypatch, bad):
    game, _, _ = make_game(monkeypatch)
    game.measure = lambda designs, grade: [bad for _ in designs]
    before = len(game.history)
    with pytest.raises(ValueError):
        game.round()
    assert len(game.history) == before
    assert not game.evidence.rows


@pytest.mark.parametrize("callback", ["compile", "measure"])
def test_game_rejects_truncated_callback_results(monkeypatch, callback):
    game, _, _ = make_game(monkeypatch)
    setattr(game, callback, lambda *args: [])
    with pytest.raises(ValueError):
        game.round()


def test_game_does_not_pool_confirmations_after_reference_changes(monkeypatch):
    game, _, _ = make_game(monkeypatch)
    game.round()
    game.incumbent["choice"] = 2
    with pytest.raises(ValueError, match="reference|incumbent"):
        game.round()


def test_evidence_does_not_alias_the_callers_design():
    design = {"tile": [64, 32]}
    evidence = xc.Evidence()
    evidence.add("a", "build", .05, design)
    design["tile"][0] = 128
    assert evidence.rows["a"]["build"]["design"] == {"tile": [64, 32]}


def test_bench_keeps_raw_samples_and_rejects_invalid_corrections():
    bench = object.__new__(xc.Bench)
    bench.graph = lambda f: f
    bench.null_us = lambda: 2.
    values = iter([10., 5., 6., 12.])
    bench.replay_us = lambda g: next(values)
    result = bench.paired("candidate", "reference", pairs=2, subtract_null=False)
    assert result["ratio"] == 2.
    assert result["raw_candidate_samples_us"] == [5., 6.]
    assert result["raw_reference_samples_us"] == [10., 12.]
    values = iter([10., 1., 1., 12.])
    with pytest.raises(ValueError, match="positive|calibrat|subtraction"):
        bench.paired("candidate", "reference", pairs=2, subtract_null=True)


def test_paired_wrapper_forwards_measurement_options(monkeypatch):
    module = importlib.import_module("xcalibur.bench")

    class FakeBench:
        def __init__(self, flush_mib):
            assert flush_mib == 32

        def paired(self, candidate, reference, pairs, subtract_null):
            assert (pairs, subtract_null) == (4, False)
            return {"ratio": 1.}

    monkeypatch.setattr(module, "Bench", FakeBench)
    assert module.paired(None, None, pairs=4, subtract_null=False, flush_mib=32) == {"ratio": 1.}
