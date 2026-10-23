"""Unit, property and regression tests for the estimators (master prompt §9)."""
import itertools

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from whitebox_ee.depth import bottleneck, memorization_level
from whitebox_ee.generality import Activity, paths
from whitebox_ee.graph import ConceptGraph, fit_psi
from whitebox_ee.info import entropy, heldout_cond_xent, normalised_vi, variation_of_information
from whitebox_ee.probe import Node, Probe
from whitebox_ee.stats import auroc, holm


# ---------------------------------------------------------------- info / stats
def test_entropy_uniform():
    assert abs(entropy(np.arange(8)) - 3.0) < 1e-12


def test_heldout_xent_deterministic_small():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 4000)
    c = (a * 3) % 4
    assert heldout_cond_xent(c, [a]) < 0.02
    assert heldout_cond_xent(c, []) > 1.9


def test_vi_invariant_to_relabelling():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, 1000)
    perm = rng.permutation(5)
    assert variation_of_information(a, perm[a]) < 1e-9
    assert normalised_vi(a, perm[a]) < 1e-9


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(2)
    s, y = rng.normal(size=300), rng.integers(0, 2, 300)
    assert abs(auroc(s, y) - roc_auc_score(y, s)) < 1e-12


def test_holm_monotone():
    p = holm([0.01, 0.04, 0.03])
    assert np.all(p >= np.array([0.01, 0.04, 0.03]))
    assert p[0] == pytest.approx(0.03)


# ---------------------------------------------------------------- bottleneck (Prop. 3.5, 3.8)
def _random_dag(rng, n_int):
    """inputs i0,i1; internal n0..; output o. Edges respect index order."""
    names = ["i0", "i1"] + [f"n{k}" for k in range(n_int)] + ["out:b"]
    kinds = ["input", "input"] + ["concept"] * n_int + ["output"]
    nodes = {}
    for j, (nm, kd) in enumerate(zip(names, kinds)):
        nodes[nm] = Node(nm, kd, [], np.zeros(1, int), j - 2 if kd != "input" else -1, meta={"behaviour": "b"})
    parents = {nm: [] for nm in names}
    for j in range(2, len(names)):
        cand = names[:j]
        k = rng.integers(1, min(3, len(cand)) + 1)
        parents[names[j]] = list(rng.choice(cand, k, replace=False))
    G = ConceptGraph(nodes, parents, {n: {} for n in names})
    return G, names


def _brute(G, w, names):
    """min over input->output paths of max internal weight (0 if no path)."""
    best = None
    children = {n: G.children(n) for n in names}

    def dfs(u, mx):
        nonlocal best
        if u == "out:b":
            best = mx if best is None else min(best, mx)
            return
        for c in children[u]:
            dfs(c, max(mx, w.get(c, 0.0)))
    for i in ("i0", "i1"):
        dfs(i, 0.0)
    return 0.0 if best is None else best


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 6))
def test_bottleneck_equals_bruteforce(seed, n_int):
    rng = np.random.default_rng(seed)
    G, names = _random_dag(rng, n_int)
    g = {n: float(rng.random()) for n in G.concepts()}
    A = Activity(np.arange(1), {c: {"b": np.ones(1, bool)} for c in G.concepts()},
                 {(p, c): np.ones(1, bool) for c, ps in G.parents.items() for p in ps}, behaviours=["b"])
    A.on_path = paths(G, A)
    w = {c: np.array([g[c]]) for c in G.concepts()}
    lam = bottleneck(G, A, w, "b")[0]
    assert abs(lam - _brute(G, g, names)) < 1e-12
    mu = memorization_level(G, A, w, "b")[0]
    assert lam <= mu + 1e-12                         # Prop. 3.8


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000))
def test_bottleneck_respects_inactive_edges(seed):
    rng = np.random.default_rng(seed)
    G, names = _random_dag(rng, 5)
    g = {n: float(rng.random()) for n in G.concepts()}
    edges = {(p, c): np.array([bool(rng.random() < 0.7)]) for c, ps in G.parents.items() for p in ps}
    A = Activity(np.arange(1), {c: {"b": np.ones(1, bool)} for c in G.concepts()}, edges, behaviours=["b"])
    A.on_path = paths(G, A)
    # brute force on the active-edge subgraph
    G2 = ConceptGraph(G.nodes, {c: [p for p in ps if edges[(p, c)][0]] for c, ps in G.parents.items()}, G.psi)
    w = {c: np.array([g[c]]) for c in G.concepts()}
    assert abs(bottleneck(G, A, w, "b")[0] - _brute(G2, g, names)) < 1e-12


# ---------------------------------------------------------------- reparametrisation (Prop. 2.13)
def test_generality_and_depth_invariant_to_relabelling():
    from validation.planted import v3_sideways
    from whitebox_ee.depth import depth_scores
    from whitebox_ee.discover import discover
    from whitebox_ee.filter import Thresholds
    from whitebox_ee.generality import compute_activity, generality
    from whitebox_ee.propose import propose_probes
    th = Thresholds()
    P = v3_sideways(seed=0)
    rng = np.random.default_rng(0)
    X = P.sample(1200, rng, train=True)
    vals = P.values(X)
    probe = Probe(P.system, X)
    G, probe, rec = discover(P.system, X, propose_probes(probe, vals), probe=probe, stability=False,
                             minimal_sets=False)
    A = compute_activity(probe, G, th, np.random.default_rng(5), rows=np.arange(300))
    g = generality(G, A)
    s1 = depth_scores(probe, G, A, g, th)
    # relabel every concept's values by a random permutation (members recoded consistently)
    for c in G.concepts():
        nd = G.nodes[c]
        k = int(nd.values.max()) + 1
        perm = rng.permutation(k)
        nd.values = perm[nd.values]
        for m in nd.members:
            m.value_ids = perm[m.value_ids]
    A2 = compute_activity(probe, G, th, np.random.default_rng(5), rows=np.arange(300))
    g2 = generality(G, A2)
    s2 = depth_scores(probe, G, A2, g2, th)
    for c in g:
        assert g[c] == pytest.approx(g2[c])
    for b in s1:
        np.testing.assert_allclose(s1[b]["lambda"], s2[b]["lambda"])


# ---------------------------------------------------------------- patching identities
def test_self_patch_is_identity_planted():
    from validation.planted import v1_addition
    P = v1_addition(3, seed=0)
    X = P.sample(200, np.random.default_rng(0))
    probe = Probe(P.system, X)
    nd = Node("s0", "concept", [], None, 1)
    from whitebox_ee.probe import Member
    nd.members = [Member("h1", P.U["s0"], np.zeros((1, P.U["s0"].shape[1])), np.array([0]))]
    lg, _ = probe.run_patched(np.arange(200), [(nd, np.arange(200))])
    for b in lg:
        np.testing.assert_allclose(lg[b], probe.logits[b], atol=1e-9)


def test_planted_networks_are_correct():
    from validation.planted import v1_addition, v2_memo, v4_scaffold
    rng = np.random.default_rng(0)
    for P in (v1_addition(3, seed=3), v2_memo(seed=3), v4_scaffold(seed=3)):
        X = P.sample(500, rng)
        tg = P.targets(X)
        lg, _ = P.system.run(X)
        for b in P.behaviours:
            assert (lg[b].argmax(-1) == tg[b]).all(), (P.system.name, b)


# ---------------------------------------------------------------- regression (planted answers)
def test_regression_v1_recovery_seed0():
    from validation.planted import v1_addition
    from whitebox_ee.discover import discover
    from whitebox_ee.graph import label_nodes
    from whitebox_ee.propose import propose_probes
    P = v1_addition(3, seed=0)
    rng = np.random.default_rng(0)
    X = P.sample(2000, rng)
    vals = P.values(X)
    probe = Probe(P.system, X)
    labels = dict(vals)
    labels["wrong"] = (vals["a0"] + vals["b1"]) % 10
    G, probe, rec = discover(P.system, X, propose_probes(probe, labels), probe=probe, stability=False,
                             minimal_sets=False)
    lab = label_nodes(G, vals, len(X))
    got = {(lab[c], tuple(sorted(lab.get(p, p) for p in G.parents[c]))) for c in G.concepts()}
    read = set(P.behaviours.values())          # M7: directly read variables are output realisers
    want = {(n, tuple(sorted(v.parents))) for n, v in P.vars.items() if v.concept and v.layer > 0 and n not in read}
    assert got == want
    for b, v in P.behaviours.items():
        assert sorted(lab.get(p, p) for p in G.parents[f"out:{b}"]) == sorted(P.vars[v].parents)


def test_regression_v4_scaffold_principles_seed0():
    from validation.planted import v4_scaffold
    from whitebox_ee.discover import discover
    from whitebox_ee.filter import Thresholds
    from whitebox_ee.generality import compute_activity, generality
    from whitebox_ee.graph import label_nodes
    from whitebox_ee.propose import propose_probes
    P = v4_scaffold(seed=0)
    rng = np.random.default_rng(0)
    X = P.sample(2000, rng)
    vals = P.values(X)
    probe = Probe(P.system, X)
    G, probe, rec = discover(P.system, X, propose_probes(probe, vals), probe=probe, stability=False,
                             minimal_sets=False)
    lab = label_nodes(G, vals, len(np.unique(X, axis=0)))
    A = compute_activity(probe, G, Thresholds(), rng, rows=np.arange(400))
    g = generality(G, A)
    prin = {lab[c] for c in g if g[c] >= 0.25}
    assert prin == set(P.meta["scaffold"])


def test_maxT_use_multiplicity():
    """M10: a candidate drawn from the random-realiser null passes criterion (i) at about the
    nominal family-wise rate, whatever the number of behaviours; a real effect passes."""
    import numpy as np
    from whitebox_ee.filter import maxT_use
    rng = np.random.default_rng(0)
    for nb in (2, 20):
        hits = 0
        for _ in range(400):
            re = rng.gamma(4, 0.01, size=(21, nb))
            hits += maxT_use(re[0], re[1:]).any()
        assert hits / 400 < 0.12
    re = rng.gamma(4, 0.01, size=(20, 20))
    A = re.mean(0).copy(); A[3] += 0.2
    assert maxT_use(A, re)[3]
