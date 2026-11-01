"""Block L release gates L2 and L3 (handbook §11), one explicit test per structure detector.

L2  every structure detector returns < 0.02 structured fraction (or an "invariant" / null-consistent
    verdict) on pure noise / on its anchors;
L3  every detector recovers ``nulls.planted`` structure of known size within 10 %.

These deliberately re-state what the per-module tests already check, so that the gate is auditable in one
file: a detector missing from here is not released as a detector. Detectors covered:
``nulls.marchenko_pastur``, ``darkmatter.residual_structure``, ``symmetry.hidden``, ``graphs.modularity``,
``topology.graph_ph``, ``topology.zigzag``, ``topology.mapper``, ``topology.hilbert_function``,
``metrics.betti_at_scale`` (via ``compute_diagrams``), ``metrics.bias_top_permutation_test``,
``gluing.explanation_obstruction``, ``invariants.check_invariance`` (anchor verdicts)."""
import numpy as np, pytest
import whiteboxtopo as wb
from whiteboxtopo import nulls, graphs, topology as T, symmetry as S
from whiteboxtopo.invariants import ReluMLP, check_invariance, weight_product
from whiteboxtopo.darkmatter import residual_structure
from whiteboxtopo.gluing import explanation_obstruction
from whiteboxtopo.core.persistence import compute_diagrams
from whiteboxtopo.metrics.betti import betti_at_scale

WITHIN = 0.10   # L3: recovered size within 10 % of planted size


def within(found, planted, tol=WITHIN):
    return abs(found - planted) <= max(tol * planted, 0.5 - 1e-12)   # a 10 % band narrower than one unit means exact


# --------------------------------------------------------------------------------------------- spectral
def test_L2_mp_pure_noise():
    fr = []
    for t in range(20):
        E, _ = nulls.planted("gaussian", (1500, 150), seed=t)
        s = np.linalg.svd(E - E.mean(0), compute_uv=False)
        fr.append(nulls.marchenko_pastur(s ** 2 / E.shape[0], aspect=E.shape[1] / E.shape[0], n=E.shape[0]).n_outliers / E.shape[1])
    assert np.mean(fr) < 0.02 and max(fr) <= 2 / 150


def test_L3_mp_recovers_planted_spikes():
    for k in (3, 5, 12):
        E, truth = nulls.planted("spikes", (3000, 200), k=k, seed=k)
        s = np.linalg.svd(E - E.mean(0), compute_uv=False)
        n = nulls.marchenko_pastur(s ** 2 / E.shape[0], aspect=E.shape[1] / E.shape[0], n=E.shape[0]).n_outliers
        assert within(n, truth["n_outliers"]), (k, n)


def test_L2_residual_structure_pure_noise():
    E = np.random.default_rng(1).normal(size=(3000, 200))
    rep = residual_structure(E, topology=False)
    assert rep.structured_fraction < 0.02 and rep.n_outliers <= 1 and rep.verdict == "neither"


def test_L3_residual_structure_recovers_planted_rank():
    for k in (5, 10):
        E, truth = nulls.planted("spikes", (3000, 200), k=k, seed=100 + k)
        rep = residual_structure(E, topology=False)
        assert within(rep.n_outliers, truth["rank"]), (k, rep.n_outliers)


# --------------------------------------------------------------------------------------------- hidden symmetry
def _mlp(rng, widths=(8,), n_in=3, n_out=1):
    dims = (n_in,) + tuple(widths) + (n_out,)
    return ReluMLP([rng.normal(size=(a, b)) for a, b in zip(dims[:-1], dims[1:])], [rng.normal(size=b) for b in dims[1:]])


def test_L2_hidden_symmetry_no_false_positives_on_generic_nets():
    """Random nets legitimately carry dead units (random biases on [-1, 2]^3 probes); the L2 anchor is therefore
    (a) no *unexplained* fiber directions beyond numerical rank slack, (b) every unit reported dead is verifiably
    dead on the probe region, (c) a net with nothing identified has excess exactly 0."""
    for seed in range(8):
        rng = np.random.default_rng(seed); m = _mlp(rng, (8, 6)); X = rng.uniform(-1, 2, size=(2000, 3))
        rep = S.hidden(m, X)
        assert rep.unexplained <= 2, rep
        acts = m.hidden_acts(X)
        for (layer, j) in rep.dead_units:
            assert acts[layer][:, j].max() <= 1e-12, (seed, layer, j)
        if rep.identified.get("explained_total", 0) == 0:
            assert rep.excess == 0 and not rep.G_incomplete


def test_L3_hidden_symmetry_recovers_planted_duplicates_and_dead():
    rng = np.random.default_rng(7); m = _mlp(rng, (10,))
    Ws = [W.copy() for W in m.Ws]; bs = [b.copy() for b in m.bs]
    for j, c in ((1, 2.0), (3, 0.5), (5, 3.0)):                # three duplicate units (positive multiples of unit 0)
        Ws[0][:, j] = c * Ws[0][:, 0]; bs[0][j] = c * bs[0][0]
    for j in (7, 8):                                          # two dead units
        bs[0][j] = -50.0
    rep = S.hidden(ReluMLP(Ws, bs))
    assert rep.G_incomplete and rep.excess > 0
    assert rep.unexplained <= WITHIN * rep.excess, rep            # >= 90 % of the planted fiber identified
    assert set(rep.dead_units) == {(0, 7), (0, 8)}, rep.dead_units
    # duplicates: the three planted units are all multiples of unit 0 -> the spanning set of pairs covers units {0,1,3,5}
    units = {u for p in rep.duplicate_pairs for u in p[1:]}
    assert units == {0, 1, 3, 5}, rep.duplicate_pairs


# --------------------------------------------------------------------------------------------- graph structure
def test_L2_modularity_random_graph_inside_rewired_null():
    rng = np.random.default_rng(3); n = 120
    A = (rng.uniform(size=(n, n)) < 0.08).astype(float); A = np.triu(A, 1); A = A + A.T
    Q, memb = graphs.modularity(A)
    q0, sd = graphs.rewired_null(A, n=10)
    assert Q <= q0 + 3 * max(sd, 1e-3)


def test_L3_modularity_recovers_planted_block_count():
    for k in (4, 10):
        A, truth = nulls.planted("blocks", 60 * k, k=k, seed=k, p_in=0.3, p_out=0.005)
        Q, memb = graphs.modularity(A)
        found = len(np.unique(memb))
        assert within(found, k), (k, found)
        # and the recovered partition agrees with the planted labels (up to relabelling)
        from sklearn.metrics import adjusted_rand_score
        assert adjusted_rand_score(truth["labels"], memb) > 0.9


def test_L2_graph_ph_tree_has_no_cycles():
    rng = np.random.default_rng(4); n = 40; A = np.zeros((n, n))
    for i in range(1, n):
        j = rng.integers(0, i); A[i, j] = A[j, i] = rng.uniform(0.2, 1.0)   # random tree
    r = T.graph_ph(A)
    assert r.betti1_essential == 0 and r.betti1_finite == 0


def test_L3_graph_ph_counts_planted_independent_cycles():
    for k in (3, 12):
        n = 5 * k; A = np.zeros((n, n))
        for c in range(k):                                       # k disjoint 5-cycles ...
            for i in range(5):
                a, b = 5 * c + i, 5 * c + (i + 1) % 5; A[a, b] = A[b, a] = 1.0
        for c in range(k - 1):                                   # ... joined into one component by bridges (no new cycles)
            A[5 * c, 5 * (c + 1)] = A[5 * (c + 1), 5 * c] = 1.0
        r = T.graph_ph(A)
        assert within(r.betti1_essential + r.betti1_finite, k), (k, r.betti1_essential, r.betti1_finite)


# --------------------------------------------------------------------------------------------- point-cloud topology
def test_L2_betti1_pure_noise_cloud_is_zero_at_planted_scale():
    rng = np.random.default_rng(5)
    for t in range(5):
        X = rng.normal(size=(150, 2)) * 0.05
        D = compute_diagrams(X, maxdim=1, backend="ripser", thresh_percentile=95.0).dgms[1]
        assert betti_at_scale(D, 0.9) == 0


def test_L3_betti_recovers_planted_circle_and_clusters():
    X, truth = nulls.planted("circle", 200, seed=0, noise=0.03)
    D1 = compute_diagrams(X, maxdim=1, backend="ripser", thresh_percentile=95.0).dgms[1]
    assert betti_at_scale(D1, 0.9) == truth["b1"]
    for k in (4, 6):
        X, truth = nulls.planted("clusters", (300, 5), k=k, sep=40.0, noise=0.5, seed=k)
        D0 = compute_diagrams(X, maxdim=0, backend="ripser", thresh_percentile=100.0).dgms[0]
        b0 = betti_at_scale(D0, 6.0) + 1                           # + the one infinite H0 bar the diagram drops
        assert within(b0, len(np.unique(truth["labels"]))), (k, b0)


def test_L3_mapper_recovers_planted_component_count():
    for k in (3, 5):
        X, truth = nulls.planted("clusters", (400, 3), k=k, sep=40.0, noise=0.5, seed=10 + k)
        r = T.mapper(X, X[:, 0], n_cubes=12, overlap=0.3)
        assert within(r.n_components, len(np.unique(truth["labels"]))), (k, r.n_components)


def test_L2_hilbert_function_noise_has_no_h1_and_L3_circle_has_one():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(150, 2)) * 0.05; f = np.linspace(0, 1, 150)
    assert T.hilbert_function(X, f, n_grid=5, scales=[0.5, 0.9]).hilbert[1].max() == 0
    X, _ = nulls.planted("circle", 150, seed=0, noise=0.03)
    h = T.hilbert_function(X, f, n_grid=5, scales=[0.5, 0.9]).hilbert[1]
    assert h.max() == 1


def test_L3_zigzag_recovers_planted_loop_extent():
    pytest.importorskip("dionysus")
    rng = np.random.default_rng(0); clouds = []; lo, hi = 3, 7
    for t in range(10):
        clouds.append(nulls.planted("circle", 60, seed=t, noise=0.03)[0] if lo <= t <= hi else rng.normal(size=(12, 2)) * 0.05)
    z = T.zigzag(clouds, scale=0.7, maxdim=1)
    ext = z.bars_in_cloud_index(1)
    long = [tuple(e) for e in ext if e[1] - e[0] + 1 >= 0.9 * (hi - lo + 1)]
    assert len(long) == 1 and long[0] == (lo, hi)
    bc = z.betti_curve(1)
    assert bc[0] == 0 and bc[-1] == 0


# --------------------------------------------------------------------------------------------- bias / gluing / invariance anchors
def test_L2_bias_top_same_distribution_fpr():
    rng = np.random.default_rng(0); hits = 0
    for t in range(10):
        X = rng.standard_normal((160, 6)); r = wb.fit_reducer(X, 4)
        _, p, _ = wb.bias_top_permutation_test(X[:80], X[80:], r, n_permutations=40, rng=rng)
        hits += p <= 0.05
    assert hits <= 1                                               # <= 10 % at alpha 0.05 over 10 draws


def test_L2_gluing_homogeneous_cover_inside_null():
    rng = np.random.default_rng(0); scores = rng.normal(0, 0.1, size=(400, 50)); scores[:, :10] += 10.0
    rep = explanation_obstruction(scores, embedding=rng.normal(size=(400, 5)), k=6, k_e=10, n_min=10, n_shuffle=60, n_boot=0, seed=0)
    assert not rep.significant_vs_null and rep.null_lo <= rep.omega <= rep.null_hi + 1e-9


def test_L2_invariance_anchor_verdicts():
    rng = np.random.default_rng(0)
    m = ReluMLP([rng.normal(size=(3, 8)), rng.normal(size=(8, 1))], [rng.normal(size=8), rng.normal(size=1)])
    X = rng.uniform(-1, 2, size=(64, 3))
    assert check_invariance(weight_product, m, n=30).verdict == "invariant"
    assert check_invariance(lambda mm: mm.hidden_acts(X)[0].mean(0), m, n=30).verdict == "artifact"
    r = check_invariance(lambda mm: np.linalg.svd(mm.Ws[0], compute_uv=False), m, n=30)
    assert r.verdict == "covariant" and abs(r.scaling_exponent - 1.0) < 1e-6
