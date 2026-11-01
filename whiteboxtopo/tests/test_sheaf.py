"""whiteboxtopo.sheaf — Čech H0/H1 of explanations (FINAL_SPEC 2.12)."""
import numpy as np
import pytest

from whiteboxtopo.sheaf import (Cover, cover, cover_from_members, local_explanations, restrict,
                                cech_H0_H1, sanity_gate)

N_FEAT = 40
A_FEATS, B_FEATS = slice(0, 20), slice(20, 40)


def two_mechanism_scores(seed=0, n=120):
    """Examples 0..59 are explained by features 0..19, examples 60..119 by 20..39."""
    rng = np.random.default_rng(seed)
    profile = np.linspace(2.0, 1.0, 20)          # a stable within-mechanism ranking
    E = 0.02 * np.abs(rng.standard_normal((n, N_FEAT)))
    E[:60, A_FEATS] += profile * (1 + 0.02 * rng.standard_normal((60, 20)))
    E[60:, B_FEATS] += profile * (1 + 0.02 * rng.standard_normal((n - 60, 20)))
    return E


def two_mechanism_cover(n=120):
    """Six regions in a ring: three inside mechanism A, three inside B, joined by two
    small cross-overlaps. The nerve has a cycle (cyclomatic 1), so H1 == 0 is not vacuous."""
    blocks = [list(range(0, 30)), list(range(20, 50)), list(range(40, 60)) + list(range(60, 65)),
              list(range(60, 90)), list(range(80, 110)), list(range(100, 120)) + list(range(0, 5))]
    M = np.zeros((n, 6), bool)
    for i, idx in enumerate(blocks):
        M[idx, i] = True
    return cover_from_members(M, n_min=5)


def single_mechanism_cover_and_scores(seed=1, n=120):
    rng = np.random.default_rng(seed)
    profile = np.linspace(2.0, 1.0, 20)
    E = 0.02 * np.abs(rng.standard_normal((n, N_FEAT)))
    E[:, A_FEATS] += profile * (1 + 0.02 * rng.standard_normal((n, 20)))   # one mechanism everywhere
    blocks = [list(range(0, 40)), list(range(30, 70)), list(range(60, 100)),
              list(range(90, 120)) + list(range(0, 10))]
    M = np.zeros((n, 4), bool)
    for i, idx in enumerate(blocks):
        M[idx, i] = True
    return cover_from_members(M, n_min=5), E


# ------------------------------------------------------------------ cover
def test_cover_from_embedding_builds_a_nerve():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.standard_normal((80, 5)), rng.standard_normal((80, 5)) + 6.0])
    cv = cover(X, k=6, overlap=0.5, seed=0, n_min=5)
    assert isinstance(cv, Cover) and cv.n_regions == 6 and cv.n_examples == 160
    assert cv.members.sum(1).min() >= 2                      # knn2: every example in >= 2 regions
    assert len(cv.edges) > 0
    d = cv.to_dict()
    assert d["n_regions"] == 6 and d["method"] == "knn2" and d["params"]["n_nearest"] == 2
    # deterministic
    assert np.array_equal(cover(X, k=6, seed=0, n_min=5).members, cv.members)


def test_cover_rejects_bad_overlap_and_method():
    X = np.random.default_rng(0).standard_normal((30, 3))
    with pytest.raises(ValueError):
        cover(X, k=3, overlap=1.5)
    with pytest.raises(ValueError):
        cover(X, k=3, method="nope")


# ------------------------------------------------- local explanations / restrict
def test_local_explanations_and_restrict():
    E = two_mechanism_scores()
    cv = two_mechanism_cover()
    ex = local_explanations(lambda idx: np.abs(E[idx]).mean(0), cv, k_e=10)
    assert len(ex) == cv.n_regions
    assert all(e.scores.shape == (N_FEAT,) for e in ex)
    assert set(ex[0].section.tolist()) <= set(range(20))      # region 0 lives in mechanism A
    assert set(ex[3].section.tolist()) <= set(range(20, 40))  # region 3 in mechanism B
    r = restrict(E, ex[0].section, cv.overlap_indices(0, 1), 5)
    assert len(r) == 5 and set(r.tolist()) <= set(ex[0].section.tolist())


def test_local_explanations_rejects_ragged_feature_axis():
    cv = two_mechanism_cover()
    with pytest.raises(ValueError):
        local_explanations(lambda idx: np.zeros(len(idx)), cv)   # length varies with the region


# ------------------------------------------------------------------ H0 / H1
def test_two_mechanisms_give_H0_2_and_boundaries():
    E, cv = two_mechanism_scores(), two_mechanism_cover()
    res = cech_H0_H1(cv, E, tau=0.5, k_e=10)
    assert res.H0 == 2, res.to_dict()
    assert len(res.boundaries) > 0
    # the boundaries are exactly the cross-mechanism overlaps
    assert {(b["i"], b["j"]) for b in res.boundaries} == {(2, 3), (0, 5)}
    assert res.cyclomatic >= 1                                # H1 == 0 is not vacuous here
    assert res.H1 == 0 and sanity_gate(res)                   # a clean split is H0, not frustration
    d = res.to_dict()
    assert d["H0"] == 2 and d["n_edges"] == len(cv.edges) and len(d["J"]) == len(cv.edges)


def test_single_mechanism_passes_the_sanity_gate():
    cv, E = single_mechanism_cover_and_scores()
    res = cech_H0_H1(cv, E, tau=0.5, k_e=10)
    assert res.H1 == 0 and sanity_gate(res) is True
    assert res.H0 == 1 and res.boundaries == []
    assert res.cyclomatic >= 1


def test_cech_shape_guard_and_empty_nerve():
    cv = two_mechanism_cover()
    with pytest.raises(ValueError):
        cech_H0_H1(cv, np.zeros((7, N_FEAT)), tau=0.5, k_e=5)
    M = np.zeros((20, 3), bool)
    M[0:6, 0] = M[6:13, 1] = M[13:20, 2] = True               # disjoint regions: no nerve edges
    disjoint = cover_from_members(M, n_min=2)
    res = cech_H0_H1(disjoint, np.random.default_rng(0).random((20, 8)), tau=0.5, k_e=3)
    assert res.n_edges == 0 and res.H0 == 3 and res.H1 == 0 and "vacuous" in res.notes
