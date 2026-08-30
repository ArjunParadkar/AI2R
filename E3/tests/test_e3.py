"""SPEC §9 controls: planted sheaves, brute-force agreement, EAP graph."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from e3lib import sheaf as sh  # noqa: E402
from e3lib import cluster as cl  # noqa: E402
from e3lib.eap import EdgeGraph  # noqa: E402


def _planted_scores(k, n_per, overlap, edge_sets, n_edges=500, seed=0):
    """Clusters with planted circuits: rows of cluster i have large |score| on edge_sets[i].
    Overlap prompts between i and j carry the union of both circuits' signals on a
    per-pair basis so that restriction keeps each side's own edges (or not)."""
    rng = np.random.default_rng(seed)
    rows, members = [], []
    for i in range(k):
        for _ in range(n_per):
            r = rng.normal(0, 0.01, n_edges); r[list(edge_sets[i])] = 1.0
            rows.append(r); m = np.zeros(k, bool); m[i] = True; members.append(m)
    for (i, j), n_ov in overlap.items():
        for _ in range(n_ov):
            r = rng.normal(0, 0.01, n_edges)
            r[list(edge_sets[i])] = 1.0; r[list(edge_sets[j])] = 1.0
            r[list(edge_sets[i] & edge_sets[j])] = 1.5   # shared edges matter most on the overlap
            rows.append(r); m = np.zeros(k, bool); m[i] = m[j] = True; members.append(m)
    return np.array(rows, np.float32), np.array(members)


def test_consistent_cover_zero():
    k, ke = 3, 20
    E = set(range(20))
    scores, members = _planted_scores(k, 30, {(0, 1): 15, (1, 2): 15, (0, 2): 15}, [E, E, E])
    edges, tris, _ = sh.nerve(members, 10)
    assert edges == [(0, 1), (0, 2), (1, 2)] and tris == []   # no triple-overlap rows planted
    J, Jraw = sh.pair_agreements(scores, members, edges, ke)
    assert np.allclose(J, 1) and np.allclose(Jraw, 1)
    ob = sh.obstruction(k, edges, J)
    assert ob["omega"] == 0 and ob["obs"].max() == 0 and ob["h0"][-1] == 1


def test_planted_triangle_h1():
    """A~B, B~C consistent, A vs C inconsistent -> exactly one frustrated cycle."""
    A = set(range(0, 20)); B = set(range(10, 30)); C = set(range(20, 40)) | set(range(100, 110))
    # restricted (top-10) sets: A|B on overlap -> both keep 10..19-ish shared? build so that
    # A,B share 10 edges, B,C share 10, A,C share 0.
    scores, members = _planted_scores(3, 30, {(0, 1): 15, (1, 2): 15, (0, 2): 15}, [A, B, C])
    edges, _, _ = sh.nerve(members, 10)
    J, _ = sh.pair_agreements(scores, members, edges, 20)   # k_r = 10
    # A∩C = {} so J_AC = 0 regardless of restriction; A∩B and B∩C non-empty
    jd = dict(zip(edges, J))
    assert jd[(0, 2)] == 0 and jd[(0, 1)] == 1 and jd[(1, 2)] == 1
    ob = sh.obstruction(3, edges, J)
    lo = min(jd[(0, 1)], jd[(1, 2)])
    tau = 0.5 * lo
    assert ob["obs"][np.searchsorted(ob["taus"], tau)] == 1
    assert np.isclose(ob["omega"], lo)          # frustrated for tau in (0, lo]
    assert sh.brute_force_obs(3, edges, J, tau) == 1


def test_two_mechanisms_no_obstruction():
    """Two consistent groups joined only by inconsistent edges: a relabelling explains it,
    obs = 0, H0 = 2 at tau between the two agreement levels."""
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3), (0, 5)]
    J = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
    ob = sh.obstruction(6, edges, J)
    assert ob["omega"] == 0
    assert ob["h0"][np.searchsorted(ob["taus"], 0.5)] == 2
    assert sh.brute_force_obs(6, edges, J, 0.5) == 0


def test_bruteforce_random_nerves():
    rng = np.random.default_rng(0)
    for trial in range(60):
        n_v = int(rng.integers(3, 8))
        all_pairs = [(i, j) for i in range(n_v) for j in range(i + 1, n_v)]
        keep = rng.random(len(all_pairs)) < 0.6
        edges = [p for p, kp in zip(all_pairs, keep) if kp]
        if len(edges) < 2:
            continue
        J = np.round(rng.random(len(edges)), 2)
        ob = sh.obstruction(n_v, edges, J)
        for tau in [0.2, 0.5, 0.8]:
            bf = sh.brute_force_obs(n_v, edges, J, tau)
            fast = int(((J < tau) & (tau <= ob["J_bott"])).sum())
            assert bf == fast, (trial, tau, edges, J)
        # integral identity: Ω = ∫ obs dτ (exact for piecewise-constant obs on a fine grid)
        taus = np.linspace(0, 1, 100001)
        obs_fine = np.array([int(((J < t) & (t <= ob["J_bott"])).sum()) for t in taus[::100]])
        assert abs(np.trapezoid(obs_fine, taus[::100]) - ob["omega"]) < 0.02


def test_edge_graph_count():
    g = EdgeGraph()
    assert g.n_edges == 32491 and g.n_up == 157 and g.n_down == 445
    assert g.edge_names[0] == "input->a0.h0<q" and g.edge_names[-1] == "m11->logits"


def test_fcm_and_cover():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 0.3, (50, 2)), rng.normal(3, 0.3, (50, 2)), rng.normal([0, 3], 0.3, (50, 2))])
    U, C, _ = cl.fcm(X, 3, n_init=3)
    assert np.allclose(U.sum(1), 1)
    hard = cl.hard_cover(U)
    assert hard.sum() == 150 and hard.sum(1).max() == 1
    soft = cl.soft_cover(U, 0.5)
    assert (soft.sum(1) >= 1).all() and soft.sum() >= 150
    M, d2, C = cl.knn_centre_cover(X, 3, 2, n_init=3)          # amendment-1 cover
    assert (M.sum(1) == 2).all() and M.sum() == 300
    assert (hard.sum(0) > 40).all()


@pytest.mark.skipif(os.environ.get("E3_MODEL_TEST", "0") != "1", reason="loads GPT-2 (set E3_MODEL_TEST=1)")
def test_eap_runs_on_model():
    import torch
    from transformer_lens import HookedTransformer
    from e3lib.eap import eap_all, ioi_metric, prepare_model
    from e3lib.prompts import make_ioi_prompts
    model = prepare_model(HookedTransformer.from_pretrained("gpt2", device="cpu"))
    prompts = make_ioi_prompts(model, n_per_cell=1)[:4]
    s, e, mc, mx = eap_all(model, prompts, ioi_metric, EdgeGraph(), batch_size=2, device="cpu", log=lambda *_: None)
    assert s.shape == (4, 32491) and np.isfinite(s).all() and (mc > mx).mean() >= 0.5
