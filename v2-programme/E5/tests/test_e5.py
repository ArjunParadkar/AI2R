import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from e5lib import load_connectome, CORE_TYPES, profile_adjacency, planted_ring, rewire_directed, rewire_undirected, HDTask, HDRNN
from e5lib.connectome import type_block_matrix
from e4lib.graph import symmetrise, threshold_density, modularity_q
import torch

def test_connectome_loads_with_expected_block_signs():
    c = load_connectome(core=True); W, types = c["W"], c["types"]
    assert W.shape == (148, 148) and set(types) == set(CORE_TYPES)
    assert (types == "EPG(PB08)").sum() == 46 and (types == "Delta7(PB15)").sum() == 42
    # Dale's law on the presynaptic axis: no row mixes signs; every Delta7 row is <= 0 and has negative mass
    assert all(not ((W[i] > 0).any() and (W[i] < 0).any()) for i in range(148))
    d7 = types == "Delta7(PB15)"; assert (W[d7] <= 0).all() and W[d7].sum() < 0
    B = type_block_matrix(W, types)
    i = {t: k for k, t in enumerate(CORE_TYPES)}
    assert B[i["EPG(PB08)"], i["Delta7(PB15)"]] > 0 and B[i["Delta7(PB15)"], i["PEN_a(PB06a)"]] < 0   # EPG->D7 excitatory, D7->PEN inhibitory
    assert B[i["EPG(PB08)"], i["PEN_a(PB06a)"]] > 0 and B[i["PEN_a(PB06a)"], i["EPG(PB08)"]] > 0     # the EPG<->PEN loop

def test_planted_ring_gets_high_Q():
    W = planted_ring(148); p = profile_adjacency(W, null_n=3, with_sweep=False)
    assert p["Q10"] > 0.45
    rng = np.random.default_rng(0); R = rng.normal(size=(148, 148)); q_rand = profile_adjacency(R, null_n=3, with_sweep=False)["Q10"]
    assert p["Q10"] > q_rand + 0.2

def test_rewiring_null_runs_and_preserves_degrees():
    c = load_connectome(core=True); W = c["W"]
    R = rewire_directed(W, seed=1)
    assert ((R != 0).sum(1) == (W != 0).sum(1)).all() and ((R != 0).sum(0) == (W != 0).sum(0)).all()
    assert np.isclose(np.abs(R).sum(), np.abs(W).sum()) and (R != W).any()
    assert np.isfinite(profile_adjacency(R, null_n=2, with_sweep=False)["Q10"])
    A = threshold_density(symmetrise(W), 0.1); U = rewire_undirected(A, seed=1)
    assert ((U > 0).sum(1) == (A > 0).sum(1)).all() and np.allclose(U, U.T) and (U != A).any()

def test_rnn_interface_and_ablation():
    task = HDTask(T=20); x, y = task.batch(8, 0); m = HDRNN(N=16, seed=0)
    logits = m(x); assert logits.shape == (8, 20, 16)
    mu = m.node_means(x); assert mu.shape == (16,)
    keep = torch.ones(16, dtype=torch.bool); keep[3] = False
    out = m(x, keep=keep, means=mu, record=True); assert torch.allclose(m._H[:, :, 3], mu[3].expand(8, 20))
    E, sc = m.edge_scores(x, y, mu); assert E.shape == (16, 16) and sc.shape == (16,)
    assert m.weight_graph().shape == (16, 16)
