"""whiteboxtopo.legibility — graph measures on planted graphs, discovery on a planted circuit,
the profile end-to-end, and the refusal of legibility_score."""
import numpy as np, pytest
pytest.importorskip("igraph"); pytest.importorskip("leidenalg"); pytest.importorskip("networkx")
torch = pytest.importorskip("torch")
from whiteboxtopo.legibility import (modularity_q, treewidth_bound, threshold_density, symmetrise, sparsity, shuffled_q_null,
                                 modularity_sweep, discover_circuit, jaccard, NodeMLP, legibility_profile, legibility_score,
                                 NotCalibratedError)


def test_planted_modular_orders_above_random():
    rng = np.random.default_rng(0); N = 60; lab = np.arange(N) // 15
    A_mod = symmetrise((lab[:, None] == lab[None, :]) * (rng.random((N, N)) < 0.6))
    A_rnd = symmetrise(rng.random((N, N)) < 0.15)
    q_mod, memb = modularity_q(A_mod); q_rnd, _ = modularity_q(A_rnd)
    assert q_mod > q_rnd + 0.2 and q_mod > 0.6
    assert len(set(memb)) == 4 and all(len(set(memb[lab == c])) == 1 for c in range(4))
    # shuffle null of a planted-modular WEIGHTED graph is below the real Q (weights placed on the block slots)
    E = A_mod * rng.uniform(0.5, 1.5, (N, N)); q, _ = modularity_q(threshold_density(symmetrise(E), 0.5))
    qn, sd = shuffled_q_null(E, 0.5, n=5); assert q >= qn - 3 * sd - 0.05


def test_treewidth_known_graphs():
    nx = pytest.importorskip("networkx")
    assert treewidth_bound(nx.to_numpy_array(nx.balanced_tree(2, 4))) == 1
    assert treewidth_bound(nx.to_numpy_array(nx.cycle_graph(10))) == 2
    assert treewidth_bound(nx.to_numpy_array(nx.complete_graph(7))) == 6
    assert 4 <= treewidth_bound(nx.to_numpy_array(nx.grid_2d_graph(4, 4))) <= 5     # exact 4; min-fill is an upper bound
    assert treewidth_bound(np.zeros((5, 5))) == 0


def test_threshold_density_and_sparsity():
    rng = np.random.default_rng(1); A = symmetrise(rng.random((30, 30)))
    B = threshold_density(A, 0.1); n_all = (np.triu(A, 1) > 0).sum(); n_keep = (np.triu(B, 1) > 0).sum()
    assert abs(n_keep - round(0.1 * n_all)) <= 1
    assert sparsity(np.ones(10)) == 0.0 and sparsity(np.eye(10)[0]) == pytest.approx(0.9) and sparsity(np.zeros(3)) == 1.0
    sw = modularity_sweep(rng.random((20, 20)), densities=(0.1, 1.0), with_tw=False)
    assert [r["density"] for r in sw] == [0.1, 1.0] and all(0 <= r["Q"] <= 1 for r in sw)


def _planted(width=24, k=6):
    """NodeMLP where only k planted neurons per layer carry the function; the rest are junk."""
    torch.manual_seed(0); m = NodeMLP(in_dim=k, width=width, n_classes=k)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        m.l1.weight[k:, :] = torch.randn(width - k, k) * 0.5
        m.l2.weight[k:, k:] = torch.randn(width - k, width - k) * 0.5
        for i in range(k): m.l1.weight[i, i] = 4.0; m.l2.weight[i, i] = 1.0; m.l3.weight[i, i] = 6.0
    x = torch.eye(k)[torch.randint(0, k, (256,))]; t = x.argmax(1)
    return m, x, t, set(range(k)) | {width + i for i in range(k)}


def test_discovery_recovers_planted_circuit():
    m, x, t, planted = _planted()
    means = m.node_means(x)
    res = discover_circuit(m, x, t, means, seed=0)
    found = set(np.flatnonzero(res["keep"].numpy()))
    assert res["F"] >= 0.9 and found <= planted and len(found) >= 5
    res2 = discover_circuit(m, x, t, means, seed=1)
    assert jaccard(res["keep"], res2["keep"]) >= 0.6 and res["cost"] <= 3 * m.n_nodes


def test_profile_end_to_end_and_guard():
    m, x, t, planted = _planted()
    prof = legibility_profile(m, (x, t), discovery_batches=[(x[:128], t[:128]), (x[128:], t[128:])], null_n=3, with_sweep=True)
    assert prof["N"] == 48 and 0 <= prof["Q10"] <= 1 and prof["tw10"] >= 0 and 0 <= prof["sparsity"] <= 1
    assert prof["learned"] and prof["accuracy"] == 1.0 and prof["S"] == prof["S_raw"] > 0.5 and prof["F"] >= 0.9
    assert len(prof["sweep"]) == 7 and "Q_auc" in prof and 0 <= prof["jaccard"] <= 1
    # an untrained model is guarded: S = 0 whatever S_raw says (E4 kill 3)
    u = NodeMLP(in_dim=6, width=24, n_classes=6, seed=3)
    pu = legibility_profile(u, (x, t), discovery_batches=[(x[:64], t[:64])], null_n=2, with_sweep=False)
    assert (not pu["learned"]) and pu["S"] == 0.0


def test_legibility_score_refuses():
    with pytest.raises(NotCalibratedError, match="not calibrated"):
        legibility_score()
    with pytest.raises(NotImplementedError):
        legibility_score(model=None, task_batch=None)


def test_tinytf_edge_scores_chunked_equals_full():
    """V AMENDMENTS #10: chunked EAP edge scores (memory budget) equal the one-shot result."""
    import torch
    from whiteboxtopo.legibility.tasks import Task
    from whiteboxtopo.legibility.e4models import TinyTF
    task = Task("sort"); m = TinyTF(task, n_layers=2, d=32, heads=4, mlp=64, seed=0).eval()
    x, y = task.batch(64, 0, "eval"); means = m.node_means(x)
    E_full, n_full = m._edge_scores_full(x, y, means)
    old = TinyTF.EDGE_BUDGET_BYTES
    try:
        TinyTF.EDGE_BUDGET_BYTES = 2 * m.nL * m.per * x.shape[1] * m.d * 4 * 7   # -> chunks of 7 batch elements
        E_ch, n_ch = m.edge_scores(x, y, means)
    finally:
        TinyTF.EDGE_BUDGET_BYTES = old
    assert torch.allclose(E_full, E_ch, rtol=1e-4, atol=1e-6 * E_full.abs().max()) and torch.allclose(n_full, n_ch, rtol=1e-4, atol=1e-6)
