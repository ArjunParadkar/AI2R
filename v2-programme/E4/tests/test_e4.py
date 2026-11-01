import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from e4lib.graph import modularity_q, treewidth_bound, threshold_density, symmetrise, sparsity
from e4lib.discovery import discover_circuit, jaccard
from e4lib.tasks import Task
from e4lib.models import build_model
from e4lib.cluster_loss import clusterability, block_mask_
from e4lib.legibility import legibility_profile

def test_planted_modular_orders_above_random():
    rng = np.random.default_rng(0); N = 60
    lab = np.arange(N) // 15
    A_mod = ((lab[:, None] == lab[None, :]) * (rng.random((N, N)) < 0.6)).astype(float)
    A_rnd = (rng.random((N, N)) < 0.15).astype(float)
    A_mod, A_rnd = symmetrise(A_mod), symmetrise(A_rnd)
    q_mod, memb = modularity_q(A_mod); q_rnd, _ = modularity_q(A_rnd)
    assert q_mod > q_rnd + 0.2
    assert q_mod > 0.6
    # planted partition recovered (4 communities, matching labels up to relabel)
    assert len(set(memb)) == 4 and all(len(set(memb[lab == c])) == 1 for c in range(4))

def test_treewidth_known_graphs():
    import networkx as nx
    tree = nx.to_numpy_array(nx.balanced_tree(2, 4)); assert treewidth_bound(tree) == 1
    cyc = nx.to_numpy_array(nx.cycle_graph(10)); assert treewidth_bound(cyc) == 2
    K = nx.to_numpy_array(nx.complete_graph(7)); assert treewidth_bound(K) == 6
    grid = nx.to_numpy_array(nx.grid_2d_graph(4, 4)); tw = treewidth_bound(grid); assert 4 <= tw <= 5  # exact 4; min-fill is an upper bound

def test_threshold_density_keeps_top_fraction():
    rng = np.random.default_rng(1); A = symmetrise(rng.random((30, 30)))
    B = threshold_density(A, 0.1); n_all = (np.triu(A, 1) > 0).sum(); n_keep = (np.triu(B, 1) > 0).sum()
    assert abs(n_keep - round(0.1 * n_all)) <= 1
    assert np.triu(B, 1)[np.triu(B, 1) > 0].min() >= np.sort(np.triu(A, 1)[np.triu(A, 1) > 0])[::-1][n_keep]

def test_sparsity_extremes():
    assert sparsity(np.ones(10)) == 0.0
    assert sparsity(np.eye(10)[0]) == 0.9

def test_clusterability_block_mask():
    W = torch.randn(16, 16); assert 0.1 < clusterability(W).item() < 0.5
    task = Task("modadd"); m = build_model("mlp", task); block_mask_(m)
    assert clusterability(m.l2.weight).item() > 0.999

def test_discovery_recovers_planted_circuit():
    """MLP where only 6 planted neurons carry the function; the rest are pure noise units."""
    torch.manual_seed(0); task = Task("copy"); m = build_model("mlp", task); W = m.width
    with torch.no_grad():
        # kill everything, then plant: layer-1 neurons 0..5 copy 6 input bits, layer-2 neurons 0..5 pass them, out reads them
        for p in m.parameters(): p.zero_()
        # random junk neurons with no downstream effect (weights into them but zero out of them)
        m.l1.weight[6:, :] = torch.randn(W - 6, m.in_dim) * 0.5
        m.l2.weight[6:, 6:] = torch.randn(W - 6, W - 6) * 0.5
        # planted: slot 0 digit bits (one-hot 8 -> we use 3 binary features? simpler: 6 one-hot inputs of slot 0)
        for i in range(6): m.l1.weight[i, i] = 4.0                          # neuron i fires on input token i at slot 0
        for i in range(6): m.l2.weight[i, i] = 1.0
        for i in range(6): m.l3.weight[i, i] = 6.0                          # slot-0 class i logit
    tokens, targets = task.batch(256, 0)
    sel = tokens[:, 0] < 6; tokens, targets = tokens[sel], targets[sel]     # inputs the planted circuit handles
    means = m.node_means(tokens)
    res = discover_circuit(m, tokens, targets, means, seed=0)
    planted = set(list(range(6)) + [W + i for i in range(6)])
    found = set(np.flatnonzero(res["keep"].numpy()))
    assert res["F"] >= 0.9
    assert found <= planted and len(found) >= 5           # nothing outside the planted circuit; nearly all of it
    res2 = discover_circuit(m, tokens, targets, means, seed=1)
    # six equal planted paths at F>=0.9: one may be dropped, which one is seed-dependent -> Jaccard ~ (k-2)/k
    assert jaccard(res["keep"], res2["keep"]) >= 0.6

def test_legibility_profile_runs_on_transformer():
    task = Task("modadd"); m = build_model("tf2", task); m.eval()
    prof = legibility_profile(m, task.batch(64, 0), with_sweep=False, null_n=2)
    assert m.n_nodes == 2 * (4 + 64) and 0 <= prof["Q10"] <= 1 and prof["tw10"] >= 0 and 0 <= prof["sparsity"] <= 1
