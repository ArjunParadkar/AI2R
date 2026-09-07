"""S4 tests: attribution graph + modularity/treewidth/sparsity on a node MLP; find_circuit is
deterministic given seed and returns a faithful circuit on a planted block-structured graph."""
import numpy as np, pytest
torch = pytest.importorskip("torch")
from whiteboxtopo import graphs, discovery, nulls
from whiteboxtopo.legibility.models import NodeMLP


def _data(seed=0, n=256):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)); y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)


def _train(seed=0):
    x, y = _data(seed); m = NodeMLP(6, width=16, seed=seed)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(300):
        opt.zero_grad(); torch.nn.functional.cross_entropy(m(x), y).backward(); opt.step()
    return m, (x, y)


def test_attribution_graph_and_measures():
    m, task = _train()
    A = graphs.attribution_graph(m, task, threshold=0.2)
    assert A.shape == (32, 32) and np.allclose(A, A.T)
    Q, memb = graphs.modularity(A); assert -0.5 <= Q <= 1 and len(memb) == 32
    assert 0 <= graphs.treewidth_ub(A) < 32
    assert 0 <= graphs.sparsity(A) <= 1


def test_modularity_recovers_planted_blocks():
    A, truth = nulls.planted("blocks", 80, k=4, seed=0)
    Q, memb = graphs.modularity(A)
    assert Q > 0.4
    q0, sd = graphs.rewired_null(A, n=5)
    assert Q > q0 + 3 * max(sd, 1e-3)


def test_find_circuit_deterministic_and_faithful():
    m, task = _train()
    r1 = discovery.find_circuit(m, task, faithfulness=0.9, seed=0)
    r2 = discovery.find_circuit(m, task, faithfulness=0.9, seed=0)
    assert np.array_equal(r1["circuit"], r2["circuit"]) and r1["size"] == r2["size"]
    assert r1["faithfulness"] >= 0.9 and r1["size"] <= 32
