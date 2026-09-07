"""S1 tests: function identity under 1000 renamings x 10 models (MLP + toy transformer), and
the hidden-symmetry detector on planted symmetries."""
import numpy as np, pytest
from whiteboxtopo import symmetry as S
from whiteboxtopo.invariants import ReluMLP


def _mlp(rng, widths=(8, 6), n_in=3):
    dims = [n_in] + list(widths) + [1]
    return ReluMLP([rng.normal(size=(dims[i], dims[i + 1])) for i in range(len(dims) - 1)],
                   [rng.normal(size=dims[i + 1]) for i in range(len(dims) - 1)])


def test_mlp_identity_1000_renamings_10_models():
    rng = np.random.default_rng(0); worst = 0.0
    for i in range(10):
        m = _mlp(rng); X = rng.uniform(-1, 2, size=(64, 3))
        for r in range(100):
            for kind in ("perm", "scale", "combined"):
                g = S.random_element(m, rng, kind=kind)
                worst = max(worst, S.output_deviation(m, g, X))
    assert worst < 1e-6


def test_torch_sequential_and_numpy_agree():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1); m = _mlp(rng)
    net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 1)).double()
    with torch.no_grad():
        for L, W, b in zip([net[0], net[2], net[4]], m.Ws, m.bs):
            L.weight.copy_(torch.tensor(W.T)); L.bias.copy_(torch.tensor(b))
    X = rng.uniform(-1, 2, size=(32, 3))
    g = S.random_element(net, rng)
    assert S.output_deviation(net, g, X) < 1e-9
    assert type(S.apply(net, g)) is torch.nn.Sequential


def test_toy_transformer_identity_all_kinds():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2); worst = 0.0
    for i in range(3):
        tf = S.ToyTransformer(vocab=5, seq_len=6, d_model=32, n_heads=4, n_layers=2, seed=i).double()
        tok = rng.integers(0, 5, size=(8, 6))
        for kind in ("perm", "scale", "rot", "combined"):
            for r in range(25):
                g = S.random_element(tf, rng, kind=kind, orthogonal=(r % 2 == 0))
                worst = max(worst, S.output_deviation(tf, g, tok))
    assert worst < 1e-6


def test_describe_known_dim():
    rng = np.random.default_rng(3); m = _mlp(rng, (8, 6))
    assert S.describe(m)["known_dim"] == 14


def test_hidden_plain_net_has_no_excess():
    rng = np.random.default_rng(4); m = _mlp(rng, (6,))
    m.bs[0][:] = 0.0                                   # zero biases: every unit fires on some probe point
    rep = S.hidden(m, seed=0)
    assert rep.dead_units == [] and rep.corank == rep.known_dim == 6 and rep.excess == 0 and not rep.G_incomplete


def test_hidden_recovers_planted_duplicate_and_dead():
    rng = np.random.default_rng(5); m = _mlp(rng, (6,))
    Ws = [W.copy() for W in m.Ws]; bs = [b.copy() for b in m.bs]
    Ws[0][:, 1] = 2 * Ws[0][:, 0]; bs[0][1] = 2 * bs[0][0]            # planted duplicate unit
    rep = S.hidden(ReluMLP(Ws, bs))
    assert rep.G_incomplete and rep.duplicate_pairs == [(0, 0, 1)] and rep.unexplained == 0
    Ws = [W.copy() for W in m.Ws]; bs = [b.copy() for b in m.bs]; bs[0][2] = -50.0   # dead unit
    rep = S.hidden(ReluMLP(Ws, bs))
    assert rep.G_incomplete and (0, 2) in rep.dead_units and rep.unexplained == 0


def test_inverse_on_units_undoes_renaming():
    rng = np.random.default_rng(6); m = _mlp(rng, (8,)); X = rng.uniform(-1, 2, size=(50, 3))
    g = S.random_element(m, rng, kind="combined")
    v = S.apply(m, g).hidden_acts(X)[0].mean(0)
    back = S.inverse_on_units(g, 0, v, axis=0, scale_power=1.0)
    assert np.allclose(back, m.hidden_acts(X)[0].mean(0))
