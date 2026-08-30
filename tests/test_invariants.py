"""whitebox.invariants — every shipped invariant re-checked by orbit sampling
(permutations x positive scalings) on hand-built, random and trained ReLU MLPs,
in numpy and torch form; plus the audit tool's controls (planted invariant
passes, planted non-invariant fails, non-symmetry is detected)."""
import numpy as np, pytest, warnings
from whitebox.invariants import (ReluMLP, as_relu_mlp, apply_gauge, sample_orbit, check_invariance, audit_metrics,
                                 cv_of, agreement_of, GROUPS, FLOAT_FLOOR, INVARIANTS, invariant_quantities,
                                 logits, standardized_tc, gate_multiset, sorted_ablation, gradxact, ablation, gate_bits,
                                 standardize_units)
from whitebox.invariants.metrics import _sorted_per_layer

X3 = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)
XBOX = np.random.default_rng(0).uniform(-1, 2, size=(120, 3))


def hand_xor3():
    """E1's hand-built XOR3: ReLU(s), ReLU(s-1), ReLU(s-2), out = h0 - 2 h1 + 2 h2 - 0.5, identity 2nd layer."""
    W1 = np.ones((3, 3)); b1 = -np.array([0., 1., 2.]); out = np.array([1., -2., 2.]).reshape(3, 1)
    return ReluMLP([W1, np.eye(3), out], [b1, np.zeros(3), np.array([-0.5])])


def random_mlp(seed, n_in=3, widths=(6, 5), n_out=1):
    rng = np.random.default_rng(seed); dims = [n_in] + list(widths) + [n_out]
    return ReluMLP([rng.normal(size=(dims[i], dims[i + 1])) for i in range(len(dims) - 1)],
                   [rng.normal(size=dims[i + 1]) for i in range(len(dims) - 1)])


class E1LikeMLP:
    """Duck-typed stand-in for the E1/E2 numpy MLP (.Ws, .bs, .copy())."""
    def __init__(self, Ws, bs): self.Ws, self.bs = [np.asarray(W) for W in Ws], [np.asarray(b) for b in bs]
    def copy(self): return E1LikeMLP([W.copy() for W in self.Ws], [b.copy() for b in self.bs])
    def logits(self, X):
        h = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]): h = np.maximum(h @ W + b, 0)
        return (h @ self.Ws[-1] + self.bs[-1])[:, 0]


@pytest.fixture(scope="module")
def trained_torch():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 1)).double()
    X = torch.tensor(X3); y = torch.tensor((X3.sum(1) % 2)[:, None])
    opt = torch.optim.Adam(net.parameters(), 3e-2)
    for _ in range(400):
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X), y); loss.backward(); opt.step()
    assert ((net(X) > 0).double() == y).all(), "toy XOR3 did not train"
    return net


# ---------------------------------------------------------------- the shipped invariants
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("name", list(INVARIANTS))
def test_shipped_invariant_is_exact_on_random_and_hand_built(name, group):
    _, f = INVARIANTS[name]
    for m in (hand_xor3(), random_mlp(1), random_mlp(2, widths=(4,), n_out=3)):
        Xp = XBOX if name == "standardized_tc" else X3
        rep = check_invariance(lambda mm: f(mm, Xp), m, n_reps=40, group=group, name=name, diagnose=False)
        assert rep.cv <= FLOAT_FLOOR and rep.verdict == "exact", (name, group, rep)
        assert rep.param_distance_median > 0.1        # the orbit really moved the parameters


@pytest.mark.parametrize("name", list(INVARIANTS))
def test_shipped_invariant_is_exact_on_trained_torch(name, trained_torch):
    _, f = INVARIANTS[name]
    Xp = XBOX if name == "standardized_tc" else X3
    rep = check_invariance(lambda mm: f(mm, Xp), trained_torch, n_reps=40, group="combined", name=name)
    assert rep.cv <= FLOAT_FLOOR and rep.broken_by == "none", (name, rep)


def test_invariants_are_not_vacuous():
    """Between-network dispersion is large: the invariants vary across nets (E2 vacuity control)."""
    q = [invariant_quantities(random_mlp(s), XBOX) for s in range(6)]
    for k in q[0]:
        if k == "gate_multiset": continue
        assert cv_of(np.stack([qq[k].ravel() for qq in q])) > 0.05, k


# ---------------------------------------------------------------- audit-tool controls
@pytest.mark.parametrize("group", GROUPS)
def test_planted_invariant_metric_passes(group):
    rep = check_invariance(lambda mm: mm.logits(X3), random_mlp(0), n_reps=50, group=group, name="logits")
    assert rep.verdict == "exact" and rep.cv < 1e-10 and rep.broken_by == "none"


@pytest.mark.parametrize("group", GROUPS)
def test_planted_non_invariant_metric_fails(group):
    rep = check_invariance(lambda mm: mm.Ws[0][0, 0], random_mlp(1), n_reps=50, group=group, name="W0[0,0]")
    assert rep.verdict == "not_invariant" and rep.broken_by != "none"


def test_diagnosis_names_the_symmetry():
    m = random_mlp(3)
    assert check_invariance(lambda mm: gradxact(mm, X3), m, 40).broken_by == "perm"          # values ok, labels not
    assert check_invariance(lambda mm: ablation(mm, X3), m, 40).broken_by == "perm"
    assert check_invariance(lambda mm: np.abs(mm.hidden_acts(X3)[0]).mean(0), m, 40).broken_by == "perm+scale"
    # raw TC of the raw cloud: permutation-exact, broken by scaling (E2 headline)
    from whitebox import compute_diagrams, topological_complexity
    raw_tc = lambda mm: topological_complexity(compute_diagrams(mm.hidden_acts(XBOX)[0], None, backend="ripser")[0])
    rep = check_invariance(raw_tc, m, 40)
    assert rep.broken_by == "scale" and rep.per_group["perm"] < 1e-10 and rep.per_group["scale"] > 0.05


def test_broken_group_is_detected():
    """A layer-0 permutation NOT propagated to layer 1 is not a symmetry: the logits move."""
    m = random_mlp(4); rng = np.random.default_rng(0)
    def broken(mm):
        p = rng.permutation(mm.widths[0]); c = mm.copy(); c.Ws[0] = c.Ws[0][:, p]; c.bs[0] = c.bs[0][p]; return c.logits(X3)
    assert cv_of(np.stack([m.logits(X3)] + [broken(m) for _ in range(30)])) > 0.05
    # and a correct apply_gauge leaves the function unchanged for every group
    for group in GROUPS:
        for _, mm, _, _, d in sample_orbit(m, 10, group):
            assert np.allclose(mm.logits(XBOX), m.logits(XBOX), atol=1e-9)


def test_apply_gauge_rejects_bad_elements():
    m = random_mlp(5)
    with pytest.raises(ValueError): apply_gauge(m, scales=[-np.ones(6), np.ones(5)])
    with pytest.raises(ValueError): apply_gauge(m, perms=[np.zeros(6, int), np.arange(5)])


def test_audit_metrics_one_pass():
    res, dists, base = audit_metrics(lambda mm: invariant_quantities(mm, X3, with_topology=False), hand_xor3(), 30, "combined")
    assert all(r["verdict"] == "exact" for r in res.values()) and np.median(dists[1:]) > 0.5


def test_cv_and_agreement_definitions():
    assert cv_of(np.array([1.0, 1.0, 1.0])) == 0.0 and cv_of(np.zeros((5, 3))) == 0.0
    assert cv_of(np.array([-1.0, 1.0])) == float("inf")
    assert abs(cv_of(np.array([[1.0, 0.0], [0.0, 1.0]])) - 1.0) < 1e-12
    assert agreement_of([np.array([1, 2]), np.array([1, 2]), np.array([2, 1])]) == pytest.approx(2 / 3)


# ---------------------------------------------------------------- model adapters
def test_e1_style_numpy_model_and_torch_agree():
    torch = pytest.importorskip("torch")
    m = random_mlp(6)
    e1 = E1LikeMLP(m.Ws, m.bs)
    seq = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(), torch.nn.Linear(6, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)).double()
    with torch.no_grad():
        for L, W, b in zip((seq[0], seq[2], seq[4]), m.Ws, m.bs): L.weight.copy_(torch.tensor(W.T)); L.bias.copy_(torch.tensor(b))
    assert np.allclose(as_relu_mlp(seq).logits(XBOX), m.logits(XBOX)) and np.allclose(as_relu_mlp(e1).logits(XBOX), m.logits(XBOX))
    # representatives keep the input's type and function
    for model, typ in ((e1, E1LikeMLP), (seq, torch.nn.Sequential)):
        for _, mm, _, _, _ in sample_orbit(model, 5, "combined"):
            assert isinstance(mm, typ)
        rep = check_invariance(lambda mm: logits(mm, XBOX), model, 20, diagnose=False); assert rep.verdict == "exact"
    assert isinstance(next(sample_orbit(seq, 2))[1], torch.nn.Sequential)
    with pytest.raises(ValueError):
        as_relu_mlp(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)))
    with pytest.raises(TypeError):
        as_relu_mlp(object())


def test_float32_torch_warns():
    torch = pytest.importorskip("torch")
    seq = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    with pytest.warns(UserWarning):
        check_invariance(lambda mm: logits(mm, X3), seq, 5, diagnose=False)


# ---------------------------------------------------------------- semantics of the metrics
def test_hand_xor3_gates_and_standardisation():
    m = hand_xor3()
    assert ((m.logits(X3)[:, 0] > 0).astype(int) == X3.sum(1) % 2).all()
    B = gate_bits(m, X3); assert B.shape == (8, 6)
    G = gate_multiset(m, X3)
    # the multiset of gates is stable under any permutation, the raw index is not
    perm = apply_gauge(m, perms=[np.array([2, 0, 1]), np.arange(3)])
    assert np.array_equal(gate_multiset(perm, X3), G) and not np.array_equal(gate_bits(perm, X3), B)
    Z = standardize_units(np.c_[XBOX, np.zeros(len(XBOX))])       # dead unit dropped, columns z-scored
    assert Z.shape == (len(XBOX), 3) and np.allclose(Z.mean(0), 0) and np.allclose(Z.std(0), 1)
    tc = standardized_tc(m, XBOX); assert tc.shape == (2, 2) and (tc >= 0).all()
    assert _sorted_per_layer(np.array([3., 1., 2., 9., 8.]), [3, 2]).tolist() == [1., 2., 3., 8., 9.]
