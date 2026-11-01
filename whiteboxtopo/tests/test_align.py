"""whiteboxtopo.align — cross-fiber circuit alignment (FINAL_SPEC 2.9, 8.3)."""
import numpy as np
import pytest

from whiteboxtopo.align import (GaugeMatch, MechanismCount, match_circuits, count_mechanisms,
                                relabel_kill, effective_circuit, residual, MATCH_TOL)
from whiteboxtopo.invariants.model import ReluMLP, apply_gauge

WIDTHS = (4, 8, 6, 3)          # n_in, hidden, hidden, n_out


def mlp(seed, widths=WIDTHS):
    r = np.random.default_rng(seed)
    Ws = [r.standard_normal((widths[i], widths[i + 1])) for i in range(len(widths) - 1)]
    bs = [r.standard_normal(widths[i + 1]) for i in range(len(widths) - 1)]
    return ReluMLP(Ws, bs)


def relabelled(model, seed=7):
    """A literal gauge relabelling: permute and positively rescale every hidden unit."""
    r = np.random.default_rng(seed)
    widths = model.widths
    perms = [r.permutation(w) for w in widths]
    scales = [np.exp(r.uniform(-1, 1, w)) for w in widths]
    return apply_gauge(model, perms, scales), perms


# ------------------------------------------------------------- match_circuits
def test_literal_relabelling_matches_exactly():
    A = mlp(1)
    B, _ = relabelled(A)
    g, r = match_circuits(A, None, B, None)
    assert isinstance(g, GaugeMatch)
    assert r < 1e-8, r
    assert g.matched and g.residual == r and len(g.perms) == len(A.widths)
    assert all(sorted(p.tolist()) == list(range(w)) for p, w in zip(g.perms, A.widths))
    d = g.to_dict()
    assert d["matched"] is True and d["size_signature"] == list(A.widths)


def test_independent_circuits_do_not_match():
    g, r = match_circuits(mlp(1), None, mlp(2), None)
    assert r > 0.1, r
    assert not g.matched and "residual" in g.reason


def test_tuple_and_subcircuit_inputs():
    A = mlp(3)
    C = [np.array([0, 2, 4, 6]), np.array([1, 3, 5])]
    g, r = match_circuits((A.Ws, A.bs), C, A, C)
    assert r < 1e-12 and g.size_signature == (4, 3)


def test_different_sizes_give_infinite_distance():
    g, r = match_circuits(mlp(1), None, mlp(2, (4, 7, 6, 3)), None)
    assert r == float("inf") and not g.matched and "sizes differ" in g.reason


def test_scale_gauge_fix_is_exact_under_pure_scaling():
    A = mlp(5)
    B = apply_gauge(A, None, [np.exp(np.linspace(-2, 2, w)) for w in A.widths])
    assert match_circuits(A, None, B, None)[1] < 1e-10


# ------------------------------------------------------- the planted kill (8.3)
@pytest.mark.parametrize("kind", ["perm", "scale", "combined"])
def test_relabel_kill(kind):
    out = relabel_kill(mlp(11), None, seed=2, kind=kind)
    assert out["passed"] is True
    assert out["residual"] < 1e-8 and out["K"] == 1


def test_relabel_kill_on_a_subcircuit():
    out = relabel_kill(mlp(12), [np.array([1, 3, 5]), np.array([0, 2])], seed=4)
    assert out["passed"] and out["residual"] < 1e-8


# ----------------------------------------------------------- count_mechanisms
def test_count_mechanisms_two_classes():
    """3 seeds: two literal relabellings of A and one independent B -> K = 2 (8.3 negative)."""
    A = mlp(21)
    A2, _ = relabelled(A, seed=9)
    B = mlp(22)
    mc = count_mechanisms([A, A2, B], [None, None, None], tau=0.1)
    assert isinstance(mc, MechanismCount)
    assert mc.K == 2 and mc.N == 3 and mc.K_over_N == pytest.approx(2 / 3)
    classes = sorted(sorted(s for s, _ in cl) for cl in mc.classes)
    assert classes == [[0, 1], [2]]
    assert mc.n_pairs == 3 and set(mc.to_dict()["residuals"]) == {"0:0|1:0", "0:0|2:0", "1:0|2:0"}


def test_count_mechanisms_reports_named_algorithms():
    A = mlp(31)
    A2, _ = relabelled(A, seed=3)
    B = mlp(32)
    circuits = [{"algorithm": "fourier"}, {"algorithm": "fourier"}, {"algorithm": "memorize"}]
    mc = count_mechanisms([A, A2, B], circuits, tau=0.1, labels=["s0", "s1", "s2"])
    assert mc.K == 2 and mc.K_alg == 2
    assert mc.algorithms == {"fourier": 2, "memorize": 1}
    assert all(isinstance(s, str) for cl in mc.classes for s, _ in cl)


def test_count_mechanisms_multiple_circuits_per_seed():
    A, B = mlp(41), mlp(42)
    CA = [np.array([0, 1, 2, 3]), np.array([0, 1, 2])]
    mc = count_mechanisms([A, B], [[{"units": CA}, {"units": CA}], [{"units": CA}]], tau=0.1)
    assert mc.N == 3 and mc.n_pairs == 2          # within-seed pair skipped (Proposition 2.1)
    mc2 = count_mechanisms([A, B], [[{"units": CA}, {"units": CA}], [{"units": CA}]],
                           tau=0.1, within_seed=True)
    assert mc2.n_pairs == 3 and mc2.K == 2        # the seed's two identical circuits merge


# ------------------------------------------------- cross-check vs the F block
def test_agrees_with_flib_crossseed():
    """The library's residual must reproduce experiments/F_gauge/flib/crossseed on a
    random case (flib's flattened output row assumes n_out == 1)."""
    from whiteboxtopo.align import _import_flib
    try:                                         # sys.path insertion happens inside the function
        flib = _import_flib()
    except Exception as e:                       # pragma: no cover - experiment tree absent
        pytest.skip(f"F_gauge experiment tree not importable: {e}")
    A, B = mlp(51, (4, 8, 6, 1)), mlp(52, (4, 8, 6, 1))

    def flib_effective(m):
        Ws, bs = flib._normalize([W.copy() for W in m.Ws], [b.copy() for b in m.bs])
        inc = [np.vstack([Ws[l], bs[l][None, :]]).T for l in range(len(Ws) - 1)]
        return inc, np.concatenate([Ws[-1].ravel(), bs[-1]])

    r_flib, _ = flib.residual(flib_effective(A), flib_effective(B))
    r_lib = match_circuits(A, None, B, None)[1]
    assert r_lib == pytest.approx(r_flib, abs=1e-10)
    # and both call a literal relabelling exact
    A2, _ = relabelled(A, seed=13)
    assert flib.residual(flib_effective(A), flib_effective(A2))[0] < 1e-8
    assert match_circuits(A, None, A2, None)[1] < 1e-8


def test_effective_circuit_and_residual_shapes():
    A = mlp(61)
    inc, out, norms, gain = effective_circuit(A)
    assert [x.shape for x in inc] == [(8, 5), (6, 9)]
    assert out[0].shape == (6, 3) and out[1].shape == (3,) and gain > 0
    assert all(np.allclose(np.linalg.norm(x, axis=1), 1.0) for x in inc)
    r, perms = residual((inc, out), (inc, out))
    assert r == pytest.approx(0.0, abs=1e-12) and len(perms) == 2


@pytest.mark.parametrize("tol", [MATCH_TOL, 1e-8])
def test_tol_is_honoured(tol):
    A = mlp(71)
    A2, _ = relabelled(A, seed=5)
    g, _ = match_circuits(A, None, A2, None, tol=tol)
    assert g.tol == tol and g.matched
