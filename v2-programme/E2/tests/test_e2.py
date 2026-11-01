"""SPEC controls 4 (planted metrics), 5 (broken group), 6 (hand-built nets), plus the
quantity-level expectations of H-E2a/d/f on a hand-built net. Expected verdicts were written
before the trained-network run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from e2lib import check_invariance, audit_metrics, all_quantities, GROUPS, param_distance
from e2lib.core import sample_orbit, cv_of, agreement_of
from e2lib.quantities import gate_agreement, scaling_sweep, fit_exponent, box_inputs
from e1lib.models import MLP
from e1lib.functions import truth_table

X3, y3 = truth_table("XOR3")


def hand_xor3(second_copy=None, scale_copy=3.0):
    """E1's hand-built XOR3 (units ReLU(s), ReLU(s-1), ReLU(s-2), out = h0-2h1+2h2-0.5) with an
    identity pass-through second layer; optional duplicated rescaled copy of the sub-circuit."""
    W1 = np.ones((3, 3)); b1 = -np.array([0., 1., 2.]); out1 = np.array([1., -2., 2.])
    if second_copy == "same":
        W1 = np.concatenate([W1, W1 * scale_copy], 1); b1 = np.concatenate([b1, b1 * scale_copy])
        out = np.concatenate([out1, out1 / scale_copy])
    else:
        out = out1
    k = W1.shape[1]
    return MLP([W1, np.eye(k), out.reshape(k, 1)], [b1, np.zeros(k), np.array([-0.5])])


def random_mlp(seed, n_in=3, widths=(6, 5)):
    rng = np.random.default_rng(seed)
    dims = [n_in] + list(widths) + [1]
    return MLP([rng.normal(size=(dims[i], dims[i + 1])) for i in range(len(dims) - 1)],
               [rng.normal(size=dims[i + 1]) for i in range(len(dims) - 1)])


# ---- control 4: planted metrics -------------------------------------------------------
@pytest.mark.parametrize("group", GROUPS)
def test_planted_invariant_metric_passes(group):
    m = random_mlp(0)
    rep = check_invariance(lambda mm: mm.logits(X3), m, n_reps=50, group=group, name="logits")
    assert rep.verdict == "exact", rep
    assert rep.cv < 1e-10


@pytest.mark.parametrize("group", GROUPS)
def test_planted_non_invariant_metric_fails(group):
    m = random_mlp(1)
    rep = check_invariance(lambda mm: mm.Ws[0][0, 0], m, n_reps=50, group=group, name="W0[0,0]")
    assert rep.verdict == "not_invariant", rep


def test_orbit_actually_moves_parameters():
    m = random_mlp(2)
    for group in GROUPS:
        d = [dd for _, _, _, _, dd in sample_orbit(m, 30, group)][1:]
        assert np.median(d) > 0.1, (group, np.median(d))


# ---- control 5: a non-symmetry must be detected -------------------------------------
def test_broken_group_is_detected():
    m = random_mlp(3)
    rng = np.random.default_rng(0)

    def broken(mm):  # permute layer-0 units WITHOUT permuting layer-1 rows
        p = rng.permutation(mm.widths[0])
        c = mm.copy(); c.Ws[0] = c.Ws[0][:, p]; c.bs[0] = c.bs[0][p]
        return c.logits(X3)
    vals = [m.logits(X3)] + [broken(m) for _ in range(30)]
    assert cv_of(np.stack(vals)) > 0.05


# ---- control 6: hand-built nets ---------------------------------------------------------
def test_hand_xor3_sanity_quantities_exact():
    m = hand_xor3()
    assert (m.predict(X3) == y3).all()
    Xb = box_inputs(3, 0)
    for group in GROUPS:
        res, dists, base = audit_metrics(lambda mm: all_quantities(mm, X3, Xb), m, 40, group, seed=1)
        for k in ("logits_tt", "input_grad_tt", "W_product"):
            assert res[k]["cv"] < 1e-10, (group, k, res[k])
        # H-E2a: everything expected-invariant is exact under perm
        if group == "perm":
            for k in ("sv_layer", "tc0_tt", "tc1_tt", "b0_tt", "tc0_box", "tc1_box", "pe0_box",
                      "act_mag_sorted", "gradxact_sorted", "ablation_sorted", "in_norm_sorted", "gates_sorted"):
                assert res[k]["cv"] < 1e-10, (k, res[k])
            assert res["act_mag"]["cv"] > 0.05      # relabelled
        if group == "scale":
            assert res["sv_layer"]["cv"] > 0.05     # H-E2c
            assert res["act_mag"]["cv"] > 0.05      # N1 broken by scaling even identity-tracked
            assert res["gradxact"]["cv"] < 1e-10    # H-E2f: alpha cancels
            assert res["ablation"]["cv"] < 1e-10
            assert res["gates"]["agreement"] == 1.0


def test_duplicated_subcircuit_changes_tc_but_not_function():
    a, b = hand_xor3(), hand_xor3("same")
    assert (a.predict(X3) == b.predict(X3)).all()
    qa, qb = all_quantities(a, X3, box_inputs(3, 0)), all_quantities(b, X3, box_inputs(3, 0))
    assert np.array_equal(np.sign(qa["logits_tt"]), np.sign(qb["logits_tt"]))  # copy doubles the logit, f unchanged
    assert not np.allclose(qa["tc0_tt"][0], qb["tc0_tt"][0])   # TC sees the extra coordinates


def test_tc_uniform_scaling_exponent_two():
    m = random_mlp(4)
    rows = scaling_sweep(m, X3, box_inputs(3, 1))
    l0 = [r for r in rows if r["layer"] == 0]
    k, r2 = fit_exponent([r["alpha"] for r in l0], [r["tc0_box"] for r in l0])
    assert abs(k - 2.0) < 1e-6 and r2 > 0.9999
    kb, _ = fit_exponent([r["alpha"] for r in l0], [r["b0_box"] for r in l0])
    assert abs(kb) < 1e-9


def test_gate_agreement_under_permutation_is_low_for_distinct_gates():
    m = hand_xor3()   # three distinct gates in layer 0, identity layer 1
    vals = []
    for _, mm, _, _, _ in sample_orbit(m, 60, "perm", seed=2):
        vals.append(all_quantities(mm, X3, with_topology=False)["gates"])
    ag = gate_agreement(vals)
    assert ag < 0.5
    # identity-tracked (scale group) it is 1
    vals = [all_quantities(mm, X3, with_topology=False)["gates"] for _, mm, _, _, _ in sample_orbit(m, 30, "scale")]
    assert gate_agreement(vals) == 1.0


def test_cv_definition():
    assert cv_of(np.array([1.0, 1.0, 1.0])) == 0.0
    assert cv_of(np.zeros((5, 3))) == 0.0
    assert cv_of(np.array([-1.0, 1.0])) == float("inf")
    assert abs(cv_of(np.array([[1.0, 0.0], [0.0, 1.0]])) - 1.0) < 1e-12
    assert agreement_of([np.array([1, 2]), np.array([1, 2]), np.array([2, 1])]) == pytest.approx(2 / 3)
